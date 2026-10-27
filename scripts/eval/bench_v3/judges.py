"""Shared library for the bench_v3 judges.

Two judges, both of which always see the SOURCE video next to the output -- the
original reward saw two output frames and nothing else, so it could not notice an
edit that replaced the person or ignored the request:

* checklist judge -- per (arm, sample): is each requested change done, is each
  preserve item kept, and what changed that nobody asked for. "Did it change the
  right things" (fidelity) and "did it keep what must stay" (preservation) are
  scored separately and never merged.
* pairwise judge -- per (sample, A, B): three separate verdicts (fidelity to the
  requirement, preservation, visual quality), each asked in both slot orders; an
  order disagreement is a tie, byte-identical videos are a tie without a call.

Frames are sampled at the same timestamps in every video. The checklist judge
spreads them over the output clip's own duration, so a 17-frame export is compared
with the first 0.85 s of its source rather than with frames it never covered. The
pairwise judge spreads them over the source clip; both outputs must therefore
cover the whole source (true for every native-duration arm here), otherwise the
shorter one repeats its last frame.

The LLM client is anything with ``complete(system_prompt=, user_text=, images_b64=)``
returning text; ``LiteLLMJudge`` from ditto_quant is the real one, tests use a fake.
Images are sent after the text in one message, so the text enumerates them in order.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[3]
DITTO_QUANT = ROOT / "scripts/eval/ditto_quant"

import sys  # noqa: E402

if str(DITTO_QUANT) not in sys.path:
    sys.path.insert(0, str(DITTO_QUANT))

from evaluate_aesthetic_reward_litellm import _extract_json_object  # noqa: E402

CHANGE_LABELS = {"done": 1.0, "partial": 0.5, "not_done": 0.0}
PRESERVE_LABELS = {"kept", "changed"}
QUESTIONS = ("fidelity_to_requirement", "preservation", "visual_quality")
VERDICTS = {"A", "B", "tie"}
DEFAULT_PRESERVE = [
    "facial identity of every person",
    "facial expressions",
    "body pose and action",
    "composition and subject placement",
    "camera framing and motion",
]


class Client(Protocol):
    def complete(self, *, system_prompt: str, user_text: str, images_b64: list[str]) -> str: ...


# --------------------------------------------------------------------------- io


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--arm must be name=dir, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def make_client(args: Any) -> Any:
    """LiteLLMJudge from CLI args; the key comes only from the environment."""
    from evaluate_aesthetic_reward_litellm import LiteLLMJudge

    api_key = os.environ.get("LOW_HIGH_JUDGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("set LOW_HIGH_JUDGE_API_KEY")
    return LiteLLMJudge(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        temperature=0.0,
        max_completion_tokens=args.max_completion_tokens,
    )


def add_client_args(parser: Any) -> None:
    parser.add_argument("--model", default=os.environ.get("LOW_HIGH_JUDGE_MODEL", "gemini-3.8-flash"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("LOW_HIGH_JUDGE_REASONING_EFFORT", "high"))
    parser.add_argument("--base-url", default=os.environ.get("LOW_HIGH_JUDGE_BASE_URL", "http://litellm-sgp.woa.com"))
    parser.add_argument("--max-completion-tokens", type=int, default=12000)


# ----------------------------------------------------------------------- frames


def _probe(path: Path) -> tuple[int, float]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else 16.0
        count = sum(1 for _ in container.decode(video=0))
    return count, fps


def _decode_at(path: Path, indices: list[int]) -> list[Any]:
    import av

    wanted = set(indices)
    last = max(indices)
    got: dict[int, Any] = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                got[index] = frame.to_image().convert("RGB")
            if index >= last:
                break
    # A short clip may end before a requested index; reuse its last frame.
    fallback = got[max(got)] if got else None
    return [got.get(i, fallback) for i in indices]


def aligned_frames(anchor: Path, others: list[Path], k: int) -> tuple[list[float], list[list[Any]]]:
    """K timestamps across ``anchor``'s duration; the nearest frame of each video.

    Returns (times, frames) where frames[0] is the anchor's and frames[i+1] is
    ``others[i]``'s. Same rounding rule for every video, as in gt_metrics.py.
    """
    n_a, fps_a = _probe(anchor)
    duration = n_a / fps_a
    times = [(j + 0.5) / k * duration for j in range(k)]
    out = []
    for path in [anchor, *others]:
        n, fps = (n_a, fps_a) if path == anchor else _probe(path)
        out.append(_decode_at(path, [min(n - 1, int(t * fps)) for t in times]))
    return times, out


def reference_frame(path: Path) -> Any:
    """A reference image; for a video, its middle frame."""
    from PIL import Image

    if path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
        n, _ = _probe(path)
        return _decode_at(path, [n // 2])[0]
    return Image.open(path).convert("RGB")


def to_b64(image: Any, width: int = 640, quality: int = 85) -> str:
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- parsing


class JudgeOutputError(ValueError):
    pass


def parse_json(text: str) -> dict[str, Any]:
    """JSON object from model text: tolerates fences, <think> blocks, prose around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    try:
        return _extract_json_object(stripped)
    except (ValueError, json.JSONDecodeError) as exc:
        raise JudgeOutputError(str(exc)) from exc


def call_with_retry(client: Client, system: str, user: str, images: list[str],
                    validate: Callable[[dict], dict]) -> tuple[dict, int]:
    """One retry on unparsable / invalid output, telling the model what was wrong."""
    last_error = ""
    for attempt in range(2):
        text = user if attempt == 0 else (
            user + f"\n\nYour previous answer was invalid ({last_error}). Return only the JSON object, exactly as specified."
        )
        raw = client.complete(system_prompt=system, user_text=text, images_b64=images)
        try:
            return validate(parse_json(raw)), attempt + 1
        except (JudgeOutputError, KeyError, TypeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:300]
    raise JudgeOutputError(last_error)


# -------------------------------------------------------------------- checklist

CHECKLIST_SYSTEM = """You are a strict video-edit auditor.
You compare SOURCE frames (the video before editing) with OUTPUT frames (after editing) taken at the same timestamps.
Judge only what is visible in the frames. Do not reward effort or intent, only visible results.
Return strict JSON only."""


def checklist_prompt(*, requirement: str, times: list[float], n_refs: int,
                     change: list[str], preserve: list[str]) -> str:
    order = []
    for t in times:
        order.append(f"SOURCE t={t:.2f}s")
        order.append(f"OUTPUT t={t:.2f}s")
    order += [f"REFERENCE {i + 1}" for i in range(n_refs)]
    image_list = "\n".join(f"Image {i + 1}: {label}" for i, label in enumerate(order))
    ref_note = (
        "\nREFERENCE images show the look the user asked for; treat them as part of the requirement.\n"
        if n_refs else ""
    )
    change_list = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(change)) or "(none)"
    preserve_list = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(preserve)) or "(none)"
    return f"""REQUIREMENT (what the user asked for):
{requirement}
{ref_note}
The attached images, in order:
{image_list}

CHANGE ITEMS -- visual states the requirement asks the OUTPUT to show:
{change_list}

For every change item, label the OUTPUT:
- "done": clearly visible in the OUTPUT frames, consistent with the requirement.
- "partial": attempted or present only weakly, in some frames, or with clearly different specifics.
- "not_done": not visible in the OUTPUT.

PRESERVE ITEMS -- things that must stay as in the SOURCE:
{preserve_list}

For every preserve item, compare OUTPUT with SOURCE at the same timestamp:
- "kept": essentially the same as in the SOURCE.
- "changed": visibly different from the SOURCE.

Then list UNREQUESTED CHANGES: concrete visible differences between SOURCE and OUTPUT that no change item asks for and the requirement does not imply.

Return JSON:
{{"change": [{{"id": 1, "label": "done|partial|not_done", "evidence": "short"}}],
  "preserve": [{{"id": 1, "label": "kept|changed", "evidence": "short"}}],
  "unrequested_changes": ["short"]}}
Give exactly {len(change)} change entries and {len(preserve)} preserve entries, ids in order."""


def validate_checklist(obj: dict, n_change: int, n_preserve: int) -> dict:
    change = obj["change"]
    preserve = obj["preserve"]
    if len(change) != n_change or len(preserve) != n_preserve:
        raise ValueError(f"expected {n_change}/{n_preserve} entries, got {len(change)}/{len(preserve)}")
    for item in change:
        if item.get("label") not in CHANGE_LABELS:
            raise ValueError(f"bad change label {item.get('label')!r}")
    for item in preserve:
        if item.get("label") not in PRESERVE_LABELS:
            raise ValueError(f"bad preserve label {item.get('label')!r}")
    extra = obj.get("unrequested_changes") or []
    if not isinstance(extra, list):
        raise ValueError("unrequested_changes must be a list")
    return {"change": change, "preserve": preserve, "unrequested_changes": [str(x) for x in extra]}


def score_checklist(result: dict) -> dict:
    change = [c["label"] for c in result["change"]]
    preserve = [p["label"] for p in result["preserve"]]
    return {
        "fidelity": sum(CHANGE_LABELS[x] for x in change) / len(change) if change else None,
        "preservation": preserve.count("kept") / len(preserve) if preserve else None,
        "n_done": change.count("done"),
        "n_partial": change.count("partial"),
        "n_not_done": change.count("not_done"),
        "n_kept": preserve.count("kept"),
        "n_changed": preserve.count("changed"),
        "n_unrequested": len(result["unrequested_changes"]),
    }


def judge_checklist(client: Client, task: dict, output_video: Path, *, frames: int = 4) -> dict:
    refs = [Path(p) for p in task.get("reference_images") or []]
    change = list(task.get("checklist") or [])
    preserve = list(task.get("preserve") or DEFAULT_PRESERVE)
    times, (out_frames, src_frames) = aligned_frames(output_video, [Path(task["source_video"])], frames)
    images = []
    for s, o in zip(src_frames, out_frames):
        images += [to_b64(s), to_b64(o)]
    images += [to_b64(reference_frame(p)) for p in refs]
    prompt = checklist_prompt(requirement=task["requirement"], times=times, n_refs=len(refs),
                              change=change, preserve=preserve)
    result, calls = call_with_retry(client, CHECKLIST_SYSTEM, prompt, images,
                                    lambda o: validate_checklist(o, len(change), len(preserve)))
    for entry, text in zip(result["change"], change):
        entry["item"] = text
    for entry, text in zip(result["preserve"], preserve):
        entry["item"] = text
    return {**result, "scores": score_checklist(result), "calls": calls, "times": times}


def summarize_checklist(rows: list[dict]) -> dict[str, dict]:
    by_arm: dict[str, list[dict]] = {}
    for row in latest(rows, ("arm", "sample_id")).values():
        by_arm.setdefault(row["arm"], []).append(row)
    out = {}
    for arm, items in by_arm.items():
        ok = [r for r in items if not r.get("error")]

        def mean(key: str) -> float | None:
            vals = [r["scores"][key] for r in ok if r["scores"].get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        out[arm] = {
            "n": len(items),
            "errors": len(items) - len(ok),
            "fidelity": mean("fidelity"),
            "preservation": mean("preservation"),
            "unrequested_per_sample": mean("n_unrequested"),
            "done_rate": _rate(ok, "n_done", ("n_done", "n_partial", "n_not_done")),
            "partial_rate": _rate(ok, "n_partial", ("n_done", "n_partial", "n_not_done")),
            "not_done_rate": _rate(ok, "n_not_done", ("n_done", "n_partial", "n_not_done")),
        }
    return out


def _rate(rows: list[dict], key: str, denom: tuple[str, ...]) -> float | None:
    num = sum(r["scores"][key] for r in rows)
    den = sum(r["scores"][k] for r in rows for k in denom)
    return num / den if den else None


# --------------------------------------------------------------------- pairwise

PAIRWISE_SYSTEM = """You compare two edited versions, A and B, of the same SOURCE video.
You answer three separate questions; a version can win one and lose another.
Judge only what is visible. Return strict JSON only."""


def pairwise_prompt(*, requirement: str, times: list[float], n_refs: int, preserve: list[str]) -> str:
    order = [f"SOURCE t={t:.2f}s" for t in times]
    order += [f"A t={t:.2f}s" for t in times]
    order += [f"B t={t:.2f}s" for t in times]
    order += [f"REFERENCE {i + 1}" for i in range(n_refs)]
    image_list = "\n".join(f"Image {i + 1}: {label}" for i, label in enumerate(order))
    ref_note = "REFERENCE images show the look the user asked for; they are part of the requirement.\n" if n_refs else ""
    preserve_list = "; ".join(preserve)
    return f"""REQUIREMENT (what the user asked for):
{requirement}
{ref_note}
The attached images, in order:
{image_list}

Questions -- answer each with "A", "B" or "tie":
1. fidelity_to_requirement: which version better realizes the requested changes{" and matches the REFERENCE" if n_refs else ""}?
2. preservation: compared with the SOURCE, which version better keeps what must not change ({preserve_list})?
3. visual_quality: ignoring the requirement, which version looks more like professional, high-production footage (no artifacts, coherent lighting and texture, temporally stable)?
Use "tie" only when the two are genuinely equivalent on that question.

Return JSON:
{{"fidelity_to_requirement": "A|B|tie", "preservation": "A|B|tie", "visual_quality": "A|B|tie", "reason": "one short sentence per question"}}"""


def validate_pairwise(obj: dict) -> dict:
    out = {}
    for q in QUESTIONS:
        value = str(obj[q]).strip()
        value = {"a": "A", "b": "B", "tie": "tie", "equal": "tie"}.get(value.lower(), value)
        if value not in VERDICTS:
            raise ValueError(f"bad verdict {q}={obj[q]!r}")
        out[q] = value
    out["reason"] = str(obj.get("reason", ""))[:600]
    return out


def combine_orders(forward: dict, reverse: dict, arm_a: str, arm_b: str) -> dict[str, str]:
    """forward: A=arm_a, B=arm_b; reverse: A=arm_b, B=arm_a. Disagreement -> tie."""
    verdict = {}
    for q in QUESTIONS:
        f = {"A": arm_a, "B": arm_b, "tie": "tie"}[forward[q]]
        r = {"A": arm_b, "B": arm_a, "tie": "tie"}[reverse[q]]
        verdict[q] = f if f == r else "tie"
    return verdict


def judge_pairwise(client: Client, task: dict, video_a: Path, video_b: Path,
                   arm_a: str, arm_b: str, *, frames: int = 4) -> dict:
    if file_digest(video_a) == file_digest(video_b):
        tie = {q: "tie" for q in QUESTIONS}
        return {"verdict": tie, "identical_bytes": True, "forward": None, "reverse": None}
    refs = [Path(p) for p in task.get("reference_images") or []]
    preserve = list(task.get("preserve") or DEFAULT_PRESERVE)
    times, (src, fa, fb) = aligned_frames(Path(task["source_video"]), [video_a, video_b], frames)
    ref_images = [to_b64(reference_frame(p)) for p in refs]
    prompt = pairwise_prompt(requirement=task["requirement"], times=times, n_refs=len(refs), preserve=preserve)
    src_b64 = [to_b64(x) for x in src]
    a_b64 = [to_b64(x) for x in fa]
    b_b64 = [to_b64(x) for x in fb]
    forward, _ = call_with_retry(client, PAIRWISE_SYSTEM, prompt, src_b64 + a_b64 + b_b64 + ref_images, validate_pairwise)
    reverse, _ = call_with_retry(client, PAIRWISE_SYSTEM, prompt, src_b64 + b_b64 + a_b64 + ref_images, validate_pairwise)
    return {"verdict": combine_orders(forward, reverse, arm_a, arm_b), "identical_bytes": False,
            "forward": forward, "reverse": reverse, "times": times}


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(sum(comb(n, j) for j in range(k + 1)) * 2 / 2**n, 1.0)


def summarize_pairwise(rows: list[dict], arm_a: str, arm_b: str) -> dict:
    items = list(latest(rows, ("sample_id",)).values())
    ok = [r for r in items if not r.get("error")]
    out: dict[str, Any] = {"arm_a": arm_a, "arm_b": arm_b, "n": len(items), "errors": len(items) - len(ok)}
    called = [r for r in ok if not r.get("identical_bytes")]
    for q in QUESTIONS:
        wa = sum(r["verdict"][q] == arm_a for r in ok)
        wb = sum(r["verdict"][q] == arm_b for r in ok)
        first = sum(r["forward"][q] == "A" for r in called) + sum(r["reverse"][q] == "A" for r in called)
        decided = sum(r["forward"][q] != "tie" for r in called) + sum(r["reverse"][q] != "tie" for r in called)
        out[q] = {
            f"wins_{arm_a}": wa,
            f"wins_{arm_b}": wb,
            "ties": len(ok) - wa - wb,
            "win_rate_a_excl_ties": wa / (wa + wb) if wa + wb else None,
            "sign_test_p": sign_test(wa, wb),
            "first_slot_win_rate": first / decided if decided else None,
        }
    out["identical_bytes"] = sum(bool(r.get("identical_bytes")) for r in ok)
    return out


# ---------------------------------------------------------------- run / resume


def latest(rows: Iterable[dict], key: tuple[str, ...]) -> dict[tuple, dict]:
    """Last record per key: on resume a retried row supersedes its earlier failure."""
    out: dict[tuple, dict] = {}
    for row in rows:
        out[tuple(row.get(k) for k in key)] = row
    return out


def task_digest(task: dict) -> str:
    """Hash of everything in a task the judge sees (requirement, items, references)."""
    payload = {k: task.get(k) for k in ("source_video", "requirement", "reference_images", "checklist", "preserve")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def input_digest(task: dict, *videos: Path) -> str:
    """Identity of a judge job's inputs: the task as the judge sees it plus the
    video file(s) by content. Stored with each row; a row whose digest differs
    from the current job's is stale and is judged again."""
    h = hashlib.sha256(task_digest(task).encode())
    for video in videos:
        stat = Path(video).stat()
        h.update(f"{Path(video).resolve()}:{stat.st_size}:{int(stat.st_mtime)}".encode())
    return h.hexdigest()[:16]


def pending(jobs: list[dict], done_rows: list[dict], key: tuple[str, ...]) -> list[dict]:
    """Jobs without a successful result made from the same inputs.

    ``key`` identifies the job; when jobs carry ``input_digest`` a stored row only
    counts if it was produced from identical inputs, so a rebuilt task file or a
    re-rendered video is judged again instead of keeping a stale verdict."""
    finished = {k: r for k, r in latest(done_rows, key).items() if not r.get("error")}
    todo = []
    for job in jobs:
        row = finished.get(tuple(job.get(k) for k in key))
        if row is None or ("input_digest" in job and row.get("input_digest") != job["input_digest"]):
            todo.append(job)
    return todo


def current_rows(rows: list[dict], jobs: list[dict], key: tuple[str, ...]) -> list[dict]:
    """Latest stored rows that match a current job's inputs (what summaries should count)."""
    wanted = {tuple(j.get(k) for k in key): j.get("input_digest") for j in jobs}
    out = []
    for k, row in latest(rows, key).items():
        if k in wanted and (wanted[k] is None or row.get("input_digest") == wanted[k]):
            out.append(row)
    return out


@dataclass
class Runner:
    """Run jobs concurrently, appending each result to ``output`` as it lands."""

    output: Path
    concurrency: int = 6
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, jobs: list[dict], work: Callable[[dict], dict], log: Callable[[str], None] = print) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)

        def one(job: dict) -> None:
            record = dict(job)
            try:
                record.update(work(job))
                record["error"] = ""
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            with self._lock:
                with self.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                log(f"[{record.get('arm', '')}] {record['sample_id']} error={bool(record['error'])}")

        with ThreadPoolExecutor(max(1, self.concurrency)) as pool:
            list(pool.map(one, jobs))


# ------------------------------------------------------------------- label kit


def make_label_pairs(sample_ids: list[str], arm_a: str, arm_b: str, n: int, seed: int) -> list[dict]:
    """N pairs with a seeded random display order; the order is the hidden key."""
    rng = random.Random(seed)
    chosen = rng.sample(sorted(sample_ids), min(n, len(sample_ids)))
    pairs = []
    for i, sid in enumerate(chosen):
        swap = rng.random() < 0.5
        pairs.append({
            "pair_id": f"p{i:03d}",
            "sample_id": sid,
            "left": arm_b if swap else arm_a,
            "right": arm_a if swap else arm_b,
        })
    return pairs


def score_human_agreement(key: list[dict], answers: dict[str, dict], judge_rows: list[dict]) -> dict:
    """Per question: how often the judge's verdict equals the human's.

    ``answers[pair_id][question]`` is "left" / "right" / "tie" as exported by the
    kit; ``key`` maps left/right to arm names; judge rows carry arm-name verdicts.
    """
    judge = {r["sample_id"]: r for r in judge_rows if not r.get("error")}
    out = {}
    for q in QUESTIONS:
        agree = total = 0
        for pair in key:
            human = (answers.get(pair["pair_id"]) or {}).get(q)
            row = judge.get(pair["sample_id"])
            if human is None or row is None:
                continue
            human_arm = {"left": pair["left"], "right": pair["right"], "tie": "tie"}[human]
            total += 1
            agree += human_arm == row["verdict"][q]
        out[q] = {"n": total, "agreement": agree / total if total else None}
    return out
