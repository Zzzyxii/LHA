#!/usr/bin/env python3
"""A/B the arms by asking the judge which of two videos is better.

The pointwise reward judge turned out to be the wrong instrument for the question
being asked of it. Two independent properties sank it:

* Quantisation. Across 196 scorings it emitted 7 distinct values (1.0, 1.5, ...,
  4.0), so the smallest difference it can express is 0.5 -- larger than most of
  the effects under test.
* Instability. Rescoring the same 30 files moved 12 of them by a full point or
  more and shifted the arm mean by 0.19. Repeat-and-median narrows that but
  cannot go below the 0.5 grid.

Asking "A or B?" sidesteps both. The judge compares two clips side by side in one
context, which is a far easier discrimination than recalling an absolute standard,
and the output is a bit rather than a number on a coarse scale. Win rate over n
pairs is then a clean binomial, and a sign test on it needs no assumption about
score distributions.

Two controls make the result trustworthy rather than merely cheap:

* Order is swapped and both directions are asked. A judge that simply favours the
  first clip would score 50/50 after cancellation, and the per-direction split is
  reported so that bias is visible rather than hidden.
* A pair where the two directions disagree is recorded as a tie, not silently
  resolved. Those are the pairs the judge genuinely cannot separate, and counting
  them as wins for either side would manufacture a result.

Identical inputs are detected up front: when two arms produced the same bytes for
a sample, the pair is a tie by construction and spending judge calls on it would
only measure noise.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
from typing import Any

import av

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_aesthetic_reward_litellm import (  # noqa: E402
    LiteLLMJudge,
    _extract_json_object,
    _generation_path,
    _read_eval_inputs,
)

_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


SYSTEM_PROMPT = """You are a strict cinematic/VFX aesthetic judge comparing two edited videos.

You will see sampled frames from video A, then sampled frames from video B. Both
were produced from the same source clip and the same editing instruction. You do
not see the source clip.

Decide which edited result is aesthetically better overall, weighing:
- how naturally the result integrates with the scene's mood and context
- consistency of the visual style with the world, era and genre
- composition, lighting hierarchy and cinematic design
- colour harmony and material/texture refinement, avoiding cheap or generic looks
- clarity of the main visual intent and focal hierarchy
- absence of artifacts: seams, flicker, warped faces, broken geometry

Rules:
- Judge the final visual result. Do not reward a video merely for changing more.
- Do not reward sharpness or saturation on their own.
- For removal or cleanup edits, a seamless and unremarkable result is a success.
- Answer "tie" only when you genuinely cannot separate them.
- Return valid JSON only, with no markdown.

Return exactly: {"winner": "A" | "B" | "tie", "reason": "one concise sentence"}
"""


def _user_prompt(instruction: str, n_a: int, n_b: int) -> str:
    return f"""Editing instruction given to both videos:
{instruction}

The first {n_a} images are frames from video A, in temporal order.
The next {n_b} images are frames from video B, in temporal order.

Which edited video is aesthetically better overall?
Return exactly: {{"winner": "A" | "B" | "tie", "reason": "one concise sentence"}}
"""


def _frames_b64(video_path: Path, *, frames: int, width: int = 640, quality: int = 85) -> list[str]:
    container = av.open(str(video_path))
    try:
        images = [frame.to_image() for frame in container.decode(video=0)]
    finally:
        container.close()
    if not images:
        raise RuntimeError(f"no decodable frames in {video_path}")
    if frames >= len(images):
        picked = images
    else:
        step = (len(images) - 1) / max(frames - 1, 1) if frames > 1 else 0
        picked = [images[min(len(images) - 1, round(i * step))] for i in range(frames)]
    out: list[str] = []
    for image in picked:
        if image.width > width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=quality)
        out.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return out


def _ask(
    judge: LiteLLMJudge,
    *,
    instruction: str,
    first: list[str],
    second: list[str],
) -> tuple[str, str]:
    """Return (winner_label, reason) where winner_label is "A", "B" or "tie"."""
    raw = judge.complete(
        system_prompt=SYSTEM_PROMPT,
        user_text=_user_prompt(instruction, len(first), len(second)),
        images_b64=first + second,
    )
    payload = _extract_json_object(raw)
    winner = str(payload.get("winner") or "").strip().upper()
    if winner not in {"A", "B"}:
        winner = "TIE"
    return winner, str(payload.get("reason") or "")


def _digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _compare_one(
    *,
    judge: LiteLLMJudge,
    row: dict[str, Any],
    arm_a: str,
    arm_b: str,
    generation_root: Path,
    frames: int,
    index: int,
    total: int,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or f"eval_{index - 1:03d}")
    instruction = str(row.get("instruction") or row.get("prompt") or "")
    record: dict[str, Any] = {"sample_id": sample_id, "arm_a": arm_a, "arm_b": arm_b}
    try:
        path_a = _generation_path(generation_root, arm_a, sample_id)
        path_b = _generation_path(generation_root, arm_b, sample_id)
        if _digest(path_a) == _digest(path_b):
            record.update(
                {
                    "verdict": "tie",
                    "identical_bytes": True,
                    "forward": "",
                    "reverse": "",
                    "reason_forward": "",
                    "reason_reverse": "",
                    "error": "",
                }
            )
            _log(f"[pairwise] {index}/{total} {sample_id}: tie (identical files)")
            return record

        frames_a = _frames_b64(path_a, frames=frames)
        frames_b = _frames_b64(path_b, frames=frames)

        # A first, then B first. Each call is independent, so a positional
        # preference shows up as the same letter winning both.
        fwd, reason_fwd = _ask(judge, instruction=instruction, first=frames_a, second=frames_b)
        rev, reason_rev = _ask(judge, instruction=instruction, first=frames_b, second=frames_a)

        # In the reverse call the slot labelled "A" held arm_b, so translate back.
        fwd_arm = {"A": arm_a, "B": arm_b}.get(fwd, "tie")
        rev_arm = {"A": arm_b, "B": arm_a}.get(rev, "tie")
        verdict = fwd_arm if fwd_arm == rev_arm else "tie"

        record.update(
            {
                "verdict": verdict,
                "identical_bytes": False,
                "forward": fwd_arm,
                "reverse": rev_arm,
                "reason_forward": reason_fwd,
                "reason_reverse": reason_rev,
                "error": "",
            }
        )
        _log(
            f"[pairwise] {index}/{total} {sample_id}: fwd={fwd_arm} rev={rev_arm} -> {verdict}"
        )
    except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the sweep
        record.update(
            {
                "verdict": "error",
                "identical_bytes": False,
                "forward": "",
                "reverse": "",
                "reason_forward": "",
                "reason_reverse": "",
                "error": repr(exc),
            }
        )
        _log(f"[pairwise] {index}/{total} {sample_id}: ERROR {exc!r}")
    return record


def _sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(sum(comb(n, j) for j in range(k + 1)) * 2 / 2**n, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise A/B comparison of two arms.")
    parser.add_argument("--eval-inputs", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--arm-a", required=True)
    parser.add_argument("--arm-b", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LOW_HIGH_JUDGE_BASE_URL", "http://litellm-sgp.woa.com"),
    )
    parser.add_argument(
        "--model", default=os.environ.get("LOW_HIGH_JUDGE_MODEL", "gemini-3.1-pro-preview")
    )
    parser.add_argument("--api-key-env", default="LOW_HIGH_JUDGE_API_KEY")
    parser.add_argument("--reasoning-effort", default=os.environ.get("LOW_HIGH_JUDGE_REASONING_EFFORT", "high"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--only-sample-ids", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing judge API key in env var: {args.api_key_env}")

    rows = _read_eval_inputs(args.eval_inputs)
    if args.only_sample_ids:
        keep = {
            line.strip()
            for line in args.only_sample_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if str(row.get("sample_id")) in keep]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no eval rows selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    judge = LiteLLMJudge(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort.strip(),
        temperature=None if args.temperature < 0 else args.temperature,
        max_completion_tokens=3000,
        timeout=args.timeout,
        retries=args.retries,
    )

    total = len(rows)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(
                _compare_one,
                judge=judge,
                row=row,
                arm_a=args.arm_a,
                arm_b=args.arm_b,
                generation_root=args.generation_root,
                frames=args.frames,
                index=index,
                total=total,
            )
            for index, row in enumerate(rows, start=1)
        ]
        results = [future.result() for future in futures]

    results.sort(key=lambda item: item["sample_id"])
    tag = f"{args.arm_a}__vs__{args.arm_b}"
    (args.output_dir / f"{tag}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n", encoding="utf-8"
    )

    scored = [r for r in results if r["verdict"] != "error"]
    wins_a = sum(1 for r in scored if r["verdict"] == args.arm_a)
    wins_b = sum(1 for r in scored if r["verdict"] == args.arm_b)
    ties = sum(1 for r in scored if r["verdict"] == "tie")
    identical = sum(1 for r in scored if r.get("identical_bytes"))
    # Order effect: how often the arm shown first won, across both directions.
    judged = [r for r in scored if not r.get("identical_bytes") and r["forward"]]
    first_slot_wins = sum(1 for r in judged if r["forward"] == args.arm_a) + sum(
        1 for r in judged if r["reverse"] == args.arm_b
    )
    disagreements = sum(1 for r in judged if r["forward"] != r["reverse"])

    summary = {
        "arm_a": args.arm_a,
        "arm_b": args.arm_b,
        "n": total,
        "errors": len(results) - len(scored),
        f"wins_{args.arm_a}": wins_a,
        f"wins_{args.arm_b}": wins_b,
        "ties": ties,
        "ties_from_identical_files": identical,
        "ties_from_direction_disagreement": disagreements,
        "win_rate_excluding_ties": (wins_a / (wins_a + wins_b)) if (wins_a + wins_b) else None,
        "sign_test_p": _sign_test(wins_a, wins_b),
        # 0.5 means no positional preference; far from 0.5 means the verdicts are
        # partly an artifact of presentation order rather than of the videos.
        "first_slot_win_rate": (first_slot_wins / (2 * len(judged))) if judged else None,
        "judge_model": args.model,
        "frames_per_video": args.frames,
    }
    (args.output_dir / f"{tag}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(json.dumps(summary, ensure_ascii=False, indent=2))
    _log(f"cost: {json.dumps(LiteLLMJudge.usage_summary() if hasattr(LiteLLMJudge, 'usage_summary') else {})}")


if __name__ == "__main__":
    main()
