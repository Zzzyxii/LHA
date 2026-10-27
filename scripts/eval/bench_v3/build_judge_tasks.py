#!/usr/bin/env python3
"""Judge task files for bench_v3: one per (track, tier), in the format run_checklist.py / run_pairwise.py read.

Each row: sample_id, source_video, requirement (what that tier's user asked for),
reference_images (shown to the judge for reference tiers), checklist, preserve.

The checklist is what "did it do what was asked" is scored against, so it must be
fixed before, and independent of, any arm's output:
  track A t3       : transformation_tags (what actually changed low -> high)
  track A t1       : generated from the intent + source frames, category level only
                     (a one-line intent admits many valid answers; no colours unless
                     the intent names them)
  track A t2_*     : generated from the reference still(s) + intent + source frames:
                     the reference attributes an edit should adopt
  track B (all)    : the style checklist built with the track
Generated checklists use a different model family (--model, default gpt-5.6-luna)
from the agent under test (gemini) so the agent is not graded by its own phrasing.
Responses are cached by request hash; reruns are offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/eval/ditto_quant"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_aesthetic_reward_litellm import LiteLLMJudge, _extract_json_object  # noqa: E402

SYSTEM = "You define acceptance checklists for video edits. Return strict JSON only."

T1_PROMPT = """A user asked a video edit model: "{request}"
The images are frames of the SOURCE clip to be edited.

List 4-6 visible changes that ANY edit faithfully satisfying this request on this clip should show.
- Category level: say what kind of change, not one specific design (e.g. "the women's modern clothing becomes period court costume", not "a green silk robe").
- Do not name colours, materials or specific objects unless the request itself names them.
- Anchor each item to something visible in the source (a person, the background, the lighting).
- Only include changes the request implies; do not add generic quality items.
Return JSON: {{"checklist": ["...", ...]}}"""

T2_PROMPT = """A user asked a video edit model: "{request}", and gave REFERENCE image(s) of the look they want.
The first {n_source} images are frames of the SOURCE clip to edit; the remaining {n_ref} are the REFERENCE.

List 4-8 visible attributes of the REFERENCE that the edited clip should adopt, each mapped onto the source:
- Be specific where the reference is clear: garment type and colour, headwear, makeup, props, set dressing, lighting quality, colour grade.
- Each item names where it applies in the source (e.g. "the woman on the right wears a green embroidered robe like the reference").
- Do not include the reference's people, faces, poses or framing.
Return JSON: {{"checklist": ["...", ...]}}"""

T1_LOOK_PROMPT = """A user asked a video edit model: "{request}"
The images are frames of the SOURCE clip to be edited. The request is about the clip's overall LOOK (grade, medium, texture), not its content.

List 3-6 visible look properties that ANY edit faithfully satisfying this request on this clip should show.
- Describe the look at the level the request implies (e.g. "the image becomes black and white", "colours turn soft and pastel"); do not name exact hex colours or numeric values.
- Only include properties the request implies; do not add content changes or generic quality items.
Return JSON: {{"checklist": ["...", ...]}}"""

T2_LOOK_PROMPT = """A user asked a video edit model: "{request}", and gave REFERENCE image(s) of the look they want.
The first {n_source} images are frames of the SOURCE clip to edit; the remaining {n_ref} are the REFERENCE.
The reference may be an illustration or a different scene: only its visual look matters, never its people, objects or layout.

List 3-7 visible look properties of the REFERENCE that the edited clip should adopt: colour palette and grade, contrast, black-and-white or tint, grain and texture, rendering medium (photo, paint, print, line art), lighting quality.
Return JSON: {{"checklist": ["...", ...]}}"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ChecklistMaker:
    def __init__(self, model: str, cache_dir: Path, offline: bool) -> None:
        self.cache_dir = cache_dir
        self.offline = offline
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.judge = None if offline else LiteLLMJudge(
            base_url=os.environ.get("LOW_HIGH_JUDGE_BASE_URL", "http://litellm-sgp.woa.com"),
            api_key=os.environ["LOW_HIGH_JUDGE_API_KEY"], model=model, reasoning_effort="high",
            max_completion_tokens=8000)
        self.model = model

    def make(self, prompt: str, images: list[str], stable_images: list[str]) -> list[str]:
        """``stable_images`` are the inputs that identify the request (reference
        stills); extracted source frames live in temp dirs, so the prompt names the
        source clip instead and those paths stay out of the cache key."""
        key = hashlib.sha256(json.dumps([self.model, prompt, stable_images]).encode()).hexdigest()[:20]
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))["checklist"]
        if self.offline:
            raise SystemExit(f"offline and no cached checklist for {key}")
        from vlm.litellm_vlm import _image_to_data_url

        b64 = [_image_to_data_url(p).split(",", 1)[1] for p in images]
        last: Exception | None = None
        for _ in range(2):
            try:
                items = _extract_json_object(self.judge.complete(system_prompt=SYSTEM, user_text=prompt, images_b64=b64))["checklist"]
                items = [" ".join(str(x).split()) for x in items if str(x).strip()]
                if 3 <= len(items) <= 10:
                    cached.write_text(json.dumps({"prompt": prompt, "images": images, "checklist": items}, ensure_ascii=False, indent=1), encoding="utf-8")
                    return items
                last = ValueError(f"bad checklist length {len(items)}")
            except Exception as exc:  # noqa: BLE001 - one retry, then fail loudly
                last = exc
        raise RuntimeError(f"checklist generation failed: {last!r}")


def source_frames(video: str, n: int = 4) -> list[str]:
    from vlm import extract_frames

    return extract_frames(video, max_frames=n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/bench_v3/judge_tasks")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    maker = ChecklistMaker(args.model, args.output_dir / "cache", args.offline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = ROOT / "scripts/eval/bench_v3/data"

    def task(r: dict, requirement: str, refs: list[str], checklist: list[str]) -> dict:
        return {"sample_id": r["sample_id"], "source_video": r["source_video"], "requirement": requirement,
                "reference_images": refs, "checklist": checklist, "preserve": r["preserve"]}

    track_a = _jsonl(data / "track_a.jsonl")

    def a_t1(r, template=T1_PROMPT):
        prompt = template.format(request=r["t1"]) + f"\n[source: {Path(r['source_video']).name}]"
        return task(r, r["t1"], [], maker.make(prompt, source_frames(r["source_video"]), []))

    def a_t2(r, field, template=T2_PROMPT):
        refs = r[field]
        frames = source_frames(r["source_video"])
        prompt = template.format(request=r["t1"], n_source=len(frames), n_ref=len(refs)) + f"\n[source: {Path(r['source_video']).name}]"
        return task(r, r["t1"], refs, maker.make(prompt, frames + refs, refs))

    jobs = {
        "track_a.t1": lambda r: a_t1(r),
        "track_a.t2_same": lambda r: a_t2(r, "t2_same_reference_images"),
        "track_a.t2_cross": lambda r: a_t2(r, "t2_cross_reference_images"),
        "track_a.t3": lambda r: task(r, r["t3"], [], r["checklist"]),
    }
    for name, fn in jobs.items():
        with ThreadPoolExecutor(args.concurrency) as pool:
            rows = list(pool.map(fn, track_a))
        (args.output_dir / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} rows, checklist items median={sorted(len(x['checklist']) for x in rows)[len(rows) // 2]}")

    for track in ("style_track", "ditto_track"):
        rows = _jsonl(data / f"{track}.jsonl")
        jobs = {
            # t3 is scored against the look checklist written from the spec; t1 and t2
            # get tier-specific checklists like track A, so a one-line request is not
            # graded against hex colours it never mentioned.
            f"{track}.t1": lambda r: a_t1(r, T1_LOOK_PROMPT),
            f"{track}.t2": lambda r: a_t2(r, "t2_reference_images", T2_LOOK_PROMPT),
            f"{track}.t3": lambda r: task(r, r["t3"], [], r["checklist"]),
        }
        for name, fn in jobs.items():
            with ThreadPoolExecutor(args.concurrency) as pool:
                out = list(pool.map(fn, rows))
            (args.output_dir / f"{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
            print(f"{name}: {len(out)} rows, checklist items median={sorted(len(x['checklist']) for x in out)[len(out) // 2]}")


if __name__ == "__main__":
    main()
