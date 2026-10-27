#!/usr/bin/env python3
"""Write edit prompts with a skill card (ON) and without it (OFF), same writer model.

Both arms get identical inputs -- the one-line intent and 6 frames of the low
video -- and identical instructions, except that the ON arm also receives the
category card from ``skills_v3.py`` (carriers, vocabulary, off-carrier policy,
model priors, failure modes). Same model, same temperature, same word budget, so
the only difference between the two prompts is the skill.

    python3 scripts/eval/bench_v3/plan_v3.py \
        --intents out/bench_v3/inputs/track_a.t1.instructions.jsonl \
        --frames-dir out/bench_v3/inputs/frames --output-dir out/bench_v3/skills_v3

Writes <output-dir>/plans_with_card.jsonl, plans_no_card.jsonl and the two
run_bench --instruction-file inputs. Reruns reuse cached responses per
(sample, arm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# The frame extractor shells out to ffmpeg/ffprobe; without the repo shim on PATH it
# silently returns zero frames and the writer then invents the scene.
_SHIM = ROOT / "scripts/bench_skill/bin"
if _SHIM.is_dir() and str(_SHIM) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_SHIM}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/eval/bench_v3"))

from skills_v3 import card_block, card_for, route  # noqa: E402
from vlm.litellm_vlm import LiteLLMVisionModel, _extract_json_object  # noqa: E402
from vlm.local_qwen import extract_frames  # noqa: E402

SYSTEM = """You are a prompt engineer writing the final prompt for a video edit model.
You get one line of what the user wants and six frames from the source video.

Write the prompt the edit model will receive. Rules:
- English, at most 90 words, plain text, no headings, no bullet lists.
- Verb first, one change per clause, most important change first.
- Only specify something you can see in the frames or that the user asked for.
- Never describe camera motion; the source motion is reused.
- No preservation boilerplate.

Return JSON only: {"prompt": "<the prompt>", "carriers_used": ["..."], "skipped": ["..."]}
where skipped lists things you deliberately did NOT specify because their carriers are not visible.
"""


def _brief(block: str) -> str:
    return ("\n\n" + block + "\n") if block else ""


GROUNDED = """You are a prompt engineer writing the final prompt for a video edit model.

Step 1: list what you actually see in the six frames -- people (how many), their garments, the setting, the light sources. Maximum 10 items.
Step 2: write the edit prompt using ONLY those items.

Rules for the prompt:
- English, at most 90 words.
- Never mention a person, garment, room or object that is not in your Step 1 list.
- If the frames show no person, do not write about a person.
- Verb first, one change per clause, most important first.
- No camera moves, no preservation boilerplate.

Return JSON only: {"visible": ["..."], "prompt": "...", "carriers_used": ["..."], "skipped": ["..."]}
"""


def _call(model, *, intent: str, frames: list[str], block: str) -> str:
    user = (
        f"User request: {intent}\n\n"
        f"The six images are frames of the source video.\n"
        f"Write the final edit prompt.{_brief(block)}"
    )
    return model.predict_text(system_prompt=SYSTEM, user_text=user, image_paths=frames, max_new_tokens=3072)


def _strip(prompt: str) -> str:
    text = re.sub(r"^(final prompt|prompt)\s*[:\-]\s*", "", prompt.strip(), flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--model", default=os.environ.get("LOW_HIGH_AGENT_LITELLM_MODEL", "gemini-3.8-flash"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.intents.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    model = LiteLLMVisionModel(model=args.model, temperature=0.0)
    frames_by_id = {}

    def work(row: dict) -> list[dict]:
        sid = row["sample_id"]
        src = row.get("source_video") or row.get("low_video")
        intent = row.get("intent") or row.get("instruction") or row.get("prompt")
        frames = frames_by_id.get(sid)
        if frames is None:
            frames = extract_frames(src, max_frames=args.frames)
            if len(frames) < args.frames:
                raise RuntimeError(f"{sid}: got {len(frames)} frames, need {args.frames} -- refusing to write an ungrounded prompt")
            frames_by_id[sid] = frames
        key = hashlib.md5(f"v3b|{sid}|{intent}|{args.model}|{args.frames}".encode()).hexdigest()
        out = []
        arms = (
            ("with_card", card_block(intent)),
            ("no_card", ""),
            ("with_card_decisive", card_block(intent, decisive=True)),
            ("with_card_exec", card_block(intent, execution_only=True)),
            ("with_card_grounded", card_block(intent)),
        )
        for arm, block in arms:
            path = cache_dir / f"{key}.{arm}.json"
            if path.exists():
                out.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            if arm == "with_card_grounded":
                user = (f"User request: {intent}\n\nThe six images are frames of the source video.\n"
                        f"Follow Step 1 and Step 2.{_brief(block)}")
                raw = model.predict_text(system_prompt=GROUNDED, user_text=user, image_paths=frames, max_new_tokens=12000)
            else:
                raw = _call(model, intent=intent, frames=frames, block=block)
            payload = _extract_json_object(raw)
            prompt = _strip(str(payload.get("prompt") or ""))
            record = {
                "sample_id": sid,
                "arm": arm,
                "category": route(intent),
                "intent": intent,
                "prompt": prompt,
                "words": len(prompt.split()),
                "carriers_used": payload.get("carriers_used") or [],
                "skipped": payload.get("skipped") or [],
                "visible": payload.get("visible") or [],
            }
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            out.append(record)
        return out

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = [item for group in pool.map(work, rows) for item in group]

    print("usage:", LiteLLMVisionModel.usage_summary())
    for arm in ("with_card", "no_card", "with_card_decisive", "with_card_exec", "with_card_grounded"):
        subset = [r for r in results if r["arm"] == arm]
        words = sorted(r["words"] for r in subset)
        over = [r["sample_id"] for r in subset if r["words"] > 110]
        print(f"{arm}: n={len(subset)} words median={words[len(words)//2]} max={words[-1]} over110={over[:3]}")
        (args.output_dir / f"plans_{arm}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in subset) + "\n", encoding="utf-8"
        )
        index = args.output_dir / f"{arm}.index.jsonl"
        index.write_text("\n".join(
            json.dumps({"sample_id": r["sample_id"], "source_video": next(x.get("source_video") for x in rows if x["sample_id"] == r["sample_id"])}, ensure_ascii=False)
            for r in subset) + "\n", encoding="utf-8")
        instr = args.output_dir / f"{arm}.instructions.jsonl"
        instr.write_text("\n".join(
            json.dumps({"sample_id": r["sample_id"], "intent": r["prompt"],
                        "source_video": next(x.get("source_video") for x in rows if x["sample_id"] == r["sample_id"])}, ensure_ascii=False)
            for r in subset) + "\n", encoding="utf-8")
        print(f"  wrote {instr}")


if __name__ == "__main__":
    main()
