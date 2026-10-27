#!/usr/bin/env python3
"""Compile an existing planner plan into a skill-disciplined edit spec.

The v3 arms that wrote a spec straight from the one-line intent scored worse than
the v2 planner pipeline on the same samples (partial run: -0.04 vs -0.004 GT
gain). The planner earns its keep through grounding and candidate planning; what
it lacks is discipline -- it specifies things whose carriers are not visible and
it runs long. So this arm keeps the plan and adds only the card:

    one-line intent + planner plan + skill card  ->  <=90-word spec

Same writer model as the other arms. Output is a run_bench instruction file.

    python3 scripts/eval/bench_v3/plan_v3_compile.py \
        --plans out/bench_v2/ablation/prompts/plan_only.jsonl \
        --intents out/bench_v3/inputs/track_a.t1.instructions.jsonl \
        --output-dir out/bench_v3/skills_v3
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
_SHIM = ROOT / "scripts/bench_skill/bin"
if _SHIM.is_dir() and str(_SHIM) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_SHIM}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/eval/bench_v3"))

from skills_v3 import card_block  # noqa: E402
from vlm.litellm_vlm import LiteLLMVisionModel, _extract_json_object  # noqa: E402
from vlm.local_qwen import extract_frames  # noqa: E402

SYSTEM = """You compress an edit plan into the final prompt for a video edit model.

Input: one line of what the user wants, a longer draft plan, and six frames from the source video.
Your job is to keep the plan's content but make it executable:
- English, at most 90 words, plain text.
- Keep only changes whose carriers are visible in the frames; drop the rest.
- Keep the plan's concrete elements where the frames support them; do not invent new ones.
- Verb first, one change per clause, most important first.
- No camera moves, no preservation boilerplate.

Return JSON only: {"prompt": "...", "kept": ["..."], "dropped": ["..."]}
"""


def _call(model, *, intent: str, plan: str, frames: list[str], block: str) -> str:
    user = (
        f"User request: {intent}\n\nDraft plan:\n{plan}\n\n"
        f"The six images are frames of the source video.\nWrite the final prompt."
        f"\n\n{block}"
    )
    return model.predict_text(system_prompt=SYSTEM, user_text=user, image_paths=frames, max_new_tokens=12000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", default="plan_card")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--model", default=os.environ.get("LOW_HIGH_AGENT_LITELLM_MODEL", "gemini-3.8-flash"))
    args = parser.parse_args()

    src = {json.loads(l)["sample_id"]: json.loads(l) for l in args.intents.read_text(encoding="utf-8").splitlines() if l.strip()}
    plans = [json.loads(l) for l in args.plans.read_text(encoding="utf-8").splitlines() if l.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cache"
    cache.mkdir(exist_ok=True)
    model = LiteLLMVisionModel(model=args.model, temperature=0.0)

    def work(row: dict) -> dict:
        sid = row["sample_id"]
        intent = src[sid]["intent"]
        plan = row["intent"].split("Detailed edit plan:", 1)[-1].strip()
        key = hashlib.md5(f"v3b|{sid}|{plan}|compile|{args.model}".encode()).hexdigest()
        path = cache / f"{key}.{args.arm}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        frames = extract_frames(src[sid]["source_video"], max_frames=6)
        if len(frames) < 6:
            raise RuntimeError(f"{sid}: got {len(frames)} frames -- refusing to compile an ungrounded spec")
        try:
            raw = _call(model, intent=intent, plan=plan, frames=frames, block=card_block(intent))
            payload = _extract_json_object(raw)
        except Exception as exc:
            return {"sample_id": sid, "intent": "", "source_video": src[sid]["source_video"],
                    "words": 0, "error": repr(exc)}
        prompt = re.sub(r"\s+", " ", str(payload.get("prompt") or "")).strip()
        record = {"sample_id": sid, "intent": prompt, "source_video": src[sid]["source_video"],
                  "words": len(prompt.split()), "kept": payload.get("kept") or [], "dropped": payload.get("dropped") or []}
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        out = list(pool.map(work, plans))
    bad = [r["sample_id"] for r in out if not r.get("intent")]
    if bad:
        print(f"failed: {bad}")
    print("usage:", LiteLLMVisionModel.usage_summary())
    out = [r for r in out if r.get("intent")]
    words = sorted(r["words"] for r in out)
    print(f"{args.arm}: n={len(out)} words median={words[len(words)//2]} max={words[-1]}")
    (args.output_dir / f"{args.arm}.instructions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / (args.arm + '.instructions.jsonl')}")


if __name__ == "__main__":
    main()
