#!/usr/bin/env python3
"""Write the planning-track user intents for bench_v2.

The expert instructions leak the target: they were written by a VLM looking at
both LOW and HIGH frames, so they carry the target's exact colours and objects.
The planning track replaces them with what a user would actually type, so that
whatever detail reaches the edit model has to come from the agent.

Two intent sets are produced:

* ``specific`` -- one line per sample that keeps the genre, era and overall look
  ("a Qing-dynasty palace drama", "a 1950s home movie") and drops every concrete
  detail. Written by an LLM that sees only the instruction text, never a frame.
* ``generic`` -- the same fixed sentence for every sample. Closest to a real
  low-cost-to-high-cost request, but it collapses every target onto one goal, so
  its GT-referenced scores mostly measure how far each target is from "generic
  cinematic".

Leakage is checked mechanically after generation: colour words, and any
garment / ornament nouns copied from the instruction, are flagged. Flagged rows
are retried once with the offending words named, then reported if they persist,
so the file is never silently contaminated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm.litellm_vlm import LiteLLMVisionModel, _extract_json_object  # noqa: E402

GENERIC_INTENT = "Make this video look like a high-budget cinematic film."

COLOR_WORDS = (
    "red", "green", "blue", "yellow", "orange", "purple", "violet", "pink", "white",
    "black", "grey", "gray", "brown", "beige", "cream", "gold", "golden", "silver",
    "teal", "cyan", "magenta", "crimson", "emerald", "jade", "ivory", "amber", "navy",
)
# Concrete items whose presence in the target is answer-bearing. A genre word
# ("palace", "period", "vintage") is allowed; the specific object is not.
DETAIL_WORDS = (
    "headpiece", "headband", "earring", "necklace", "collar", "robe", "gown", "sleeve",
    "embroider", "gemstone", "candelabra", "curtain", "drapery", "lantern", "hairpin",
    "makeup", "lipstick", "eyeliner", "floral", "pattern",
)

SYSTEM_PROMPT = """You turn detailed video-editing instructions into the short request a
non-expert user would type before any prompt engineering happens.

Rules:
- One sentence, at most 15 words, in English.
- Keep only the overall goal: genre, era, or overall look (e.g. "a Qing-dynasty palace
  drama", "a 1950s home movie", "a dreamy pastel memory").
- Remove every concrete detail: no colours, no specific garments, ornaments, props,
  makeup, lighting setups, or background objects.
- Do not mention preservation, identity, or camera framing.
- Return JSON only: {"intent": "..."}"""


def _leaks(intent: str, instruction: str) -> list[str]:
    words = set(re.findall(r"[a-z]+", intent.lower()))
    hits = [w for w in COLOR_WORDS if w in words]
    lowered_instruction = instruction.lower()
    hits += [w for w in DETAIL_WORDS if w in intent.lower() and w in lowered_instruction]
    return hits


def _write_one(model: LiteLLMVisionModel, row: dict) -> dict:
    instruction = row["expert_instruction"]
    user = f"Detailed instruction:\n{instruction}\n\nWrite the short user request."
    intent, leaks, attempts = "", [], 0
    for attempts in (1, 2):
        text = model.predict_text(system_prompt=SYSTEM_PROMPT, user_text=user, image_paths=[], max_new_tokens=2048)
        intent = str(_extract_json_object(text).get("intent") or "").strip()
        leaks = _leaks(intent, instruction)
        if intent and not leaks and len(intent.split()) <= 18:
            break
        user = (
            f"Detailed instruction:\n{instruction}\n\n"
            f"Your previous answer was: {intent!r}. It leaked {leaks or 'too many words'}. "
            "Write the short user request again, obeying every rule."
        )
    return {
        "sample_id": row["sample_id"],
        "intent": intent,
        "category": row["category"],
        "leak_flags": leaks,
        "word_count": len(intent.split()),
        "attempts": attempts,
        "source": "llm_from_instruction_text_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write bench_v2 planning-track intents.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = LiteLLMVisionModel()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        specific = list(pool.map(lambda r: _write_one(model, r), rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "intents_specific.jsonl").open("w", encoding="utf-8") as handle:
        for item in specific:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_dir / "intents_generic.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"sample_id": row["sample_id"], "intent": GENERIC_INTENT,
                                     "category": row["category"], "source": "fixed"}, ensure_ascii=False) + "\n")

    flagged = [s for s in specific if s["leak_flags"] or not s["intent"]]
    print(f"specific intents: {len(specific)}  flagged: {len(flagged)}  usage: {LiteLLMVisionModel.usage_summary()}")
    for s in flagged:
        print(f"  FLAG {s['sample_id']}: {s['leak_flags']} -> {s['intent']}")


if __name__ == "__main__":
    main()
