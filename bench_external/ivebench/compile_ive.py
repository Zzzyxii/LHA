"""Compile the IVEBench arms.

`direct` is the benchmark's own edit prompt verbatim, which is what "single-pass
editing with the base model" means on this benchmark.

`skill` is the same instruction put through the project's skill compiler: one
decisive global clause, then the preservation guardrails the skill system attaches
to every dimension, and nothing else. Deliberately short -- bench_v3 measured that
the edit model does *less* the longer the prompt gets, and a style edit is the case
where that hurts most, so the arm spends its words on the restyle and on the rule
that keeps the restyle from rewriting the shot.

`skill_attr` is the point-editing counterpart, for `attribute_editing` clips
("change the sky to black", "reduce the size of the statue"). Here "be decisive"
applied globally is exactly wrong: it restyles the whole frame. The compiler keeps
the edit verbatim as the *only* change and spends its guardrail budget on
everything else staying identical -- which is also what the benchmark measures, so
this is the arm where a skill system has to show it can be strong locally and
conservative globally at the same time.

No clip-specific grounding here: IVEBench's prompts are already explicit, and the
point of the external run is that the same compiler that helps on the low->high
benchmark also helps out of the box on someone else's.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HOLD = ("Keep every subject's identity, pose, position, and count unchanged, "
        "and keep the motion and camera unchanged")
APPEARANCE = "change appearance and palette only; do not add or remove anything"


def compile_skill(instruction: str) -> str:
    text = instruction.strip().rstrip(".")
    low = text.lower()
    for prefix in ("convert the video to", "convert the video style to",
                   "transform the video into", "transform the video to",
                   "change the style of the video to", "change the video style to",
                   "change the video to", "turn the video into", "make the video"):
        if low.startswith(prefix):
            rest = text[len(prefix):].strip()
            break
    else:
        rest = None
    if rest:
        head = ("Restyle the entire clip decisively as " + rest[0].lower() + rest[1:]
                if not rest.lower().startswith(("a ", "an ", "the "))
                else "Restyle the entire clip decisively as " + rest)
    else:
        head = "Restyle the entire clip decisively as follows: " + text[0].lower() + text[1:]
    return f"{head}. {HOLD}; {APPEARANCE}."


ATTR_HOLD = ("Change only that; every other object, its colour, material and size, "
             "the background, the motion, the camera and the lighting stay exactly as "
             "they are")


def compile_attr(instruction: str) -> str:
    text = instruction.strip().rstrip(".")
    return f"{text}. Apply it completely and only to that subject. {ATTR_HOLD}."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("direct", "skill", "skill_attr"), required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    with Path(args.out).open("w") as fh:
        for r in rows:
            ins = r["instruction"]
            if args.mode == "skill":
                ins = compile_skill(ins)
            elif args.mode == "skill_attr":
                ins = compile_attr(ins)
            fh.write(json.dumps({"sample_id": r["sample_id"], "instruction": ins,
                                 "category": r["category"], "subcategory": r["subcategory"],
                                 "source_prompt": r["source_prompt"],
                                 "target_prompt": r["target_prompt"],
                                 "target_span": r["target_span"],
                                 "target_phrase": r["target_phrase"]},
                                ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {args.out} (mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
