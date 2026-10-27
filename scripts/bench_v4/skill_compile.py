#!/usr/bin/env python3
"""Deterministic skill compiler: one-line intent -> edit prompt, no LLM in the loop.

Why deterministic. bench_v3's prompts came from an LLM writer, which made the
agent arm impossible to reproduce: two runs of the same model give different
prompts, and when the judges disagreed there was no way to tell whether the
prompt or the judge moved the number. This compiler is a pure function of
(intent, skill card), so the arm can be re-derived from its inputs.

What it encodes, all of it measured in bench_v3 (see scripts/eval/bench_v3/skills_v3.py
and LESSONS.md):

1. The edit model under-edits as the prompt grows (change 0.21 -> 0.17 -> 0.12 as
   sections were added), so the budget is a hard 90 words and carriers are dropped
   from the bottom of the card's order until it fits -- never the global look clause.
2. Change volume correlates with getting closer to the target, so the clauses are
   written as replacements ("replace X with Y") rather than softeners ("add a hint
   of Y"), and the off-carrier rule is one sentence at the end instead of a
   qualifier on every clause.
3. Carriers the frames do not show are the main source of "invented detail" failures,
   so every carrier clause is conditional and one sentence says what to do when the
   carrier is absent. The compiler cannot see frames; the edit model can, and it is
   the one that has to apply the condition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval" / "bench_v3"))

from skills_v3 import CARDS, route  # noqa: E402

WORD_BUDGET = 90
OFF_CARRIER = ("Only change what is visible; if it is not in frame, carry the look through "
               "grade and light instead.")
DECISIVE = "Replace, do not tint: the old element must be gone, not visible underneath."
HOLD = ("Keep faces, expressions, poses, framing and the number of people unchanged.")


def _words(text: str) -> int:
    return len([w for w in text.replace("\n", " ").split() if w])


def _global_clause(intent: str) -> str:
    """Turn the user's one-liner into a whole-clip restyle clause."""
    text = intent.strip().rstrip(".")
    low = text.lower()
    for prefix in ("make it look like", "make this look like", "make this video look like",
                   "make the video look like", "turn this into", "turn it into",
                   "make it look", "make this"):
        if low.startswith(prefix):
            rest = text[len(prefix):].strip()
            return "Restyle the whole clip as " + rest + "."
    return "Restyle the whole clip as follows: " + text + "."


def _short_name(name: str) -> str:
    """First few words of a carrier description, used as the clause's subject."""
    head = name.split(" on ")[0].split(" of ")[0].strip()
    return " ".join(head.split()[:4])


def _clause(name: str, action: str) -> str:
    """One imperative clause per carrier, with the carrier named as its subject.

    Card actions are written as fragments ("upgrade to period headdress: ...",
    "period hat, styled hair with ...") because they were originally advice to an
    LLM writer, not prompt text. Emitting them bare produced subjectless prompts,
    so the carrier is attached explicitly.
    """
    verb = action.strip().rstrip(".")
    return "On the " + _short_name(name) + ", " + verb[0].lower() + verb[1:] + "."


def compile_prompt(intent: str, *, max_carriers: int = 4, decisive: bool = True,
                   include_hold: bool = True, budget: int = WORD_BUDGET) -> dict:
    card = CARDS[route(intent)]
    # Guardrails first, carriers last. The earlier ordering let four carrier
    # clauses eat the whole budget: 47/50 compiled prompts came out with no
    # preservation clause at all and 0/50 kept the off-carrier rule, which is
    # exactly the failure mode the cards exist to prevent. Carriers are the
    # expendable part -- they are domain detail, the rest is what keeps the edit
    # from drifting away from the shot.
    clauses = [_global_clause(intent)]
    guardrails = [HOLD] if include_hold else []
    guardrails.append(OFF_CARRIER)
    for sentence in guardrails:
        trial = " ".join(clauses + [sentence])
        if _words(trial) > budget:
            break
        clauses.append(sentence)
    used = []
    for name, _looks, action in card.carriers[:max_carriers]:
        clause = _clause(name, action)
        trial = " ".join(clauses + [clause])
        if _words(trial) > budget:
            break
        clauses.append(clause)
        used.append(name)
    if decisive:
        trial = " ".join(clauses + [DECISIVE])
        if _words(trial) <= budget:
            clauses.append(DECISIVE)
    prompt = " ".join(clauses)
    return {
        "prompt": prompt,
        "words": _words(prompt),
        "card": route(intent),
        "carriers_used": used,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents", required=True, help="jsonl {sample_id, intent}")
    ap.add_argument("--out", required=True, help="output jsonl {sample_id, instruction}")
    ap.add_argument("--max-carriers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=WORD_BUDGET)
    ap.add_argument("--no-decisive", action="store_true")
    ap.add_argument("--no-hold", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.intents).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            compiled = compile_prompt(
                str(row.get("intent") or row.get("instruction") or ""),
                max_carriers=args.max_carriers,
                decisive=not args.no_decisive,
                include_hold=not args.no_hold,
                budget=args.budget,
            )
            fh.write(json.dumps(
                {"sample_id": row["sample_id"], "instruction": compiled["prompt"],
                 "intent": row.get("intent") or row.get("instruction") or "",
                 "card": compiled["card"], "words": compiled["words"],
                 "carriers_used": compiled["carriers_used"]},
                ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} prompts -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
