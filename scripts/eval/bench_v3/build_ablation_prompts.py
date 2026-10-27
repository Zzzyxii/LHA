#!/usr/bin/env python3
"""Build per-sample prompts for the planning-track ablation.

Reuses the prompts the agent arm already sent (planner output is not re-run) and
removes sections from them, so each arm differs from the agent arm by exactly one
thing:

  agentprompt : the agent prompt verbatim     (vs agent arm: no reference image)
  plan_craft  : minus "Preserve unchanged"    (vs agentprompt: the preserve list)
  plan_only   : minus "Craft direction" too   (vs plan_craft: the craft lines)

Sections are deleted from the sent text rather than recomposed, so the plan and
craft lines are byte-identical to what the agent arm used. Output rows carry the
prompt in the ``intent`` field because that is what run_bench.py --instruction-file
reads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from utils.prompt_budget import count_tokens  # noqa: E402

HEADERS = ("Detailed edit plan", "Craft direction", "Preserve unchanged")
ARMS = {
    "agentprompt": ("Detailed edit plan", "Craft direction", "Preserve unchanged"),
    "plan_craft": ("Detailed edit plan", "Craft direction"),
    "plan_only": ("Detailed edit plan",),
}


def split_sections(prompt: str) -> tuple[str, dict[str, str]]:
    parts = re.split(r"\n\n(?=(?:%s):\n)" % "|".join(HEADERS), prompt.strip())
    sections = {}
    for part in parts[1:]:
        head, _, body = part.partition(":\n")
        sections[head] = body.strip()
    return parts[0].strip(), sections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-arm-dir", type=Path, default=ROOT / "out/bench_v2/planning/ours_intent_agent")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/bench_v2/ablation/prompts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for attempt in sorted(args.agent_arm_dir.glob("loop/*/attempt_001")):
        sample_id = attempt.parent.name
        intent, sections = split_sections((attempt / "edit.prompt.txt").read_text(encoding="utf-8"))
        if "Detailed edit plan" not in sections:
            raise SystemExit(f"{sample_id}: agent prompt has no plan section")
        for arm, keep in ARMS.items():
            body = [intent] + [f"{h}:\n{sections[h]}" for h in keep if h in sections]
            prompt = "\n\n".join(body)
            rows[arm].append({"sample_id": sample_id, "intent": prompt, "tokens": count_tokens(prompt)})

    for arm, items in rows.items():
        path = args.output_dir / f"{arm}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
        tokens = sorted(r["tokens"] for r in items)
        print(f"{arm:12s} n={len(items)} tokens median={tokens[len(tokens) // 2]} max={tokens[-1]} -> {path}")


if __name__ == "__main__":
    main()
