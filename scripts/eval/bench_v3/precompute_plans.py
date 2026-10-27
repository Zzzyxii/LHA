#!/usr/bin/env python3
"""Precompute planner plans for the one-line tier (T1 agent) and write a render-ready instruction file.

The T1 agent prompt is the intent followed by the planner's own plan, with no
appended skill blocks and no memory (the "plan_only" composition, chosen because
the planning-track ablation showed the skill blocks and the memory-driven
reference image only damp the edit). Planning is API-only, so it runs here
concurrently, ahead of and separate from rendering.

Output: <output-root>/plans/<sample_id>.json (cached; reruns skip planned samples)
and <input-stem>_agent.{instructions,index}.jsonl next to the input stem
(e.g. track_a.t1 -> track_a.t1_agent), ready for run_bench.py raw arms.
Prompts are fitted to utils.prompt_budget.token_limit() tokens by fit_sections.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-stem", type=Path, required=True, help="make_instruction_file.py stem of a t1 tier")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    os.environ["LOW_HIGH_AGENT_PROMPT_MODE"] = "expand"
    from planner.core.video_planner import plan_video
    from utils.prompt_budget import fit_sections, token_limit

    rows = [json.loads(l) for l in Path(f"{args.input_stem}.instructions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    plan_dir = args.output_root / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)

    def run(row: dict) -> dict:
        path = plan_dir / f"{row['sample_id']}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_video(sample_id=row["sample_id"], prompt=row["intent"], low_video=row["source_video"],
                              memory_context={}, memory_store=None, state_path=Path(tmp) / "planner.state.json")
        text = " ".join(plan.planner_plan.split())
        if not text:
            raise RuntimeError(f"{row['sample_id']}: planner returned no plan")
        prompt, report = fit_sections(instruction=row["intent"], plan=text)
        record = {"sample_id": row["sample_id"], "intent": row["intent"], "plan": text,
                  "prompt": prompt, "budget": report}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    with ThreadPoolExecutor(args.concurrency) as pool:
        records = list(pool.map(run, rows))
    out = Path(f"{args.input_stem}_agent.instructions.jsonl")
    lines = [{"sample_id": row["sample_id"], "source_video": row["source_video"], "intent": rec["prompt"]}
             for row, rec in zip(rows, records)]
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    Path(f"{args.input_stem}_agent.index.jsonl").write_text(Path(f"{args.input_stem}.index.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    cut = sum(r["budget"].get("dropped_plan_sentences", 0) > 0 for r in records)
    print(f"{out}: {len(records)} rows; plans trimmed to fit {token_limit()} tokens: {cut}")


if __name__ == "__main__":
    main()
