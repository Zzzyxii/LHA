#!/usr/bin/env python3
"""Pairwise judge: per sample, A vs B on three separate questions
(fidelity_to_requirement, preservation, visual_quality), with the SOURCE visible.

    python3 scripts/eval/bench_v3/run_pairwise.py \
      --tasks tasks.jsonl --arm-a ours_raw=out/.../a --arm-b intent_direct=out/.../b \
      --output-dir out/bench_v3/pairwise

Each sample is asked in both slot orders; disagreement is a tie. Byte-identical
videos are a tie without a call. Resumable like run_checklist.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judges  # noqa: E402


def build_jobs(tasks: list[dict], dir_a: Path, dir_b: Path, limit: int = 0) -> tuple[list[dict], list[str]]:
    jobs, missing = [], []
    for task in tasks[: limit or None]:
        a, b = dir_a / f"{task['sample_id']}.mp4", dir_b / f"{task['sample_id']}.mp4"
        if a.exists() and b.exists():
            jobs.append({"sample_id": task["sample_id"], "video_a": str(a), "video_b": str(b),
                         "input_digest": judges.input_digest(task, a, b)})
        else:
            missing.append(task["sample_id"])
    return jobs, missing


def write_summary(output_dir: Path, stem: str, rows: list[dict], arm_a: str, arm_b: str) -> dict:
    summary = judges.summarize_pairwise(rows, arm_a, arm_b)
    (output_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"{arm_a} vs {arm_b}: n={summary['n']} errors={summary['errors']} identical={summary['identical_bytes']}", "",
             f"| question | {arm_a} wins | {arm_b} wins | ties | sign p | first-slot win rate |", "|---|---|---|---|---|---|"]
    for q in judges.QUESTIONS:
        s = summary[q]
        fs = "—" if s["first_slot_win_rate"] is None else f"{s['first_slot_win_rate']:.2f}"
        lines.append(f"| {q} | {s[f'wins_{arm_a}']} | {s[f'wins_{arm_b}']} | {s['ties']} | {s['sign_test_p']:.3g} | {fs} |")
    (output_dir / f"{stem}.summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--arm-a", required=True, help="name=dir")
    parser.add_argument("--arm-b", required=True, help="name=dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    judges.add_client_args(parser)
    args = parser.parse_args()

    (arm_a, dir_a), (arm_b, dir_b) = judges.parse_arm(args.arm_a), judges.parse_arm(args.arm_b)
    tasks = judges.read_jsonl(args.tasks)
    by_id = {t["sample_id"]: t for t in tasks}
    jobs, missing = build_jobs(tasks, dir_a, dir_b, args.limit)
    if missing:
        print(f"skipping {len(missing)} samples without both videos, e.g. {missing[:3]}")
    stem = f"{arm_a}__vs__{arm_b}"
    out = args.output_dir / f"{stem}.jsonl"
    key = ("sample_id",)
    todo = judges.pending(jobs, judges.read_jsonl(out), key)
    print(f"{len(jobs)} pairs, {len(todo)} to run")
    client = judges.make_client(args)
    judges.Runner(out, args.concurrency).run(
        todo,
        lambda job: judges.judge_pairwise(client, by_id[job["sample_id"]], Path(job["video_a"]),
                                          Path(job["video_b"]), arm_a, arm_b, frames=args.frames),
    )
    summary = write_summary(args.output_dir, stem, judges.current_rows(judges.read_jsonl(out), jobs, key), arm_a, arm_b)
    print(json.dumps(summary, indent=2))
    print(f"cost=${getattr(client, 'total_cost', 0.0):.2f}")


if __name__ == "__main__":
    main()
