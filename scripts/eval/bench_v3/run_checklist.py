#!/usr/bin/env python3
"""Checklist judge: per (arm, sample), which requested changes are done and which
preserve items are kept, judged against the SOURCE video.

    python3 scripts/eval/bench_v3/run_checklist.py \
      --tasks tasks.jsonl --arm ours_raw=out/.../generations --arm noop=... \
      --output-dir out/bench_v3/checklist

Task rows: {"sample_id", "source_video", "requirement", "reference_images": [],
"checklist": [...], "preserve": [...]}. An arm dir holds <sample_id>.mp4.
Re-running with the same --output-dir resumes: rows already scored are skipped,
failed rows are retried. Requires LOW_HIGH_JUDGE_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judges  # noqa: E402


def build_jobs(tasks: list[dict], arms: list[tuple[str, Path]], limit: int = 0) -> tuple[list[dict], list[str]]:
    jobs, missing = [], []
    for task in tasks[: limit or None]:
        for arm, folder in arms:
            video = folder / f"{task['sample_id']}.mp4"
            if video.exists():
                jobs.append({"arm": arm, "sample_id": task["sample_id"], "video": str(video),
                             "input_digest": judges.input_digest(task, video)})
            else:
                missing.append(f"{arm}/{task['sample_id']}")
    return jobs, missing


def write_summary(output_dir: Path, rows: list[dict]) -> dict:
    summary = judges.summarize_checklist(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["| arm | n | errors | fidelity | preservation | done | partial | not_done | unrequested/sample |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    for arm, s in summary.items():
        lines.append(f"| {arm} | {s['n']} | {s['errors']} | {fmt(s['fidelity'])} | {fmt(s['preservation'])} | "
                     f"{fmt(s['done_rate'])} | {fmt(s['partial_rate'])} | {fmt(s['not_done_rate'])} | "
                     f"{fmt(s['unrequested_per_sample'])} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, help="name=dir; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    judges.add_client_args(parser)
    args = parser.parse_args()

    tasks = judges.read_jsonl(args.tasks)
    by_id = {t["sample_id"]: t for t in tasks}
    jobs, missing = build_jobs(tasks, [judges.parse_arm(a) for a in args.arm], args.limit)
    if missing:
        print(f"skipping {len(missing)} missing videos, e.g. {missing[:3]}")
    out = args.output_dir / "checklist.jsonl"
    key = ("arm", "sample_id")
    todo = judges.pending(jobs, judges.read_jsonl(out), key)
    print(f"{len(jobs)} jobs, {len(todo)} to run")
    client = judges.make_client(args)
    judges.Runner(out, args.concurrency).run(
        todo, lambda job: judges.judge_checklist(client, by_id[job["sample_id"]], Path(job["video"]), frames=args.frames)
    )
    summary = write_summary(args.output_dir, judges.current_rows(judges.read_jsonl(out), jobs, key))
    print(json.dumps(summary, indent=2))
    print(f"cost=${getattr(client, 'total_cost', 0.0):.2f}")


if __name__ == "__main__":
    main()
