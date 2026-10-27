#!/usr/bin/env python3
"""Re-pick each closed-loop sample's representative video from attempts on disk.

The first bench pass kept the final attempt whenever the loop exhausted its
budget, but refinement frequently regresses, so that systematically published a
worse clip than the critic's own scores justified. Attempt videos and their
evaluations are all retained, so the selection can be corrected without
regenerating anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "bench_skill"))

from run_bench import _accepted_video  # noqa: E402


def _loop_result(loop_dir: Path) -> dict:
    """Reconstruct just enough of the loop result to resolve an accepted attempt."""
    trace_path = loop_dir / "loop.trace.json"
    if not trace_path.exists():
        return {}
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    for attempt in trace.get("attempts") or []:
        if attempt.get("action") == "accept":
            return {"attempt_index": attempt.get("attempt_index")}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-select bench videos from stored attempts.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for arm in args.arms:
        arm_dir = args.run_root / arm
        loop_root = arm_dir / "loop"
        gen_dir = arm_dir / "generations"
        if not loop_root.is_dir():
            print(f"[reselect] {arm}: no loop/ directory, skipping")
            continue

        changed = 0
        checked = 0
        for sample_dir in sorted(loop_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            target = gen_dir / f"{sample_dir.name}.mp4"
            if not target.exists():
                continue
            checked += 1
            best = _accepted_video(sample_dir, _loop_result(sample_dir))
            if best is None:
                continue
            if best.read_bytes() == target.read_bytes():
                continue
            changed += 1
            print(f"[reselect] {arm} {sample_dir.name}: -> {best.parent.name}")
            if not args.dry_run:
                shutil.copyfile(best, target)

        verb = "would change" if args.dry_run else "changed"
        print(f"[reselect] {arm}: {verb} {changed}/{checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
