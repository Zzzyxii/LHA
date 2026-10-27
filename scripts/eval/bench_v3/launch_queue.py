#!/usr/bin/env python3
"""Run run_bench.py jobs across GPUs: every (arm, input, shard) is a job, each GPU works a queue.

    python3 scripts/eval/bench_v3/launch_queue.py --gpus 0,1,2,3,4,5,6,7 --shards 4 \
        --job ours_t2_agent:out/bench_v3/inputs/track_a.t2_same:out/bench_v3/runs/track_a.t2_same ...

Each --job is ARM:INPUT_STEM:OUTPUT_ROOT[:SEED], where INPUT_STEM.instructions.jsonl and
INPUT_STEM.index.jsonl come from make_instruction_file.py (SEED defaults to run_bench's 42).
Jobs are split into --shards shards and dealt round-robin to GPUs; a GPU runs its
jobs one after another (the 14B model needs a whole card). run_bench skips
finished samples, so re-running the same command resumes. Exits non-zero if any
job failed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_jobs(specs: list[str], shards: int) -> list[tuple[str, str, str, str, int]]:
    """(arm, input_stem, output_root, seed, shard) per shard of every ARM:STEM:ROOT[:SEED] spec."""
    jobs = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) not in (3, 4) or not all(parts[:3]):
            raise SystemExit(f"bad --job {spec!r}: expected ARM:INPUT_STEM:OUTPUT_ROOT[:SEED]")
        arm, stem, out_root = parts[:3]
        seed = parts[3] if len(parts) == 4 else ""
        jobs += [(arm, stem, out_root, seed, shard) for shard in range(shards)]
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="comma-separated GPU ids")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--job", action="append", required=True, help="ARM:INPUT_STEM:OUTPUT_ROOT[:SEED]")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "out/bench_v3/logs")
    args = parser.parse_args()

    jobs = parse_jobs(args.job, args.shards)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    queues = {g: jobs[i::len(gpus)] for i, g in enumerate(gpus)}
    args.log_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    def work(gpu: str) -> None:
        for arm, stem, out_root, seed, shard in queues[gpu]:
            tag = f"{Path(out_root).name}.{arm}.shard{shard}"
            cmd = [sys.executable, str(ROOT / "scripts/bench_skill/run_bench.py"), "--arm", arm,
                   "--sample-index", f"{stem}.index.jsonl", "--instruction-file", f"{stem}.instructions.jsonl",
                   "--output-root", out_root, "--shard-index", str(shard), "--shard-count", str(args.shards),
                   "--max-attempts", "1"] + (["--seed", seed] if seed else [])
            with open(args.log_dir / f"{tag}.log", "a", encoding="utf-8") as log:
                code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
            print(f"[{time.strftime('%H:%M:%S')}] gpu{gpu} {tag} exit={code}", flush=True)
            if code:
                failures.append(tag)

    threads = [threading.Thread(target=work, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"ALL DONE failures={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
