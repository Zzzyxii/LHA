#!/usr/bin/env python3
"""Precompute T2 reference translations into <output-root>/translations/ before rendering.

run_bench.py's translate arms read the same cache (planner.reference_translator.
load_or_translate). Filling it up front (API only, concurrent) means arms
rendering the same sample on different GPUs never race to translate it, and every
translate arm under one output root renders the identical instruction. The
--output-root here must equal the OUTPUT_ROOT given to launch_queue.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from planner.reference_translator import load_or_translate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.instruction_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache_dir = args.output_root / "translations"

    def run(row: dict) -> str:
        try:
            load_or_translate(cache_dir=cache_dir, sample_id=row["sample_id"], intent=row["intent"],
                              low_video=row["source_video"], reference_images=row["reference_images"])
        except Exception as exc:  # noqa: BLE001 - report and continue; run_bench retries a missing entry
            return f"error {row['sample_id']}: {type(exc).__name__}: {exc}"[:300]
        return "ok"

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(run, rows))
    errors = [r for r in results if r != "ok"]
    print(f"{cache_dir}: ok={results.count('ok')} errors={len(errors)}")
    for e in errors:
        print(" ", e)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
