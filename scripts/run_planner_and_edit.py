#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from loop import run_planner_and_edit_loop
from _dataset import DataSample, get_sample_by_id, get_sample_by_index, smoke_sample


DEFAULT_SAMPLE_ID = "2"
DEFAULT_PROMPT_FILE = "smoke-data/prompt2.txt"
DEFAULT_LOW_VIDEO = "smoke-data/2_low-cost.mp4"
DEFAULT_OUTPUT_DIR = "out/sample_2/loop_run_wan14b_4step"
DEFAULT_VARIANT = "wan14b+4steplora"
DEFAULT_MEMORY_PATH = "out/memory/global_memory.json"
DEFAULT_MAX_ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run planner -> edit -> critic loop.")
    parser.add_argument("--pair-dir", default="", type=str,
                        help="Path to a data directory with pair.json or manifest.json.")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index of the sample to use (when --pair-dir is set). 0-based.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID,
                        help="Sample ID (when --pair-dir is set). Overrides --sample-index.")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--low-video", default=DEFAULT_LOW_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser.parse_args()


def _resolve_sample(args: argparse.Namespace) -> DataSample:
    if args.pair_dir:
        pair_dir = (PROJECT_ROOT / args.pair_dir).resolve()
        if args.sample_id and args.sample_id != DEFAULT_SAMPLE_ID:
            return get_sample_by_id(pair_dir, args.sample_id)
        return get_sample_by_index(pair_dir, args.sample_index)
    n = int(args.sample_id) if args.sample_id.isdigit() else 2
    return smoke_sample(index=n - 1)


def _resolve_output_dir(raw_output_dir: str | Path, pair_dir: str, sample_id: str) -> Path:
    output_dir = Path(raw_output_dir)
    if pair_dir:
        output_dir = output_dir / pair_dir.replace("datas/", "") / sample_id
    if not output_dir.exists():
        return output_dir
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"rerun_{timestamp}"


def main() -> None:
    args = parse_args()
    sample = _resolve_sample(args)
    os.environ.setdefault("LOW_HIGH_AGENT_KEEP_VLM_ON_GPU", "1")
    os.environ.setdefault("LOW_HIGH_AGENT_KEEP_EDIT_ON_GPU", "1")
    resolved_output_dir = _resolve_output_dir(args.output_dir, args.pair_dir, sample.sample_id)
    print(
        f"[run_planner_and_edit] start sample_id={sample.sample_id} variant={args.variant} max_attempts={args.max_attempts} output_dir={resolved_output_dir}",
        flush=True,
    )
    prompt = args.prompt.strip() if args.prompt else sample.prompt_text
    result = run_planner_and_edit_loop(
        sample_id=sample.sample_id,
        prompt=prompt,
        low_video=str(sample.low_video),
        output_dir=resolved_output_dir,
        variant=args.variant,
        memory_path=args.memory_path or None,
        max_attempts=args.max_attempts,
    )
    print(f"[run_planner_and_edit] done status={result.get('status', '')}", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
