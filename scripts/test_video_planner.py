#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from planner.core.video_planner import plan_video, planner_state_template
from _dataset import DataSample, get_sample_by_id, get_sample_by_index, load_samples, smoke_sample


DEFAULT_SAMPLE_ID = "1"
DEFAULT_PROMPT_FILE = str(PROJECT_ROOT / "smoke-data" / "prompt1.txt")
DEFAULT_LOW_VIDEO = str(PROJECT_ROOT / "smoke-data" / "1_low-cost.mp4")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "out" / "sample_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone video planner.")
    # --pair-dir implies loading from pair.json/manifest.json
    parser.add_argument("--pair-dir", default="", type=str,
                        help="Path to a data directory with pair.json or manifest.json. "
                             "Supports: datas/ditto, datas/ditto_face, datas/custom.")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index of the sample to use (when --pair-dir is set). 0-based.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID,
                        help="Sample ID (when --pair-dir is set). Overrides --sample-index.")
    # Manual overrides (smoke-data or custom)
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--low-video", default=DEFAULT_LOW_VIDEO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve_sample(args: argparse.Namespace) -> DataSample:
    """Resolve a DataSample from --pair-dir or fall back to smoke-data defaults."""
    if args.pair_dir:
        pair_dir = (PROJECT_ROOT / args.pair_dir).resolve()
        if args.sample_id and args.sample_id != DEFAULT_SAMPLE_ID:
            return get_sample_by_id(pair_dir, args.sample_id)
        return get_sample_by_index(pair_dir, args.sample_index)
    # smoke-data fallback
    n = int(args.sample_id) if args.sample_id.isdigit() else 1
    return smoke_sample(index=n - 1)


def main() -> None:
    args = parse_args()
    sample = _resolve_sample(args)
    output_dir = Path(args.output_dir)
    if args.pair_dir:
        output_dir = output_dir / args.pair_dir.replace("datas/", "") / sample.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_context: dict = {}

    request_payload = {
        "sample_id": sample.sample_id,
        "prompt": sample.prompt_text,
        "prompt_file": str(args.prompt_file) if not args.pair_dir else "",
        "low_video": str(sample.low_video),
        "memory_context": memory_context,
    }
    (output_dir / "planner.request.json").write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    state_template = planner_state_template()
    state_template["request"] = {
        "sample_id": sample.sample_id,
        "prompt": sample.prompt_text,
        "low_video": str(sample.low_video),
        "memory_context": memory_context,
    }
    (output_dir / "planner.template.json").write_text(json.dumps(state_template, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "planner.state.json").write_text(json.dumps(state_template, ensure_ascii=False, indent=2), encoding="utf-8")

    plan = plan_video(
        sample_id=sample.sample_id,
        prompt=sample.prompt_text,
        low_video=str(sample.low_video),
        memory_context=memory_context,
        state_path=output_dir / "planner.state.json",
    )

    (output_dir / "planner.response.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "planner.raw.txt").write_text(plan.raw_vlm_response, encoding="utf-8")
    print(str(output_dir))


if __name__ == "__main__":
    main()
