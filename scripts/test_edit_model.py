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

from edit import EDIT_VARIANTS, EditRequest, list_variant_status, render_edit_video
from _dataset import DataSample, get_sample_by_id, get_sample_by_index, smoke_sample


DEFAULT_SAMPLE_ID = "1"
DEFAULT_LOW_VIDEO = str(PROJECT_ROOT / "smoke-data" / "1_low-cost.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local edit-model variant on a low video.")
    parser.add_argument("--pair-dir", default="", type=str,
                        help="Path to a data directory with pair.json or manifest.json.")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index of the sample to use (when --pair-dir is set). 0-based.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID,
                        help="Sample ID (when --pair-dir is set). Overrides --sample-index.")
    parser.add_argument("--variant", default="wan14b", choices=sorted(EDIT_VARIANTS))
    parser.add_argument("--low-video", required=False, default="")
    parser.add_argument("--output-video", required=False, default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--planner-response", default="")
    parser.add_argument("--negative-prompt", default="low quality, blurry, deformed face, extra limbs, identity drift, cartoon, anime, oversaturated neon, broken geometry")
    parser.add_argument("--control-video", default="")
    parser.add_argument("--reference-image", default="")
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--list-variants", action="store_true")
    return parser.parse_args()


def _resolve_sample(args: argparse.Namespace) -> DataSample:
    if args.pair_dir:
        pair_dir = (PROJECT_ROOT / args.pair_dir).resolve()
        if args.sample_id and args.sample_id != DEFAULT_SAMPLE_ID:
            return get_sample_by_id(pair_dir, args.sample_id)
        return get_sample_by_index(pair_dir, args.sample_index)
    n = int(args.sample_id) if args.sample_id.isdigit() else 1
    return smoke_sample(index=n - 1)


def _resolve_prompt(args: argparse.Namespace, sample: DataSample) -> str:
    if args.prompt:
        return args.prompt.strip()
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.planner_response:
        payload = json.loads(Path(args.planner_response).read_text(encoding="utf-8"))
        prompt = str(payload.get("normalized_prompt", "")).strip()
        if not prompt:
            raise ValueError("planner.response.json does not contain normalized_prompt")
        return prompt
    return sample.prompt_text


def main() -> None:
    args = parse_args()
    if args.list_variants:
        print(json.dumps(list_variant_status(), ensure_ascii=False, indent=2))
        return

    sample = _resolve_sample(args)
    low_video = Path(args.low_video) if args.low_video else sample.low_video
    if not low_video:
        raise ValueError("--low-video is required unless --list-variants is used")

    if not args.output_video:
        # Auto-generate output path under out/ if not specified
        args.output_video = str(PROJECT_ROOT / "out" / "edit" / f"{sample.sample_id}_{args.variant}.mp4")

    prompt = _resolve_prompt(args, sample)
    request = EditRequest(
        low_video=low_video,
        output_video=Path(args.output_video),
        positive_prompt=prompt,
        negative_prompt=args.negative_prompt,
        variant=args.variant,
        control_video=Path(args.control_video) if args.control_video else None,
        reference_image=Path(args.reference_image) if args.reference_image else None,
        num_frames=args.num_frames or None,
        fps=args.fps or None,
        seed=args.seed,
        device=args.device,
    )
    result = render_edit_video(request)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
