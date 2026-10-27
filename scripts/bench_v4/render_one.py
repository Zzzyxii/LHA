#!/usr/bin/env python3
"""Render one (sample, instruction, variant) triple through the shared edit backend.

bench_v4 entry point for GPU work. Kept deliberately thin: everything that can
differ between arms (variant, prompt, seed, frame count) is a flag, so an arm is
fully described by the command line that produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "Ditto"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed face, extra limbs, identity drift, cartoon, anime, "
    "oversaturated neon, broken geometry"
)
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "reports" / "top50_video_bundle_20260725_nodup" / "videos" / "source_low"


def load_instructions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        out[str(item["sample_id"])] = str(item.get("instruction") or item.get("intent") or "").strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--instruction-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    ap.add_argument("--verify-existing-metadata", action="store_true",
                    help="only resume an existing output if its sidecar matches this request")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from edit import EditRequest, render_edit_video

    instructions = load_instructions(args.instruction_file)
    if args.sample_id not in instructions:
        raise SystemExit("no instruction for %s in %s" % (args.sample_id, args.instruction_file))
    low_video = args.source_dir / ("%s.mp4" % args.sample_id)
    if not low_video.exists():
        raise SystemExit("source missing: %s" % low_video)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / ("%s.mp4" % args.sample_id)
    if final.exists() and not args.overwrite:
        if args.verify_existing_metadata:
            meta_path = out_dir / ("%s.meta.json" % args.sample_id)
            if not final.is_file() or final.stat().st_size == 0 or not meta_path.is_file():
                raise SystemExit(f"existing output lacks a valid video/metadata pair: {final}")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = {
                "sample_id": args.sample_id,
                "variant": args.variant,
                "seed": args.seed,
                "source": str(low_video),
                "instruction": instructions[args.sample_id],
                "negative_prompt": args.negative_prompt,
                "requested_num_frames": args.num_frames,
                "requested_fps": args.fps,
            }
            mismatches = [
                key for key, value in expected.items() if metadata.get(key) != value
            ]
            if mismatches:
                raise SystemExit(
                    f"existing output request mismatch for {args.sample_id}: "
                    + ", ".join(mismatches)
                )
        print("skip %s (exists)" % args.sample_id)
        return 0

    stage = out_dir / "stage" / ("%s.mp4" % args.sample_id)
    stage.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = render_edit_video(
        EditRequest(
            low_video=low_video,
            output_video=stage,
            positive_prompt=instructions[args.sample_id],
            negative_prompt=args.negative_prompt,
            variant=args.variant,
            num_frames=args.num_frames if args.num_frames > 0 else None,
            fps=args.fps if args.fps > 0 else None,
            seed=args.seed,
            device=args.device,
        )
    )
    elapsed = time.time() - started
    final.write_bytes(stage.read_bytes())
    (out_dir / ("%s.meta.json" % args.sample_id)).write_text(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "variant": args.variant,
                "seed": args.seed,
                "source": str(low_video),
                "instruction": instructions[args.sample_id],
                "negative_prompt": args.negative_prompt,
                "requested_num_frames": args.num_frames,
                "requested_fps": args.fps,
                "seconds": round(elapsed, 1),
                "response": payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("done %s variant=%s seconds=%.1f -> %s" % (args.sample_id, args.variant, elapsed, final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
