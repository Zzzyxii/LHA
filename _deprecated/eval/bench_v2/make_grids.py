#!/usr/bin/env python3
"""Side-by-side grids for the planning track: low | target | direct | agent.

Each panel is resized to 416x240 and labelled; all four clips are resampled to
the source clip's frame count so they play in lockstep. Output plays at the
source fps. Use for eyeballing what the GT metric numbers mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "reports/top50_video_bundle_20260725_nodup/videos"
PLANNING = ROOT / "out/bench_v2/planning"
FONT = "/usr/share/fonts/cjk"
W, H = 416, 240


def _frames(path: Path) -> tuple[list[np.ndarray], float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 16)
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return frames, fps


def _font(size: int):
    for candidate in sorted(Path(FONT).rglob("*.tt[cf]")) + sorted(Path(FONT).rglob("*.otf")):
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel(frame: np.ndarray, label: str, font) -> np.ndarray:
    image = Image.fromarray(frame).resize((W, H), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, W, 30], fill=(0, 0, 0))
    draw.text((8, 3), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--form", default="intent", choices=("intent", "generic"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/bench_v2/grids")
    args = parser.parse_args()

    gt = {
        arm: {r["sample_id"]: r for r in map(json.loads, open(ROOT / f"out/bench_v2/gt_planning/{arm}.jsonl"))}
        for arm in (f"{args.form}_direct", f"{args.form}_agent")
    }
    intents_file = "intents_specific.jsonl" if args.form == "intent" else "intents_generic.jsonl"
    intents = {
        r["sample_id"]: r["intent"]
        for r in map(json.loads, open(ROOT / "scripts/eval/bench_v3/data" / intents_file))
    }
    pairwise_path = ROOT / f"out/bench_v2/pairwise_planning/{args.form}_agent__vs__{args.form}_direct.jsonl"
    verdict_names = {f"{args.form}_agent": "judge选助手", f"{args.form}_direct": "judge选直接喂", "tie": "judge平局"}
    verdicts = {
        r["sample_id"]: verdict_names.get(r.get("verdict"), str(r.get("verdict")))
        for r in map(json.loads, open(pairwise_path))
    } if pairwise_path.exists() else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font, small = _font(20), _font(18)

    for sample_id in args.sample_ids:
        d = gt[f"{args.form}_direct"][sample_id]
        a = gt[f"{args.form}_agent"][sample_id]
        clips = [
            (BUNDLE / "source_low" / f"{sample_id}.mp4", "原视频 low"),
            (BUNDLE / "target_high" / f"{sample_id}.mp4", "标准答案 high"),
            (PLANNING / f"ours_{args.form}_direct/generations/{sample_id}.mp4",
             f"直接喂  gain={d['gain_dino']:+.3f}  改动={1 - d['src_dino']:.2f}"),
            (PLANNING / f"ours_{args.form}_agent/generations/{sample_id}.mp4",
             f"助手扩写  gain={a['gain_dino']:+.3f}  改动={1 - a['src_dino']:.2f}"),
        ]
        decoded = [(_frames(path), label) for path, label in clips]
        (low_frames, fps), _ = decoded[0]
        n = len(low_frames)
        caption = Image.new("RGB", (2 * W, 36), (30, 30, 30))
        verdict = verdicts.get(sample_id, "")
        ImageDraw.Draw(caption).text(
            (8, 7), f"{sample_id[-4:]} [{verdict}] {intents[sample_id]}", fill=(255, 220, 120), font=small
        )
        caption_arr = np.asarray(caption)

        out_path = args.output_dir / f"{args.form}_{sample_id}.mp4"
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7, macro_block_size=1)
        for t in range(n):
            panels = []
            for (frames, _), label in decoded:
                frame = frames[min(len(frames) - 1, round(t * len(frames) / n))]
                panels.append(_panel(frame, label, font))
            grid = np.vstack([caption_arr, np.hstack(panels[:2]), np.hstack(panels[2:])])
            writer.append_data(grid)
        writer.close()
        print(out_path)


if __name__ == "__main__":
    main()
