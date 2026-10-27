#!/usr/bin/env python3
"""Adversarial controls for the bench_v3 judges: outputs that should score badly.

  recolor : the source clip with only a global warm grade (no content edit). A
            judge that rewards "something changed" would give it fidelity credit.
  swap    : another sample's target_high clip (polished, on-genre, wrong scene).
            A judge that only looks at the output would love it; a judge that
            checks the source must fail it on preservation.
Writes out/bench_v3/eval/_adv_<name>/<sample_id>.mp4 for the track A samples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out/bench_v3/eval"


def recolor(src: Path, dst: Path) -> None:
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        stream_in = inp.streams.video[0]
        rate = stream_in.average_rate or 16
        stream = out.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = stream_in.width, stream_in.height, "yuv420p"
        gain = np.array([1.12, 1.0, 0.82], dtype=np.float32)
        for frame in inp.decode(video=0):
            img = np.clip(frame.to_ndarray(format="rgb24").astype(np.float32) * gain, 0, 255).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "scripts/eval/bench_v3/data/track_a.jsonl").read_text().splitlines() if l.strip()]
    (OUT / "_adv_recolor").mkdir(parents=True, exist_ok=True)
    (OUT / "_adv_swap").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        dst = OUT / "_adv_recolor" / f"{r['sample_id']}.mp4"
        if not dst.exists():
            recolor(Path(r["source_video"]), dst)
        other = rows[(i + 7) % len(rows)]  # fixed derangement: never the sample itself
        link = OUT / "_adv_swap" / f"{r['sample_id']}.mp4"
        if not link.exists():
            os.symlink(other["target_video"], link)
    print("wrote", len(rows), "recolor +", len(rows), "swap controls")


if __name__ == "__main__":
    main()
