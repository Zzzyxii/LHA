"""Convert rendered IVEBench mp4s into the frame folders IVEBench's metrics read.

Same layout as IVEBench's own `mp42frames_batch.py` (`<stem>/%05d.png`, 0-based),
written with cv2 because the workstation's ffmpeg is missing libssl.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    src = Path(args.mp4_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for mp4 in sorted(src.glob("*.mp4")):
        stem = mp4.stem
        fd = out / stem
        if fd.exists() and any(fd.glob("*.png")):
            n += 1
            continue
        fd.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(mp4))
        k = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(fd / f"{k:05d}.png"), frame)
            k += 1
        cap.release()
        if k:
            n += 1
    print(f"wrote frame folders for {n} clips -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
