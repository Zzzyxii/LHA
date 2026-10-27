"""Build the IVEBench working set: one 832x480, 32-frame mp4 per clip plus the
frame folder IVEBench's metrics expect.

Why downscale and cap: IVEBench DB is 720P-8K and 32-1024 frames. Our edit model
renders at 832x480 and its cost scales with frame count, and IVEBench ships
`resize_batch.py --max_frame` for exactly this reason. Both arms and the source go
through the identical transform, so the comparison is unaffected.

Frame naming follows IVEBench's own `mp42frames_batch.py`: `<stem>/%05d.png`, 0-based.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TARGET_W, TARGET_H = 832, 480
N_FRAMES = 32
FPS = 16


def crop_resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(TARGET_W / w, TARGET_H / h)
    nw, nh = round(w * scale), round(h * scale)
    small = frame if scale == 1.0 else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x = max(0, (nw - TARGET_W) // 2)
    y = max(0, (nh - TARGET_H) // 2)
    return small[y:y + TARGET_H, x:x + TARGET_W]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--out-frames", required=True)
    ap.add_argument("--prompts-out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = json.loads(Path(args.subset).read_text())
    mp4_dir = Path(args.out_mp4); mp4_dir.mkdir(parents=True, exist_ok=True)
    fr_dir = Path(args.out_frames); fr_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    for r in rows:
        src = Path(args.src_dir) / r["src_video_name"]
        stem = Path(r["src_video_name"]).stem
        out_mp4 = mp4_dir / f"{stem}.mp4"
        if not src.exists():
            print(f"MISSING {src}")
            continue
        if not out_mp4.exists():
            cap = cv2.VideoCapture(str(src))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                cap.release(); print(f"NOFRAMES {src}"); continue
            idx = np.linspace(0, total - 1, min(N_FRAMES, total)).astype(int)
            frames = []
            for i in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok:
                    frames.append(crop_resize(f))
            cap.release()
            if not frames:
                print(f"NOREAD {src}"); continue
            w = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                (TARGET_W, TARGET_H))
            for f in frames:
                w.write(f)
            w.release()
            fd = fr_dir / stem; fd.mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(frames):
                cv2.imwrite(str(fd / f"{k:05d}.png"), f)
        n_frames = len(list((fr_dir / stem).glob("*.png"))) if (fr_dir / stem).exists() else N_FRAMES
        kept.append(dict(sample_id=stem, src_video_name=r["src_video_name"],
                         category=r["category"], subcategory=r["subcategory"],
                         instruction=r["edit_prompt"], source_prompt=r["source_prompt"],
                         target_prompt=r["target_prompt"], target_span=r["target_span"],
                         target_phrase=r["target_phrase"], n_frames=n_frames))
        if args.limit and len(kept) >= args.limit:
            break

    with Path(args.prompts_out).open("w") as fh:
        for k in kept:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"prepared {len(kept)} clips -> {mp4_dir} | {fr_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
