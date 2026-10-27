"""Does a published method's output cover the whole clip, or only its head?

The archived method bundles have different frame counts and fps from the release
source (81 frames @15.75 fps): most are 17 frames @20 fps, which is 0.85 s if the
timestamps are real but could also be 17 uniformly sampled frames spanning the
whole 5 s clip. bench_v5 compares an output's axis vector against the *source's*,
so the answer decides whether those methods can be scored at all.

For each output frame this finds the source frame with the highest CLIP image
similarity: a slope near (n_src / n_out) means the output spans the clip, a slope
near 1 means it only covers the head.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import featurize  # noqa: E402


def all_frames(path: Path) -> np.ndarray:
    return featurize.image_matrix(featurize.read_frames(path, 512), batch_size=32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    args = ap.parse_args()

    src = all_frames(Path(args.source))
    print(f"source {Path(args.source).name}: {src.shape[0]} frames", flush=True)
    for other in args.others:
        out = all_frames(Path(other))
        sim = out @ src.T
        idx = sim.argmax(axis=1)
        slope = float(np.polyfit(np.arange(len(idx)), idx, 1)[0])
        print(f"{Path(other).parent.name}: n={len(idx)} first->{idx[0]} last->{idx[-1]} "
              f"slope={slope:.2f} span_slope={src.shape[0] / max(1, len(idx)):.2f} "
              f"mean_max_sim={sim.max(axis=1).mean():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
