#!/usr/bin/env python3
"""JEDi for every method in the original-metrics manifest.

Same features and formula as the original pipeline
(eval_bundles/patches/scripts/eval/jedi): V-JEPA ViT-H/16 with the SSv2 probe,
16 frames at 224, then a degree-2 polynomial MMD (x100) against the target_high
clips of the *same* sample ids. The original extractor is hard-wired to a fixed
30-sample set, so this one only reuses its decode and scoring functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PATCHES = ROOT / "out/hf_eval_bundle/eval_bundles/patches/scripts/eval/jedi"
sys.path.insert(0, str(PATCHES))

from aggregate_val30_jedi import official_jedi  # noqa: E402
from extract_val30_features import decode_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/vjepa")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import torch
    from videojedi import VJEPA

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_method: dict[str, dict[str, str]] = {}
    targets: dict[str, str] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["sample_id"]] = row["edited_video"]
        targets[row["sample_id"]] = row["target_video"]
    by_method["_high"] = targets

    feat_dir = args.output_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    model = VJEPA(model_dir=str(args.model_dir), config_fname=str(args.model_dir / "vith16_ssv2_16x2x3.yaml"))
    features: dict[str, dict[str, np.ndarray]] = {}
    for method, videos in by_method.items():
        cache = feat_dir / f"{method}.npz"
        if cache.exists():
            data = np.load(cache)
            features[method] = {sid: data[sid] for sid in data.files}
            continue
        ids = sorted(videos)
        out: dict[str, np.ndarray] = {}
        for start in range(0, len(ids), args.batch_size):
            batch = ids[start : start + args.batch_size]
            clips = torch.stack([decode_video(Path(videos[sid]), args.frames, args.resolution) for sid in batch]).cuda()
            with torch.inference_mode():
                feats = model.get_feats(clips)
            for sid, vec in zip(batch, np.asarray(feats)):
                out[sid] = vec
        np.savez(cache, **out)
        features[method] = out
        print(f"[jedi] {method}: {len(out)} clips", flush=True)

    results = []
    for method, feats in features.items():
        if method == "_high":
            continue
        ids = sorted(feats)
        reference = np.stack([features["_high"][sid] for sid in ids])
        test = np.stack([feats[sid] for sid in ids])
        results.append({"method": method, "n": len(ids), "jedi": official_jedi(reference, test)})
        print(f"[jedi] {method:20s} n={len(ids)} jedi={results[-1]['jedi']:.3f}", flush=True)
    (args.output_dir / "jedi.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
