"""Freeze a calibration split and estimate the low->high axis profile from it.

Why a split at all: the profile says *which axes a low->high upgrade moves and by
how much*, and that is exactly the kind of prior a skill would carry. Estimating
it on the same clips you then report on would let the method memorise the answer
per sample. So 15 clips are frozen as calibration (every third clip within each
provenance stratum, decided before any profile was computed) and only the other 35
are reported as held-out.

The profile itself is a *global* prior: one direction plus one per-axis scale for
the whole task family. It carries no per-sample target information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sources", default="scripts/bench_v5/data/top50_sources.jsonl")
    ap.add_argument("--axis-validation", default="scripts/bench_v5/data/axis_validation.json")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--prov", default="low_high_cost",
                    help="stratum the profile is estimated in and stratified over; "
                         "everything else goes to held-out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-out", default="")
    args = ap.parse_args()

    val = json.loads(Path(args.axis_validation).read_text())
    keep = set(val["keep"])
    import axes
    names = [n for n in axes.axis_names() if n in keep]

    prov = {}
    for l in Path(args.sources).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            prov[r["sample_id"]] = r.get("source_dataset", "?")

    cache = {}
    for f in Path(args.cache_dir).glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        cache[d["key"]] = d

    samples = sorted(k.split("/", 1)[1] for k in cache if k.startswith("src/"))
    samples = [s for s in samples if f"tgt/{s}" in cache]

    # Stratify inside the task stratum only. The other strata (ditto) are a
    # different task -- pooling them in moved the prior's skin_finish and headwear
    # terms by more than their own magnitude.
    calib, held = [], []
    task = sorted(s for s in samples if prov.get(s, "?") == args.prov)
    for i, s in enumerate(task):
        (calib if i % args.every == 0 else held).append(s)
    held += sorted(s for s in samples if prov.get(s, "?") != args.prov)

    # Index into the FULL axis vector, not into the kept subset. names.index(n)
    # silently returns the position within the kept list, which silently shifted
    # every axis by one once three axes were dropped.
    full = axes.axis_names()
    idx = [full.index(n) for n in names]
    def tp_vec(s):
        a = np.array(cache[f"src/{s}"]["axis"])[idx]
        b = np.array(cache[f"tgt/{s}"]["axis"])[idx]
        return b - a

    T = np.array([tp_vec(s) for s in calib])
    mean_tp = T.mean(0)
    unit_dir = mean_tp / np.linalg.norm(mean_tp)
    scale = float(np.linalg.norm(mean_tp))

    out = dict(
        calibration=calib, held_out=held,
        axis_names=names,
        mean_tp=[float(x) for x in mean_tp],
        unit_direction=[float(x) for x in unit_dir],
        scale=scale,
        note="global prior estimated on the calibration split only",
    )
    Path(args.out).write_text(json.dumps(out, indent=1))
    if args.split_out:
        Path(args.split_out).write_text(json.dumps(
            dict(calibration=calib, held_out=held), indent=1))
    print(f"calibration n={len(calib)}  held-out n={len(held)}  |profile|={scale:.4f}")
    for n, v in sorted(zip(names, mean_tp), key=lambda kv: -abs(kv[1])):
        print(f"  {n:18s} {v:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
