"""Decide which candidate axes actually carry the low->high change.

An axis is usable only if `margin(target) - margin(source)` points the same way on
most of the 50 pairs. That is a stronger requirement than "the mean moved": a mean
can be dragged by five samples while the axis is noise on the other 45, and an axis
that is noise contributes nothing but variance to the profile vector.

Reports, per axis: mean shift, paired t statistic, win rate, and Cohen's d.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--min-t", type=float, default=2.0,
                    help="absolute paired t required to keep an axis")
    ap.add_argument("--sources", default="scripts/bench_v5/data/top50_sources.jsonl")
    ap.add_argument("--prov", default="ALL",
                    help="only use pairs of this provenance (e.g. low_high_cost)")
    ap.add_argument("--split", default="",
                    help="json from make_profile.py: restrict to its calibration ids")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cpath = Path(args.cache)
    if cpath.is_dir():
        cache = {}
        for f in cpath.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            cache[d["key"]] = d
    else:
        cache = json.loads(cpath.read_text())
    samples = sorted(k.split("/", 1)[1] for k in cache if k.startswith("src/"))
    samples = [s for s in samples if f"tgt/{s}" in cache]
    if args.prov != "ALL":
        prov = {}
        for l in Path(args.sources).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                prov[r["sample_id"]] = r.get("source_dataset", "?")
        samples = [s for s in samples if prov.get(s, "?") == args.prov]
    if args.split:
        calib = set(json.loads(Path(args.split).read_text())["calibration"])
        samples = [s for s in samples if s in calib]
    names = axes.axis_names()

    S = np.array([cache[f"src/{s}"]["axis"] for s in samples])
    T = np.array([cache[f"tgt/{s}"]["axis"] for s in samples])
    D = T - S
    n = len(samples)
    print(f"n pairs = {n}\n")
    print(f"{'axis':18s} {'mean':>8s} {'sd':>8s} {'t':>7s} {'d':>6s} {'wins':>7s}  keep")
    keep = []
    rows = []
    for j, name in enumerate(names):
        d = D[:, j]
        mean = d.mean()
        sd = d.std(ddof=1)
        t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
        cohen = mean / sd if sd > 0 else 0.0
        wins = int((d > 0).sum())
        ok = abs(t) >= args.min_t
        if ok:
            keep.append(name)
        print(f"{name:18s} {mean:+8.4f} {sd:8.4f} {t:+7.2f} {cohen:+6.2f} "
              f"{wins:>3d}/{n:<3d}  {'yes' if ok else 'NO'}")
        rows.append(dict(axis=name, mean=float(mean), sd=float(sd), t=float(t),
                         cohen=float(cohen), wins=wins, n=n, keep=bool(ok)))

    print(f"\nkept {len(keep)}/{len(names)}: {keep}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(n=n, axes=rows, keep=keep), indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
