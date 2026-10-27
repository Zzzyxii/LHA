"""Split-half reliability of the axis-profile metrics.

Each clip's axis vector is also computed on its even frames and its odd frames
separately (`axis_a` / `axis_b` in the cache). If a metric were mostly sampling
noise, two halves of the same clip would disagree, no arm could ever be separated,
and adding frames would not tighten anything. This reports how much of the variance
in the *paired difference between two arms* is real.

Spearman-Brown then gives what the full-frame estimate's reliability is, which is
the ceiling on any p-value the comparison could reach.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(cdir: Path) -> dict:
    out = {}
    for f in cdir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        out[d["key"]] = d
    return out


def delivery(src, tgt, out, idx):
    tp = (tgt - src)[idx]
    cv = (out - src)[idx]
    n = float(np.linalg.norm(tp))
    if n < 1e-12:
        return None
    return float(cv @ (tp / n)) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--axis-validation", required=True)
    ap.add_argument("--baseline", default="direct_intent")
    ap.add_argument("--prov", default="low_high_cost")
    ap.add_argument("--sources", default="scripts/bench_v5/data/top50_sources.jsonl")
    args = ap.parse_args()

    val = json.loads(Path(args.axis_validation).read_text())
    import axes
    full = axes.axis_names()
    idx = np.array([full.index(n) for n in full if n in set(val["keep"])])

    cache = load(Path(args.cache_dir))
    prov = {}
    for l in Path(args.sources).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            prov[r["sample_id"]] = r.get("source_dataset", "?")

    samples = sorted(k.split("/", 1)[1] for k in cache if k.startswith("src/"))
    samples = [s for s in samples if f"tgt/{s}" in cache and prov.get(s, "?") == args.prov
               and cache[f"src/{s}"].get("axis_a")]
    if not samples:
        print("no samples with split-half axis vectors -- re-run extract_cache.py")
        return 1

    arms = sorted({k.split("/", 1)[0] for k in cache
                   if "/" in k and not k.startswith(("src/", "tgt/"))})
    # An arm rendered on a subset (archived bundle, or still rendering) has no entry
    # for samples[0]; keep it if it covers *any* sample of this provenance.
    arms = [a for a in arms if any(f"{a}/{s}" in cache for s in samples)]

    print(f"n={len(samples)} ({args.prov}), axes={len(idx)}, arms={arms}\n")
    print(f"{'arm':16s} {'r(delivery_a, delivery_b)':>26s} {'r(diff_a, diff_b)':>20s}")
    for arm in arms:
        da, db = [], []
        for s in samples:
            if f"{arm}/{s}" not in cache:
                continue
            src = np.array(cache[f"src/{s}"]["axis"]); tgt = np.array(cache[f"tgt/{s}"]["axis"])
            oa = np.array(cache[f"{arm}/{s}"].get("axis_a") or [])
            ob = np.array(cache[f"{arm}/{s}"].get("axis_b") or [])
            sa = np.array(cache[f"src/{s}"]["axis_a"]); sb = np.array(cache[f"src/{s}"]["axis_b"])
            ta = np.array(cache[f"tgt/{s}"]["axis_a"]); tb = np.array(cache[f"tgt/{s}"]["axis_b"])
            if min(len(oa), len(ob)) == 0:
                continue
            da.append(delivery(sa, ta, oa, idx)); db.append(delivery(sb, tb, ob, idx))
        da = np.array([x for x in da if x is not None]); db = np.array([x for x in db if x is not None])
        r = stats.pearsonr(da, db)[0] if len(da) > 3 else float("nan")
        line = f"{arm:16s} {r:>26.3f}"
        if arm != args.baseline:
            ba, bb = [], []
            for s in samples:
                if f"{arm}/{s}" not in cache or f"{args.baseline}/{s}" not in cache:
                    continue
                sa = np.array(cache[f"src/{s}"]["axis_a"]); sb = np.array(cache[f"src/{s}"]["axis_b"])
                ta = np.array(cache[f"tgt/{s}"]["axis_a"]); tb = np.array(cache[f"tgt/{s}"]["axis_b"])
                oa = np.array(cache[f"{arm}/{s}"].get("axis_a") or [])
                ob = np.array(cache[f"{arm}/{s}"].get("axis_b") or [])
                pa = np.array(cache[f"{args.baseline}/{s}"].get("axis_a") or [])
                pb = np.array(cache[f"{args.baseline}/{s}"].get("axis_b") or [])
                if min(len(oa), len(ob), len(pa), len(pb)) == 0:
                    continue
                xa, xb = delivery(sa, ta, oa, idx), delivery(sb, tb, ob, idx)
                ya, yb = delivery(sa, ta, pa, idx), delivery(sb, tb, pb, idx)
                if None in (xa, xb, ya, yb):
                    continue
                ba.append(xa - ya); bb.append(xb - yb)
            rd = stats.pearsonr(ba, bb)[0] if len(ba) > 3 else float("nan")
            sb_full = 2 * rd / (1 + rd) if rd == rd and rd > -1 else float("nan")
            line += f" {rd:>20.3f}   (full-frame reliability ~{sb_full:.3f})"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
