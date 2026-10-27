"""Rank every arm (ours and the archived published methods) on the profile metrics.

`report_profile.py` prints arms in the order they were scored, which is useless once
the comparison includes 16 published methods. This sorts by `profile_error` -- the
only scalar of the set that neither under- nor over-editing can win -- and prints
the paired difference against a chosen baseline next to it, so a rank can always be
read together with its significance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline", default="direct_intent")
    ap.add_argument("--prov", default="ALL")
    ap.add_argument("--metric", default="profile_error")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scores).read_text().splitlines() if l.strip()]
    sub = [r for r in rows if args.prov == "ALL" or r["provenance"] == args.prov]
    # Union over rows, not row 0: an arm rendered on a subset (the archived bundle is
    # missing clips for some methods, a new arm is still rendering) is absent from
    # the first row and would be silently dropped from the ranking.
    arms = sorted({k for r in sub for k, v in r.items()
                   if isinstance(v, dict) and "delivery" in v})

    def vals(arm, metric):
        return np.array([r[arm][metric] for r in sub if arm in r])

    scored = []
    for a in arms:
        v = vals(a, args.metric)
        scored.append((a, v.mean(), len(v)))
    scored.sort(key=lambda t: t[1])

    print(f"metric = {args.metric}   provenance = {args.prov}   n = {len(sub)}")
    print("NOTE: direct_intent and agent arms are same-information Ditto runs; "
          "archived pub_* rows use answer-aware expert instructions and are "
          "reference rows, not same-information controls.")
    print(f"{'arm':22s} {'mean':>8s} {'n':>4s}   vs {args.baseline}: {'mean d':>8s} "
          f"{'wins':>7s} {'p_t':>9s}")
    base_v = vals(args.baseline, args.metric)
    for arm, mean, n in scored:
        if arm == args.baseline:
            print(f"{arm:22s} {mean:+8.4f} {n:>4d}")
            continue
        pairs = [(r[arm][args.metric], r[args.baseline][args.metric])
                 for r in sub if arm in r and args.baseline in r]
        if len(pairs) < 3:
            print(f"{arm:22s} {mean:+8.4f} {n:>4d}   (no paired samples)")
            continue
        a_v = np.array([p[0] for p in pairs]); b_v = np.array([p[1] for p in pairs])
        d = a_v - b_v
        p = stats.ttest_rel(a_v, b_v).pvalue
        print(f"{arm:22s} {mean:+8.4f} {n:>4d}   {'':>{len(args.baseline) + 3}} "
              f"{d.mean():+8.4f} {int((d < 0).sum()) if args.metric != 'delivery' else int((d > 0).sum()):>3d}/{len(d):<3d} "
              f"{p:>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
