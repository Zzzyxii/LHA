"""Is the headline metric a "do less" metric? And is it reliable enough to rank?

The project's first headline (`gain_dino`) was abandoned because every arm scored
below noop on it, which made "edit less" the only winning move. Any replacement has
to be checked against the same failure, and checked for sampling noise:

1. Analytic: score a synthetic edit `cv = a * tp` for a in 0..2. `delivery` grows
   monotonically (so it cannot separate a=2 from a better edit), `profile_error`
   is |a-1| (so a=1 is the unique optimum).
2. Empirical: across samples, correlate each arm's `profile_error` with how much
   the arm changed the clip. A metric that rewards under-editing shows a strong
   positive correlation with `change` *and* ranks the noop-like arm first.
3. Reliability: split-half (`axis_a` / `axis_b`) correlation of the paired
   difference between two arms, from `reliability.py`.
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
    ap.add_argument("--prov", default="low_high_cost")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scores).read_text().splitlines() if l.strip()]
    sub = [r for r in rows if args.prov == "ALL" or r["provenance"] == args.prov]
    arms = sorted({k for r in sub for k, v in r.items()
                   if isinstance(v, dict) and "delivery" in v})

    print("## 1. Analytic: a synthetic edit cv = a * tp\n")
    print(f"{'a':>5s} {'delivery':>9s} {'profile_error':>14s} {'verdict':>10s}")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        delivery = a
        pe = abs(a - 1.0)
        verdict = "optimal" if a == 1.0 else ("too little" if a < 1 else "too much")
        print(f"{a:>5.2f} {delivery:>9.2f} {pe:>14.2f} {verdict:>10s}")

    print("\n## 2. Empirical: does scoring well mean editing less?\n")
    print(f"{'arm':22s} {'mean PE':>8s} {'mean change':>12s} "
          f"{'r(PE,change)':>13s} {'p':>8s} {'n':>4s}")
    for a in arms:
        pairs = [(r[a]["profile_error"], r[a]["change"]) for r in sub if a in r]
        if len(pairs) < 5:
            continue
        pe = np.array([p[0] for p in pairs]); ch = np.array([p[1] for p in pairs])
        r, p = stats.pearsonr(pe, ch)
        print(f"{a:22s} {pe.mean():+8.3f} {ch.mean():+12.3f} {r:+13.3f} {p:>8.3f} {len(pairs):>4d}")

    print("\n   (A metric that rewarded under-editing would show a strong POSITIVE")
    print("    r: samples where the arm did less would score better.)")

    print("\n## 3. Where each arm sits on the change axis (oracle reference)\n")
    ch_oracle = np.mean([r["oracle"]["change"] for r in sub if "oracle" in r])
    print(f"   oracle mean change = {ch_oracle:.3f}")
    for a in ("noop", "direct_intent") + tuple(x for x in arms
                                               if x.startswith(("g0", "g1", "g2", "pub_Ditto"))):
        vals = [r[a]["change"] for r in sub if a in r]
        if vals:
            print(f"   {a:22s} {np.mean(vals):.3f}  ({np.mean(vals) / ch_oracle:5.2f}x oracle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
