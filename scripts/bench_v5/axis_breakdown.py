"""Per-axis diagnostic: which knobs does each arm under- or over-deliver?

`delivery` says how much of the target change landed; this says *where* it landed,
which is what tells you what to change in the skill. An arm can hit 0.58 overall by
over-pushing grade and light while barely moving costume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--prov", default="low_high_cost")
ap.add_argument("--arms", default="",
                help="comma-separated subset; the full table is ~200 columns wide")
args = ap.parse_args()

rows = [json.loads(l) for l in Path(args.scores).read_text().splitlines() if l.strip()]
meta = json.loads(Path(str(args.scores) + ".meta.json").read_text())
names = meta["axis_names"]
# Anchors are constant (0 and tp) and add nothing but width; the arm names that
# carry them are not always literally "noop"/"oracle".
arms = [a for a in meta["arms"] if not a.startswith(("noop", "oracle"))]
if args.arms:
    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in arms if a in want]

sub = [r for r in rows if args.prov == "ALL" or r["provenance"] == args.prov]
tp = np.array([r["tp"] for r in sub])
print(f"provenance={args.prov}  n={len(sub)}\n")

print(f"{'axis':18s} {'target':>8s} " + " ".join(f"{a[:12]:>13s}" for a in arms)
      + "   " + " ".join(f"{a[:12]:>7s}" for a in arms))
print(" " * 19 + " " * 8 + " ".join(f"{'cv':>13s}" for a in arms)
      + "   " + " ".join(f"{'cv/tp':>7s}" for a in arms))
for j, name in enumerate(names):
    t = tp[:, j].mean()
    line = f"{name:18s} {t:+8.4f} "
    cvs = []
    for a in arms:
        # Arms are not all rendered on the same sample set (the archived bundle is
        # missing clips for some methods); an arm's mean is over the clips it has.
        cv = np.array([r[a + "__cv"][j] for r in sub if a + "__cv" in r])
        cvs.append(cv.mean())
        line += f"{cv.mean():+13.4f} "
    line += "  "
    for cv in cvs:
        line += f"{cv / t if abs(t) > 1e-9 else float('nan'):+7.2f} "
    print(line)

# Is the target direction itself stable enough for `delivery` to mean anything?
U = tp / np.linalg.norm(tp, axis=1, keepdims=True)
g = U.mean(0)
g /= np.linalg.norm(g)
cos = U @ g
print(f"\ntarget-direction consistency: mean cos with the mean direction = {cos.mean():.3f} "
      f"(min {cos.min():.3f}, {int((cos > 0).sum())}/{len(cos)} positive)")
