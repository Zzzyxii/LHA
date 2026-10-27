"""Report the axis-profile comparison between arms.

Reads the per-sample json from score_profile.py and prints:

1. the target's own axis profile -- what the low->high change actually consists of,
   which is the fact the whole method has to be built around;
2. per-arm means with the noop/oracle anchors;
3. paired arm-vs-baseline tests, split by data provenance.

Paired tests only: every arm is rendered on the same sample with the same seed, so
the sample is the blocking factor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes  # noqa: E402

METRICS = ("delivery", "profile_error", "alignment", "waste", "magnitude",
           "change", "adherence")


def paired(a, b, label):
    """a and b are aligned per-sample arrays."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 3:
        return None
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    ci = (mean - 1.96 * sd / np.sqrt(n), mean + 1.96 * sd / np.sqrt(n)) if sd > 0 else (mean, mean)
    p_t = float(stats.ttest_rel(a, b).pvalue)
    try:
        p_w = float(stats.wilcoxon(a, b).pvalue)
    except Exception:  # noqa: BLE001 - all-zero differences have no sign distribution
        p_w = float("nan")
    sign_p = float(stats.binomtest(int((d > 0).sum()), n, 0.5).pvalue)
    return dict(label=label, mean=mean, ci=ci, wins=int((d > 0).sum()), n=n,
                p_t=p_t, p_w=p_w, p_sign=sign_p, median=float(np.median(d)))


def fmt_p(p):
    return "nan" if p != p else (f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline", default="direct_intent")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scores).read_text().splitlines() if l.strip()]
    meta_path = Path(str(args.scores) + ".meta.json")
    names = (json.loads(meta_path.read_text())["axis_names"] if meta_path.exists()
             else axes.axis_names())

    # Arm names are any dict-valued field that carries a 'delivery' key.
    arms = sorted({k for r in rows for k, v in r.items()
                   if isinstance(v, dict) and "delivery" in v})
    print(f"n samples = {len(rows)}   arms = {arms}\n")

    kept = sorted({k for k in names})  # axis names present in tp
    print("## 1. What the target actually changes (mean over all pairs)")
    tp = np.array([r["tp"] for r in rows])
    # Rebuild which axis indices survived: tp length tells us, but names are needed.
    print(f"   (profile has {tp.shape[1]} axes)\n")

    print("## 2. Per-arm means (anchored)\n")
    print(f"{'arm':16s} " + " ".join(f"{m[:9]:>10s}" for m in METRICS))
    for arm in arms:
        vals = [np.mean([r[arm][m] for r in rows if arm in r]) for m in METRICS]
        n = sum(1 for r in rows if arm in r)
        print(f"{arm:16s} " + " ".join(f"{v:+10.4f}" for v in vals) + f"   (n={n})")

    print("\n## 3. Paired vs baseline, by provenance\n")
    provs = sorted({r["provenance"] for r in rows} | {"ALL"})
    out_lines = []
    for prov in provs:
        sub = rows if prov == "ALL" else [r for r in rows if r["provenance"] == prov]
        for arm in arms:
            if arm == args.baseline:
                continue
            pairs = [(r[arm], r[args.baseline]) for r in sub if arm in r and args.baseline in r]
            if len(pairs) < 3:
                continue
            print(f"### {arm} vs {args.baseline} -- {prov} (n={len(pairs)})")
            print(f"{'metric':12s} {'mean d':>9s} {'95% CI':>22s} {'wins':>8s} {'p_t':>9s} {'p_w':>9s}")
            for m in METRICS:
                a = [p[0][m] for p in pairs]
                b = [p[1][m] for p in pairs]
                r = paired(a, b, m)
                if not r:
                    continue
                ci = f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]"
                print(f"{m:12s} {r['mean']:+9.4f} {ci:>22s} "
                      f"{r['wins']:>3d}/{r['n']:<4d} {fmt_p(r['p_t']):>9s} {fmt_p(r['p_w']):>9s}")
                out_lines.append(dict(prov=prov, arm=arm, **r))
            print()
    if args.out:
        Path(args.out).write_text(json.dumps(out_lines, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
