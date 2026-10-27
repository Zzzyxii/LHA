#!/usr/bin/env python3
"""Aggregate bench_v4 scores into RESULTS.md with paired tests.

Headline metric is `gain_dino` (DINO cosine to target_high minus the source's):
an edit that moves the shot toward the real high-cost version scores positive,
one that just changes it scores negative, and `noop` is 0 by construction.
`change` is reported beside it so a low-gain arm cannot hide behind "I edited
carefully": an arm that beats the baseline on gain_dino while changing less has
to say so explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

HEADLINE = "gain_dino"
SUPPORT = ["change", "dino_change", "aes_gain", "zsq_gain", "clip_t_gain", "preservation"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sign_test(diffs: list[float]) -> tuple[int, int, float]:
    """Two-sided exact binomial p for the paired sign test, ties dropped."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, p)


def wilcoxon(diffs: list[float]) -> float:
    """Normal-approximation Wilcoxon signed-rank p, ties averaged, zeros dropped.

    The sign test throws away magnitude, so it is the conservative number; this one
    uses the ranks and is reported beside it. Implemented here rather than pulled
    from scipy so the report script has no dependency beyond the stdlib.
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 5:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return min(1.0, p)


def trimmed(diffs: list[float], drop: int = 2) -> float:
    """Mean after dropping the `drop` most extreme values on each side."""
    if len(diffs) <= 2 * drop:
        return _mean(diffs)
    ordered = sorted(diffs)
    return _mean(ordered[drop:len(ordered) - drop])


def bootstrap_ci(diffs: list[float], draws: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(_mean([rng.choice(diffs) for _ in diffs]) for _ in range(draws))
    lo = means[int(0.025 * draws)]
    hi = means[int(0.975 * draws) - 1]
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline", default="direct_intent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-by", default=None,
                    help="jsonl with sample_id + source_dataset; when given, the low_high_cost\n"
                         "subset is reported as the primary split (the task the benchmark is for)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    arms = sorted({r["arm"] for r in rows})
    by_arm = {a: {r["sample_id"]: r for r in rows if r["arm"] == a} for a in arms}

    subsets = {"all samples": None}
    if args.split_by:
        prov = {}
        for line in Path(args.split_by).read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                prov[str(d["sample_id"])] = str(d.get("source_dataset") or "?")
        subsets = {"low_high_cost (the task)": {s for s, v in prov.items() if v == "low_high_cost"},
                   "ditto style-transfer (other task)": {s for s, v in prov.items() if v != "low_high_cost"},
                   "all samples": None}

    lines = ["# bench_v4 results", "",
             f"Headline: `{HEADLINE}` (DINO cosine to `target_high` minus the source's). "
             "`noop` is 0 by construction, `oracle` is the ceiling.", ""]

    metrics = [HEADLINE] + [m for m in SUPPORT if any(m in r for a in arms for r in by_arm[a].values())]
    lines.append("## Per-arm means")
    lines.append("")
    lines.append("| arm | n | " + " | ".join(metrics) + " |")
    lines.append("|---|--|" + "|".join(["--"] * len(metrics)) + "|")
    for arm in arms:
        recs = list(by_arm[arm].values())
        cells = []
        for m in metrics:
            vals = [r[m] for r in recs if m in r]
            cells.append(f"{_mean(vals):.4f}" if vals else "-")
        lines.append(f"| {arm} | {len(recs)} | " + " | ".join(cells) + " |")
    lines.append("")

    base = by_arm.get(args.baseline, {})
    for label, keep in subsets.items():
        lines.append(f"## Paired vs `{args.baseline}` — {label}")
        lines.append("")
        for arm in arms:
            if arm == args.baseline or arm in ("noop", "oracle"):
                continue
            shared = sorted(set(by_arm[arm]) & set(base))
            if keep is not None:
                shared = [s for s in shared if s in keep]
            if not shared:
                continue
            lines.append(f"### {arm} vs {args.baseline} (n={len(shared)})")
            if HEADLINE in metrics:
                hd = [by_arm[arm][x][HEADLINE] - base[x][HEADLINE] for x in shared
                      if HEADLINE in by_arm[arm][x] and HEADLINE in base[x]]
                orc = [r[HEADLINE] for r in by_arm.get("oracle", {}).values() if HEADLINE in r]
                base_g = [base[x][HEADLINE] for x in shared if HEADLINE in base[x]]
                if hd and orc and base_g:
                    gap = abs(_mean(orc) - _mean(base_g))
                    lines.append("")
                    lines.append(f"`{HEADLINE}`: the method closes "
                                 f"{100 * _mean(hd) / gap:.1f}% of the baseline's distance to the oracle "
                                 f"(oracle {_mean(orc):+.4f}, baseline {_mean(base_g):+.4f}, gap {gap:.4f}).")
            lines.append("")
            lines.append("| metric | Δ(a−b) | 95% CI | a/b | p (sign) | p (wilcoxon) | median Δ | Δ w/o 2 extremes |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for m in metrics:
                diffs = [by_arm[arm][s][m] - base[s][m] for s in shared
                         if m in by_arm[arm][s] and m in base[s]]
                if not diffs:
                    continue
                lo, hi = bootstrap_ci(diffs)
                pos, neg, p = sign_test(diffs)
                med = sorted(diffs)[len(diffs) // 2]
                lines.append(f"| {m} | {_mean(diffs):+.4f} | [{lo:+.4f}, {hi:+.4f}] | {pos}/{neg} | {p:.4g} | {wilcoxon(diffs):.4g} | {med:+.4f} | {trimmed(diffs):+.4f} |")
            if HEADLINE in metrics:
                hd = [by_arm[arm][x][HEADLINE] - base[x][HEADLINE] for x in shared
                      if HEADLINE in by_arm[arm][x] and HEADLINE in base[x]]
                orc = [r[HEADLINE] for r in by_arm.get("oracle", {}).values() if HEADLINE in r]
                base_g = [base[x][HEADLINE] for x in shared if HEADLINE in base[x]]
                if hd and orc and base_g:
                    gap = abs(_mean(orc) - _mean(base_g))
                    lines.append("")
                    lines.append(f"`{HEADLINE}`: the method closes "
                                 f"{100 * _mean(hd) / gap:.1f}% of the baseline's distance to the oracle "
                                 f"(oracle {_mean(orc):+.4f}, baseline {_mean(base_g):+.4f}, gap {gap:.4f}).")
            lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
