"""Assemble the paper-ready results into one markdown file.

`report_profile.py`, `leaderboard.py` and `metric_audit.py` each answer one
question; this stitches them into the document that goes into the paper, so the
numbers in it are regenerated rather than transcribed. Run it after the arms are
rendered and scored:

    python3 scripts/bench_v5/final_report.py \\
        --scores scripts/bench_v5/data/scores_final.jsonl \\
        --out out/bench_v5/RESULTS_FINAL.md

Anything it cannot find (an arm that is still rendering, an IVEBench summary that
has not been produced) is reported as missing instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = ("delivery", "profile_error", "alignment", "waste", "magnitude", "change", "adherence")
AGENT_ARMS = ("g0_grounded_clean", "g1_axis_prior", "g2_axis_feedback")
BASELINES = ("direct_intent", "direct_expert")
ANCHORS = ("noop", "oracle")
REFERENCE = ("pub_Ditto", "pub_ours_full")


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def paired(sub, a, b, metric):
    pairs = [(r[a][metric], r[b][metric]) for r in sub if a in r and b in r]
    if len(pairs) < 3:
        return None
    x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    d = x - y
    p_t = float(stats.ttest_rel(x, y).pvalue)
    sd = d.std(ddof=1)
    ci = (d.mean() - 1.96 * sd / np.sqrt(len(d)), d.mean() + 1.96 * sd / np.sqrt(len(d)))
    better = int((d < 0).sum()) if metric in ("profile_error", "waste") else int((d > 0).sum())
    return dict(mean=float(d.mean()), ci=ci, wins=f"{better}/{len(d)}", p=p_t, n=len(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--prov", default="low_high_cost")
    ap.add_argument("--ids", default="",
                    help="file of sample ids to restrict to (e.g. the curated 30)")
    ap.add_argument("--ive", nargs="*", default=[], help="ivebench summary.json paths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scores).read_text().splitlines() if l.strip()]
    sub = [r for r in rows if args.prov == "ALL" or r["provenance"] == args.prov]
    if args.ids:
        keep = {l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()}
        sub = [r for r in sub if r["sample_id"] in keep]
    arms = sorted({k for r in sub for k, v in r.items()
                   if isinstance(v, dict) and "delivery" in v})
    order = [a for a in ANCHORS + BASELINES + AGENT_ARMS + REFERENCE if a in arms]
    order += [a for a in arms if a not in order]

    L = [f"# bench_v5 final results\n",
         f"provenance = {args.prov}, n samples = {len(sub)}, axes = "
         f"{len(sub[0]['tp']) if sub else 0}\n",
         "Information parity: `direct_intent` and the agent arms use the same "
         "Ditto checkpoint and render protocol. Archived `pub_Ditto` and "
         "`pub_ours_full` are answer-aware reference outputs whose detailed "
         "instructions were written after viewing low and high videos; they are "
         "not same-information controls. See `INFORMATION_AUDIT.md`.\n",
         "## 1. All arms\n"]
    header = ["arm", "n"] + list(METRICS)
    body = []
    for a in order:
        have = [r for r in sub if a in r]
        if not have:
            continue
        body.append([a, len(have)]
                    + [f"{np.mean([r[a][m] for r in have]):+.4f}" for m in METRICS])
    L.append(_md_table(header, body))
    L.append("\n`profile_error`: noop 1, oracle 0. Lower is better; it is the only "
             "scalar of the set that neither under- nor over-editing can win.\n")

    # An arm that is still rendering covers a subset, and a mean over that subset is
    # not comparable with a mean over all clips -- the first clips of a run are not
    # a random sample of the benchmark. Print the baselines restricted to the same
    # samples next to it.
    partial = [a for a in order
               if sum(1 for r in sub if a in r) < len(sub)
               and any(a in r for r in sub)]
    if partial:
        L.append("\n## 1b. Arms with partial coverage (baselines on the same samples)\n")
        for a in partial:
            have = [r for r in sub if a in r]
            L.append(f"**{a}** (n={len(have)})\n")
            rows_p = []
            for b in [x for x in ("direct_intent", "direct_expert", "pub_Ditto", a)
                      if any(x in r for r in have)]:
                rows_p.append([b, len([r for r in have if b in r])]
                              + [f"{np.mean([r[b][m] for r in have if b in r]):+.4f}"
                                 for m in ("profile_error", "delivery", "waste", "change")])
            L.append(_md_table(["arm", "n", "profile_error", "delivery", "waste", "change"],
                               rows_p))
            L.append("")

    L.append("\n## 2. Agent arms against the baselines (paired)\n")
    L.append("The primary causal baseline is `direct_intent`. Comparisons against "
             "`pub_Ditto` or `pub_ours_full` are gap-to-answer-aware-reference "
             "comparisons, not same-information tests.\n")
    for base in [b for b in BASELINES if b in arms] + [r for r in REFERENCE if r in arms]:
        for a in [x for x in AGENT_ARMS if x in arms]:
            if a == base:
                continue
            res = {m: paired(sub, a, base, m) for m in METRICS}
            res = {m: v for m, v in res.items() if v}
            if not res:
                continue
            L.append(f"### {a} vs {base} (n={max(v['n'] for v in res.values())})\n")
            L.append(_md_table(["metric", "mean d", "95% CI", "wins", "p"],
                               [[m, f"{v['mean']:+.4f}",
                                 f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]",
                                 v["wins"], f"{v['p']:.4f}"] for m, v in res.items()]))
            L.append("")

    if args.ive:
        L.append("\n## 3. IVEBench (external)\n")
        ive_rows = []
        for p in args.ive:
            p = Path(p)
            if not p.exists():
                ive_rows.append([p.parent.name + " (MISSING)", "-", "-", "-", "-"])
                continue
            d = json.loads(p.read_text())
            m = d["metrics"]
            ive_rows.append([p.parent.name.replace("ive_eval_", ""),
                             f"{m.get('overall_semantic_consistency_score', float('nan')):.4f}",
                             f"{m.get('phrase_semantic_consistency_score', float('nan')):.4f}",
                             f"{m.get('semantic_fidelity_score', float('nan')):.4f}",
                             f"{m.get('temporal_flickering_score', float('nan')):.4f}"])
        L.append(_md_table(["arm", "OSC", "PSC", "SF", "TF"], ive_rows))
        L.append("\nSource anchor: OSC 0.2363, PSC 0.1210, SF 1.0000, TF 0.9668. "
                 "Published Ditto (short subset): OSC 0.25, PSC 0.24, SF 0.89, TF 0.97.\n")

    missing = [a for a in AGENT_ARMS + BASELINES if a not in arms]
    if missing:
        L.append(f"\n## Missing\n\nNot scored (still rendering or not rendered): "
                 f"{', '.join(missing)}\n")

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out} ({len(sub)} samples, {len(order)} arms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
