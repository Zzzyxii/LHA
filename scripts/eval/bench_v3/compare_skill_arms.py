#!/usr/bin/env python3
"""One-shot comparison of every skill arm we rendered on the 50-sample bench."""

from __future__ import annotations

import json
import statistics as st
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GT = ROOT / "out/bench_v3/eval_local/skillv3_all.gt"
PLAN = ROOT / "out/bench_v3/skills_v3/plans_with_card.jsonl"
ARMS = {
    "direct": "out/bench_v2/planning/ours_intent_direct/generations",
    "plan_only": "out/bench_v2/ablation/ours_abl_plan_only/generations",
    "no_card": "out/bench_v3/runs/skillv3_nocard/ours_t1_direct/generations",
    "card": "out/bench_v3/runs/skillv3_card/ours_t1_direct/generations",
    "card_decisive": "out/bench_v3/runs/skillv3_decisive/ours_t1_direct/generations",
    "card_exec": "out/bench_v3/runs/skillv3_exec/ours_t1_direct/generations",
    "plan_card": "out/bench_v3/runs/skillv3_plancard/ours_t1_direct/generations",
}


def load(name: str) -> dict:
    path = GT / f"{name}.jsonl"
    return {r["sample_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def sign_p(w: int, l: int) -> float:
    n = w + l
    if not n:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    data = {}
    for name in ARMS:
        try:
            data[name] = load(name)
        except FileNotFoundError:
            pass
    cats = {json.loads(l)["sample_id"]: json.loads(l)["category"] for l in PLAN.read_text(encoding="utf-8").splitlines() if l.strip()}
    base = data.get("plan_only") or next(iter(data.values()))
    ids = sorted(base)
    present = [n for n in ARMS if n in data]
    print(f"{'arm':<15}{'n':>4}{'gain':>9}{'change':>8}{'src':>7}   per-category gain")
    for name in present:
        d = data[name]
        sub = [s for s in ids if s in d]
        per = "  ".join(
            f"{c[:9]}={st.fmean(d[s]['gain_dino'] for s in sub if cats[s]==c):+.3f}" if any(cats[s]==c for s in sub) else ""
            for c in ("chinese_period", "style_filter", "western_period_fantasy"))
        print(f"{name:<15}{len(sub):>4}{st.fmean(d[s]['gain_dino'] for s in sub):>+9.3f}"
              f"{st.fmean(1-d[s]['src_dino'] for s in sub):>8.2f}{st.fmean(d[s]['src_dino'] for s in sub):>7.3f}   {per}")
    print()
    for a, b in (("no_card", "card"), ("plan_only", "card"), ("plan_only", "plan_card"),
                 ("card", "card_decisive"), ("card", "card_exec"), ("direct", "plan_card")):
        if a not in data or b not in data:
            continue
        sub = [s for s in ids if s in data[a] and s in data[b]]
        d = [data[b][s]["gain_dino"] - data[a][s]["gain_dino"] for s in sub]
        w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
        print(f"{b:<14} vs {a:<12} Δ={st.fmean(d):+.3f}  {w}/{l}  p={sign_p(w, l):.3g}  n={len(sub)}")


if __name__ == "__main__":
    main()
