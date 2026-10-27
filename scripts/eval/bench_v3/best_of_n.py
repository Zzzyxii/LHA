#!/usr/bin/env python3
"""Best-of-N without new renders: pick one of the seed replicates per sample.

The closed loop spends its extra budget rewriting the prompt between rounds, and
on the expert-instruction arm three rounds left GT gain unchanged (25/23 vs round
one). Best-of-N spends the same budget differently: N independent samples of the
same prompt, then keep one. Every seed replicate of an arm is already rendered and
scored, so the comparison is offline.

Selection uses the Gemini judge's checklist (fidelity + preservation). Gemini is
the agent's own model family and is excluded from conclusions, so it plays the
agent's critic; the outcome is then read off the judges that never saw the
selection (GPT, Claude) and the judge-free GT metrics. A GT-oracle pick is
reported as the ceiling: how much a perfect selector could gain from the same
candidates.

    python3 scripts/eval/bench_v3/best_of_n.py --tier track_a.t1 \
        --group agent=t1_agent,t1_agent_s1055,t1_agent_s2068 \
        --group direct=t1_direct,t1_direct_s1055,t1_direct_s2068
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate as ag  # noqa: E402

SELECTOR = ("fidelity_gem", "preservation_gem")
OUTCOMES = ("fidelity_luna", "fidelity_sonnet", "preservation_luna", "preservation_sonnet",
            "unrequested_luna", "unrequested_sonnet", "gain_dino", "gain_clip", "change")


def pick(rows: list[dict], key) -> dict:
    # Ties resolve to the first-listed arm (the original seed), so a selector that
    # cannot separate candidates degrades to "no selection" rather than to noise.
    return max(rows, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline best-of-N over seed replicates.")
    parser.add_argument("--tier", required=True)
    parser.add_argument("--group", action="append", required=True, help="name=arm1,arm2,...")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    groups = {}
    for spec in args.group:
        name, _, arms = spec.partition("=")
        groups[name.strip()] = [a.strip() for a in arms.split(",") if a.strip()]
    data = ag.load_tier(args.tier, {a for arms in groups.values() for a in arms})

    derived: dict[str, dict[str, dict]] = {}
    report = {"tier": args.tier, "selector": "+".join(SELECTOR), "groups": {}}
    for name, arms in groups.items():
        missing = [a for a in arms if a not in data]
        if missing:
            raise SystemExit(f"{name}: arms not scored in {args.tier}: {missing}")
        ids = sorted(set.intersection(*(set(data[a]) for a in arms)))
        ids = [s for s in ids if all(all(data[a][s].get(k) is not None for k in SELECTOR + OUTCOMES) for a in arms)]
        single, best, oracle, choice = {}, {}, {}, {}
        for s in ids:
            rows = [{"arm": a, **data[a][s]} for a in arms]
            single[s] = rows[0]
            chosen = pick(rows, lambda r: sum(r[k] for k in SELECTOR))
            best[s] = chosen
            oracle[s] = pick(rows, lambda r: r["gain_dino"])
            choice[s] = chosen["arm"]
        derived[f"{name}_single"] = single
        derived[f"{name}_best{len(arms)}"] = best
        derived[f"{name}_oracle{len(arms)}"] = oracle
        picks = {a: list(choice.values()).count(a) for a in arms}
        report["groups"][name] = {"arms": arms, "n": len(ids), "picked": picks}

    def mean(arm, key):
        return statistics.fmean(r[key] for r in derived[arm].values())

    report["means"] = {arm: {k: mean(arm, k) for k in OUTCOMES} for arm in derived}
    comparisons = []
    names = list(groups)
    for name in names:
        n = len(groups[name])
        comparisons.append((f"{name}_best{n}", f"{name}_single"))
    if len(names) == 2:
        a, b = names
        comparisons.append((f"{a}_best{len(groups[a])}", f"{b}_best{len(groups[b])}"))
        comparisons.append((f"{a}_single", f"{b}_single"))
    report["paired"] = {}
    for a, b in comparisons:
        report["paired"][f"{a} vs {b}"] = {k: ag.paired(derived, a, b, k) for k in OUTCOMES}

    print(f"tier {args.tier}; selector = Gemini judge ({' + '.join(SELECTOR)})")
    for name, g in report["groups"].items():
        print(f"  {name}: n={g['n']} picked {g['picked']}")
    print(f"\n{'arm':<18}" + "".join(f"{k:>18}" for k in OUTCOMES))
    for arm, m in report["means"].items():
        print(f"{arm:<18}" + "".join(f"{m[k]:>18.3f}" for k in OUTCOMES))
    for label, res in report["paired"].items():
        print(f"\n{label}")
        for k, r in res.items():
            if r:
                print(f"  {k:<20} Δ={r['delta']:+.3f} CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] better/worse {r['wins']}/{r['losses']} p={r['p']:.2g}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
