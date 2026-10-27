#!/usr/bin/env python3
"""Aggregate bench_v3 results into out/bench_v3/RESULTS.md (+ results.json).

Per tier (eval_config.json): one row per arm with the three questions kept apart
  改对了吗 : checklist fidelity (per judge), GT gain_dino / gain_clip where a target exists
  保住了吗 : checklist preservation (per judge), unrequested changes per sample
  改了多少 : 1 - src_dino (to tell "did not break it" from "got it right")
then, for every configured pair, paired checklist / GT deltas with sign tests and
the pairwise verdicts of every judge plus their consensus (samples on which all
judges pick the same side). Conclusions should rest on consensus, not on one judge.
"""

from __future__ import annotations

import json
import random
import statistics
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "out/bench_v3/eval"
CFG = json.loads((ROOT / "scripts/eval/bench_v3/eval_config.json").read_text(encoding="utf-8"))
JUDGES = CFG["judges"]
QUESTIONS = ("fidelity_to_requirement", "preservation", "visual_quality")
SHORT = {"gemini-3.8-flash": "gem", "gpt-5.6-luna": "luna", "claude-sonnet-5": "sonnet"}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def judge_dir(tier: str, stage: str, model: str) -> Path:
    return EVAL / (f"{tier}.{stage}" if model == JUDGES[0] else f"{tier}.{stage}@{model}")


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(values: list[float], iters: int = 4000) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def task_ids(tier: str) -> set[str]:
    path = ROOT / "out/bench_v3/judge_tasks" / f"{tier}.jsonl"
    return {json.loads(l)["sample_id"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}


def load_tier(tier: str, arms: set[str]) -> dict[str, dict[str, dict]]:
    """arm -> sample_id -> metrics; checklist keys are suffixed with the judge's short name.

    Only arms configured for the tier are read, so a stale output left in a judge or
    GT dir by an earlier configuration cannot leak into the table."""
    ids = task_ids(tier)
    data: dict[str, dict[str, dict]] = {}
    for model in JUDGES:
        tag = SHORT.get(model, model)
        latest = {(r["arm"], r["sample_id"]): r for r in _jsonl(judge_dir(tier, "checklist", model) / "checklist.jsonl")}
        for (arm, sid), r in latest.items():
            if arm in arms and sid in ids and not r.get("error") and r.get("scores"):
                s = r["scores"]
                data.setdefault(arm, {}).setdefault(sid, {}).update(
                    {f"fidelity_{tag}": s["fidelity"], f"preservation_{tag}": s["preservation"], f"unrequested_{tag}": s["n_unrequested"]})
    gt_dir = EVAL / f"{tier}.gt"
    for path in sorted(gt_dir.glob("*.jsonl")) if gt_dir.exists() else []:
        if path.stem not in arms:
            continue
        for r in _jsonl(path):
            if r["sample_id"] in ids:
                data.setdefault(path.stem, {}).setdefault(r["sample_id"], {}).update(
                    {"gain_dino": r["gain_dino"], "gain_clip": r["gain_clip"], "change": 1 - r["src_dino"]})
    return data


def mean(rows: dict[str, dict], key: str) -> float | None:
    vals = [r[key] for r in rows.values() if r.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def fmt(v, digits=3, signed=False):
    if v is None:
        return "—"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


LOWER_IS_BETTER = ("unrequested",)


def paired(data, a, b, key):
    """Paired a−b over common samples. wins/losses count samples where a is *better*,
    so for lower-is-better keys a win is a negative difference."""
    ids = [s for s in data.get(a, {}) if s in data.get(b, {}) and data[a][s].get(key) is not None and data[b][s].get(key) is not None]
    if len(ids) < 3:
        return None
    d = [data[a][s][key] - data[b][s][key] for s in ids]
    sign = -1 if key.startswith(LOWER_IS_BETTER) else 1
    w, l = sum(sign * x > 1e-9 for x in d), sum(sign * x < -1e-9 for x in d)
    lo, hi = boot_ci(d)
    return {"n": len(ids), "delta": statistics.fmean(d), "ci": [lo, hi], "wins": w, "losses": l, "p": sign_p(w, l)}


def pairwise_verdicts(tier: str, a: str, b: str) -> dict[str, dict[str, dict]]:
    """judge -> sample_id -> {question: winner arm or 'tie'}"""
    out = {}
    for model in JUDGES:
        rows = _jsonl(judge_dir(tier, "pairwise", model) / f"{a}__vs__{b}.jsonl")
        latest = {r["sample_id"]: r for r in rows if r.get("verdict")}
        if latest:
            out[model] = {sid: r["verdict"] for sid, r in latest.items()}
    return out


def tally_question(verdicts: dict[str, dict[str, dict]], q: str, a: str, b: str,
                   consensus_of: list[str] | None = None) -> dict:
    """Per-judge a/b/tie counts for one question and the consensus: samples every
    consensus judge saw on which all of them name the same winner. ``consensus_of``
    defaults to every judge present; judges absent from ``verdicts`` are ignored."""
    per: dict = {}
    for model, v in verdicts.items():
        wins_a = sum(x[q] == a for x in v.values())
        wins_b = sum(x[q] == b for x in v.values())
        per[model] = {"a": wins_a, "b": wins_b, "tie": len(v) - wins_a - wins_b}
    pool = [m for m in (consensus_of or list(verdicts)) if m in verdicts]
    if len(pool) > 1:
        common = set.intersection(*(set(verdicts[m]) for m in pool))
        both_a = sum(all(verdicts[m][s][q] == a for m in pool) for s in common)
        both_b = sum(all(verdicts[m][s][q] == b for m in pool) for s in common)
        per["consensus"] = {"a": both_a, "b": both_b, "n": len(common), "p": sign_p(both_a, both_b), "judges": pool}
    return per


def oracle_check(arm_means: dict[str, dict], tags: list[str]) -> list[str]:
    """Arms that out-score the oracle on a judge's fidelity: if any do, that judge's
    checklist is not measuring 'reached the target' for this tier."""
    oracle = arm_means.get("oracle")
    if not oracle:
        return []
    notes = []
    for t in tags:
        ceiling = oracle.get(f"fidelity_{t}")
        if ceiling is None:
            continue
        above = [a for a, m in arm_means.items() if a != "oracle" and (m.get(f"fidelity_{t}") or -1) > ceiling]
        if above:
            notes.append(f"{t}: oracle fidelity {ceiling:.3f} is below {', '.join(above)}")
    return notes


def main() -> None:
    tags = [SHORT.get(m, m) for m in JUDGES]
    consensus_of = CFG.get("consensus_judges") or JUDGES
    out = ["# bench_v3 results", "",
           f"Auto-generated by `scripts/eval/bench_v3/aggregate.py`. Judges: {', '.join(f'{m} ({SHORT.get(m, m)})' for m in JUDGES)}.",
           f"Consensus = samples on which all of {', '.join(SHORT.get(m, m) for m in consensus_of)} agree "
           "(the agent's own model family is reported but excluded). Only consensus results should be relied on.", ""]
    results: dict = {}
    for tier, spec in CFG["tiers"].items():
        data = load_tier(tier, set(spec["arms"]))
        if not data:
            continue
        results[tier] = {"arms": {}, "comparisons": []}
        head = ["arm", "n"] + [f"改对 {t}" for t in tags] + ["gain_dino", "gain_clip"] + [f"保住 {t}" for t in tags] + [f"多改 {t}" for t in tags] + ["改了多少"]
        out += [f"## {tier}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        order = [a for a in spec["arms"] if a in data] + sorted(set(data) - set(spec["arms"]))
        for arm in order:
            rows = data[arm]
            keys = [f"fidelity_{t}" for t in tags] + ["gain_dino", "gain_clip"] + [f"preservation_{t}" for t in tags] + [f"unrequested_{t}" for t in tags] + ["change"]
            m = {k: mean(rows, k) for k in keys}
            results[tier]["arms"][arm] = {"n": len(rows), **m}
            cells = [fmt(m[k], signed=k.startswith("gain")) if not k.startswith(("unrequested", "change")) else fmt(m[k], 2) for k in keys]
            out.append(f"| {arm} | {len(rows)} | " + " | ".join(cells) + " |")
        out.append("")
        warnings = oracle_check(results[tier]["arms"], tags)
        results[tier]["oracle_warnings"] = warnings
        if warnings:
            out += ["> ⚠ checklist ceiling: " + "; ".join(warnings), ""]
        for a, b in spec.get("pairwise", []):
            if a not in data or b not in data:
                continue
            keys = [f"fidelity_{t}" for t in tags] + [f"preservation_{t}" for t in tags] + [f"unrequested_{t}" for t in tags] + ["gain_dino"]
            comp = {"a": a, "b": b, "paired": {k: paired(data, a, b, k) for k in keys}}
            out += [f"**{a} vs {b}**", "", "| 指标 | n | Δ(a−b) | 95% CI | a更好/b更好 | p |", "|---|---|---|---|---|---|"]
            for k, v in comp["paired"].items():
                if v:
                    out.append(f"| {k} | {v['n']} | {v['delta']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {v['wins']}/{v['losses']} | {v['p']:.2g} |")
            verdicts = pairwise_verdicts(tier, a, b)
            if verdicts:
                comp["judge"] = {}
                has_consensus = len([m for m in consensus_of if m in verdicts]) > 1
                cols = [f"{SHORT.get(m, m)} a/b/平" for m in verdicts] + (["共识 a/b (p)"] if has_consensus else [])
                out += ["", "| 成对 judge | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
                for q in QUESTIONS:
                    per = tally_question(verdicts, q, a, b, consensus_of)
                    comp["judge"][q] = per
                    cells = [f"{per[m]['a']}/{per[m]['b']}/{per[m]['tie']}" for m in verdicts]
                    if "consensus" in per:
                        c = per["consensus"]
                        cells.append(f"{c['a']}/{c['b']} ({c['p']:.2g})")
                    out.append(f"| {q} | " + " | ".join(cells) + " |")
            results[tier]["comparisons"].append(comp)
            out.append("")
    (ROOT / "out/bench_v3/RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (ROOT / "out/bench_v3/results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
