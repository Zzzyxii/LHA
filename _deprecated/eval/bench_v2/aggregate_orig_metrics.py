#!/usr/bin/env python3
"""Collect the original-paper metrics for every arm into one table.

Reads the per-method outputs written by run_orig_metrics.sh, orig_metrics_jedi.py
and the substitute reward judge, and writes summary.csv / summary.md, plus a
low-high vs Ditto split (top50_sources.jsonl) for the headline columns.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OM = ROOT / "out/bench_v2/orig_metrics"
SOURCES = {r["sample_id"]: r["source_dataset"] for r in map(json.loads, open(ROOT / "scripts/eval/bench_v3/data/top50_sources.jsonl"))}

ORDER = [
    "oracle", "noop",
    "pub_ours_full", "pub_Bernini", "pub_Ditto", "pub_VACE",
    "expert_ours_raw", "expert_bernini_raw", "expert_skill_v1", "expert_skill_v2",
    "intent_direct", "intent_agent", "generic_direct", "generic_agent",
    "abl_agentprompt", "abl_plan_craft", "abl_plan_only",
]
# (column, source, per-sample field) -- direction noted in the md header
COLUMNS = [
    ("CLIP-T", "clip", "clip_t"),
    ("CLIP-F", "clip", "clip_f"),
    ("TCLIPFrame", "supp", "tclip_frame"),
    ("DINO-F", "supp", "dino_frame"),
    ("CLIP Src-Edit", "supp", "clip_src_edit"),
    ("LPIPS", "trad", "lpips_source_edit"),
    ("LAION", "laion", "laion_aesthetic"),
    ("MUSIQ", "trad", "pyiqa_musiq"),
    ("NIQE", "trad", "pyiqa_niqe"),
    ("PickScore", "pick", "pickscore"),
    ("Reward*", "reward", "overall_aesthetic_score"),
]
PUBLISHED = {  # reports/top50_video_bundle_20260725_nodup + si_stage2 report
    "pub_ours_full": {"CLIP-F": 0.9967, "DINO-F": 0.9985, "PickScore": 18.83, "JEDi": 0.48, "Reward*": 3.87},
    "pub_Bernini": {"CLIP-F": 0.9953, "DINO-F": 0.9954, "PickScore": 19.32, "JEDi": 0.50, "Reward*": 3.77},
    "pub_Ditto": {"CLIP-F": 0.9911, "DINO-F": 0.9913, "PickScore": 18.83, "JEDi": 0.16, "Reward*": 3.22},
    "pub_VACE": {"CLIP-F": 0.9884, "DINO-F": 0.9896, "PickScore": 18.09, "JEDi": 0.21, "Reward*": 2.31},
}


def _float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def per_sample(method: str) -> dict[str, dict[str, float]]:
    d = OM / method
    out: dict[str, dict[str, float]] = {}

    def put(sid, key, value):
        v = _float(value)
        if v is not None:
            out.setdefault(sid, {})[key] = v

    for r in map(json.loads, open(d / "clip/per_sample_metrics.jsonl")):
        put(r["sample_id"], "clip_t", r.get("clip_t"))
        put(r["sample_id"], "clip_f", r.get("clip_f"))
    for name, path in (("supp", d / "supp.csv"), ("laion", d / "laion.csv"), ("trad", d / "traditional.csv"),
                       ("pick", d / f"pickscore/{method}/per_sample.csv")):
        if path.exists():
            for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                for _, src, field in COLUMNS:
                    if src == name and field in r:
                        put(r["sample_id"], field, r[field])
    reward = OM / f"reward_gemini38flash/{method}.aesthetic_reward.jsonl"
    if reward.exists():
        for r in map(json.loads, open(reward)):
            put(r["sample_id"], "overall_aesthetic_score", r.get("overall_aesthetic_score"))
    return out


def mean(rows, field, ids=None):
    vals = [r[field] for sid, r in rows.items() if field in r and (ids is None or sid in ids)]
    return statistics.fmean(vals) if vals else None


def main() -> None:
    jedi = {r["method"]: r["jedi"] for r in json.load(open(OM / "jedi/jedi.json"))} if (OM / "jedi/jedi.json").exists() else {}
    table = []
    for m in ORDER:
        rows = per_sample(m)
        row = {"method": m, "n": len(rows)}
        for col, _, field in COLUMNS:
            row[col] = mean(rows, field)
        row["JEDi"] = jedi.get(m)
        for src in ("low_high_cost", "ditto"):
            ids = {s for s, v in SOURCES.items() if v == src}
            for col in ("CLIP-T", "PickScore", "Reward*", "LAION"):
                field = next(f for c, _, f in COLUMNS if c == col)
                row[f"{col}|{src}"] = mean(rows, field, ids)
        table.append(row)

    cols = ["n"] + [c for c, _, _ in COLUMNS] + ["JEDi"]
    with open(OM / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["method"] + cols + [k for k in table[0] if "|" in k])
        w.writeheader()
        w.writerows(table)

    def fmt(v, col):
        if v is None:
            return "—"
        if col in ("CLIP-F", "TCLIPFrame", "DINO-F", "CLIP Src-Edit", "LPIPS"):
            return f"{v / 100 if col == 'CLIP-F' else v:.4f}"
        return f"{v:.2f}" if col not in ("n",) else str(int(v))

    lines = ["| 组 | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for row in table:
        lines.append(f"| {row['method']} | " + " | ".join(fmt(row[c], c) for c in cols) + " |")
    lines += ["", "原表对照(published):", "", "| 组 | CLIP-F | DINO-F | PickScore | JEDi | Reward |", "|---|---|---|---|---|---|"]
    for m, p in PUBLISHED.items():
        lines.append(f"| {m} | " + " | ".join(str(p[k]) for k in ("CLIP-F", "DINO-F", "PickScore", "JEDi", "Reward*")) + " |")
    lines += ["", "按数据源拆分:", "", "| 组 | CLIP-T lh | CLIP-T ditto | PickScore lh | PickScore ditto | Reward* lh | Reward* ditto |",
              "|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append(f"| {row['method']} | " + " | ".join(
            fmt(row[f"{c}|{s}"], c) for c in ("CLIP-T", "PickScore", "Reward*") for s in ("low_high_cost", "ditto")) + " |")
    (OM / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
