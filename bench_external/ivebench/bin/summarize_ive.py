"""Per-metric means and IVEBench dimension averages for one arm's result CSV.

`get_average_score.py` in the upstream repo is broken (parse_args returns 2 values,
main unpacks 5), and it would silently average whatever columns happen to exist.
This prints each metric separately and then the dimension means over the metrics
that were actually computed, so a run that only has three of the twelve metrics
cannot be mistaken for a full IVEBench score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DIMENSION = {
    "subject_consistency_score": "quality",
    "background_consistency_score": "quality",
    "temporal_flickering_score": "quality",
    "motion_smoothness_score": "quality",
    "vtss_score": "quality",
    "overall_semantic_consistency_score": "compliance",
    "phrase_semantic_consistency_score": "compliance",
    "instruction_satisfaction_score": "compliance",
    "quantity_accuracy_score": "compliance",
    "semantic_fidelity_score": "fidelity",
    "motion_fidelity_score": "fidelity",
    "content_fidelity_score": "fidelity",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    metrics = [c for c in df.columns if c in DIMENSION]
    means = {m: float(pd.to_numeric(df[m], errors="coerce").mean()) for m in metrics}
    dims: dict[str, list[float]] = {}
    for m, v in means.items():
        dims.setdefault(DIMENSION[m], []).append(v)

    print(f"{Path(args.csv).name}  n={len(df)}")
    for m in metrics:
        print(f"  {m:36s} {means[m]:.4f}   [{DIMENSION[m]}]")
    print("  -- dimension means (over computed metrics only) --")
    for d, vs in dims.items():
        print(f"  {d:12s} {sum(vs) / len(vs):.4f}   ({len(vs)}/"
              f"{sum(1 for k, v in DIMENSION.items() if v == d)} metrics)")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"csv": args.csv, "n": len(df), "metrics": means,
             "dimensions": {d: sum(v) / len(v) for d, v in dims.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
