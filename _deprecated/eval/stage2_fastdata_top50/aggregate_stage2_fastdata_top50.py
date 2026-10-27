#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    ("clip_frame_t", "CLIP-F"),
    ("dino_frame", "DINO-F"),
    ("pickscore_mean", "PickScore"),
    ("jedi", "JEDi"),
    ("reward_overall", "Reward overall"),
    ("reward_narrative", "Narr./emotion"),
    ("reward_style", "Style/world"),
    ("reward_composition", "Comp./lighting"),
    ("reward_color", "Color/texture"),
    ("reward_hierarchy", "Hierarchy/readability"),
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    full_csv = args.run_root / "full_metrics_aggregate" / "summary.csv"
    pick_csv = args.run_root / "pickscore" / "summary.csv"
    jedi_csv = args.run_root / "jedi" / "results" / "jedi_val50.csv"
    methods = json.loads((args.run_root / "methods.json").read_text(encoding="utf-8"))
    method_order = [item["method"] for item in methods]

    merged: dict[str, dict[str, Any]] = {method: {"method": method} for method in method_order}
    for row in read_csv_rows(full_csv):
        method = row.get("method", "")
        if method in merged:
            merged[method].update(row)
    for row in read_csv_rows(pick_csv):
        method = row.get("method", "")
        if method in merged:
            merged[method].update(row)
    for row in read_csv_rows(jedi_csv):
        method = row.get("method", "")
        if method in merged:
            merged[method]["jedi"] = row.get("jedi_mmd_x100", "")

    records = []
    for method in method_order:
        row = dict(merged[method])
        clip_f = to_float(row.get("clip_frame_t"))
        if clip_f is not None and clip_f > 1.5:
            row["clip_frame_t"] = clip_f / 100.0
        records.append(row)

    output_csv = args.run_root / "stage2_fastdata_top50_scores.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", *[key for key, _ in COLUMNS]]
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Stage2 Fastdata Top50 Scores",
        "",
        "| Method | " + " | ".join(label for _, label in COLUMNS) + " |",
        "| --- | " + " | ".join("---:" for _ in COLUMNS) + " |",
    ]
    for row in records:
        values = [fmt(to_float(row.get(key))) for key, _ in COLUMNS]
        lines.append("| " + " | ".join([row["method"], *values]) + " |")
    output_md = args.run_root / "stage2_fastdata_top50_scores.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
