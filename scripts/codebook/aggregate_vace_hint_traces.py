#!/usr/bin/env python3
"""Aggregate vace_hint routing traces into category/layer heatmap CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.trace_glob))
    if not paths:
        raise FileNotFoundError(f"No traces matched: {args.trace_glob}")

    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        trace = read_trace(path)
        category = str(trace.get("category") or "unknown")
        for layer in trace.get("summary", {}).get("layers", []):
            buckets[(category, int(layer["hint_index"]))].append(layer)

    heatmap_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for (category, hint_index), layers in sorted(buckets.items()):
        num_tokens = len(layers[0]["avg_weights"])
        weights = [0.0 for _ in range(num_tokens)]
        for layer in layers:
            for i, value in enumerate(layer["avg_weights"]):
                weights[i] += float(value)
        weights = [value / len(layers) for value in weights]
        row = {"category": category, "hint_index": hint_index, "samples": len(layers)}
        row.update({f"token_{i}": weights[i] for i in range(num_tokens)})
        row["top_token"] = int(max(range(num_tokens), key=lambda i: weights[i]))
        row["top_weight"] = max(weights)
        heatmap_rows.append(row)

        metric_rows.append(
            {
                "category": category,
                "hint_index": hint_index,
                "samples": len(layers),
                "avg_entropy": sum(float(x["avg_entropy"]) for x in layers) / len(layers),
                "avg_delta_norm": sum(float(x["avg_delta_norm"]) for x in layers) / len(layers),
                "avg_hint_norm": sum(float(x["avg_hint_norm"]) for x in layers) / len(layers),
                "avg_delta_to_hint_norm": sum(float(x["avg_delta_to_hint_norm"]) for x in layers) / len(layers),
            }
        )

    token_fields = [k for k in heatmap_rows[0] if k.startswith("token_")]
    write_csv(args.output_dir / "routing_heatmap.csv", heatmap_rows, ["category", "hint_index", "samples", *token_fields, "top_token", "top_weight"])
    write_csv(args.output_dir / "delta_norms.csv", metric_rows, ["category", "hint_index", "samples", "avg_entropy", "avg_delta_norm", "avg_hint_norm", "avg_delta_to_hint_norm"])
    summary = {"trace_count": len(paths), "output_dir": str(args.output_dir)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
