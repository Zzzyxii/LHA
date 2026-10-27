#!/usr/bin/env python3
"""Aggregate latent-memory routing traces into category/slot heatmap CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_trace(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    max_tokens = 0
    for path in paths:
        trace = read_trace(path)
        mode = str(trace.get("mode") or "vace_hint")
        category = str(trace.get("category") or "unknown")
        for slot in trace.get("summary", {}).get("slots") or trace.get("summary", {}).get("layers", []):
            slot_index = int(slot.get("slot", slot.get("hint_index", 0)))
            buckets.setdefault((mode, category, slot_index), []).append(slot)
            max_tokens = max(max_tokens, len(slot.get("avg_weights", [])))

    token_fields = [f"token_{i}" for i in range(max_tokens)]
    heatmap_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for (mode, category, slot_index), slots in sorted(buckets.items()):
        num_tokens = max(len(x.get("avg_weights", [])) for x in slots)
        avg_weights = [0.0 for _ in range(num_tokens)]
        for slot in slots:
            for i, value in enumerate(slot.get("avg_weights", [])):
                avg_weights[i] += float(value)
        avg_weights = [value / len(slots) for value in avg_weights]
        row = {"mode": mode, "category": category, "slot": slot_index, "samples": len(slots)}
        for i in range(max_tokens):
            row[f"token_{i}"] = avg_weights[i] if i < len(avg_weights) else ""
        row["top_token"] = int(max(range(num_tokens), key=lambda i: avg_weights[i]))
        row["top_weight"] = max(avg_weights)
        heatmap_rows.append(row)

        entropies = [float(x["avg_entropy"]) for x in slots if x.get("avg_entropy") is not None]
        metric_rows.append(
            {
                "mode": mode,
                "category": category,
                "slot": slot_index,
                "samples": len(slots),
                "avg_entropy": sum(entropies) / len(entropies) if entropies else "",
                "avg_delta_norm": sum(float(x["avg_delta_norm"]) for x in slots) / len(slots),
                "avg_base_norm": sum(float(x.get("avg_base_norm", x.get("avg_hint_norm", 0.0))) for x in slots) / len(slots),
                "avg_delta_to_base_norm": sum(float(x.get("avg_delta_to_base_norm", x.get("avg_delta_to_hint_norm", 0.0))) for x in slots) / len(slots),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "routing_heatmap.csv", heatmap_rows, ["mode", "category", "slot", "samples", *token_fields, "top_token", "top_weight"])
    write_csv(args.output_dir / "delta_norms.csv", metric_rows, ["mode", "category", "slot", "samples", "avg_entropy", "avg_delta_norm", "avg_base_norm", "avg_delta_to_base_norm"])
    summary = {"trace_count": len(paths), "output_dir": str(args.output_dir)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
