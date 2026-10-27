#!/usr/bin/env python3
"""Aggregate Ditto paper-style per-sample metrics without rerunning models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate import METRIC_VERSION, summarize


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(payload)
    return rows


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Ditto Paper-Style Quantitative Summary", "", f"metric_version: `{METRIC_VERSION}`", ""]
    lines.append("| method | n | errors | CLIP-T | CLIP-F | VLM Overall | VLM Instr | VLM Source | VLM Quality | VLM Safety |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method, payload in summary["methods"].items():
        def fmt(metric: str) -> str:
            value = payload[metric]["mean"]
            return "" if value is None else f"{value:.4f}"

        lines.append(
            f"| {method} | {payload['num_rows']} | {payload['num_errors']} | "
            f"{fmt('clip_t')} | {fmt('clip_f')} | {fmt('vlm_overall')} | "
            f"{fmt('vlm_instruction_fidelity')} | {fmt('vlm_source_preservation')} | "
            f"{fmt('vlm_visual_quality')} | {fmt('vlm_safety')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Ditto paper-style per-sample metrics.")
    parser.add_argument("--metrics-jsonl", required=True, type=Path, nargs="+", help="One or more per_sample_metrics.jsonl files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for summary.json and summary.md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in args.metrics_jsonl:
        rows.extend(read_jsonl(path))
    summary = summarize(rows)
    write_summary(args.output_dir, summary)
    print(f"[ditto_quant.report] wrote summary to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
