#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--expected-samples", type=int, default=50)
    args = parser.parse_args()

    methods = read_jsonl(args.run_root / "methods.json") if False else json.loads(
        (args.run_root / "methods.json").read_text(encoding="utf-8")
    )
    target_by_sample = {
        row["sample_id"]: row["target_video"]
        for row in read_jsonl(args.run_root / "high_manifest.top50.jsonl")
    }

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in methods:
        method = item["method"]
        manifests = []
        legacy_manifest = args.run_root / "infer" / method / "manifests" / "quant_manifest.jsonl"
        if legacy_manifest.is_file():
            manifests.append(legacy_manifest)
        chunk_root = args.run_root / "infer_chunks" / method
        if chunk_root.is_dir():
            manifests.extend(sorted(chunk_root.glob("chunk_*/manifests/quant_manifest.jsonl")))
        if not manifests:
            missing.append(str(legacy_manifest))
            continue
        for manifest in manifests:
            for row in read_jsonl(manifest):
                output = Path(str(row.get("edited_video", "")))
                if not output.is_file():
                    missing.append(str(output))
                    continue
                row["target_video"] = target_by_sample[row["sample_id"]]
                rows.append(row)
    if missing:
        preview = "\n".join(missing[:30])
        raise FileNotFoundError(f"missing inference outputs ({len(missing)}):\n{preview}")

    expected_methods = len(methods)
    counts = Counter(row["method"] for row in rows)
    bad = {method: count for method, count in counts.items() if count != args.expected_samples}
    if len(counts) != expected_methods or bad:
        raise RuntimeError(
            f"expected {expected_methods} methods x {args.expected_samples}; "
            f"found methods={len(counts)} bad={bad}"
        )

    full = args.run_root / "full_metrics"
    manifest_all = full / "manifest_all.jsonl"
    write_jsonl(manifest_all, rows)
    subprocess.run(
        [
            sys.executable,
            str(args.project_root / "scripts/eval/val30_full_metrics/prepare_full_metrics.py"),
            "--manifest",
            str(manifest_all),
            "--output-root",
            str(full),
            "--num-shards",
            str(args.num_shards),
            "--expected-methods",
            str(expected_methods),
            "--expected-samples",
            str(args.expected_samples),
        ],
        check=True,
    )
    print(json.dumps({"rows": len(rows), "methods": expected_methods, "full_metrics": str(full)}, indent=2))


if __name__ == "__main__":
    main()
