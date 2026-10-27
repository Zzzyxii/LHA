from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit eval inputs, reference pairs, generations, and metric details for duplicate/cached samples.")
    parser.add_argument("--eval-inputs", type=Path, required=True)
    parser.add_argument("--reference-pairs-dir", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--generation-dir", action="append", default=[], help="METHOD=DIR containing eval50_*.mp4 generations.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def duplicate_groups(items: list[tuple[Any, str]]) -> list[list[str]]:
    groups: dict[Any, list[str]] = collections.defaultdict(list)
    for key, item_id in items:
        groups[key].append(item_id)
    return [values for values in groups.values() if len(values) > 1]


def audit_eval_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {}
    for name, fields in {
        "source_instruction": ("source_video", "instruction"),
        "source_target_instruction": ("source_video", "target_video", "instruction"),
        "target_instruction": ("target_video", "instruction"),
    }.items():
        items = [(tuple(row.get(field, "") for field in fields), str(row.get("sample_id", ""))) for row in rows]
        dups = duplicate_groups(items)
        checks[name] = {
            "groups": len(items) - sum(len(group) - 1 for group in dups),
            "duplicate_groups": len(dups),
            "duplicate_rows": sum(len(group) for group in dups),
            "examples": dups[:20],
        }
    return checks


def audit_video_dir(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("eval50_*.mp4"))
    items = []
    sizes = {}
    for file in files:
        digest = sha256_file(file)
        items.append((digest, file.stem))
        sizes[file.stem] = file.stat().st_size
    dups = duplicate_groups(items)
    return {
        "path": str(path),
        "files": len(files),
        "unique_hashes": len(files) - sum(len(group) - 1 for group in dups),
        "duplicate_hash_groups": len(dups),
        "duplicate_hash_rows": sum(len(group) for group in dups),
        "duplicate_examples": dups[:20],
        "sizes": sizes,
    }


def audit_reference_pairs(path: Path) -> dict[str, Any]:
    out = {"path": str(path)}
    for subdir in ["low_videos", "high_videos"]:
        video_dir = path / subdir
        out[subdir] = audit_video_dir(video_dir) if video_dir.exists() else {"path": str(video_dir), "missing": True}
    return out


def load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def audit_metrics(path: Path) -> dict[str, Any]:
    rows = load_metrics(path)
    by_method = collections.defaultdict(list)
    for row in rows:
        by_method[row.get("method", "")].append(row)
    method_summary = {}
    for method, method_rows in sorted(by_method.items()):
        method_summary[method] = {
            "rows": len(method_rows),
            "sample_ids": len({row.get("sample_id", "") for row in method_rows}),
            "edited_video_paths": len({row.get("edited_video", "") for row in method_rows}),
            "target_video_paths": len({row.get("target_video", "") for row in method_rows}),
        }
    metric_dup_checks = {}
    metric_columns = [
        "clip_t",
        "laion_aesthetic",
        "musiq",
        "niqe",
        "our_reward_overall_aesthetic_score",
    ]
    for column in metric_columns:
        items = [((row.get("method", ""), row.get(column, ""), row.get("source_video", ""), row.get("instruction", "")), row.get("sample_id", "")) for row in rows]
        dups = duplicate_groups(items)
        metric_dup_checks[column] = {
            "duplicate_groups": len(dups),
            "duplicate_examples": dups[:20],
        }
    return {
        "path": str(path),
        "rows": len(rows),
        "methods": method_summary,
        "metric_duplicate_checks": metric_dup_checks,
    }


def parse_generation_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--generation-dir must be METHOD=DIR, got: {value}")
    method, path = value.split("=", 1)
    return method.strip(), Path(path)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.eval_inputs)
    report: dict[str, Any] = {
        "eval_inputs": {
            "path": str(args.eval_inputs),
            "rows": len(rows),
            "sample_ids": len({row.get("sample_id", "") for row in rows}),
            "duplicate_checks": audit_eval_inputs(rows),
        }
    }
    if args.reference_pairs_dir is not None:
        report["reference_pairs"] = audit_reference_pairs(args.reference_pairs_dir)
    if args.metrics_csv is not None:
        report["metrics"] = audit_metrics(args.metrics_csv)
    if args.generation_dir:
        generations = {}
        for item in args.generation_dir:
            method, path = parse_generation_arg(item)
            generations[method] = audit_video_dir(path)
        report["generations"] = generations

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
