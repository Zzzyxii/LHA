from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CODE_FILES = [
    "scripts/eval/ditto_quant/run_inference.py",
    "scripts/eval/ditto_quant/evaluate.py",
    "scripts/eval/ditto_quant/evaluate_aesthetic_reward.py",
    "scripts/eval/ditto_quant/audit_eval_dataset.py",
    "scripts/eval/ditto_quant/dedupe_eval_inputs.py",
    "scripts/training/edit-model/train.py",
    "scripts/training/edit-model/build_eval_heldout_split.py",
    "scripts/training/edit-model/build_grpo_candidates_from_metrics.py",
    "Ditto/inference/infer_ditto.py",
    "Ditto/diffsynth/models/latent_memory_runtime.py",
    "Ditto/examples/wanvideo/model_training/train.py",
]


DATA_FILES = [
    "out/eval_newsplit50/metric_details/per_video_all_metrics/per_video_all_metrics.csv",
    "out/eval_newsplit50/metric_details/eval_inputs.jsonl",
    "out/eval_newsplit50/reference_pairs/manifest.jsonl",
    "out/new_split_critic_inputs/combined_high_ditto_1k_sample.csv",
    "out/new_split_critic_inputs/combined_high_ditto_50_fine_coverage_sample.csv",
    "out/edit_model_face_stage1/shared_records.rebuild_merged.jsonl",
    "out/edit_model_face_stage1/shared_records.low_high_cost.qwen3vl32b_low2high.jsonl",
    "out/edit_model_face_stage1/new_eval_split_50/stage1_rebuild_merged.train_no_eval50.jsonl",
    "out/edit_model_face_stage1/new_eval_split_50/stage2_low_high_cost.train_no_eval50.jsonl",
]

MODEL_FILES = [
    "models/Ditto_models/models/ditto_global.safetensors",
    "models/Ditto_models/models/ditto_global_style.safetensors",
    "out/edit_model_face_stage1/newsplit50_stage1_8gpu_latent_text/real_train_output/step-15207.safetensors",
    "out/edit_model_face_stage1/newsplit50_stage2_low_high_cost_8gpu_latent_text_si/real_train_output/step-11573.safetensors",
    "out/critic_aesthetic_stage1/evaluator_sft_qwen8b_aesthetic5/real_sft_checkpoint/adapter_model.safetensors",
]


VIDEO_FIELDS = [
    "low_video_path",
    "high_video_path",
    "source_video",
    "target_video",
    "control_video",
    "edited_video_path",
    "edited_video",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local/remote repo consistency, duplicate eval mappings, and video path integrity.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-hash-mb", type=int, default=512, help="Skip hashing files larger than this size; report size only.")
    parser.add_argument("--hash-videos", action="store_true", help="Hash video paths inside JSONL/CSV rows. Slower but catches duplicate renamed videos.")
    parser.add_argument("--max-video-hash", type=int, default=5000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(root: Path, rel: str, max_hash_bytes: int) -> dict[str, Any]:
    path = root / rel
    rec: dict[str, Any] = {"path": rel, "exists": path.exists()}
    if not path.exists():
        return rec
    rec["size"] = path.stat().st_size
    if path.is_file() and rec["size"] <= max_hash_bytes:
        rec["sha256"] = sha256_file(path)
    else:
        rec["sha256"] = ""
        rec["hash_skipped"] = True
    return rec


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def duplicate_summary(rows: list[dict[str, Any]], key_fields: list[str], max_examples: int = 20) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[str]] = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        key = tuple(norm(row.get(field)) for field in key_fields)
        if not any(key):
            continue
        sample_id = str(row.get("sample_id") or row.get("source_record_id") or idx)
        groups[key].append(sample_id)
    dups = [values for values in groups.values() if len(values) > 1]
    return {
        "key_fields": key_fields,
        "groups": len(groups),
        "duplicate_groups": len(dups),
        "duplicate_rows": sum(len(values) for values in dups),
        "examples": dups[:max_examples],
    }


def path_values(rows: list[dict[str, Any]]) -> list[str]:
    values = []
    for row in rows:
        for field in VIDEO_FIELDS:
            value = norm(row.get(field))
            if value and value.endswith((".mp4", ".mov", ".mkv", ".webm")):
                values.append(value)
    return values


def resolve_video_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def audit_video_paths(root: Path, rows: list[dict[str, Any]], hash_videos: bool, max_video_hash: int) -> dict[str, Any]:
    values = path_values(rows)
    counter = collections.Counter(values)
    repeated_paths = [(path, count) for path, count in counter.items() if count > 1]
    missing = []
    hashes: dict[str, list[str]] = collections.defaultdict(list)
    checked = 0
    for value in sorted(counter):
        path = resolve_video_path(root, value)
        if not path.exists():
            missing.append(value)
            continue
        if hash_videos and checked < max_video_hash:
            hashes[sha256_file(path)].append(value)
            checked += 1
    duplicate_hash_groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "video_path_values": len(values),
        "unique_video_paths": len(counter),
        "repeated_video_paths": len(repeated_paths),
        "repeated_path_examples": repeated_paths[:20],
        "missing_paths": len(missing),
        "missing_examples": missing[:30],
        "hashed_existing_paths": checked,
        "duplicate_video_hash_groups": len(duplicate_hash_groups),
        "duplicate_video_hash_examples": duplicate_hash_groups[:20],
    }


def audit_table(root: Path, rel: str, hash_videos: bool, max_video_hash: int) -> dict[str, Any]:
    path = root / rel
    if rel.endswith(".jsonl"):
        rows = load_jsonl(path)
    elif rel.endswith(".csv"):
        rows = load_csv(path)
    else:
        rows = []
    out: dict[str, Any] = {"path": rel, "exists": path.exists(), "rows": len(rows)}
    if not rows:
        return out
    out["sample_ids"] = len({str(row.get("sample_id", "")) for row in rows if row.get("sample_id")})
    out["duplicates"] = [
        duplicate_summary(rows, ["sample_id"]),
        duplicate_summary(rows, ["source_video", "instruction"]),
        duplicate_summary(rows, ["low_video_path", "raw_prompt"]),
        duplicate_summary(rows, ["low_video_path", "high_video_path", "raw_prompt"]),
        duplicate_summary(rows, ["source_path", "instruction"]),
        duplicate_summary(rows, ["item_path"]),
        duplicate_summary(rows, ["edited_path"]),
    ]
    out["video_paths"] = audit_video_paths(root, rows, hash_videos=hash_videos, max_video_hash=max_video_hash)
    return out


def audit_reference_pairs(root: Path) -> dict[str, Any]:
    base = root / "out/eval_newsplit50/reference_pairs"
    out: dict[str, Any] = {"path": str(base), "exists": base.exists()}
    for sub in ["low_videos", "high_videos"]:
        d = base / sub
        files = sorted(d.glob("*.mp4")) if d.exists() else []
        hashes: dict[str, list[str]] = collections.defaultdict(list)
        for p in files:
            hashes[sha256_file(p)].append(p.name)
        dups = [v for v in hashes.values() if len(v) > 1]
        out[sub] = {
            "files": len(files),
            "unique_hashes": len(hashes),
            "duplicate_hash_groups": len(dups),
            "duplicate_hash_rows": sum(len(v) for v in dups),
            "duplicate_examples": dups[:30],
        }
    return out


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    max_hash_bytes = args.max_hash_mb * 1024 * 1024
    report: dict[str, Any] = {
        "repo_root": str(root),
        "cwd": os.getcwd(),
        "code": [file_record(root, rel, max_hash_bytes) for rel in CODE_FILES],
        "data_files": [file_record(root, rel, max_hash_bytes) for rel in DATA_FILES],
        "model_files": [file_record(root, rel, max_hash_bytes) for rel in MODEL_FILES],
        "tables": [audit_table(root, rel, hash_videos=args.hash_videos, max_video_hash=args.max_video_hash) for rel in DATA_FILES if rel.endswith((".jsonl", ".csv"))],
        "reference_pairs": audit_reference_pairs(root),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
