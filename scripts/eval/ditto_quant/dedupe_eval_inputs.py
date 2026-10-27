from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate eval inputs by generation condition. "
            "The default key is source video content hash plus instruction, because target_video is not used by generation."
        )
    )
    parser.add_argument("--eval-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--removed-output", type=Path, default=None)
    parser.add_argument(
        "--reference-low-dir",
        type=Path,
        default=None,
        help="Optional low_videos dir containing <sample_id>.mp4. If provided, use file hash instead of source path.",
    )
    parser.add_argument("--key", choices=["source_instruction", "source_hash_instruction"], default="source_hash_instruction")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity(row: dict[str, Any], reference_low_dir: Path | None) -> str:
    sample_id = str(row.get("sample_id", "") or "")
    if reference_low_dir is not None:
        low_path = reference_low_dir / f"{sample_id}.mp4"
        if low_path.exists():
            return "sha256:" + sha256_file(low_path)
    return "path:" + str(row.get("source_video") or row.get("control_video") or row.get("low_video_path") or "")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.eval_inputs)
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
        if args.key == "source_hash_instruction":
            source_key = source_identity(row, args.reference_low_dir)
        else:
            source_key = "path:" + str(row.get("source_video") or row.get("control_video") or row.get("low_video_path") or "")
        key = (source_key, instruction)
        sample_id = str(row.get("sample_id", "") or "")
        if key in seen:
            removed_row = dict(row)
            removed_row["_dedupe_reason"] = "duplicate_generation_condition"
            removed_row["_dedupe_kept_sample_id"] = seen[key]
            removed.append(removed_row)
            continue
        seen[key] = sample_id
        kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    removed_output = args.removed_output or args.output.with_suffix(".removed.jsonl")
    removed_output.parent.mkdir(parents=True, exist_ok=True)
    with removed_output.open("w", encoding="utf-8") as f:
        for row in removed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input": str(args.eval_inputs),
        "output": str(args.output),
        "removed_output": str(removed_output),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "key": args.key,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
