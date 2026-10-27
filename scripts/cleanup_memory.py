#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory import cleanup_memory, load_global_memory, load_memory_policy, save_global_memory


DEFAULT_MEMORY_PATH = "out/memory/global_memory.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup memory banks using retention/decay thresholds from configs/memory.json.")
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, help="Input memory JSON path.")
    parser.add_argument("--output-memory-path", default="", help="Optional output path. Defaults to <memory>.cleaned.json unless --in-place is set.")
    parser.add_argument("--config", default="", help="Optional memory config path. Defaults to configs/memory.json or LOW_HIGH_MEMORY_CONFIG.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite --memory-path with cleaned memory.")
    parser.add_argument("--backup-path", default="", help="Optional backup path used with --in-place.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print summary without writing output.")
    return parser.parse_args()


def _resolve_output_path(args: argparse.Namespace, memory_path: Path) -> Path:
    if args.output_memory_path:
        return Path(args.output_memory_path).expanduser().resolve()
    if args.in_place:
        return memory_path
    return memory_path.with_name(f"{memory_path.stem}.cleaned{memory_path.suffix}").resolve()


def _collection_keys(payload: dict) -> list[str]:
    return [key for key, value in payload.items() if key != "schema_version" and isinstance(value, list)]


def _item_identity(item: dict) -> str:
    for key in [
        "record_id",
        "skill_id",
        "summary_key",
        "recipe_id",
        "prior_key",
        "failure_key",
        "execution_key",
        "sample_id",
        "archetype_id",
        "pattern",
    ]:
        text = str(item.get(key, "")).strip()
        if text:
            return text
    return json.dumps(item, ensure_ascii=False, sort_keys=True)[:120]


def _removed_examples(before: dict, after: dict, key: str, *, limit: int = 20) -> list[str]:
    before_items = {
        _item_identity(item)
        for item in (before.get(key) or [])
        if isinstance(item, dict)
    }
    after_items = {
        _item_identity(item)
        for item in (after.get(key) or [])
        if isinstance(item, dict)
    }
    removed = sorted(before_items - after_items)
    return removed[:limit]


def _summary(before: dict, after: dict, *, config: dict, input_path: Path, output_path: Path) -> dict:
    keys = sorted(set(_collection_keys(before)) | set(_collection_keys(after)))
    return {
        "input_memory_path": str(input_path),
        "output_memory_path": str(output_path),
        "config": config,
        "before": {key: len(before.get(key) or []) for key in keys},
        "after": {key: len(after.get(key) or []) for key in keys},
        "removed_examples": {
            key: _removed_examples(before, after, key)
            for key in [
                "custom_skill_records",
                "promotable_skill_records",
                "runtime_skill_records",
                "quarantined_skill_records",
                "skill_execution_records",
                "skill_execution_summaries",
                "prompt_compilation_recipes",
            ]
        },
    }


def main() -> None:
    args = parse_args()
    memory_path = Path(args.memory_path).expanduser().resolve()
    if not memory_path.exists():
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    source_memory = load_global_memory(memory_path)
    cleaned_memory = cleanup_memory(source_memory, config_path=args.config or None)
    output_path = _resolve_output_path(args, memory_path)
    policy = load_memory_policy(args.config or None)
    summary = _summary(source_memory, cleaned_memory, config=policy, input_path=memory_path, output_path=output_path)

    if args.dry_run:
        print(json.dumps({**summary, "written": False}, ensure_ascii=False, indent=2))
        return

    if args.in_place and args.backup_path:
        backup_path = Path(args.backup_path).expanduser().resolve()
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(memory_path, backup_path)

    save_global_memory(output_path, cleaned_memory)
    print(json.dumps({**summary, "written": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
