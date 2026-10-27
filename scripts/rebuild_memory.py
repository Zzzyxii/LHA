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

from memory import cleanup_memory, load_memory_policy, load_global_memory, rebuild_memory, save_global_memory


DEFAULT_MEMORY_PATH = "out/memory/global_memory.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild memory banks and stats from an existing global_memory.json.")
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH, help="Input memory JSON path.")
    parser.add_argument("--output-memory-path", default="", help="Optional output path. Defaults to <memory>.rebuilt.json unless --in-place is set.")
    parser.add_argument("--config", default="", help="Optional memory config path. Defaults to configs/memory.json or LOW_HIGH_MEMORY_CONFIG.")
    parser.add_argument("--cleanup-after", action="store_true", help="Run memory cleanup after rebuild using the same config.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite --memory-path with rebuilt memory.")
    parser.add_argument("--backup-path", default="", help="Optional backup path used with --in-place.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print summary without writing output.")
    return parser.parse_args()


def _resolve_output_path(args: argparse.Namespace, memory_path: Path) -> Path:
    if args.output_memory_path:
        return Path(args.output_memory_path).expanduser().resolve()
    if args.in_place:
        return memory_path
    return memory_path.with_name(f"{memory_path.stem}.rebuilt{memory_path.suffix}").resolve()


def _bank_ids(payload: dict, key: str) -> list[str]:
    return [
        str(item.get("skill_id", "")).strip()
        for item in (payload.get(key) or [])
        if isinstance(item, dict) and str(item.get("skill_id", "")).strip()
    ]


def _summary(
    before: dict,
    rebuilt: dict,
    final: dict,
    *,
    config: dict,
    input_path: Path,
    output_path: Path,
    cleanup_applied: bool,
) -> dict:
    summary = {
        "input_memory_path": str(input_path),
        "output_memory_path": str(output_path),
        "config": config,
        "cleanup_applied": cleanup_applied,
        "before": {
            "custom_skill_records": len(before.get("custom_skill_records") or []),
            "promotable_skill_records": len(before.get("promotable_skill_records") or []),
            "runtime_skill_records": len(before.get("runtime_skill_records") or []),
            "quarantined_skill_records": len(before.get("quarantined_skill_records") or []),
            "skill_execution_records": len(before.get("skill_execution_records") or []),
            "skill_execution_summaries": len(before.get("skill_execution_summaries") or []),
        },
        "after_rebuild": {
            "custom_skill_records": len(rebuilt.get("custom_skill_records") or []),
            "promotable_skill_records": len(rebuilt.get("promotable_skill_records") or []),
            "runtime_skill_records": len(rebuilt.get("runtime_skill_records") or []),
            "quarantined_skill_records": len(rebuilt.get("quarantined_skill_records") or []),
            "skill_execution_records": len(rebuilt.get("skill_execution_records") or []),
            "skill_execution_summaries": len(rebuilt.get("skill_execution_summaries") or []),
        },
        "final": {
            "custom_skill_records": len(final.get("custom_skill_records") or []),
            "promotable_skill_records": len(final.get("promotable_skill_records") or []),
            "runtime_skill_records": len(final.get("runtime_skill_records") or []),
            "quarantined_skill_records": len(final.get("quarantined_skill_records") or []),
            "skill_execution_records": len(final.get("skill_execution_records") or []),
            "skill_execution_summaries": len(final.get("skill_execution_summaries") or []),
        },
    }
    summary["final_skill_ids"] = {
        "promotable": _bank_ids(final, "promotable_skill_records")[:20],
        "runtime": _bank_ids(final, "runtime_skill_records")[:20],
        "quarantine": _bank_ids(final, "quarantined_skill_records")[:20],
    }
    if cleanup_applied:
        summary["after_cleanup"] = dict(summary["final"])
    return summary


def main() -> None:
    args = parse_args()
    memory_path = Path(args.memory_path).expanduser().resolve()
    if not memory_path.exists():
        raise FileNotFoundError(f"Memory file not found: {memory_path}")

    source_memory = load_global_memory(memory_path)
    rebuilt_memory = rebuild_memory(source_memory, config_path=args.config or None)
    final_memory = cleanup_memory(rebuilt_memory, config_path=args.config or None) if args.cleanup_after else rebuilt_memory
    output_path = _resolve_output_path(args, memory_path)
    policy = load_memory_policy(args.config or None)
    summary = _summary(
        source_memory,
        rebuilt_memory,
        final_memory,
        config=policy,
        input_path=memory_path,
        output_path=output_path,
        cleanup_applied=bool(args.cleanup_after),
    )

    if args.dry_run:
        print(json.dumps({**summary, "written": False}, ensure_ascii=False, indent=2))
        return

    if args.in_place and args.backup_path:
        backup_path = Path(args.backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(memory_path, backup_path)

    save_global_memory(output_path, final_memory)
    print(json.dumps({**summary, "written": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
