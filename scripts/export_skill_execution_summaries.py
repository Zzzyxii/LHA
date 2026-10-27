#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory import load_global_memory


DEFAULT_MEMORY_PATH = "out/memory/global_memory.json"
DEFAULT_OUTPUT_DIR = "out/skill_execution_summaries"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export aggregated skill execution summaries from memory.")
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--style-family", default="")
    parser.add_argument("--archetype-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory = load_global_memory(args.memory_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [item for item in (memory.get("skill_execution_summaries") or []) if isinstance(item, dict)]
    if args.style_family:
        summaries = [item for item in summaries if str(item.get("style_family", "")).strip() == args.style_family]
    if args.archetype_id:
        summaries = [item for item in summaries if str(item.get("archetype_id", "")).strip() == args.archetype_id]

    exported: list[dict] = []
    for item in summaries:
        style_family = str(item.get("style_family", "")).strip() or "unknown_family"
        archetype_id = str(item.get("archetype_id", "")).strip() or "unknown_archetype"
        family_dir = output_dir / style_family
        family_dir.mkdir(parents=True, exist_ok=True)
        out_path = family_dir / f"{archetype_id}.json"
        out_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append({"path": str(out_path), "style_family": style_family, "archetype_id": archetype_id})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"exported": exported}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"exported_count": len(exported), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
