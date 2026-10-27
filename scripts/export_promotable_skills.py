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
DEFAULT_OUTPUT_DIR = "out/promoted_skills"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export promotable custom skills from memory into JSON files.")
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-score", type=float, default=0.78)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--accepted-only", action="store_true", default=True)
    parser.add_argument("--allow-existing-skill-id", action="store_true")
    return parser.parse_args()


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _slug(text: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "custom_skill"


def _is_promotable(record: dict, *, min_score: float, min_count: int, accepted_only: bool) -> bool:
    if bool(record.get("memory_only", False)):
        return False
    if accepted_only and not bool(record.get("accepted", False)):
        return False
    if float(record.get("overall_score", 0.0) or 0.0) < min_score:
        return False
    if int(record.get("count", 1) or 1) < min_count:
        return False
    skill_payload = record.get("skill_payload", {})
    if not isinstance(skill_payload, dict):
        return False
    required_fields = [
        "strategy_id",
        "style_family",
        "title",
        "description",
        "environment",
        "prop",
        "lighting",
        "costume",
        "subject_surface",
        "subject_integration",
        "atmosphere",
        "material",
    ]
    return all(_clean_text(skill_payload.get(field, "")) for field in required_fields)


def _export_filename(record: dict) -> tuple[str, str]:
    style_family = _clean_text(record.get("style_family", "")) or "custom_family"
    skill_id = _clean_text(record.get("skill_id", "")) or _slug(style_family)
    return style_family, f"{skill_id}.json"


def main() -> None:
    args = parse_args()
    memory = load_global_memory(args.memory_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict] = []
    promotable_records = memory.get("promotable_skill_records") or []
    if not promotable_records and not memory.get("quarantined_skill_records"):
        promotable_records = memory.get("custom_skill_records", [])
    for record in promotable_records:
        if not isinstance(record, dict):
            continue
        if not _is_promotable(
            record,
            min_score=args.min_score,
            min_count=args.min_count,
            accepted_only=bool(args.accepted_only),
        ):
            continue
        style_family, filename = _export_filename(record)
        family_dir = output_dir / style_family
        family_dir.mkdir(parents=True, exist_ok=True)
        export_path = family_dir / filename
        payload = {
            "skill_id": record.get("skill_id", ""),
            "style_family": record.get("style_family", ""),
            "source": record.get("source", ""),
            "memory_only": bool(record.get("memory_only", False)),
            "sample_id": record.get("sample_id", ""),
            "attempt_index": record.get("attempt_index", 0),
            "variant": record.get("variant", ""),
            "archetype_id": record.get("archetype_id", ""),
            "overall_score": record.get("overall_score", 0.0),
            "accepted": bool(record.get("accepted", False)),
            "count": int(record.get("count", 1) or 1),
            "selected_for_candidate": bool(record.get("selected_for_candidate", False)),
            "selected_strategy": record.get("selected_strategy", ""),
            "reflection_action": record.get("reflection_action", ""),
            "skill_payload": record.get("skill_payload", {}),
        }
        export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append({"path": str(export_path), "skill_id": payload["skill_id"], "style_family": payload["style_family"]})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"exported": exported}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"exported_count": len(exported), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
