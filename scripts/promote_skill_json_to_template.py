#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_INPUT_DIR = "out/promoted_skills"
DEFAULT_SKILLS_ROOT = "src/skills/templates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert promoted skill JSON files into skill template Python files.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--skills-root", default=DEFAULT_SKILLS_ROOT)
    parser.add_argument("--style-family", default="")
    parser.add_argument("--skill-id", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    return slug or "custom_skill"


def _template_py(payload: dict) -> str:
    skill_payload = dict(payload.get("skill_payload", {}) or {})
    return "SKILLS = [\n    " + json.dumps(skill_payload, ensure_ascii=False, indent=4) + "\n]\n"


def _iter_json_files(root: Path) -> list[Path]:
    return sorted([path for path in root.rglob("*.json") if path.name != "manifest.json"])


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    skills_root = Path(args.skills_root)
    json_files = _iter_json_files(input_dir)
    written: list[str] = []

    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        style_family = str(payload.get("style_family", "")).strip()
        skill_id = str(payload.get("skill_id", "")).strip()
        if args.style_family and style_family != args.style_family:
            continue
        if args.skill_id and skill_id != args.skill_id:
            continue
        if not style_family or not skill_id:
            continue

        family_dir = skills_root / style_family
        family_dir.mkdir(parents=True, exist_ok=True)
        init_file = family_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("SKILLS = []\n", encoding="utf-8")

        out_path = family_dir / f"{_slug(skill_id)}.py"
        if out_path.exists() and not args.overwrite:
            continue
        out_path.write_text(_template_py(payload), encoding="utf-8")
        written.append(str(out_path))

    print(json.dumps({"written_count": len(written), "written": written}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
