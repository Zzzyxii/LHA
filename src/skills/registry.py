from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from skills.base import SkillTemplate

if TYPE_CHECKING:
    from planner.core.planner_types import StrategyCandidate


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, list[SkillTemplate]] = {}
        self._load_all_skills()

    def _load_all_skills(self) -> None:
        templates_dir = Path(__file__).parent / "templates"
        for style_dir in templates_dir.iterdir():
            if not style_dir.is_dir():
                continue
            style_name = style_dir.name
            self._skills[style_name] = []
            for py_file in style_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                module_name = f"skills.templates.{style_name}.{py_file.stem}"
                module = importlib.import_module(module_name)
                for skill_data in getattr(module, "SKILLS", []):
                    self._skills[style_name].append(SkillTemplate(**skill_data))

    def get_skills_by_style(self, style_family: str) -> list[SkillTemplate]:
        return list(self._skills.get(style_family, []))

    def get_all_skills(self) -> list[SkillTemplate]:
        items: list[SkillTemplate] = []
        for skills in self._skills.values():
            items.extend(skills)
        return items

    def rank_skills_for_context(
        self,
        *,
        style_family: str,
        archetype_id: str = "",
        available_anchor_types: list[str] | None = None,
    ) -> list[SkillTemplate]:
        available_anchor_types = [str(item).strip() for item in (available_anchor_types or []) if str(item).strip()]
        candidates = self.get_skills_by_style(style_family)

        def score(skill: SkillTemplate) -> tuple[int, int, int]:
            archetype_match = int(bool(archetype_id and archetype_id in skill.best_for_archetypes))
            archetype_penalty = int(bool(archetype_id and archetype_id in skill.high_risk_archetypes))
            anchor_match = sum(1 for item in skill.anchor_requirements if item in available_anchor_types)
            return (archetype_match, anchor_match, -archetype_penalty)

        return sorted(candidates, key=score, reverse=True)

_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry


def templates_for_style(style_family: str) -> list[dict]:
    return [skill.to_dict() for skill in get_skill_registry().get_skills_by_style(style_family)]


def candidate_models_for_style(style_family: str, limit: int = 3) -> list["StrategyCandidate"]:
    from planner.core.planner_types import StrategyCandidate

    candidates = templates_for_style(style_family)[:limit]
    return [StrategyCandidate(**item) for item in candidates]


def templates_for_context(
    style_family: str,
    *,
    archetype_id: str = "",
    available_anchor_types: list[str] | None = None,
    limit: int = 3,
) -> list[dict]:
    skills = get_skill_registry().rank_skills_for_context(
        style_family=style_family,
        archetype_id=archetype_id,
        available_anchor_types=available_anchor_types,
    )
    return [skill.to_dict() for skill in skills[:limit]]
