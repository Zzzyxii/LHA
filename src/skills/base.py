"""Style-family skill template: what a world looks like, dimension by dimension.

A template answers one question -- "in this world, what does each dimension look
like" -- for the 11 canonical dimensions in ``skills.dimensions``. The facts that
are the same in every world (what must be preserved when a dimension is edited,
which words put it in play, how hard it is normally pushed) live in that module,
not here. Splitting it that way is what keeps 98 styles from needing 98 copies of
the preservation rules.

The dimension fields are English by requirement, not by preference: the edit
backend's text encoder is umT5 on a predominantly English-trained checkpoint, and
the previous generation of templates reached it in Chinese after a translation
pass, which was part of what licensed the model to paraphrase freely.

``reference_image`` exists for the ``reference`` delivery mode, where the world is
carried by an image instead of by prose. It is the only field that is not text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skills.dimensions import sorted_dimensions

# Canonical dimension order, global -> local. Imported rather than repeated so a
# new dimension cannot be added to one module and forgotten in the other.
CANONICAL_DIMENSIONS: tuple[str, ...] = tuple(
    policy.dimension for policy in sorted_dimensions()
)


@dataclass
class SkillTemplate:
    strategy_id: str
    style_family: str
    title: str
    description: str
    # --- canonical dimensions, named to match skills.dimensions.DIMENSIONS ---
    environment: str = ""
    lighting: str = ""
    costume: str = ""
    headwear: str = ""
    jewelry: str = ""
    subject_surface: str = ""
    subject_integration: str = ""
    material: str = ""
    atmosphere: str = ""
    color_grade: str = ""
    # Legacy spelling of `props`. Retained while the 13 existing templates are
    # migrated; `dimension_fields()` folds it into `props` so callers only ever
    # see the canonical name. Remove once every template uses `props`.
    prop: str = ""
    props: str = ""
    # Path to an image that carries this world, used by the reference delivery
    # mode. Relative to the repo root.
    reference_image: str = ""
    tags: list[str] = field(default_factory=list)
    intensity: str = "normal"
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    forbidden_mixes: list[str] = field(default_factory=list)
    anchor_requirements: list[str] = field(default_factory=list)
    best_for_archetypes: list[str] = field(default_factory=list)
    high_risk_archetypes: list[str] = field(default_factory=list)
    world_rewrite_cost: str = "medium"
    identity_risk: str = "medium"
    prompt_compilation_notes: list[str] = field(default_factory=list)
    common_failure_modes: list[str] = field(default_factory=list)
    recommended_carriers: list[str] = field(default_factory=list)

    def dimension_fields(self) -> dict[str, str]:
        """{canonical dimension: this world's text}, empty entries omitted.

        Ordered global -> local so a composer that runs short of budget can drop
        from the end and lose the local detail before the global look.
        """
        values = {
            dimension: (getattr(self, dimension, "") or "").strip()
            for dimension in CANONICAL_DIMENSIONS
        }
        # Legacy alias folded in here; remove with the `prop` field.
        if not values.get("props"):
            values["props"] = (self.prop or "").strip()
        return {
            dimension: text for dimension, text in values.items() if text
        }

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "style_family": self.style_family,
            "title": self.title,
            "description": self.description,
            "environment": self.environment,
            "lighting": self.lighting,
            "costume": self.costume,
            "headwear": self.headwear,
            "jewelry": self.jewelry,
            "subject_surface": self.subject_surface,
            "subject_integration": self.subject_integration,
            "material": self.material,
            "atmosphere": self.atmosphere,
            "color_grade": self.color_grade,
            "prop": self.prop,
            "props": self.props,
            "reference_image": self.reference_image,
            "tags": self.tags[:6],
            "intensity": self.intensity,
            "pros": self.pros[:2],
            "cons": self.cons[:2],
            "forbidden_mixes": self.forbidden_mixes[:4],
            "anchor_requirements": self.anchor_requirements[:6],
            "best_for_archetypes": self.best_for_archetypes[:6],
            "high_risk_archetypes": self.high_risk_archetypes[:6],
            "world_rewrite_cost": self.world_rewrite_cost,
            "identity_risk": self.identity_risk,
            "prompt_compilation_notes": self.prompt_compilation_notes[:6],
            "common_failure_modes": self.common_failure_modes[:6],
            "recommended_carriers": self.recommended_carriers[:6],
        }
