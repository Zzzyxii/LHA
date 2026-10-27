from __future__ import annotations

from planner.core.planner_types import GroundingEvidence, StrategyCandidate, VideoPlan


def resolve_constraint_conflicts(
    *,
    hard_preserve: list[str],
    soft_preserve: list[str],
    rewrite_targets: list[str],
    light_touch_targets: list[str],
    forbidden_changes: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    resolutions: list[str] = []
    hard_lower = {item.lower(): item for item in hard_preserve}
    rewrite_filtered: list[str] = []
    light_touch_filtered = list(light_touch_targets)
    for item in rewrite_targets:
        lowered = item.lower()
        if "costume" in lowered and any("costume" in key or "clothing" in key or "wardrobe" in key for key in hard_lower):
            light_touch_filtered.append("limit costume preservation to identity-safe silhouette cues, not original everyday clothing details")
            resolutions.append("Moved conflicting costume preserve intent out of hard preserve so costume can be redesigned.")
            continue
        if "face" in lowered or "facial" in lowered or "skin" in lowered:
            light_touch_filtered.append(item)
            resolutions.append("Demoted facial or skin rewrite into light-touch targets to protect identity.")
            continue
        rewrite_filtered.append(item)
    dedup = lambda items: list(dict.fromkeys([item for item in items if item]))
    return (
        dedup(hard_preserve),
        dedup(soft_preserve),
        dedup(rewrite_filtered),
        dedup(light_touch_filtered),
        dedup(forbidden_changes),
        dedup(resolutions),
    )


def validate_selected_strategy(selected_strategy: str, candidates: list[StrategyCandidate]) -> str:
    candidate_ids = [item.strategy_id for item in candidates]
    if selected_strategy in candidate_ids:
        return selected_strategy
    if not candidate_ids:
        # No candidates survive when the scene falls outside every registered
        # style family. Keep the model's own choice rather than crashing, so the
        # run degrades to an unskilled plan instead of losing the sample.
        return str(selected_strategy or "").strip()
    return candidate_ids[0]


def validate_carriers(grounding: GroundingEvidence, primary: list[str], secondary: list[str]) -> tuple[list[str], list[str], list[str]]:
    allowed = set(grounding.carrier_candidates)
    aliases = {
        "functional_scene_props": set(grounding.visible_props),
        "environment": {"environment"},
        "lighting_system_design": {"lighting system"},
        "atmospheric_depth_layer": {"atmospheric layer"},
        "subject_environment_integration": {"subject edge light"},
        "material_system_design": set(),
    }

    def _is_allowed(item: str) -> bool:
        if item in allowed:
            return True
        mapped = aliases.get(item, set())
        return bool(mapped & allowed)

    primary_valid = [item for item in primary if _is_allowed(item)]
    secondary_valid = [item for item in secondary if _is_allowed(item)]
    notes: list[str] = []
    if not primary_valid and allowed:
        primary_valid = [next(iter(allowed))]
        notes.append("Replaced invalid primary carrier list with the first grounded visible carrier.")
    return primary_valid, secondary_valid, notes


def ensure_plan_is_productive(plan: VideoPlan) -> VideoPlan:
    if len(plan.region_edits) < 4:
        raise ValueError("planner produced too few region_edits for a production-ready edit plan")
    return plan
