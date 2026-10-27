from __future__ import annotations

import re

from skills.registry import templates_for_context

from planner.core.common import clean_string_list, clean_text


def family_options(available_style_families: list[str], registry) -> list[dict]:
    options: list[dict] = []
    for family in available_style_families:
        skills = registry.get_skills_by_style(family)
        options.append(
            {
                "style_family": family,
                "summary": " / ".join(skill.title for skill in skills[:3]),
            }
        )
    return options


def skill_briefs(skills: list) -> list[dict]:
    return [
        {
            "strategy_id": skill.strategy_id,
            "style_family": skill.style_family,
            "title": skill.title,
            "description": skill.description,
        }
        for skill in skills
    ]


def context_archetype_id(prompt: str, memory_context: dict) -> str:
    scene_hints = list(memory_context.get("scene_archetype_hints", []) or [])
    if scene_hints and isinstance(scene_hints[0], dict):
        archetype_id = clean_text(scene_hints[0].get("archetype_id", ""))
        if archetype_id:
            return archetype_id
    prompt_lower = prompt.lower()
    if any(token in prompt_lower for token in ["室内", "indoor", "房间", "录音", "control room", "room"]):
        return "single_person_indoor_closeup"
    return ""


def grounding_scene_archetype(prompt: str, grounding_payload: dict) -> dict:
    visible_people = clean_string_list(grounding_payload.get("visible_people") or [])
    visible_world_anchors = clean_string_list(grounding_payload.get("visible_world_anchors") or [])
    visible_props = clean_string_list(grounding_payload.get("visible_props") or [])
    prompt_lower = prompt.lower()
    subject_count = len(visible_people)
    indoor = any(token in prompt_lower for token in ["室内", "indoor", "房间", "录音", "control room", "room"])
    if indoor and subject_count <= 1:
        archetype_id = "single_person_indoor_closeup"
    elif subject_count >= 3:
        archetype_id = "multi_person_group_scene"
    elif visible_world_anchors:
        archetype_id = "world_anchor_driven_scene"
    else:
        archetype_id = "general_edit_scene"
    return {
        "archetype_id": archetype_id,
        "scene_hint": archetype_id.replace("_", " "),
        "subject_count": subject_count,
        "world_anchor_count": len(visible_world_anchors),
        "prop_count": len(visible_props),
        "evidence": {
            "visible_people": visible_people[:5],
            "visible_world_anchors": visible_world_anchors[:5],
            "visible_props": visible_props[:5],
        },
    }


def context_anchor_types(prompt: str, archetype_id: str, memory_context: dict) -> list[str]:
    prompt_lower = prompt.lower()
    anchors: list[str] = []
    if archetype_id == "single_person_indoor_closeup":
        anchors.extend(["foreground_device_anchor", "indoor_background_depth", "single_subject"])
    if any(token in prompt_lower for token in ["环", "ring", "portal", "传送门"]):
        anchors.append("ring_like_prop")
    if any(token in prompt_lower for token in ["麦克风", "microphone", "device", "装置"]):
        anchors.append("foreground_device_anchor")
    for item in list(memory_context.get("failure_patterns", []) or []):
        for carrier in item.get("recommended_carriers", []) or []:
            if isinstance(carrier, str) and carrier.strip():
                anchors.append(carrier.strip())
    execution_bias = memory_context.get("execution_adaptive_bias") or {}
    if isinstance(execution_bias, dict):
        for carrier in execution_bias.get("preferred_carriers", []) or []:
            if isinstance(carrier, str) and carrier.strip():
                anchors.append(carrier.strip())
    return list(dict.fromkeys(anchors))


def contextual_skill_briefs(*, selected_family: str, prompt: str, memory_context: dict, limit: int = 5) -> list[dict]:
    archetype_id = context_archetype_id(prompt, memory_context)
    anchor_types = context_anchor_types(prompt, archetype_id, memory_context)
    ranked = templates_for_context(
        selected_family,
        archetype_id=archetype_id,
        available_anchor_types=anchor_types,
        limit=max(limit * 2, 6),
    )
    execution_bias = memory_context.get("execution_adaptive_bias") or {}
    boosted = {str(item).strip() for item in (execution_bias.get("boosted_skill_ids") or []) if str(item).strip()} if isinstance(execution_bias, dict) else set()
    penalized = {str(item).strip() for item in (execution_bias.get("penalized_skill_ids") or []) if str(item).strip()} if isinstance(execution_bias, dict) else set()
    if ranked:
        def _brief_score(item: dict) -> tuple[int, int, int]:
            skill_id = clean_text(item.get("strategy_id", ""))
            archetype_match = int(bool(archetype_id and archetype_id in (item.get("best_for_archetypes") or [])))
            anchor_match = sum(1 for value in (item.get("anchor_requirements") or []) if value in anchor_types)
            bias_bonus = int(skill_id in boosted) - int(skill_id in penalized)
            return (bias_bonus, archetype_match, anchor_match)

        ranked = sorted(ranked, key=_brief_score, reverse=True)[:limit]
    if ranked:
        briefs = []
        for item in ranked:
            reasons: list[str] = []
            skill_id = clean_text(item.get("strategy_id", ""))
            if archetype_id and archetype_id in (item.get("best_for_archetypes") or []):
                reasons.append(f"匹配 archetype={archetype_id}")
            anchor_hits = [value for value in (item.get("anchor_requirements") or []) if value in anchor_types]
            if anchor_hits:
                reasons.append(f"匹配锚点={','.join(anchor_hits[:3])}")
            if skill_id in boosted:
                reasons.append("历史执行统计升权")
            if skill_id in penalized:
                reasons.append("历史执行统计降权")
            briefs.append(
                {
                    "strategy_id": item.get("strategy_id", ""),
                    "style_family": item.get("style_family", selected_family),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "best_for_archetypes": item.get("best_for_archetypes", []),
                    "anchor_requirements": item.get("anchor_requirements", []),
                    "common_failure_modes": item.get("common_failure_modes", []),
                    "context_reason": "；".join(reasons),
                }
            )
        return briefs
    return []


def skill_full(skills: list) -> list[dict]:
    results: list[dict] = []
    for skill in skills:
        if isinstance(skill, dict):
            results.append(dict(skill))
        else:
            results.append(skill.to_dict())
    return results


def skill_brief_from_dict(skill: dict, *, default_family: str) -> dict:
    return {
        "strategy_id": clean_text(skill.get("strategy_id", "")),
        "style_family": clean_text(skill.get("style_family", "")) or default_family,
        "title": clean_text(skill.get("title", "")),
        "description": clean_text(skill.get("description", "")),
        "best_for_archetypes": clean_string_list(skill.get("best_for_archetypes")),
        "anchor_requirements": clean_string_list(skill.get("anchor_requirements")),
        "common_failure_modes": clean_string_list(skill.get("common_failure_modes")),
        "context_reason": clean_text(skill.get("context_reason", "") or skill.get("_retrieval_reason", "")),
    }


def merge_frontloaded_skills(base_skills: list[dict], extra_skills: list[dict], *, limit: int = 5) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for item in extra_skills + base_skills:
        if not isinstance(item, dict):
            continue
        strategy_id = clean_text(item.get("strategy_id", ""))
        if not strategy_id or strategy_id in seen:
            continue
        merged.append(dict(item))
        seen.add(strategy_id)
        if len(merged) >= limit:
            break
    return merged


def candidate_strategy_preferences(memory_context: dict) -> tuple[list[str], list[str]]:
    state = memory_context.get("attempt_adaptation_state") or {}
    adaptive_bias = state.get("adaptive_bias", {}) if isinstance(state.get("adaptive_bias", {}), dict) else {}
    preferred = clean_string_list(
        list(state.get("last_preferred_candidate_strategies") or [])
        + list(adaptive_bias.get("preferred_candidate_strategies") or [])
        + list((memory_context.get("execution_adaptive_bias") or {}).get("boosted_skill_ids") or [])
    )
    disfavored = clean_string_list(
        list(state.get("last_disfavored_candidate_strategies") or [])
        + list(adaptive_bias.get("forbidden_candidate_strategies") or [])
        + list((memory_context.get("execution_adaptive_bias") or {}).get("penalized_skill_ids") or [])
    )
    return preferred, disfavored


def apply_candidate_strategy_bias(skills: list[dict], memory_context: dict, *, limit: int = 5) -> list[dict]:
    preferred, disfavored = candidate_strategy_preferences(memory_context)
    if not preferred and not disfavored:
        return skills[:limit]

    preferred_set = set(preferred)
    disfavored_set = set(disfavored)
    preferred_items: list[dict] = []
    neutral_items: list[dict] = []
    disfavored_items: list[dict] = []

    for item in skills:
        if not isinstance(item, dict):
            continue
        strategy_id = clean_text(item.get("strategy_id", ""))
        if not strategy_id:
            continue
        if strategy_id in preferred_set:
            preferred_items.append(item)
        elif strategy_id in disfavored_set:
            disfavored_items.append(item)
        else:
            neutral_items.append(item)

    merged = preferred_items + neutral_items
    if len(merged) < limit:
        merged.extend(disfavored_items)
    return merged[:limit]


def adaptive_runtime_skills(memory_context: dict, *, selected_family: str) -> tuple[list[dict], list[dict]]:
    state = memory_context.get("attempt_adaptation_state") or {}
    if not isinstance(state, dict):
        return [], []
    extra_full: list[dict] = []
    materialized = state.get("last_materialized_skill") or {}
    if isinstance(materialized, dict):
        materialized_family = clean_text(materialized.get("style_family", "")) or selected_family
        if materialized_family == selected_family and clean_text(materialized.get("strategy_id", "")):
            extra_full.append(dict(materialized))
    for item in state.get("last_accepted_custom_skills") or []:
        if isinstance(item, dict):
            family = clean_text(item.get("style_family", "")) or selected_family
            if family == selected_family and clean_text(item.get("strategy_id", "")):
                extra_full.append(dict(item))
    extra_briefs = [skill_brief_from_dict(item, default_family=selected_family) for item in extra_full]
    return extra_briefs, extra_full


def normalize_custom_skill_candidate(item: dict, *, selected_family: str) -> dict:
    style_family = clean_text(item.get("style_family", "")) or selected_family or "custom_family"
    title = clean_text(item.get("title", "")) or "Custom Skill"
    strategy_id = clean_text(item.get("strategy_id", ""))
    if not strategy_id:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "custom_skill"
        strategy_id = f"{style_family}_{slug}"
    return {
        "strategy_id": strategy_id,
        "style_family": style_family,
        "title": title,
        "description": clean_text(item.get("description", "")) or f"Custom skill for {style_family} adaptation.",
        "environment": clean_text(item.get("environment", "")),
        "prop": clean_text(item.get("prop", "")),
        "lighting": clean_text(item.get("lighting", "")),
        "costume": clean_text(item.get("costume", "")),
        "subject_surface": clean_text(item.get("subject_surface", "")),
        "subject_integration": clean_text(item.get("subject_integration", "")),
        "atmosphere": clean_text(item.get("atmosphere", "")),
        "material": clean_text(item.get("material", "")),
        "tags": clean_string_list(item.get("tags")),
        "intensity": clean_text(item.get("intensity", "")) or "normal",
        "pros": clean_string_list(item.get("pros")),
        "cons": clean_string_list(item.get("cons")),
        "forbidden_mixes": clean_string_list(item.get("forbidden_mixes")),
        "anchor_requirements": clean_string_list(item.get("anchor_requirements")),
        "best_for_archetypes": clean_string_list(item.get("best_for_archetypes")),
        "high_risk_archetypes": clean_string_list(item.get("high_risk_archetypes")),
        "world_rewrite_cost": clean_text(item.get("world_rewrite_cost", "")) or "medium",
        "identity_risk": clean_text(item.get("identity_risk", "")) or "medium",
        "prompt_compilation_notes": clean_string_list(item.get("prompt_compilation_notes")),
        "common_failure_modes": clean_string_list(item.get("common_failure_modes")),
        "recommended_carriers": clean_string_list(item.get("recommended_carriers")),
    }


def custom_skill_required_field_names() -> list[str]:
    return [
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


def validate_custom_skill_candidates(custom_skill_candidates: list[dict], *, selected_family: str) -> tuple[list[dict], list[dict]]:
    normalized = [
        normalize_custom_skill_candidate(item, selected_family=selected_family)
        for item in custom_skill_candidates
        if isinstance(item, dict)
    ]
    diagnostics: list[dict] = []
    required_fields = custom_skill_required_field_names()
    for item in normalized:
        missing_fields = [field for field in required_fields if not clean_text(item.get(field, ""))]
        if missing_fields:
            diagnostics.append(
                {
                    "strategy_id": item.get("strategy_id", ""),
                    "title": item.get("title", ""),
                    "missing_fields": missing_fields,
                }
            )
    return normalized, diagnostics


def merge_custom_skill_candidates(base_items: list[dict], repair_items: list[dict], *, selected_family: str) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in base_items:
        normalized = normalize_custom_skill_candidate(item, selected_family=selected_family)
        merged[normalized["strategy_id"]] = normalized
    for item in repair_items:
        normalized = normalize_custom_skill_candidate(item, selected_family=selected_family)
        strategy_id = normalized["strategy_id"]
        current = dict(merged.get(strategy_id, {}))
        current.update({key: value for key, value in normalized.items() if value not in ("", [], {}, None)})
        merged[strategy_id] = current
    return list(merged.values())


def selected_skills_with_custom(
    *,
    family_skills: list,
    matched_skill_ids: list[str],
    custom_skill_candidates: list[dict],
    selected_family: str,
) -> list[dict]:
    normalized_custom_skills, _ = validate_custom_skill_candidates(custom_skill_candidates, selected_family=selected_family)
    family_skill_dicts = [skill.to_dict() for skill in family_skills]
    family_by_id = {
        str(item.get("strategy_id", "")).strip(): item
        for item in family_skill_dicts
        if str(item.get("strategy_id", "")).strip()
    }
    custom_by_id = {
        str(item.get("strategy_id", "")).strip(): item
        for item in normalized_custom_skills
        if str(item.get("strategy_id", "")).strip()
    }

    selected: list[dict] = []
    seen: set[str] = set()

    def _append(item: dict | None) -> None:
        if not item:
            return
        strategy_id = str(item.get("strategy_id", "")).strip()
        if not strategy_id or strategy_id in seen:
            return
        selected.append(dict(item))
        seen.add(strategy_id)

    for strategy_id in matched_skill_ids:
        _append(family_by_id.get(strategy_id))
        _append(custom_by_id.get(strategy_id))

    for item in family_skill_dicts:
        _append(item)
        if len(selected) >= 5:
            return selected[:5]

    for item in normalized_custom_skills:
        _append(item)
        if len(selected) >= 5:
            return selected[:5]

    return selected[:5]


def candidate_strategy_map(candidate_planning: dict) -> dict[str, dict]:
    strategies = candidate_planning.get("candidate_strategies") or []
    mapping: dict[str, dict] = {}
    for item in strategies:
        if isinstance(item, dict):
            strategy_id = clean_text(item.get("strategy_id", ""))
            if strategy_id:
                mapping[strategy_id] = item
    return mapping


def materialize_candidate_strategy_as_skill(
    candidate_strategy: dict,
    *,
    selected_family: str,
    fallback_templates: list[dict],
    video_analysis: dict,
) -> dict:
    strategy_id = clean_text(candidate_strategy.get("strategy_id", "")) or "custom_strategy"
    style_family = clean_text(candidate_strategy.get("style_family", "")) or selected_family or "custom_family"
    base_template = next(
        (
            item
            for item in fallback_templates
            if isinstance(item, dict)
            and (clean_text(item.get("style_family", "")) or selected_family or "") == style_family
        ),
        fallback_templates[0] if fallback_templates else {},
    )

    prop = clean_text(candidate_strategy.get("prop", ""))
    environment = clean_text(candidate_strategy.get("environment", ""))
    lighting = clean_text(candidate_strategy.get("lighting", ""))
    costume = clean_text(candidate_strategy.get("costume", ""))
    subject_surface = clean_text(candidate_strategy.get("subject_surface", ""))
    subject_integration = clean_text(candidate_strategy.get("subject_integration", ""))
    atmosphere = clean_text(candidate_strategy.get("atmosphere", ""))
    material = clean_text(candidate_strategy.get("material", ""))
    rationale = clean_text(candidate_strategy.get("rationale", ""))
    title = clean_text(candidate_strategy.get("title", "")) or strategy_id

    visible_props = clean_string_list(video_analysis.get("visible_props"))
    visible_anchors = clean_string_list(video_analysis.get("visible_world_anchors"))
    primary_fx_carriers = clean_string_list(video_analysis.get("primary_fx_carriers"))
    carrier_seed = primary_fx_carriers or ([prop] if prop else []) or visible_props[:2] or visible_anchors[:2]

    def _merged_list(key: str, extra: list[str]) -> list[str]:
        values = []
        values.extend(clean_string_list(base_template.get(key)))
        values.extend(clean_string_list(extra))
        return list(dict.fromkeys([item for item in values if item]))

    return {
        "strategy_id": strategy_id,
        "style_family": style_family,
        "title": title,
        "description": rationale or clean_text(base_template.get("description", "")) or f"Candidate-derived skill for {style_family}.",
        "environment": environment or clean_text(base_template.get("environment", "")),
        "prop": prop or clean_text(base_template.get("prop", "")),
        "lighting": lighting or clean_text(base_template.get("lighting", "")),
        "costume": costume or clean_text(base_template.get("costume", "")),
        "subject_surface": subject_surface or clean_text(base_template.get("subject_surface", "")),
        "subject_integration": subject_integration or clean_text(base_template.get("subject_integration", "")),
        "atmosphere": atmosphere or clean_text(base_template.get("atmosphere", "")),
        "material": material or clean_text(base_template.get("material", "")),
        "tags": _merged_list("tags", [style_family, prop, atmosphere]),
        "intensity": clean_text(base_template.get("intensity", "")) or "normal",
        "pros": clean_string_list(candidate_strategy.get("pros")) or clean_string_list(base_template.get("pros")),
        "cons": clean_string_list(candidate_strategy.get("cons")) or clean_string_list(base_template.get("cons")),
        "forbidden_mixes": clean_string_list(base_template.get("forbidden_mixes")),
        "anchor_requirements": _merged_list("anchor_requirements", carrier_seed),
        "best_for_archetypes": clean_string_list(base_template.get("best_for_archetypes")),
        "high_risk_archetypes": clean_string_list(base_template.get("high_risk_archetypes")),
        "world_rewrite_cost": clean_text(base_template.get("world_rewrite_cost", "")) or "medium",
        "identity_risk": clean_text(base_template.get("identity_risk", "")) or "low",
        "prompt_compilation_notes": _merged_list(
            "prompt_compilation_notes",
            [
                "Materialized from a candidate strategy before final planning.",
                "Keep all FX bound to visible carriers from the video.",
                "Use the candidate rationale as the high-level template basis, then ground it to the visible scene.",
            ],
        ),
        "common_failure_modes": clean_string_list(base_template.get("common_failure_modes")),
        "recommended_carriers": _merged_list("recommended_carriers", carrier_seed),
    }


def custom_strategy_from_base(base: dict, suffix: str, title: str, rationale: str, environment: str, prop: str, lighting: str, costume: str, atmosphere: str) -> dict:
    return {
        "strategy_id": f"{base.get('strategy_id', 'custom')}_{suffix}",
        "style_family": base.get("style_family", "sci_fi"),
        "title": title,
        "rationale": rationale,
        "environment": environment,
        "prop": prop,
        "lighting": lighting,
        "costume": costume,
        "subject_surface": base.get("subject_surface", ""),
        "subject_integration": base.get("subject_integration", ""),
        "atmosphere": atmosphere,
        "material": base.get("material", ""),
        "pros": ["Better fit to current video anchors", "Keeps conversion cost controlled"],
        "cons": ["Custom style needs stronger execution discipline"],
    }


def synthesize_candidate_payload(*, selected_family: str, selected_skills: list[dict], video_analysis: dict, prompt: str) -> dict:
    family_label = selected_family or "custom_family"
    base = selected_skills[0] if selected_skills else {
        "strategy_id": f"{family_label}_core",
        "style_family": selected_family,
        "title": "Custom Core Style",
        "rationale": "Fallback synthesized candidate from selected family and video anchors.",
        "environment": f"Upgrade the current scene into a cinematic {family_label} staging ground.",
        "prop": "Transform the yellow ring structure into a functional VFX carrier.",
        "lighting": f"Replace plain lighting with motivated cinematic {family_label} lighting.",
        "costume": f"Upgrade each subject into a coherent {family_label} wardrobe.",
        "subject_surface": f"Add light-touch {family_label} surface treatment while preserving identity.",
        "subject_integration": "Bind all subjects into the new world with edge light, reflections, and staged interaction.",
        "atmosphere": "Use haze, light shafts, and energy residue to support scale.",
        "material": "Use brushed metal, emissive panels, engineered composites, and grounded wear.",
    }
    visible_people = video_analysis.get("visible_people") or ["wheelchair subject", "standing male subject", "standing female subject"]
    visible_props = video_analysis.get("visible_props") or ["yellow ring structure", "wheelchair"]
    visible_world = video_analysis.get("visible_world_anchors") or ["grassy field", "mist", "trees"]
    strategies = [base]
    if len(strategies) < 2:
        strategies.append(
            custom_strategy_from_base(
                base,
                "field_array",
                "Field Array Gateway",
                "Custom variant that uses the yellow ring and open field as a contained energy-test site.",
                f"Turn {' / '.join(visible_world[:3])} into a controlled sci-fi test field with distant power pylons and restrained set extensions.",
                f"Convert {' / '.join(visible_props[:2])} into a calibrated gateway device with emitter rails and diagnostics.",
                "Introduce motivated rim light, localized energy pulses, and atmospheric backlight tied to the ring device.",
                f"Redesign {' / '.join(visible_people[:3])} into coordinated field-team gear with utility straps and lightweight armor layers.",
                "Use mist, dust drift, and energy shimmer around the device rather than detached random FX.",
            )
        )
    if len(strategies) < 3:
        strategies.append(
            custom_strategy_from_base(
                base,
                "expedition",
                "Expedition Sci-Fi Team",
                "Custom variant that prioritizes subject redesign and believable mission-world integration over large world replacement.",
                "Keep the outdoor scene readable but elevate it into a cinematic anomaly-zone perimeter with subtle tech infrastructure.",
                "Upgrade the yellow structure into a scanning or portal-frame apparatus and the wheelchair into a rugged mobility support system.",
                "Use directional search-light logic, suit-mounted practicals, and controlled glow only on real carriers.",
                f"Give each subject a differentiated but related expedition role costume based on current pose and placement: {' / '.join(visible_people[:3])}.",
                "Keep the atmosphere grounded with low fog, environmental depth, and limited energy residue.",
            )
        )
    plan_candidates = []
    candidate_scores = []
    for index, strategy in enumerate(strategies[:3], start=1):
        candidate_id = f"candidate_{index}"
        plan_candidates.append(
            {
                "candidate_id": candidate_id,
                "skill_basis": [strategy["strategy_id"]],
                "environment_plan": strategy["environment"],
                "prop_plan": strategy["prop"],
                "lighting_plan": strategy["lighting"],
                "costume_plan": strategy["costume"],
                "subject_surface_plan": strategy["subject_surface"],
                "integration_plan": strategy["subject_integration"],
            }
        )
        score = 0.92 if index == 1 else 0.84 - (index - 2) * 0.03
        candidate_scores.append(
            {
                "candidate_id": candidate_id,
                "prompt_fit": score,
                "video_anchor_fit": score,
                "identity_safety": 0.9,
                "physical_plausibility": 0.88 if index == 1 else 0.83,
                "effect_impact": 0.9 if index == 1 else 0.86,
                "total": score,
                "why_selected_or_rejected": "Strongest conversion fit with the yellow ring, outdoor field, and current subject layout." if index == 1 else "Viable alternative, but less directly grounded than the strongest candidate.",
            }
        )
    return {
        "candidate_strategies": strategies[:3],
        "plan_candidates": plan_candidates,
        "candidate_scores": candidate_scores,
        "selected_candidate_id": "candidate_1",
        "selected_strategy": strategies[0]["strategy_id"],
        "selection_reason": f"Selected {strategies[0]['strategy_id']} because it best matches the requested {family_label} direction and most efficiently converts the current video anchors into a coherent effects world.",
        "focus_regions": ["environment", "functional_scene_props", "lighting_system_design", "costume_design", "subject_environment_integration"],
    }


def selected_family_or_default(state: dict, available_style_families: list[str]) -> str:
    selected = str(state["skill_style_family"].get("selected_style_family", "")).strip()
    if selected:
        return selected
    prompt = str(state.get("request", {}).get("prompt", ""))
    if "科幻" in prompt or "sci-fi" in prompt.lower() or "scifi" in prompt.lower():
        return "sci_fi" if "sci_fi" in available_style_families else (available_style_families[0] if available_style_families else "sci_fi")
    return available_style_families[0] if available_style_families else "sci_fi"


def selected_skill_ids_or_default(state: dict, family_skills: list) -> list[str]:
    matched_ids = [str(item).strip() for item in (state["skill_style_family"].get("matched_skill_candidates") or []) if str(item).strip()]
    available_ids = {skill.strategy_id for skill in family_skills}
    selected = [item for item in matched_ids if item in available_ids]
    if len(selected) >= 3:
        return selected[:3]
    for skill in family_skills:
        if skill.strategy_id not in selected:
            selected.append(skill.strategy_id)
        if len(selected) >= 3:
            break
    return selected[:3]
