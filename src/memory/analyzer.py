from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: object) -> list[str]:
    items: list[str] = []
    if isinstance(values, list):
        raw_items = values
    elif isinstance(values, str):
        raw_items = values.split(",")
    else:
        raw_items = []
    for item in raw_items:
        text = _clean_text(item)
        if text:
            items.append(text)
    return list(dict.fromkeys(items))


def _salient_prompt_terms(prompt: str, *, limit: int = 8) -> list[str]:
    terms: list[str] = []
    for token in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", str(prompt or "").lower()):
        token = token.strip()
        if len(token) < 2:
            continue
        terms.append(token)
    return list(dict.fromkeys(terms))[:limit]


def _infer_scene_archetype(prompt: str, planner_payload: dict, evaluation_payload: dict) -> dict:
    direct = planner_payload.get("scene_archetype") or {}
    if isinstance(direct, dict) and _clean_text(direct.get("archetype_id", "")):
        return {
            "archetype_id": _clean_text(direct.get("archetype_id", "")),
            "scene_hint": _clean_text(direct.get("scene_hint", "")) or _clean_text(direct.get("archetype_id", "")).replace("_", " "),
            "subject_count": int(direct.get("subject_count", 0) or 0),
            "world_anchor_count": int(direct.get("world_anchor_count", 0) or 0),
            "prop_count": int(direct.get("prop_count", 0) or 0),
            "evidence": dict(direct.get("evidence") or {}),
        }
    visible_people = list(planner_payload.get("grounding_evidence", {}).get("visible_people", []) or [])
    visible_props = list(planner_payload.get("grounding_evidence", {}).get("visible_props", []) or [])
    world_anchors = list(planner_payload.get("grounding_evidence", {}).get("visible_world_anchors", []) or [])
    prompt_lower = prompt.lower()
    subject_count = len(visible_people)
    indoor = any(token in prompt_lower for token in ["室内", "indoor", "control room", "room"])
    closeup = subject_count <= 1
    if indoor and closeup:
        archetype_id = "single_person_indoor_closeup"
    elif subject_count >= 3:
        archetype_id = "multi_person_group_scene"
    elif world_anchors:
        archetype_id = "world_anchor_driven_scene"
    else:
        archetype_id = "general_edit_scene"
    return {
        "archetype_id": archetype_id,
        "scene_hint": archetype_id.replace("_", " "),
        "subject_count": subject_count,
        "world_anchor_count": len(world_anchors),
        "prop_count": len(visible_props),
        "evidence": {
            "visible_people": visible_people[:5],
            "visible_props": visible_props[:5],
            "visible_world_anchors": world_anchors[:5],
            "overall_score": evaluation_payload.get("scores", {}).get("overall_score", 0.0),
        },
    }


def _model_behavior_priors(variant: str, evaluation_payload: dict) -> list[dict]:
    scores = evaluation_payload.get("scores", {}) or {}
    failure_tags = list(evaluation_payload.get("failure_tags", []) or [])
    priors: list[dict] = []
    if not variant:
        return priors
    if "under_transformation" in failure_tags:
        priors.append(
            {
                "variant": variant,
                "pattern": "under_transformation",
                "observation": "This variant can preserve structure while failing to push visible transformation strongly enough.",
                "evidence_score": scores.get("transformation_strength", 0.0),
            }
        )
    if "weak_world_realization" in failure_tags:
        priors.append(
            {
                "variant": variant,
                "pattern": "weak_world_realization",
                "observation": "This variant benefits from explicit world-building instructions rather than generic atmosphere language.",
                "evidence_score": scores.get("world_realization", 0.0),
            }
        )
    return priors


def _style_convertibility_priors(
    *,
    prompt: str,
    planner_payload: dict,
    evaluation_payload: dict,
    archetype: dict,
) -> list[dict]:
    scores = evaluation_payload.get("scores", {}) or {}
    failure_tags = list(evaluation_payload.get("failure_tags", []) or [])
    style_family = str(planner_payload.get("style_family", "")).strip()
    if not style_family:
        return []

    observation_parts: list[str] = []
    if "weak_world_realization" in failure_tags:
        observation_parts.append("world realization tends to be weak unless world-building is made visually explicit")
    if "under_transformation" in failure_tags:
        observation_parts.append("conversion requires stronger visible transformation rather than subtle polish")
    if "detached_fx" in failure_tags:
        observation_parts.append("effects should be tied to concrete visible carriers instead of diffuse background glow")
    if not observation_parts:
        if float(scores.get("overall_score", 0.0) or 0.0) >= 0.75:
            observation_parts.append("this archetype converts well when the selected style is expressed through concrete prop and costume changes")
        else:
            observation_parts.append("this archetype has moderate convertibility and needs concrete object-level world cues")

    return [
        {
            "style_family": style_family,
            "archetype_id": archetype.get("archetype_id", ""),
            "scene_hint": archetype.get("scene_hint", ""),
            "when_prompt_contains": prompt[:200],
            "observation": "; ".join(observation_parts),
            "evidence_score": float(scores.get("overall_score", 0.0) or 0.0),
            "failure_tags": failure_tags[:8],
        }
    ]


def _failure_patterns(prompt: str, planner_payload: dict, evaluation_payload: dict, archetype: dict) -> list[dict]:
    notes = list(evaluation_payload.get("notes", []) or [])
    recommended_carriers = _clean_list(
        list(planner_payload.get("primary_fx_carriers") or [])
        + list((planner_payload.get("grounding_evidence") or {}).get("carrier_candidates") or [])
    )[:6]
    patterns = []
    for tag in list(evaluation_payload.get("failure_tags", []) or []):
        patterns.append(
            {
                "pattern": tag,
                "scene_hint": archetype.get("scene_hint", ""),
                "archetype_id": archetype.get("archetype_id", ""),
                "when_prompt_contains": prompt[:200],
                "notes": notes[:3],
                "recommended_carriers": recommended_carriers,
            }
        )
    return patterns


def _recipe_candidates(prompt: str, variant: str, reflection_payload: dict, archetype: dict) -> list[dict]:
    prompt_patch = list(reflection_payload.get("prompt_patch", []) or [])
    if not prompt_patch:
        return []
    return [
        {
            "recipe_id": f"{archetype.get('archetype_id', 'general')}__{variant or 'default'}__prompt_patch",
            "variant": variant,
            "archetype_id": archetype.get("archetype_id", ""),
            "when_prompt_contains": _salient_prompt_terms(prompt),
            "patches": prompt_patch[:6],
            "source_action": reflection_payload.get("action", ""),
            "notes": "Retry-driven prompt compilation recipe distilled from reflection prompt_patch.",
        }
    ]


def _custom_skill_records(
    *,
    sample_id: str,
    attempt_index: int,
    variant: str,
    prompt: str,
    planner_payload: dict,
    evaluation_payload: dict,
    reflection_payload: dict,
    archetype: dict,
) -> list[dict]:
    style_family = _clean_text(planner_payload.get("style_family", ""))
    selected_strategy = _clean_text(planner_payload.get("selected_strategy", ""))
    overall_score = float(evaluation_payload.get("scores", {}).get("overall_score", 0.0) or 0.0)
    accepted = reflection_payload.get("action", "") == "accept"
    failure_tags = [_clean_text(tag) for tag in (evaluation_payload.get("failure_tags", []) or []) if _clean_text(tag)]
    carrier_grounding = float(evaluation_payload.get("scores", {}).get("carrier_grounding", 0.0) or 0.0)
    transformation_strength = float(evaluation_payload.get("scores", {}).get("transformation_strength", 0.0) or 0.0)
    world_realization = float(evaluation_payload.get("scores", {}).get("world_realization", 0.0) or 0.0)

    analysis_custom_skills = [
        item for item in (planner_payload.get("custom_skill_candidates", []) or []) if isinstance(item, dict)
    ]
    analysis_custom_ids = {
        _clean_text(item.get("strategy_id", ""))
        for item in analysis_custom_skills
        if _clean_text(item.get("strategy_id", ""))
    }

    candidate_strategies = [
        item for item in (planner_payload.get("candidate_strategies", []) or []) if isinstance(item, dict)
    ]
    provided_style_templates = [
        item for item in (planner_payload.get("provided_style_templates", []) or []) if isinstance(item, dict)
    ]
    registry_strategy_ids = {
        _clean_text(item.get("strategy_id", ""))
        for item in provided_style_templates
        if _clean_text(item.get("strategy_id", ""))
    }
    registry_strategy_ids.update(
        _clean_text(item)
        for item in (planner_payload.get("matched_skill_candidates", []) or [])
        if _clean_text(item)
    )
    records: list[dict] = []

    for item in analysis_custom_skills:
        strategy_id = _clean_text(item.get("strategy_id", ""))
        if not strategy_id:
            continue
        recommended_carriers = _clean_list(item.get("recommended_carriers"))
        anchor_requirements = _clean_list(item.get("anchor_requirements"))
        best_for_archetypes = _clean_list(item.get("best_for_archetypes"))
        high_risk_archetypes = _clean_list(item.get("high_risk_archetypes"))
        records.append(
            {
                "record_id": f"{style_family}::{strategy_id}::analysis_stage_custom",
                "skill_id": strategy_id,
                "style_family": style_family or _clean_text(item.get("style_family", "")),
                "source": "analysis_stage_custom",
                "sample_id": sample_id,
                "attempt_index": attempt_index,
                "variant": variant,
                "archetype_id": archetype.get("archetype_id", ""),
                "prompt": prompt[:400],
                "skill_payload": dict(item),
                "selected_for_candidate": strategy_id == selected_strategy,
                "selected_strategy": selected_strategy,
                "overall_score": overall_score,
                "reflection_action": reflection_payload.get("action", ""),
                "accepted": accepted,
                "evidence_score": overall_score,
                "recommended_carriers": recommended_carriers,
                "anchor_requirements": anchor_requirements,
                "best_for_archetypes": best_for_archetypes,
                "high_risk_archetypes": high_risk_archetypes,
                "failure_tags": failure_tags[:8],
                "lineage": {
                    "origin": "analysis_stage_custom",
                    "parent_skill_ids": [],
                    "generation": 0,
                    "change_summary": "Structured custom skill generated during analysis-stage skill completion.",
                },
                "judgment": {
                    "selected": strategy_id == selected_strategy,
                    "applied": strategy_id == selected_strategy,
                    "accepted": accepted,
                    "failure_tags": failure_tags[:8],
                    "carrier_grounding": carrier_grounding,
                    "transformation_strength": transformation_strength,
                    "world_realization": world_realization,
                },
            }
        )

    for item in candidate_strategies:
        strategy_id = _clean_text(item.get("strategy_id", ""))
        strategy_family = _clean_text(item.get("style_family", "")) or style_family
        if not strategy_id or strategy_id in analysis_custom_ids:
            continue
        if strategy_id in registry_strategy_ids:
            continue
        if strategy_family != style_family:
            continue
        recommended_carriers = _clean_list(item.get("recommended_carriers"))
        anchor_requirements = _clean_list(item.get("anchor_requirements"))
        best_for_archetypes = _clean_list(item.get("best_for_archetypes"))
        high_risk_archetypes = _clean_list(item.get("high_risk_archetypes"))
        records.append(
            {
                "record_id": f"{strategy_family}::{strategy_id}::candidate_stage_custom",
                "skill_id": strategy_id,
                "style_family": strategy_family,
                "source": "candidate_stage_custom",
                "sample_id": sample_id,
                "attempt_index": attempt_index,
                "variant": variant,
                "archetype_id": archetype.get("archetype_id", ""),
                "prompt": prompt[:400],
                "skill_payload": dict(item),
                "selected_for_candidate": strategy_id == selected_strategy,
                "selected_strategy": selected_strategy,
                "overall_score": overall_score,
                "reflection_action": reflection_payload.get("action", ""),
                "accepted": accepted,
                "evidence_score": overall_score,
                "recommended_carriers": recommended_carriers,
                "anchor_requirements": anchor_requirements,
                "best_for_archetypes": best_for_archetypes,
                "high_risk_archetypes": high_risk_archetypes,
                "failure_tags": failure_tags[:8],
                "lineage": {
                    "origin": "candidate_stage_custom",
                    "parent_skill_ids": [],
                    "generation": 0,
                    "change_summary": "Candidate-stage custom strategy materialized as a reusable skill-shaped object.",
                    "used_in_attempt": strategy_id == selected_strategy,
                    "selected_as_final_basis": strategy_id == selected_strategy,
                },
                "judgment": {
                    "selected": strategy_id == selected_strategy,
                    "applied": strategy_id == selected_strategy,
                    "accepted": accepted,
                    "failure_tags": failure_tags[:8],
                    "carrier_grounding": carrier_grounding,
                    "transformation_strength": transformation_strength,
                    "world_realization": world_realization,
                },
            }
        )

    materialized_selected_skill = planner_payload.get("materialized_selected_skill")
    if isinstance(materialized_selected_skill, dict):
        strategy_id = _clean_text(materialized_selected_skill.get("strategy_id", ""))
        strategy_family = _clean_text(materialized_selected_skill.get("style_family", "")) or style_family
        existing_record = next(
            (
                item
                for item in records
                if isinstance(item, dict)
                and _clean_text(item.get("skill_id", "")) == strategy_id
                and _clean_text(item.get("style_family", "")) == strategy_family
            ),
            None,
        )
        if existing_record is not None:
            lineage = existing_record.get("lineage", {}) if isinstance(existing_record.get("lineage", {}), dict) else {}
            lineage["materialized_for_final_plan"] = True
            lineage["used_in_attempt"] = True
            lineage["selected_as_final_basis"] = True
            lineage["child_sources"] = list(
                dict.fromkeys(list(lineage.get("child_sources") or []) + ["materialized_selected_skill"])
            )
            existing_record["lineage"] = lineage
            judgment = existing_record.get("judgment", {}) if isinstance(existing_record.get("judgment", {}), dict) else {}
            judgment["applied"] = True
            judgment["selected"] = True
            existing_record["judgment"] = judgment
            existing_record["recommended_carriers"] = _clean_list(materialized_selected_skill.get("recommended_carriers"))
            existing_record["anchor_requirements"] = _clean_list(materialized_selected_skill.get("anchor_requirements"))
            existing_record["best_for_archetypes"] = _clean_list(materialized_selected_skill.get("best_for_archetypes"))
            existing_record["high_risk_archetypes"] = _clean_list(materialized_selected_skill.get("high_risk_archetypes"))
            existing_record["failure_tags"] = failure_tags[:8]
        elif strategy_id and strategy_family == style_family:
            recommended_carriers = _clean_list(materialized_selected_skill.get("recommended_carriers"))
            anchor_requirements = _clean_list(materialized_selected_skill.get("anchor_requirements"))
            best_for_archetypes = _clean_list(materialized_selected_skill.get("best_for_archetypes"))
            high_risk_archetypes = _clean_list(materialized_selected_skill.get("high_risk_archetypes"))
            records.append(
                {
                    "record_id": f"{strategy_family}::{strategy_id}::materialized_selected_skill",
                    "skill_id": strategy_id,
                    "style_family": strategy_family,
                    "source": "materialized_selected_skill",
                    "memory_only": True,
                    "sample_id": sample_id,
                    "attempt_index": attempt_index,
                    "variant": variant,
                    "archetype_id": archetype.get("archetype_id", ""),
                    "prompt": prompt[:400],
                    "skill_payload": dict(materialized_selected_skill),
                    "selected_for_candidate": True,
                    "selected_strategy": selected_strategy,
                    "overall_score": overall_score,
                    "reflection_action": reflection_payload.get("action", ""),
                    "accepted": accepted,
                    "evidence_score": overall_score,
                    "recommended_carriers": recommended_carriers,
                    "anchor_requirements": anchor_requirements,
                    "best_for_archetypes": best_for_archetypes,
                    "high_risk_archetypes": high_risk_archetypes,
                    "failure_tags": failure_tags[:8],
                    "lineage": {
                        "origin": "materialized_selected_skill",
                        "parent_skill_ids": [selected_strategy] if selected_strategy and selected_strategy != strategy_id else [],
                        "generation": 1 if selected_strategy and selected_strategy != strategy_id else 0,
                        "change_summary": "Selected candidate strategy was materialized into a full runtime skill template for final planning.",
                        "materialized_for_final_plan": True,
                        "used_in_attempt": True,
                        "selected_as_final_basis": True,
                    },
                    "judgment": {
                        "selected": True,
                        "applied": True,
                        "accepted": accepted,
                        "failure_tags": failure_tags[:8],
                        "carrier_grounding": carrier_grounding,
                        "transformation_strength": transformation_strength,
                        "world_realization": world_realization,
                    },
                }
            )

    return records


@dataclass
class MemoryAnalysisResult:
    sample_id: str
    attempt_index: int
    variant: str
    scene_archetype: dict
    run_record: dict
    evaluation_record: dict
    failure_patterns: list[dict] = field(default_factory=list)
    style_convertibility_priors: list[dict] = field(default_factory=list)
    model_behavior_priors: list[dict] = field(default_factory=list)
    prompt_compilation_recipe_candidates: list[dict] = field(default_factory=list)
    custom_skill_records: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_loop_attempt(
    *,
    sample_id: str,
    attempt_index: int,
    variant: str,
    prompt: str,
    planner_payload: dict,
    edit_payload: dict,
    evaluation_payload: dict,
    reflection_payload: dict,
) -> MemoryAnalysisResult:
    archetype = _infer_scene_archetype(prompt, planner_payload, evaluation_payload)
    run_record = {
        "sample_id": sample_id,
        "attempt_index": attempt_index,
        "variant": variant,
        "selected_strategy": planner_payload.get("selected_strategy", ""),
        "style_family": planner_payload.get("style_family", ""),
        "overall_score": evaluation_payload.get("scores", {}).get("overall_score", 0.0),
        "action": reflection_payload.get("action", ""),
        "output_video": edit_payload.get("output_video", ""),
        "archetype_id": archetype.get("archetype_id", ""),
    }
    evaluation_record = {
        "sample_id": sample_id,
        "attempt_index": attempt_index,
        "variant": variant,
        "scores": dict(evaluation_payload.get("scores", {}) or {}),
        "failure_tags": list(evaluation_payload.get("failure_tags", []) or []),
        "notes": list(evaluation_payload.get("notes", []) or []),
    }
    return MemoryAnalysisResult(
        sample_id=sample_id,
        attempt_index=attempt_index,
        variant=variant,
        scene_archetype=archetype,
        run_record=run_record,
        evaluation_record=evaluation_record,
        failure_patterns=_failure_patterns(prompt, planner_payload, evaluation_payload, archetype),
        style_convertibility_priors=_style_convertibility_priors(
            prompt=prompt,
            planner_payload=planner_payload,
            evaluation_payload=evaluation_payload,
            archetype=archetype,
        ),
        model_behavior_priors=_model_behavior_priors(variant, evaluation_payload),
        prompt_compilation_recipe_candidates=_recipe_candidates(prompt, variant, reflection_payload, archetype),
        custom_skill_records=_custom_skill_records(
            sample_id=sample_id,
            attempt_index=attempt_index,
            variant=variant,
            prompt=prompt,
            planner_payload=planner_payload,
            evaluation_payload=evaluation_payload,
            reflection_payload=reflection_payload,
            archetype=archetype,
        ),
    )
