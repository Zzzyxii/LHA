from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from memory import build_memory_query, retrieve_execution_adaptive_bias, retrieve_memory_context, retrieve_runtime_skills
from planner.core.common import (
    clean_string_list as _clean_string_list,
    clean_text as _clean_text,
    merge_payload_dict as _merge_payload_dict,
    parse_bool as _parse_bool,
    parse_float as _parse_float,
    parse_section_fields as _parse_section_fields,
    parse_tagged_blocks as _parse_tagged_blocks,
    planner_log as _planner_log,
    planner_state_template,
    split_items as _split_items,
    state_diagnostics as _state_diagnostics,
    update_section as _update_section,
    write_state as _write_state,
)
from planner.core.pipeline import build_plan
from planner.core.planner_types import VideoPlan
from planner.core.skill_runtime import (
    adaptive_runtime_skills as _adaptive_runtime_skills,
    apply_candidate_strategy_bias as _apply_candidate_strategy_bias,
    candidate_strategy_map as _candidate_strategy_map,
    contextual_skill_briefs as _contextual_skill_briefs,
    family_options as _family_options,
    grounding_scene_archetype as _grounding_scene_archetype,
    materialize_candidate_strategy_as_skill as _materialize_candidate_strategy_as_skill,
    merge_custom_skill_candidates as _merge_custom_skill_candidates,
    merge_frontloaded_skills as _merge_frontloaded_skills,
    selected_family_or_default as _selected_family_or_default,
    selected_skill_ids_or_default as _selected_skill_ids_or_default,
    selected_skills_with_custom as _selected_skills_with_custom,
    skill_brief_from_dict as _skill_brief_from_dict,
    skill_briefs as _skill_briefs,
    skill_full as _skill_full,
    synthesize_candidate_payload as _synthesize_candidate_payload,
    validate_custom_skill_candidates as _validate_custom_skill_candidates,
    context_archetype_id as _context_archetype_id,
)
from planner.core.stage_runner import (
    call_repair_step as _call_repair_step,
    call_step as _call_step,
)
from planner.prompts.planner_prompts import (
    build_candidate_planning_system_prompt,
    build_candidate_planning_user_prompt,
    build_candidate_scoring_system_prompt,
    build_candidate_scoring_user_prompt,
    build_family_selection_system_prompt,
    build_family_selection_user_prompt,
    build_final_plan_system_prompt,
    build_final_plan_user_prompt,
    build_skill_selection_system_prompt,
    build_skill_selection_user_prompt,
    build_video_grounding_system_prompt,
    build_video_grounding_user_prompt,
)
from skills.registry import get_skill_registry
from vlm import resolve_vision_model
from vlm.local_qwen import extract_frames


def _available_style_families(registry) -> list[str]:
    return sorted({skill.style_family for skill in registry.get_all_skills()})


def _is_explicit_no(value: object) -> bool:
    lowered = _clean_text(value).lower()
    return lowered in {"", "no", "none", "null", "n/a", "无", "没有"}


def _split_items_or_empty(value: str) -> list[str]:
    return [] if _is_explicit_no(value) else _split_items(value)


def _parse_family_selection_text(text: str) -> dict:
    fields = _parse_section_fields(
        text,
        [
            "用户需求分析",
            "USER_INTENT_SUMMARY",
            "SELECTED_STYLE_FAMILY",
            "STYLE_FAMILY_CONFIDENCE",
            "STYLE_FAMILY_REASON",
            "IS_PROMPT_VAGUE",
        ],
    )
    selected_style_family = _clean_text(fields.get("SELECTED_STYLE_FAMILY", ""))
    if not selected_style_family:
        lowered = text.lower()
        family_hits = []
        for family in ("sci_fi", "anime", "historical_fantasy"):
            score = 0
            if family in lowered:
                score += 3
            if f"选择的风格家族。应该是{family}".lower() in lowered:
                score += 5
            if f"应该是{family}".lower() in lowered:
                score += 4
            if f"directly correspond to {family}".lower() in lowered:
                score += 4
            family_hits.append((score, family))
        if "科幻" in text:
            family_hits.append((10, "sci_fi"))
        if "动漫" in text or "动画" in text:
            family_hits.append((10, "anime"))
        if "仙侠" in text or "武侠" in text or "中世纪" in text:
            family_hits.append((10, "historical_fantasy"))
        selected_style_family = max(family_hits)[1] if any(score > 0 for score, _ in family_hits) else ""
    style_family_reason = _clean_text(fields.get("STYLE_FAMILY_REASON", ""))
    if not style_family_reason:
        reason_match = re.search(r"(从prompt中.*?更合适。|因为用户明确.*?。|这明确提到了.*?。)", text, flags=re.DOTALL)
        style_family_reason = _clean_text(reason_match.group(1)) if reason_match else ""
    confidence = _parse_float(fields.get("STYLE_FAMILY_CONFIDENCE", ""))
    if confidence == 0.0 and selected_style_family:
        confidence = 0.9
    user_intent_summary = _clean_text(fields.get("用户需求分析", "") or fields.get("USER_INTENT_SUMMARY", ""))
    style_family_candidates = [selected_style_family] if selected_style_family else []
    return {
        "user_intent_summary": user_intent_summary,
        "style_family_candidates": style_family_candidates,
        "selected_style_family": selected_style_family,
        "style_family_confidence": confidence,
        "style_family_reason": style_family_reason,
        "style_intensity": "",
        "is_prompt_vague": _parse_bool(fields.get("IS_PROMPT_VAGUE", "")),
        "prompt_safety_notes": [],
        "suggested_prompt_rewrites": [],
    }


def _parse_video_analysis_text(text: str) -> dict:
    fields = _parse_section_fields(
        text,
        [
            "VIDEO_BRIEF",
            "VISIBLE_PEOPLE",
            "VISIBLE_PROPS",
            "VISIBLE_WORLD_ANCHORS",
            "PRIMARY_FX_CARRIERS",
            "GROUNDING_NOTES",
            "MATCHED_SKILL_CANDIDATES",
            "REJECTED_SKILL_CANDIDATES",
            "NEEDS_CUSTOM_SKILL",
            "CUSTOM_SKILL_SEED",
            "CUSTOM_SKILL_CANDIDATES",
        ],
    )
    custom_skill_candidates = []
    for item in _parse_tagged_blocks(text, "CUSTOM_SKILL_CANDIDATE"):
        strategy_id = _clean_text(item.get("strategy_id", ""))
        style_family = _clean_text(item.get("style_family", ""))
        title = _clean_text(item.get("title", ""))
        description = _clean_text(item.get("description", ""))
        if not strategy_id or not title:
            continue
        custom_skill_candidates.append(
            {
                "strategy_id": strategy_id,
                "style_family": style_family,
                "title": title,
                "description": description,
                "environment": _clean_text(item.get("environment", "")),
                "prop": _clean_text(item.get("prop", "")),
                "lighting": _clean_text(item.get("lighting", "")),
                "costume": _clean_text(item.get("costume", "")),
                "subject_surface": _clean_text(item.get("subject_surface", "")),
                "subject_integration": _clean_text(item.get("subject_integration", "")),
                "atmosphere": _clean_text(item.get("atmosphere", "")),
                "material": _clean_text(item.get("material", "")),
                "tags": _split_items_or_empty(item.get("tags", "")),
                "intensity": _clean_text(item.get("intensity", "")) or "normal",
                "pros": _split_items_or_empty(item.get("pros", "")),
                "cons": _split_items_or_empty(item.get("cons", "")),
                "forbidden_mixes": _split_items_or_empty(item.get("forbidden_mixes", "")),
                "anchor_requirements": _split_items_or_empty(item.get("anchor_requirements", "")),
                "best_for_archetypes": _split_items_or_empty(item.get("best_for_archetypes", "")),
                "high_risk_archetypes": _split_items_or_empty(item.get("high_risk_archetypes", "")),
                "world_rewrite_cost": _clean_text(item.get("world_rewrite_cost", "")) or "medium",
                "identity_risk": _clean_text(item.get("identity_risk", "")) or "medium",
                "prompt_compilation_notes": _split_items_or_empty(item.get("prompt_compilation_notes", "")),
                "common_failure_modes": _split_items_or_empty(item.get("common_failure_modes", "")),
                "recommended_carriers": _split_items_or_empty(item.get("recommended_carriers", "")),
            }
        )
    matched_skill_candidates = _split_items_or_empty(fields.get("MATCHED_SKILL_CANDIDATES", ""))
    if custom_skill_candidates:
        custom_ids = [item.get("strategy_id", "") for item in custom_skill_candidates if item.get("strategy_id")]
        matched_skill_candidates = list(dict.fromkeys(matched_skill_candidates + custom_ids))
    if not matched_skill_candidates:
        known_ids = [
            "wormhole_portal",
            "industrial_wasteland",
            "cyberpunk_dystopia",
            "alien_biomech",
            "clean_lab",
            "star_wars_scifi",
            "mad_max_wasteland",
            "studio_ghibli",
            "cyberpunk_anime",
            "fantasy_anime",
            "xianxia",
            "wuxia",
            "medieval_fantasy",
        ]
        lowered = text.lower()
        matched_skill_candidates = [skill_id for skill_id in known_ids if skill_id.lower() in lowered][:3]
    visible_people = _split_items_or_empty(fields.get("VISIBLE_PEOPLE", ""))
    if not visible_people:
        lowered = text.lower()
        if "三个人" in text or "three people" in lowered:
            visible_people = ["wheelchair subject", "standing male subject", "standing female subject"]
        elif "轮椅" in text or "wheelchair" in lowered:
            visible_people = ["wheelchair subject"]
    visible_props = _split_items_or_empty(fields.get("VISIBLE_PROPS", ""))
    if not visible_props:
        lowered = text.lower()
        if "黄色" in text or "yellow" in lowered:
            visible_props.append("yellow ring structure")
        if "轮椅" in text or "wheelchair" in lowered:
            visible_props.append("wheelchair")
    visible_world_anchors = _split_items_or_empty(fields.get("VISIBLE_WORLD_ANCHORS", ""))
    if not visible_world_anchors:
        lowered = text.lower()
        if "草地" in text or "grass" in lowered:
            visible_world_anchors.append("grassy field")
        if "雾" in text or "fog" in lowered or "mist" in lowered:
            visible_world_anchors.append("mist")
        if "树" in text or "tree" in lowered:
            visible_world_anchors.append("trees")
        if "山" in text or "hill" in lowered:
            visible_world_anchors.append("hills")
    primary_fx_carriers = _split_items_or_empty(fields.get("PRIMARY_FX_CARRIERS", ""))
    if not primary_fx_carriers:
        primary_fx_carriers = [item for item in ["yellow ring structure", "environment", "mist layer"] if item in visible_props + visible_world_anchors or item in {"environment", "mist layer"}]
    video_brief = _clean_text(fields.get("VIDEO_BRIEF", ""))
    if not video_brief:
        if visible_people or visible_props or visible_world_anchors:
            video_brief = "Outdoor group scene with three subjects, a yellow ring structure, wheelchair, grassy field, and mist."
    return {
        "video_brief": video_brief,
        "visible_people": visible_people,
        "visible_props": visible_props,
        "visible_world_anchors": visible_world_anchors,
        "primary_fx_carriers": primary_fx_carriers,
        "grounding_notes": _split_items_or_empty(fields.get("GROUNDING_NOTES", "")),
        "matched_skill_candidates": matched_skill_candidates,
        "rejected_skill_candidates": _split_items_or_empty(fields.get("REJECTED_SKILL_CANDIDATES", "")),
        "needs_custom_skill": _parse_bool(fields.get("NEEDS_CUSTOM_SKILL", "")) or bool(custom_skill_candidates),
        "custom_skill_seed": "" if _is_explicit_no(fields.get("CUSTOM_SKILL_SEED", "")) else _clean_text(fields.get("CUSTOM_SKILL_SEED", "")),
        "custom_skill_candidates": custom_skill_candidates,
    }


def _parse_video_grounding_text(text: str) -> dict:
    fields = _parse_section_fields(
        text,
        [
            "VIDEO_BRIEF",
            "VISIBLE_PEOPLE",
            "VISIBLE_PROPS",
            "VISIBLE_WORLD_ANCHORS",
            "PRIMARY_FX_CARRIERS",
            "GROUNDING_NOTES",
        ],
    )
    return {
        "video_brief": _clean_text(fields.get("VIDEO_BRIEF", "")),
        "visible_people": _split_items_or_empty(fields.get("VISIBLE_PEOPLE", "")),
        "visible_props": _split_items_or_empty(fields.get("VISIBLE_PROPS", "")),
        "visible_world_anchors": _split_items_or_empty(fields.get("VISIBLE_WORLD_ANCHORS", "")),
        "primary_fx_carriers": _split_items_or_empty(fields.get("PRIMARY_FX_CARRIERS", "")),
        "grounding_notes": _split_items_or_empty(fields.get("GROUNDING_NOTES", "")),
    }


def _parse_skill_selection_text(text: str) -> dict:
    fields = _parse_section_fields(
        text,
        [
            "MATCHED_SKILL_CANDIDATES",
            "REJECTED_SKILL_CANDIDATES",
            "NEEDS_CUSTOM_SKILL",
            "CUSTOM_SKILL_SEED",
            "CUSTOM_SKILL_CANDIDATES",
        ],
    )
    custom_skill_candidates = []
    for item in _parse_tagged_blocks(text, "CUSTOM_SKILL_CANDIDATE"):
        strategy_id = _clean_text(item.get("strategy_id", ""))
        title = _clean_text(item.get("title", ""))
        if not strategy_id or not title:
            continue
        custom_skill_candidates.append(
            {
                "strategy_id": strategy_id,
                "style_family": _clean_text(item.get("style_family", "")),
                "title": title,
                "description": _clean_text(item.get("description", "")),
                "environment": _clean_text(item.get("environment", "")),
                "prop": _clean_text(item.get("prop", "")),
                "lighting": _clean_text(item.get("lighting", "")),
                "costume": _clean_text(item.get("costume", "")),
                "subject_surface": _clean_text(item.get("subject_surface", "")),
                "subject_integration": _clean_text(item.get("subject_integration", "")),
                "atmosphere": _clean_text(item.get("atmosphere", "")),
                "material": _clean_text(item.get("material", "")),
                "tags": _split_items_or_empty(item.get("tags", "")),
                "intensity": _clean_text(item.get("intensity", "")) or "normal",
                "pros": _split_items_or_empty(item.get("pros", "")),
                "cons": _split_items_or_empty(item.get("cons", "")),
                "forbidden_mixes": _split_items_or_empty(item.get("forbidden_mixes", "")),
                "anchor_requirements": _split_items_or_empty(item.get("anchor_requirements", "")),
                "best_for_archetypes": _split_items_or_empty(item.get("best_for_archetypes", "")),
                "high_risk_archetypes": _split_items_or_empty(item.get("high_risk_archetypes", "")),
                "world_rewrite_cost": _clean_text(item.get("world_rewrite_cost", "")) or "medium",
                "identity_risk": _clean_text(item.get("identity_risk", "")) or "medium",
                "prompt_compilation_notes": _split_items_or_empty(item.get("prompt_compilation_notes", "")),
                "common_failure_modes": _split_items_or_empty(item.get("common_failure_modes", "")),
                "recommended_carriers": _split_items_or_empty(item.get("recommended_carriers", "")),
            }
        )
    matched_skill_candidates = _split_items_or_empty(fields.get("MATCHED_SKILL_CANDIDATES", ""))
    custom_skill_field = fields.get("CUSTOM_SKILL_CANDIDATES", "")
    return {
        "matched_skill_candidates": _clean_string_list(matched_skill_candidates),
        "rejected_skill_candidates": _split_items_or_empty(fields.get("REJECTED_SKILL_CANDIDATES", "")),
        "needs_custom_skill": _parse_bool(fields.get("NEEDS_CUSTOM_SKILL", "")) or bool(custom_skill_candidates),
        "custom_skill_seed": "" if _is_explicit_no(fields.get("CUSTOM_SKILL_SEED", "")) else _clean_text(fields.get("CUSTOM_SKILL_SEED", "")),
        "custom_skill_candidates": [] if _is_explicit_no(custom_skill_field) and not custom_skill_candidates else custom_skill_candidates,
    }


def _analysis_structure_diagnostics(raw_text: str, payload: dict) -> list[str]:
    diagnostics: list[str] = []
    required_headings = [
        "VIDEO_BRIEF:",
        "VISIBLE_PEOPLE:",
        "VISIBLE_PROPS:",
        "VISIBLE_WORLD_ANCHORS:",
        "PRIMARY_FX_CARRIERS:",
        "GROUNDING_NOTES:",
        "MATCHED_SKILL_CANDIDATES:",
        "REJECTED_SKILL_CANDIDATES:",
        "NEEDS_CUSTOM_SKILL:",
        "CUSTOM_SKILL_SEED:",
        "CUSTOM_SKILL_CANDIDATES:",
    ]
    for heading in required_headings:
        if heading not in raw_text:
            diagnostics.append(f"missing heading {heading.rstrip(':')}")
    if "<think>" in raw_text.lower() or "</think>" in raw_text.lower():
        diagnostics.append("contains_think_tags")
    if payload.get("needs_custom_skill") and not payload.get("custom_skill_candidates"):
        diagnostics.append("needs_custom_skill_without_custom_skill_candidates")
    if payload.get("custom_skill_candidates") and "[CUSTOM_SKILL_CANDIDATE]" not in raw_text:
        diagnostics.append("custom_skill_candidates_missing_block_tags")
    return diagnostics


def _build_prompt_memory_context(
    *,
    memory_context: dict,
    selected_family: str,
    scene_archetype: dict | None = None,
    archetype_id: str = "",
    memory_store: dict | None = None,
    prompt: str = "",
    variant: str = "",
    grounding_payload: dict | None = None,
) -> tuple[dict, list[dict]]:
    prompt_memory_context = {}
    effective_archetype_id = _clean_text(archetype_id or (scene_archetype or {}).get("archetype_id", ""))
    grounding_payload = grounding_payload or {}
    if memory_store:
        refined_query = build_memory_query(
            prompt=prompt,
            variant=variant,
            selected_family=selected_family,
            scene_archetype=effective_archetype_id or scene_archetype or "",
            visible_people=grounding_payload.get("visible_people") or [],
            visible_props=grounding_payload.get("visible_props") or [],
            visible_world_anchors=grounding_payload.get("visible_world_anchors") or [],
            carrier_candidates=grounding_payload.get("primary_fx_carriers") or grounding_payload.get("carrier_candidates") or [],
            primary_fx_carriers=grounding_payload.get("primary_fx_carriers") or [],
        )
        prompt_memory_context = retrieve_memory_context(memory_store, query=refined_query)
    elif memory_context:
        prompt_memory_context = dict(memory_context)

    for key in [
        "critic_replan_directives",
        "critic_failure_hints",
        "critic_prompt_patch",
        "attempt_adaptation_state",
    ]:
        if key in memory_context:
            prompt_memory_context[key] = deepcopy(memory_context[key])
    if scene_archetype:
        existing_scene_hints = [item for item in (prompt_memory_context.get("scene_archetype_hints") or []) if isinstance(item, dict)]
        prompt_memory_context["scene_archetype_hints"] = [scene_archetype] + existing_scene_hints
    execution_adaptive_bias = retrieve_execution_adaptive_bias(
        prompt_memory_context,
        selected_family=selected_family,
        archetype_id=effective_archetype_id,
    )
    if execution_adaptive_bias:
        prompt_memory_context["execution_adaptive_bias"] = execution_adaptive_bias
    retrieved_runtime_skills = retrieve_runtime_skills(
        prompt_memory_context,
        selected_family=selected_family,
        archetype_id=effective_archetype_id,
    )
    prompt_memory_context["runtime_skill_records"] = retrieved_runtime_skills
    return prompt_memory_context, retrieved_runtime_skills


def _update_analysis_sections(
    *,
    state: dict,
    analysis_payload: dict,
    scene_archetype: dict,
) -> None:
    analysis_section_payload = dict(analysis_payload)
    analysis_section_payload["scene_archetype"] = scene_archetype
    _update_section(
        state,
        "prompt_video_analysis",
        analysis_section_payload,
        ["video_brief", "visible_people", "visible_props"],
    )
    _update_section(
        state,
        "skill_style_family",
        analysis_payload,
        [],
    )
    state["skill_style_family"]["status"] = "completed"
    state["skill_style_family"]["validation_errors"] = []


def _analyze_and_select_skills(
    *,
    model,
    state: dict,
    state_path: str | Path | None,
    sample_id: str,
    prompt: str,
    image_paths: list[str],
    selected_family: str,
    family_skills: list,
    memory_context: dict,
    memory_store: dict | None,
    variant: str,
    adaptive_skill_briefs: list[dict],
) -> tuple[dict, list[dict]]:
    grounding_system_prompt = build_video_grounding_system_prompt()
    grounding_user_text = build_video_grounding_user_prompt(
        sample_id=sample_id,
        prompt=prompt,
    )
    grounding_payload, grounding_raw = _call_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_002a_grounding",
        system_prompt=grounding_system_prompt,
        user_text=grounding_user_text,
        image_paths=image_paths,
        parser=_parse_video_grounding_text,
        max_new_tokens=8192,
    )
    grounding_repair_payload, grounding_repair_raw = _call_repair_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_002a_grounding_repair",
        section_name="prompt_video_grounding",
        current_payload=grounding_payload,
        required_fields=["video_brief", "visible_people", "visible_props"],
        system_prompt=grounding_system_prompt,
        user_text=grounding_user_text,
        image_paths=image_paths,
        parser=_parse_video_grounding_text,
        max_new_tokens=768,
    )
    if grounding_repair_payload:
        grounding_payload = _merge_payload_dict(grounding_payload, grounding_repair_payload)
        state["raw_model_outputs"]["call_002a_grounding_repair"] = {"parsed": grounding_repair_payload, "raw_response": grounding_repair_raw}
    state["raw_model_outputs"]["call_002a_grounding"] = {"parsed": grounding_payload, "raw_response": grounding_raw}

    current_scene_archetype = _grounding_scene_archetype(prompt, grounding_payload)
    prompt_memory_context, retrieved_runtime_skills = _build_prompt_memory_context(
        memory_context=memory_context,
        selected_family=selected_family,
        scene_archetype=current_scene_archetype,
        memory_store=memory_store,
        prompt=prompt,
        variant=variant,
        grounding_payload=grounding_payload,
    )
    retrieved_runtime_skill_briefs = [
        _skill_brief_from_dict(item, default_family=selected_family) for item in retrieved_runtime_skills
    ]
    skill_briefs = _contextual_skill_briefs(
        selected_family=selected_family,
        prompt=prompt,
        memory_context=prompt_memory_context,
        limit=5,
    ) or _skill_briefs(family_skills)
    skill_briefs = _merge_frontloaded_skills(
        skill_briefs,
        adaptive_skill_briefs + retrieved_runtime_skill_briefs,
        limit=5,
    )

    skill_selection_system_prompt = build_skill_selection_system_prompt()
    skill_selection_user_text = build_skill_selection_user_prompt(
        sample_id=sample_id,
        prompt=prompt,
        selected_style_family=selected_family,
        video_grounding=grounding_payload,
        skill_briefs=skill_briefs,
        memory_context=prompt_memory_context,
    )
    skill_selection_payload, skill_selection_raw = _call_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_002b_skill_selection",
        system_prompt=skill_selection_system_prompt,
        user_text=skill_selection_user_text,
        image_paths=[],
        parser=_parse_skill_selection_text,
        max_new_tokens=16000,
    )
    state["raw_model_outputs"]["call_002b_skill_selection"] = {"parsed": skill_selection_payload, "raw_response": skill_selection_raw}

    analysis_payload = {
        **grounding_payload,
        **skill_selection_payload,
    }
    analysis_raw = grounding_raw + "\n\n" + skill_selection_raw
    analysis_structure_errors = _analysis_structure_diagnostics(
        "VIDEO_BRIEF:\n" + grounding_raw + "\nMATCHED_SKILL_CANDIDATES:\n" + skill_selection_raw,
        analysis_payload,
    )
    if analysis_structure_errors:
        structure_repair_user_text = (
            skill_selection_user_text.rstrip()
            + "\n\n当前输出的结构不合格，需要只修复第二阶段B的最终输出结构。\n"
            + f"structure_errors: {json.dumps(analysis_structure_errors, ensure_ascii=False)}\n"
            + f"current_parsed_payload: {json.dumps(analysis_payload, ensure_ascii=False)}\n"
            + "只返回第二阶段B要求的字段和 [CUSTOM_SKILL_CANDIDATE] 块，不要输出分析过程，不要输出额外解释。\n"
        )
        analysis_structure_repair_payload, analysis_structure_repair_raw = _call_step(
            model=model,
            state_path=state_path,
            prefix="planner.call_002c_analysis_structure_repair",
            system_prompt=skill_selection_system_prompt.rstrip()
            + "\n\n修复模式：只修复第二阶段B输出结构。必须输出完整字段和块结构，不能输出思维过程。",
            user_text=structure_repair_user_text,
            image_paths=[],
            parser=_parse_skill_selection_text,
            max_new_tokens=16000,
        )
        analysis_payload = _merge_payload_dict(analysis_payload, analysis_structure_repair_payload)
        state["raw_model_outputs"]["call_002c_analysis_structure_repair"] = {
            "parsed": analysis_structure_repair_payload,
            "raw_response": analysis_structure_repair_raw,
            "structure_errors": analysis_structure_errors,
        }

    normalized_custom_skills, custom_skill_diagnostics = _validate_custom_skill_candidates(
        analysis_payload.get("custom_skill_candidates", []),
        selected_family=selected_family,
    )
    analysis_payload["custom_skill_candidates"] = normalized_custom_skills
    if custom_skill_diagnostics:
        custom_skill_repair_user_text = (
            skill_selection_user_text.rstrip()
            + "\n\n当前 custom skill 候选存在缺失字段，需要只修复 CUSTOM_SKILL_CANDIDATES。\n"
            + f"selected_style_family: {selected_family}\n"
            + f"custom_skill_validation_errors: {json.dumps(custom_skill_diagnostics, ensure_ascii=False)}\n"
            + f"current_custom_skill_candidates: {json.dumps(normalized_custom_skills, ensure_ascii=False)}\n"
            + "只返回需要修复后的 CUSTOM_SKILL_CANDIDATES 与 [CUSTOM_SKILL_CANDIDATE] 块；不要重写其它 section。\n"
        )
        custom_skill_repair_payload, custom_skill_repair_raw = _call_step(
            model=model,
            state_path=state_path,
            prefix="planner.call_002c_custom_skill_repair",
            system_prompt=skill_selection_system_prompt.rstrip() + "\n\n修复模式：只修复 CUSTOM_SKILL_CANDIDATES 中缺失的关键字段。",
            user_text=custom_skill_repair_user_text,
            image_paths=image_paths,
            parser=_parse_skill_selection_text,
            max_new_tokens=3072,
        )
        repaired_custom_skills = custom_skill_repair_payload.get("custom_skill_candidates", [])
        merged_custom_skills = _merge_custom_skill_candidates(
            normalized_custom_skills,
            repaired_custom_skills,
            selected_family=selected_family,
        )
        merged_custom_skills, remaining_diagnostics = _validate_custom_skill_candidates(
            merged_custom_skills,
            selected_family=selected_family,
        )
        analysis_payload["custom_skill_candidates"] = merged_custom_skills
        if analysis_payload.get("needs_custom_skill") or merged_custom_skills:
            analysis_payload["needs_custom_skill"] = True
        if merged_custom_skills:
            analysis_payload["matched_skill_candidates"] = _clean_string_list(
                list(analysis_payload.get("matched_skill_candidates", []))
                + [item.get("strategy_id", "") for item in merged_custom_skills]
            )
        state["raw_model_outputs"]["call_002c_custom_skill_repair"] = {
            "parsed": {
                "custom_skill_candidates": merged_custom_skills,
                "remaining_validation_errors": remaining_diagnostics,
            },
            "raw_response": custom_skill_repair_raw,
        }

    state["raw_model_outputs"]["call_002_analysis"] = {"parsed": analysis_payload, "raw_response": analysis_raw}
    _update_analysis_sections(
        state=state,
        analysis_payload=analysis_payload,
        scene_archetype=current_scene_archetype,
    )
    return prompt_memory_context, retrieved_runtime_skills


def _parse_candidate_planning_text(text: str) -> dict:
    strategies = []
    for item in _parse_tagged_blocks(text, "CANDIDATE_STRATEGY"):
        strategies.append(
            {
                "strategy_id": _clean_text(item.get("strategy_id", "")),
                "style_family": _clean_text(item.get("style_family", "")),
                "title": _clean_text(item.get("title", "")),
                "rationale": _clean_text(item.get("rationale", "")),
                "environment": _clean_text(item.get("environment", "")),
                "prop": _clean_text(item.get("prop", "")),
                "lighting": _clean_text(item.get("lighting", "")),
                "costume": _clean_text(item.get("costume", "")),
                "subject_surface": _clean_text(item.get("subject_surface", "")),
                "subject_integration": _clean_text(item.get("subject_integration", "")),
                "atmosphere": _clean_text(item.get("atmosphere", "")),
                "material": _clean_text(item.get("material", "")),
                "pros": _split_items(item.get("pros", "")),
                "cons": _split_items(item.get("cons", "")),
            }
        )
    plan_candidates = []
    for item in _parse_tagged_blocks(text, "PLAN_CANDIDATE"):
        plan_candidates.append(
            {
                "candidate_id": _clean_text(item.get("candidate_id", "")),
                "skill_basis": _split_items(item.get("skill_basis", "")),
                "environment_plan": _clean_text(item.get("environment_plan", "")),
                "prop_plan": _clean_text(item.get("prop_plan", "")),
                "lighting_plan": _clean_text(item.get("lighting_plan", "")),
                "costume_plan": _clean_text(item.get("costume_plan", "")),
                "subject_surface_plan": _clean_text(item.get("subject_surface_plan", "")),
                "integration_plan": _clean_text(item.get("integration_plan", "")),
            }
        )
    candidate_scores = []
    for item in _parse_tagged_blocks(text, "CANDIDATE_SCORE"):
        candidate_scores.append(
            {
                "candidate_id": _clean_text(item.get("candidate_id", "")),
                "prompt_fit": _parse_float(item.get("prompt_fit", "")),
                "video_anchor_fit": _parse_float(item.get("video_anchor_fit", "")),
                "identity_safety": _parse_float(item.get("identity_safety", "")),
                "physical_plausibility": _parse_float(item.get("physical_plausibility", "")),
                "effect_impact": _parse_float(item.get("effect_impact", "")),
                "total": _parse_float(item.get("total", "")),
                "why_selected_or_rejected": _clean_text(item.get("why_selected_or_rejected", "")),
            }
        )
    fields = _parse_section_fields(text, ["SELECTED_CANDIDATE_ID", "SELECTED_STRATEGY", "SELECTION_REASON", "FOCUS_REGIONS"])
    return {
        "candidate_strategies": strategies,
        "plan_candidates": plan_candidates,
        "candidate_scores": candidate_scores,
        "selected_candidate_id": _clean_text(fields.get("SELECTED_CANDIDATE_ID", "")),
        "selected_strategy": _clean_text(fields.get("SELECTED_STRATEGY", "")).strip("\"'"),
        "selection_reason": _clean_text(fields.get("SELECTION_REASON", "")),
        "focus_regions": _split_items(fields.get("FOCUS_REGIONS", "")),
    }


def _parse_candidate_scoring_text(text: str) -> dict:
    candidate_scores = []
    for item in _parse_tagged_blocks(text, "CANDIDATE_SCORE"):
        candidate_scores.append(
            {
                "candidate_id": _clean_text(item.get("candidate_id", "")),
                "prompt_fit": _parse_float(item.get("prompt_fit", "")),
                "video_anchor_fit": _parse_float(item.get("video_anchor_fit", "")),
                "identity_safety": _parse_float(item.get("identity_safety", "")),
                "physical_plausibility": _parse_float(item.get("physical_plausibility", "")),
                "effect_impact": _parse_float(item.get("effect_impact", "")),
                "total": _parse_float(item.get("total", "")),
                "why_selected_or_rejected": _clean_text(item.get("why_selected_or_rejected", "")),
            }
        )
    fields = _parse_section_fields(text, ["SELECTED_CANDIDATE_ID", "SELECTED_STRATEGY", "SELECTION_REASON"])
    return {
        "candidate_scores": candidate_scores,
        "selected_candidate_id": _clean_text(fields.get("SELECTED_CANDIDATE_ID", "")),
        "selected_strategy": _clean_text(fields.get("SELECTED_STRATEGY", "")).strip("\"'"),
        "selection_reason": _clean_text(fields.get("SELECTION_REASON", "")),
    }


def _parse_final_plan_text(text: str) -> dict:
    fields = _parse_section_fields(
        text,
        [
            "STYLE_FAMILY",
            "STYLE_FAMILY_CONFIDENCE",
            "SELECTED_STRATEGY",
            "VISIBLE_PEOPLE",
            "VISIBLE_PROPS",
            "VISIBLE_WORLD_ANCHORS",
            "CARRIER_CANDIDATES",
            "IMMUTABLE_OBJECTS",
            "MUTABLE_REGIONS",
            "GROUNDING_NOTES",
            "HARD_PRESERVE_CONSTRAINTS",
            "SOFT_PRESERVE_CONSTRAINTS",
            "REWRITE_TARGETS",
            "LIGHT_TOUCH_TARGETS",
            "FORBIDDEN_CHANGES",
            "PRIMARY_FX_CARRIERS",
            "SECONDARY_FX_CARRIERS",
            "NORMALIZED_PROMPT",
            "SELF_CHECK_NOTES",
            "CONFLICT_RESOLUTIONS",
            "COMPILED_EDIT_PROMPT",
        ],
    )
    edits = []
    for item in _parse_tagged_blocks(text, "REGION_EDIT"):
        edits.append(
            {
                "edit_id": _clean_text(item.get("edit_id", "")),
                "region": _clean_text(item.get("region", "")),
                "target_objects": _split_items(item.get("target_objects", "")),
                "action": _clean_text(item.get("action", "")),
                "new_description": _clean_text(item.get("new_description", "")),
                "template_basis": _clean_text(item.get("template_basis", "")),
                "video_specific_adaptation": _clean_text(item.get("video_specific_adaptation", "")),
                "preserve": _split_items(item.get("preserve", "")),
                "why_this_carrier": _clean_text(item.get("why_this_carrier", "")),
                "strength": _parse_float(item.get("strength", "")),
            }
        )
    return {
        "style_family": _clean_text(fields.get("STYLE_FAMILY", "")),
        "style_family_confidence": _parse_float(fields.get("STYLE_FAMILY_CONFIDENCE", "")),
        "selected_strategy": _clean_text(fields.get("SELECTED_STRATEGY", "")),
        "grounding_evidence": {
            "visible_people": _split_items(fields.get("VISIBLE_PEOPLE", "")),
            "visible_props": _split_items(fields.get("VISIBLE_PROPS", "")),
            "visible_world_anchors": _split_items(fields.get("VISIBLE_WORLD_ANCHORS", "")),
            "carrier_candidates": _split_items(fields.get("CARRIER_CANDIDATES", "")),
            "immutable_objects": _split_items(fields.get("IMMUTABLE_OBJECTS", "")),
            "mutable_regions": _split_items(fields.get("MUTABLE_REGIONS", "")),
        },
        "grounding_notes": _split_items(fields.get("GROUNDING_NOTES", "")),
        "hard_preserve_constraints": _split_items(fields.get("HARD_PRESERVE_CONSTRAINTS", "")),
        "soft_preserve_constraints": _split_items(fields.get("SOFT_PRESERVE_CONSTRAINTS", "")),
        "rewrite_targets": _split_items(fields.get("REWRITE_TARGETS", "")),
        "light_touch_targets": _split_items(fields.get("LIGHT_TOUCH_TARGETS", "")),
        "forbidden_changes": _split_items(fields.get("FORBIDDEN_CHANGES", "")),
        "primary_fx_carriers": _split_items(fields.get("PRIMARY_FX_CARRIERS", "")),
        "secondary_fx_carriers": _split_items(fields.get("SECONDARY_FX_CARRIERS", "")),
        "normalized_prompt": _clean_text(fields.get("NORMALIZED_PROMPT", "")),
        "region_edits": edits,
        "self_check_notes": _split_items(fields.get("SELF_CHECK_NOTES", "")),
        "conflict_resolutions": _split_items(fields.get("CONFLICT_RESOLUTIONS", "")),
        "compiled_edit_prompt": _clean_text(fields.get("COMPILED_EDIT_PROMPT", "")),
    }



def _flatten_state_payload(state: dict, style_family_options: dict[str, list[dict]]) -> dict:
    theme = dict(state.get("skill_style_family") or {})
    prompt_video_analysis = dict(state.get("prompt_video_analysis") or {})
    candidate_planning = dict(state.get("candidate_planning") or {})
    final_plan = dict(state.get("final_plan") or {})

    selected_family = str(final_plan.get("style_family") or theme.get("selected_style_family") or "").strip()
    matched_ids = [str(item).strip() for item in (theme.get("matched_skill_candidates") or []) if str(item).strip()]
    provided_style_templates = [item for item in style_family_options.get(selected_family, []) if item.get("strategy_id") in matched_ids] or style_family_options.get(selected_family, [])[:3]
    raw_outputs = state.get("raw_model_outputs") or {}

    def _call_meta_for(key: str) -> dict:
        item = raw_outputs.get(key) or {}
        parsed = item.get("parsed") or {}
        if isinstance(parsed, dict):
            meta = parsed.get("_call_meta") or {}
            if isinstance(meta, dict):
                return {
                    "truncation_applied": bool(meta.get("truncation_applied", False)),
                    "pretrim_response_available": bool(meta.get("pretrim_response_available", False)),
                }
        return {
            "truncation_applied": False,
            "pretrim_response_available": False,
        }

    section_diagnostics = {
        "analysis_structure_repair": {
            "invoked": "call_002c_analysis_structure_repair" in raw_outputs,
            "structure_errors": list((raw_outputs.get("call_002c_analysis_structure_repair", {}) or {}).get("structure_errors") or []),
            "remaining_validation_errors": list((((raw_outputs.get("call_002c_analysis_structure_repair", {}) or {}).get("parsed") or {}).get("remaining_validation_errors") or [])),
        },
        "custom_skill_repair": {
            "invoked": "call_002c_custom_skill_repair" in raw_outputs,
            "remaining_validation_errors": list((((raw_outputs.get("call_002c_custom_skill_repair", {}) or {}).get("parsed") or {}).get("remaining_validation_errors") or [])),
        },
        "candidate_generation": _call_meta_for("call_003_candidates"),
        "candidate_scoring": _call_meta_for("call_003b_scoring"),
        "final_plan": _call_meta_for("call_004_final"),
        "final_repair": _call_meta_for("call_004b_final_repair"),
    }

    return {
        "user_intent_summary": theme.get("user_intent_summary", ""),
        "style_family_candidates": theme.get("style_family_candidates", []),
        "selected_style_family": theme.get("selected_style_family", selected_family),
        "style_family_reason": theme.get("style_family_reason", ""),
        "matched_skill_candidates": matched_ids,
        "rejected_skill_candidates": theme.get("rejected_skill_candidates", []),
        "needs_custom_skill": bool(theme.get("needs_custom_skill", False)),
        "custom_skill_seed": theme.get("custom_skill_seed", ""),
        "custom_skill_candidates": prompt_video_analysis.get("custom_skill_candidates", []),
        "scene_archetype": prompt_video_analysis.get("scene_archetype", {}),
        "style_family": selected_family,
        "style_family_confidence": final_plan.get("style_family_confidence", theme.get("style_family_confidence", 0.0)),
        "grounding_evidence": final_plan.get("grounding_evidence", {}),
        "grounding_notes": final_plan.get("grounding_notes", prompt_video_analysis.get("grounding_notes", [])),
        "hard_preserve_constraints": final_plan.get("hard_preserve_constraints", []),
        "soft_preserve_constraints": final_plan.get("soft_preserve_constraints", []),
        "rewrite_targets": final_plan.get("rewrite_targets", []),
        "light_touch_targets": final_plan.get("light_touch_targets", []),
        "forbidden_changes": final_plan.get("forbidden_changes", []),
        "primary_fx_carriers": final_plan.get("primary_fx_carriers", prompt_video_analysis.get("primary_fx_carriers", [])),
        "secondary_fx_carriers": final_plan.get("secondary_fx_carriers", []),
        "plan_candidates": candidate_planning.get("plan_candidates", []),
        "candidate_strategies": candidate_planning.get("candidate_strategies", []),
        "selected_strategy": final_plan.get("selected_strategy", candidate_planning.get("selected_strategy", "")),
        "candidate_scores": candidate_planning.get("candidate_scores", []),
        "selected_candidate_id": candidate_planning.get("selected_candidate_id", ""),
        "selection_reason": candidate_planning.get("selection_reason", ""),
        "materialized_selected_skill": candidate_planning.get("materialized_selected_skill", {}),
        "normalized_prompt": final_plan.get("normalized_prompt", state.get("compiled_edit_prompt", "")),
        "compiled_edit_prompt": final_plan.get("compiled_edit_prompt", state.get("compiled_edit_prompt", "")),
        "focus_regions": candidate_planning.get("focus_regions", []),
        "region_edits": final_plan.get("region_edits", []),
        "self_check_notes": final_plan.get("self_check_notes", []),
        "conflict_resolutions": final_plan.get("conflict_resolutions", []),
        "provided_style_templates": provided_style_templates,
        "section_diagnostics": section_diagnostics,
        "raw_state": state,
    }


def _hydrate_state_from_plan_payload(
    *,
    sample_id: str,
    prompt: str,
    low_video: str,
    memory_context: dict,
    available_style_families: list[str],
    planner_payload: dict,
) -> dict:
    state = planner_state_template()
    state["request"] = {
        "sample_id": sample_id,
        "prompt": prompt,
        "prompt_file": "",
        "low_video": low_video,
        "memory_context": memory_context,
        "created_at": "",
        "available_style_families": available_style_families,
    }
    state["skill_style_family"].update(
        {
            "status": "completed",
            "selected_style_family": planner_payload.get("selected_style_family", planner_payload.get("style_family", "")),
            "style_family_candidates": list(planner_payload.get("style_family_candidates") or []),
            "style_family_confidence": planner_payload.get("style_family_confidence", 0.0),
            "style_family_reason": planner_payload.get("style_family_reason", ""),
            "matched_skill_candidates": list(planner_payload.get("matched_skill_candidates") or []),
            "rejected_skill_candidates": list(planner_payload.get("rejected_skill_candidates") or []),
            "needs_custom_skill": bool(planner_payload.get("needs_custom_skill", False)),
            "custom_skill_seed": planner_payload.get("custom_skill_seed", ""),
            "validation_errors": [],
        }
    )
    grounding = planner_payload.get("grounding_evidence") or {}
    state["prompt_video_analysis"].update(
        {
            "status": "completed",
            "scene_archetype": dict(planner_payload.get("scene_archetype") or {}),
            "video_brief": planner_payload.get("video_brief", ""),
            "visible_people": list((grounding.get("visible_people") or planner_payload.get("visible_people") or [])),
            "visible_props": list((grounding.get("visible_props") or planner_payload.get("visible_props") or [])),
            "visible_world_anchors": list((grounding.get("visible_world_anchors") or planner_payload.get("visible_world_anchors") or [])),
            "primary_fx_carriers": list(planner_payload.get("primary_fx_carriers") or []),
            "grounding_notes": list(planner_payload.get("grounding_notes") or []),
            "custom_skill_candidates": list(planner_payload.get("custom_skill_candidates") or []),
            "validation_errors": [],
        }
    )
    state["candidate_planning"].update(
        {
            "status": "completed",
            "candidate_strategies": list(planner_payload.get("candidate_strategies") or []),
            "plan_candidates": list(planner_payload.get("plan_candidates") or []),
            "candidate_scores": list(planner_payload.get("candidate_score_details") or planner_payload.get("candidate_scores") or []),
            "selected_candidate_id": planner_payload.get("selected_candidate_id", ""),
            "selected_strategy": planner_payload.get("selected_strategy", ""),
            "selection_reason": planner_payload.get("selection_reason", ""),
            "focus_regions": list(planner_payload.get("focus_regions") or []),
            "materialized_selected_skill": dict(planner_payload.get("materialized_selected_skill") or {}),
            "validation_errors": [],
        }
    )
    state["final_plan"].update(
        {
            "status": "completed",
            "style_family": planner_payload.get("style_family", planner_payload.get("selected_style_family", "")),
            "style_family_confidence": planner_payload.get("style_family_confidence", 0.0),
            "selected_strategy": planner_payload.get("selected_strategy", ""),
            "grounding_evidence": dict(grounding),
            "grounding_notes": list(planner_payload.get("grounding_notes") or []),
            "hard_preserve_constraints": list(planner_payload.get("hard_preserve_constraints") or []),
            "soft_preserve_constraints": list(planner_payload.get("soft_preserve_constraints") or []),
            "rewrite_targets": list(planner_payload.get("rewrite_targets") or []),
            "light_touch_targets": list(planner_payload.get("light_touch_targets") or []),
            "forbidden_changes": list(planner_payload.get("forbidden_changes") or []),
            "primary_fx_carriers": list(planner_payload.get("primary_fx_carriers") or []),
            "secondary_fx_carriers": list(planner_payload.get("secondary_fx_carriers") or []),
            "normalized_prompt": planner_payload.get("normalized_prompt", ""),
            "region_edits": list(planner_payload.get("region_edits") or []),
            "self_check_notes": list(planner_payload.get("self_check_notes") or []),
            "conflict_resolutions": list(planner_payload.get("conflict_resolutions") or []),
            "validation_errors": [],
            "compiled_edit_prompt": planner_payload.get("compiled_edit_prompt", planner_payload.get("normalized_prompt", "")),
        }
    )
    state["compiled_edit_prompt"] = planner_payload.get("compiled_edit_prompt", planner_payload.get("normalized_prompt", ""))
    return state


def _finalize_state_to_plan(
    *,
    state: dict,
    registry,
    sample_id: str,
    prompt: str,
    image_paths: list[str],
    state_path: str | Path | None,
) -> VideoPlan:
    state["validation_diagnostics"] = _state_diagnostics(state)
    _write_state(state_path, state)
    flattened = _flatten_state_payload(
        state,
        {family: [skill.to_dict() for skill in registry.get_skills_by_style(family)] for family in _available_style_families(registry)},
    )
    flattened["frame_paths"] = image_paths
    flattened["frame_count"] = len(image_paths)
    flattened["used_low_video_only"] = True
    plan = build_plan(sample_id=sample_id, prompt=prompt, raw_vlm_payload=flattened)
    state["compiled_edit_prompt"] = plan.compiled_edit_prompt or plan.normalized_prompt
    state["validation_diagnostics"] = _state_diagnostics(state)
    _write_state(state_path, state)
    return plan


def _run_candidate_planning(
    *,
    model,
    state: dict,
    state_path: str | Path | None,
    sample_id: str,
    prompt: str,
    image_paths: list[str],
    selected_family: str,
    family_skills: list,
    prompt_memory_context: dict,
    adaptive_full_skills: list[dict],
    retrieved_runtime_skills: list[dict],
    log_label: str,
) -> list[dict]:
    matched_skill_ids = [str(item).strip() for item in (state["skill_style_family"].get("matched_skill_candidates") or []) if str(item).strip()]
    selected_skill_ids = _selected_skill_ids_or_default(state, family_skills)
    custom_skill_candidates = [item for item in (state["prompt_video_analysis"].get("custom_skill_candidates") or []) if isinstance(item, dict)]
    selected_skills = _selected_skills_with_custom(
        family_skills=family_skills,
        matched_skill_ids=matched_skill_ids or selected_skill_ids,
        custom_skill_candidates=custom_skill_candidates,
        selected_family=selected_family,
    )
    selected_skills = _merge_frontloaded_skills(
        _skill_full(selected_skills),
        adaptive_full_skills + retrieved_runtime_skills,
        limit=5,
    )
    selected_skills = _apply_candidate_strategy_bias(selected_skills, prompt_memory_context, limit=5)

    candidate_system_prompt = build_candidate_planning_system_prompt()
    candidate_user_text = build_candidate_planning_user_prompt(
        sample_id=sample_id,
        prompt=prompt,
        video_analysis=state["prompt_video_analysis"],
        full_skills=_skill_full(selected_skills),
        memory_context=prompt_memory_context,
    )
    _planner_log(f"section.candidate_planning.{log_label}.start")
    candidate_payload, candidate_raw = _call_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_003_candidates",
        system_prompt=candidate_system_prompt,
        user_text=candidate_user_text,
        image_paths=image_paths,
        parser=_parse_candidate_planning_text,
        max_new_tokens=80000,
    )
    if not candidate_payload.get("candidate_strategies") or not candidate_payload.get("plan_candidates"):
        candidate_payload = _synthesize_candidate_payload(
            selected_family=selected_family,
            selected_skills=_skill_full(selected_skills),
            video_analysis=state["prompt_video_analysis"],
            prompt=prompt,
        )
    elif not candidate_payload.get("candidate_scores"):
        scoring_payload, scoring_raw = _call_step(
            model=model,
            state_path=state_path,
            prefix="planner.call_003b_scoring",
            system_prompt=build_candidate_scoring_system_prompt(),
            user_text=build_candidate_scoring_user_prompt(
                sample_id=sample_id,
                prompt=prompt,
                video_analysis=state["prompt_video_analysis"],
                candidate_planning=candidate_payload,
                memory_context=prompt_memory_context,
            ),
            image_paths=image_paths,
            parser=_parse_candidate_scoring_text,
            max_new_tokens=8000,
        )
        state["raw_model_outputs"]["call_003b_scoring"] = {"parsed": scoring_payload, "raw_response": scoring_raw}
        if scoring_payload.get("candidate_scores"):
            candidate_payload["candidate_scores"] = scoring_payload.get("candidate_scores", [])
        if scoring_payload.get("selected_candidate_id"):
            candidate_payload["selected_candidate_id"] = scoring_payload.get("selected_candidate_id", "")
        if scoring_payload.get("selected_strategy"):
            candidate_payload["selected_strategy"] = scoring_payload.get("selected_strategy", "")
        if scoring_payload.get("selection_reason"):
            candidate_payload["selection_reason"] = scoring_payload.get("selection_reason", "")
    candidate_repair_payload, candidate_repair_raw = _call_repair_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_003c_candidate_repair",
        section_name="candidate_planning",
        current_payload=candidate_payload,
        required_fields=["candidate_strategies", "plan_candidates", "selection_reason"],
        system_prompt=candidate_system_prompt,
        user_text=candidate_user_text,
        image_paths=image_paths,
        parser=_parse_candidate_planning_text,
        max_new_tokens=4096,
    )
    if candidate_repair_payload:
        candidate_payload = _merge_payload_dict(candidate_payload, candidate_repair_payload)
        state["raw_model_outputs"]["call_003c_candidate_repair"] = {"parsed": candidate_repair_payload, "raw_response": candidate_repair_raw}
    state["raw_model_outputs"]["call_003_candidates"] = {"parsed": candidate_payload, "raw_response": candidate_raw}
    _update_section(
        state,
        "candidate_planning",
        candidate_payload,
        ["candidate_strategies", "plan_candidates", "selection_reason"],
    )
    if state["candidate_planning"].get("candidate_strategies") and state["candidate_planning"].get("plan_candidates") and state["candidate_planning"].get("selection_reason"):
        state["candidate_planning"]["status"] = "completed"
        state["candidate_planning"]["validation_errors"] = []
    _planner_log(f"section.candidate_planning.{log_label}.done")
    _write_state(state_path, state)
    return _skill_full(selected_skills)


def _run_final_plan(
    *,
    model,
    state: dict,
    state_path: str | Path | None,
    sample_id: str,
    prompt: str,
    image_paths: list[str],
    selected_family: str,
    prompt_memory_context: dict,
    selected_skills_for_final: list[dict],
    log_label: str,
) -> None:
    selected_strategy = str(state["candidate_planning"].get("selected_strategy", "")).strip()
    selected_skill = next((skill for skill in selected_skills_for_final if skill.get("strategy_id") == selected_strategy), None)
    if selected_skill is None:
        candidate_strategy = _candidate_strategy_map(state["candidate_planning"]).get(selected_strategy)
        if candidate_strategy is not None:
            selected_skill = _materialize_candidate_strategy_as_skill(
                candidate_strategy,
                selected_family=selected_family,
                fallback_templates=selected_skills_for_final,
                video_analysis=state["prompt_video_analysis"],
            )
            state["candidate_planning"]["materialized_selected_skill"] = dict(selected_skill)
    if selected_skill is None and selected_skills_for_final:
        selected_skill = selected_skills_for_final[0]

    final_system_prompt = build_final_plan_system_prompt()
    final_user_text = build_final_plan_user_prompt(
        sample_id=sample_id,
        prompt=prompt,
        video_analysis=state["prompt_video_analysis"],
        candidate_planning=state["candidate_planning"],
        selected_skill=selected_skill or {},
        memory_context=prompt_memory_context,
    )
    _planner_log(f"section.final_plan.{log_label}.start")
    final_payload, final_raw = _call_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_004_final",
        system_prompt=final_system_prompt,
        user_text=final_user_text,
        image_paths=image_paths,
        parser=_parse_final_plan_text,
        max_new_tokens=8192,
    )
    final_repair_payload, final_repair_raw = _call_repair_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_004b_final_repair",
        section_name="final_plan",
        current_payload=final_payload,
        required_fields=["region_edits", "normalized_prompt"],
        system_prompt=final_system_prompt,
        user_text=final_user_text,
        image_paths=image_paths,
        parser=_parse_final_plan_text,
        max_new_tokens=4096,
    )
    if final_repair_payload:
        final_payload = _merge_payload_dict(final_payload, final_repair_payload)
        state["raw_model_outputs"]["call_004b_final_repair"] = {"parsed": final_repair_payload, "raw_response": final_repair_raw}
    state["raw_model_outputs"]["call_004_final"] = {"parsed": final_payload, "raw_response": final_raw}
    _update_section(
        state,
        "final_plan",
        final_payload,
        ["region_edits", "normalized_prompt"],
    )
    if isinstance(final_payload.get("compiled_edit_prompt"), str) and final_payload.get("compiled_edit_prompt", "").strip():
        state["compiled_edit_prompt"] = final_payload.get("compiled_edit_prompt", "")
    elif isinstance(final_payload.get("normalized_prompt"), str):
        state["compiled_edit_prompt"] = final_payload.get("normalized_prompt", "")
    _planner_log(f"section.final_plan.{log_label}.done")


def replan_video_from_existing(
    *,
    sample_id: str,
    prompt: str,
    low_video: str,
    planner_payload: dict,
    memory_context: dict | None = None,
    memory_store: dict | None = None,
    attempt_adaptation_state: dict | None = None,
    state_path: str | Path | None = None,
    scope: str = "final_only",
    variant: str = "",
) -> VideoPlan:
    memory_context = memory_context or {}
    if attempt_adaptation_state:
        memory_context = dict(memory_context)
        memory_context["attempt_adaptation_state"] = attempt_adaptation_state
    registry = get_skill_registry()
    available_style_families = _available_style_families(registry)
    state = _hydrate_state_from_plan_payload(
        sample_id=sample_id,
        prompt=prompt,
        low_video=low_video,
        memory_context=memory_context,
        available_style_families=available_style_families,
        planner_payload=planner_payload,
    )
    _write_state(state_path, state)
    model = resolve_vision_model()
    image_paths = extract_frames(low_video, max_frames=6)
    selected_family = _selected_family_or_default(state, available_style_families)
    scene_archetype = state.get("prompt_video_analysis", {}).get("scene_archetype", {}) or {}
    archetype_id = _clean_text(scene_archetype.get("archetype_id", "")) or _context_archetype_id(prompt, memory_context)
    family_skills = registry.get_skills_by_style(selected_family)
    adaptive_skill_briefs, adaptive_full_skills = _adaptive_runtime_skills(
        memory_context,
        selected_family=selected_family,
    )
    prompt_memory_context, retrieved_runtime_skills = _build_prompt_memory_context(
        memory_context=memory_context,
        selected_family=selected_family,
        scene_archetype=scene_archetype,
        archetype_id=archetype_id,
        memory_store=memory_store,
        prompt=prompt,
        variant=variant,
        grounding_payload=state.get("prompt_video_analysis", {}) or {},
    )

    if scope == "candidate_and_final":
        selected_skills_for_final = _run_candidate_planning(
            model=model,
            state=state,
            state_path=state_path,
            sample_id=sample_id,
            prompt=prompt,
            image_paths=image_paths,
            selected_family=selected_family,
            family_skills=family_skills,
            prompt_memory_context=prompt_memory_context,
            adaptive_full_skills=adaptive_full_skills,
            retrieved_runtime_skills=retrieved_runtime_skills,
            log_label="rerun",
        )
    else:
        selected_skills_for_final = _merge_frontloaded_skills(
            _skill_full(family_skills),
            adaptive_full_skills
            + retrieved_runtime_skills
            + [item for item in (state["prompt_video_analysis"].get("custom_skill_candidates") or []) if isinstance(item, dict)],
            limit=8,
        )
    _run_final_plan(
        model=model,
        state=state,
        state_path=state_path,
        sample_id=sample_id,
        prompt=prompt,
        image_paths=image_paths,
        selected_family=selected_family,
        prompt_memory_context=prompt_memory_context,
        selected_skills_for_final=selected_skills_for_final,
        log_label=scope,
    )
    return _finalize_state_to_plan(
        state=state,
        registry=registry,
        sample_id=sample_id,
        prompt=prompt,
        image_paths=image_paths,
        state_path=state_path,
    )


def plan_video(
    *,
    sample_id: str,
    prompt: str,
    low_video: str,
    memory_context: dict | None = None,
    memory_store: dict | None = None,
    attempt_adaptation_state: dict | None = None,
    state_path: str | Path | None = None,
    variant: str = "",
) -> VideoPlan:
    memory_context = memory_context or {}
    if attempt_adaptation_state:
        memory_context = dict(memory_context)
        memory_context["attempt_adaptation_state"] = attempt_adaptation_state
    registry = get_skill_registry()
    available_style_families = _available_style_families(registry)
    state = planner_state_template()
    state["request"] = {
        "sample_id": sample_id,
        "prompt": prompt,
        "prompt_file": "",
        "low_video": low_video,
        "memory_context": memory_context,
        "created_at": "",
        "available_style_families": available_style_families,
    }
    _write_state(state_path, state)

    model = resolve_vision_model()
    image_paths = extract_frames(low_video, max_frames=6)

    family_options = _family_options(available_style_families, registry)
    family_system_prompt = build_family_selection_system_prompt()
    family_user_text = build_family_selection_user_prompt(sample_id=sample_id, prompt=prompt, family_options=family_options)
    _planner_log("section.skill_style_family.start")
    family_payload, family_raw = _call_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_001_family",
        system_prompt=family_system_prompt,
        user_text=family_user_text,
        image_paths=[],
        parser=_parse_family_selection_text,
        max_new_tokens=2048,
    )
    family_repair_payload, family_repair_raw = _call_repair_step(
        model=model,
        state_path=state_path,
        prefix="planner.call_001b_family_repair",
        section_name="skill_style_family",
        current_payload=family_payload,
        required_fields=["selected_style_family", "style_family_reason"],
        system_prompt=family_system_prompt,
        user_text=family_user_text,
        image_paths=[],
        parser=_parse_family_selection_text,
        max_new_tokens=512,###改成512
    )
    if family_repair_payload:
        family_payload = _merge_payload_dict(family_payload, family_repair_payload)
        state["raw_model_outputs"]["call_001b_family_repair"] = {"parsed": family_repair_payload, "raw_response": family_repair_raw}
    state["raw_model_outputs"]["call_001_family"] = {"parsed": family_payload, "raw_response": family_raw}
    _update_section(
        state,
        "skill_style_family",
        family_payload,
        ["selected_style_family", "style_family_reason"],
    )
    _planner_log("section.skill_style_family.done")
    _write_state(state_path, state)

    selected_family = _selected_family_or_default(state, available_style_families)
    family_skills = registry.get_skills_by_style(selected_family)
    adaptive_skill_briefs, adaptive_full_skills = _adaptive_runtime_skills(
        memory_context,
        selected_family=selected_family,
    )
    _planner_log("section.prompt_video_analysis.start")
    prompt_memory_context, retrieved_runtime_skills = _analyze_and_select_skills(
        model=model,
        state=state,
        state_path=state_path,
        sample_id=sample_id,
        prompt=prompt,
        image_paths=image_paths,
        selected_family=selected_family,
        family_skills=family_skills,
        memory_context=memory_context,
        memory_store=memory_store,
        variant=variant,
        adaptive_skill_briefs=adaptive_skill_briefs,
    )
    _planner_log("section.prompt_video_analysis.done")
    _write_state(state_path, state)

    selected_skills_for_final = _run_candidate_planning(
        model=model,
        state=state,
        state_path=state_path,
        sample_id=sample_id,
        prompt=prompt,
        image_paths=image_paths,
        selected_family=selected_family,
        family_skills=family_skills,
        prompt_memory_context=prompt_memory_context,
        adaptive_full_skills=adaptive_full_skills,
        retrieved_runtime_skills=retrieved_runtime_skills,
        log_label="main",
    )
    _run_final_plan(
        model=model,
        state=state,
        state_path=state_path,
        sample_id=sample_id,
        prompt=prompt,
        image_paths=image_paths,
        selected_family=selected_family,
        prompt_memory_context=prompt_memory_context,
        selected_skills_for_final=selected_skills_for_final,
        log_label="main",
    )
    return _finalize_state_to_plan(
        state=state,
        registry=registry,
        sample_id=sample_id,
        prompt=prompt,
        image_paths=image_paths,
        state_path=state_path,
    )
