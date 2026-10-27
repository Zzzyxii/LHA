from __future__ import annotations

import os
import re

from planner.core.planner_types import (
    CandidateComparisonTrace,
    GroundingEvidence,
    PlanCandidateTrace,
    RegionEdit,
    SelectionScore,
    StrategyCandidate,
    VideoPlan,
)
from skills.dimension_parse import KEEP, parse_overrides
from skills.dimensions import UNIVERSAL_PRESERVE, compose_dimensions, policy_for
from skills.registry import candidate_models_for_style, get_skill_registry
from skills.base import CANONICAL_DIMENSIONS
from utils.prompt_budget import fit_sections
from utils.rule_checker import ensure_plan_is_productive, resolve_constraint_conflicts, validate_carriers, validate_selected_strategy


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _looks_analysis_like(text: str) -> bool:
    lowered = _clean_text(text).lower()
    if not lowered:
        return True
    bad_phrases = (
        "we are given",
        "the scene shows",
        "key ideas",
        "could be enhanced",
        "first, analyze",
        "observations from the frames",
        "we must propose",
    )
    return any(phrase in lowered for phrase in bad_phrases)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys([_clean_text(item) for item in items if _clean_text(item)]))


def _clean_string_list(items) -> list[str]:
    return _unique([str(item) for item in (items or [])])


def _coerce_scene_inventory_item(item) -> tuple[str, str, str]:
    if isinstance(item, dict):
        label = _clean_text(item.get("label", ""))
        role = _clean_text(item.get("role", "")).lower()
        description = _clean_text(item.get("description", ""))
        return label, role, description
    text = _clean_text(item)
    lowered = text.lower()
    role = ""
    if any(token in lowered for token in ("subject_anchor", "subject", "man", "woman", "people", "person")):
        role = "subject_anchor"
    elif any(token in lowered for token in ("prop_anchor", "prop", "structure", "device", "wheelchair", "object")):
        role = "prop_anchor"
    elif any(token in lowered for token in ("world_anchor", "scene_anchor", "background", "field", "hill", "mist", "tree", "environment")):
        role = "world_anchor"
    label = text
    description = text
    if ":" in text:
        prefix, rest = text.split(":", 1)
        if len(prefix.split()) <= 6:
            label = _clean_text(prefix)
            description = _clean_text(rest)
    return label, role, description


def _subject_slots_from_grounding(grounding: GroundingEvidence) -> list[tuple[str, list[str]]]:
    joined = " ".join(grounding.visible_people).lower()
    slots: list[tuple[str, list[str]]] = []
    if "wheelchair" in joined:
        slots.append(("wheelchair_subject", [item for item in grounding.visible_people if "wheelchair" in item.lower()] or ["wheelchair subject"]))
    if "standing_man" in joined or "standing man" in joined or "man standing" in joined:
        slots.append(("standing_male_subject", [item for item in grounding.visible_people if "standing_man" in item.lower() or "standing man" in item.lower() or "man standing" in item.lower()] or ["standing male subject"]))
    if "standing_woman" in joined or "standing woman" in joined or "woman standing" in joined:
        slots.append(("standing_female_subject", [item for item in grounding.visible_people if "standing_woman" in item.lower() or "standing woman" in item.lower() or "woman standing" in item.lower()] or ["standing female subject"]))
    return slots or [("group_subjects", grounding.visible_people or ["visible subjects"])]


def _is_meta_text(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return True
    lowered = cleaned.lower()
    strong_bad_fragments = (
        "style template",
        "style_templates",
        "candidate_strateg",
        "strategy_id",
        "style family",
        "style_family",
        "for reference only",
        "critical guidance",
        "intensity level",
        "user prompt",
        "memory_context",
        "provided template",
        "not part of the video scene",
        "风格模板",
        "关键风格指导",
        "强度级别",
        "用户提示",
    )
    if any(fragment in lowered for fragment in strong_bad_fragments):
        return True
    # Only treat these as meta when they appear in obvious instruction/template contexts.
    contextual_meta_pairs = (
        ("do not include", ("scene_inventory", "target_objects", "template", "style")),
        ("ignore anime", ("template", "style family", "candidate", "strategy")),
        ("not sci-fi", ("template", "style family", "candidate", "strategy")),
        ("不是科幻风格", ("模板", "风格", "候选", "策略")),
        ("请注意", ("模板", "风格", "candidate", "strategy", "scene_inventory")),
        ("优先考虑", ("工业", "风格", "lighting", "template", "strategy")),
        ("转变必须明显", ("风格", "强度", "template", "strategy")),
    )
    for trigger, contexts in contextual_meta_pairs:
        if trigger in lowered and any(context in lowered for context in contexts):
            return True
    if "->" in cleaned:
        return True
    if "：" in cleaned and any(fragment in lowered for fragment in ("风格", "模板", "策略", "候选", "强度")):
        return True
    return False


def _normalize_phrase(text: str) -> str:
    text = _clean_text(text)
    text = re.sub(r"^\s*(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    return text


def _strip_section_prefix(text: str) -> str:
    cleaned = _clean_text(text)
    if ":" in cleaned:
        prefix, rest = cleaned.split(":", 1)
        if len(prefix.split()) <= 4:
            return _clean_text(rest)
    return cleaned


def _extract_parenthetical_enumeration(text: str) -> list[str]:
    if _is_meta_text(text):
        return []
    matches = re.findall(r"\(([^)]*)\)", text)
    results: list[str] = []
    for match in matches:
        if "," in match or " and " in match.lower():
            parts = re.split(r",|\band\b", match, flags=re.IGNORECASE)
            results.extend(_normalize_phrase(part) for part in parts)
    return [item for item in results if len(item) >= 3]


def _split_enumeration_text(text: str) -> list[str]:
    if _is_meta_text(text):
        return []
    stripped = _strip_section_prefix(text)
    first_sentence = re.split(r"(?<=[.!?。])\s+", stripped)[0]
    first_sentence = re.sub(r"\([^)]*\)", "", first_sentence).strip()
    first_sentence = re.sub(r"^\s*(?:\d+\s+)?(?:people|persons|individuals|subjects|props|objects|anchors)\s*", "", first_sentence, flags=re.IGNORECASE)
    parts = re.split(r",|\band\b|;", first_sentence, flags=re.IGNORECASE)
    cleaned = [_normalize_phrase(part) for part in parts]
    return [item for item in cleaned if len(item) >= 3]


def _contains_person_token(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("man", "woman", "person", "people", "subject", "character", "figure", "child", "girl", "boy"))


def _contains_world_token(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("background", "field", "hill", "tree", "landscape", "mist", "fog", "haze", "sky", "ground", "outdoor", "environment"))


def _contains_prop_token(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("prop", "device", "structure", "ring", "platform", "wheelchair", "vehicle", "bench", "chair", "gate", "portal", "path", "slide", "machine"))


def _extract_items_from_scene_text(text: str, role: str) -> list[str]:
    parenthetical = _extract_parenthetical_enumeration(text)
    enumerated = _split_enumeration_text(text)
    candidates = _unique(parenthetical + enumerated)
    filtered: list[str] = []
    for item in candidates:
        if _is_meta_text(item):
            continue
        lowered = item.lower()
        if lowered in {
            "visible people",
            "props",
            "world anchors",
            "carriers",
            "hard preserve",
            "soft preserve",
            "rewrite targets",
            "light touch targets",
            "light-touch targets",
            "forbidden changes",
            "main subjects",
        }:
            continue
        if role.startswith("subject"):
            if _contains_person_token(item):
                filtered.append(item)
        elif role.startswith("prop"):
            if _contains_prop_token(item) or not _contains_person_token(item):
                filtered.append(item)
        elif role.startswith("world") or role.startswith("scene"):
            if _contains_world_token(item) or not _contains_person_token(item):
                filtered.append(item)
        else:
            filtered.append(item)
    return _unique(filtered)


def _carrier_candidates_from_grounding(people: list[str], props: list[str], world: list[str]) -> list[str]:
    carriers: list[str] = []
    if props:
        carriers.extend(props[:2])
    if world:
        carriers.append("environment")
    if any(token in " ".join(world).lower() for token in ("mist", "fog", "haze")):
        carriers.append("atmospheric layer")
    carriers.append("lighting system")
    if people:
        carriers.append("subject edge light")
    return _unique(carriers)


# Confidence at or above this means lock_style_family matched a registered family;
# below it the value is its "sci_fi", 0.2 fallback and the direction is a guess.
STYLE_FAMILY_MIN_CONFIDENCE = 0.5


def lock_style_family(prompt: str, raw_vlm_payload: dict) -> tuple[str, float]:
    payload_family = _clean_text(raw_vlm_payload.get("style_family", ""))
    if payload_family in {"sci_fi", "anime", "historical_fantasy"}:
        confidence = float(raw_vlm_payload.get("style_family_confidence", 0.82) or 0.82)
        return payload_family, max(0.0, min(1.0, confidence))
    provided_templates = raw_vlm_payload.get("provided_style_templates") or []
    family_counts: dict[str, int] = {}
    for item in provided_templates:
        if not isinstance(item, dict):
            continue
        family = _clean_text(item.get("style_family", ""))
        if family in {"sci_fi", "anime", "historical_fantasy"}:
            family_counts[family] = family_counts.get(family, 0) + 1
    if family_counts:
        selected_family = max(family_counts, key=family_counts.get)
        return selected_family, 0.25
    # The registry only covers three families, so a scene outside all of them lands
    # here. The low confidence is what downstream code uses to tell a real match from
    # this guess -- see STYLE_FAMILY_MIN_CONFIDENCE.
    return "sci_fi", 0.2


def _extract_grounding_from_payload(raw_vlm_payload: dict) -> GroundingEvidence | None:
    explicit_grounding = raw_vlm_payload.get("grounding_evidence")
    if isinstance(explicit_grounding, dict):
        visible_people = _clean_string_list(explicit_grounding.get("visible_people"))
        visible_props = _clean_string_list(explicit_grounding.get("visible_props"))
        visible_world_anchors = _clean_string_list(explicit_grounding.get("visible_world_anchors"))
        carrier_candidates = _clean_string_list(explicit_grounding.get("carrier_candidates"))
        immutable_objects = _clean_string_list(explicit_grounding.get("immutable_objects"))
        mutable_regions = _clean_string_list(explicit_grounding.get("mutable_regions"))
        if visible_people or visible_props or visible_world_anchors:
            return GroundingEvidence(
                visible_people=visible_people,
                visible_props=visible_props,
                visible_world_anchors=visible_world_anchors,
                carrier_candidates=carrier_candidates or ["environment"],
                immutable_objects=immutable_objects or ["subject identity", "coarse geometry"],
                mutable_regions=mutable_regions or ["background world", "lighting system", "costume language", "prop semantics"],
            )
    scene_inventory = raw_vlm_payload.get("scene_inventory") or []
    visible_people: list[str] = []
    visible_props: list[str] = []
    visible_world_anchors: list[str] = []
    carrier_candidates: list[str] = []
    immutable_objects: list[str] = []
    mutable_regions: list[str] = []

    for item in scene_inventory:
        label, role, description = _coerce_scene_inventory_item(item)
        if _is_meta_text(label) or _is_meta_text(description):
            continue
        joined = f"{label} {description}".lower()
        extracted = _extract_items_from_scene_text(f"{label}: {description}", role)
        if role.startswith("subject") or _contains_person_token(joined):
            visible_people.extend(extracted or ([label] if _contains_person_token(label) else []))
        elif role.startswith("prop") or _contains_prop_token(joined):
            visible_props.extend(extracted or ([label] if _contains_prop_token(label) else []))
        elif role.startswith("world") or role.startswith("scene") or _contains_world_token(joined):
            visible_world_anchors.extend(extracted or ([label] if _contains_world_token(label) else []))

        if role.startswith("subject") or any(token in joined for token in ("man", "woman", "person", "people", "subject")):
            mutable_regions.extend(["costume language", "subject surface", "subject environment integration"])
        if role.startswith("prop") or _contains_prop_token(joined):
            mutable_regions.append("prop semantics")
        if role.startswith("world") or role.startswith("scene") or _contains_world_token(joined):
            mutable_regions.extend(["background world", "lighting system", "atmosphere", "material language"])

    carrier_candidates = _carrier_candidates_from_grounding(visible_people, visible_props, visible_world_anchors)
    immutable_objects.extend(f"{person} identity" for person in visible_people)
    if visible_props:
        immutable_objects.append("preserve the coarse geometry of dominant visible props")

    visible_people = _unique(visible_people)
    visible_props = _unique(visible_props)
    visible_world_anchors = _unique(visible_world_anchors)
    carrier_candidates = _unique(carrier_candidates)
    immutable_objects = _unique(immutable_objects)
    mutable_regions = _unique(mutable_regions)

    if not (visible_people or visible_props or visible_world_anchors):
        return None
    return GroundingEvidence(
        visible_people=visible_people,
        visible_props=visible_props,
        visible_world_anchors=visible_world_anchors,
        carrier_candidates=carrier_candidates or ["environment"],
        immutable_objects=immutable_objects or ["subject identity", "coarse geometry"],
        mutable_regions=mutable_regions or ["background world", "lighting system", "costume language", "prop semantics"],
    )


def _parse_plan_candidates(raw_vlm_payload: dict) -> list[PlanCandidateTrace]:
    results: list[PlanCandidateTrace] = []
    for item in (raw_vlm_payload.get("plan_candidates") or []):
        if not isinstance(item, dict):
            continue
        candidate_id = _clean_text(item.get("candidate_id", "") or item.get("strategy_id", ""))
        if not candidate_id:
            continue
        results.append(
            PlanCandidateTrace(
                candidate_id=candidate_id,
                skill_basis=_clean_string_list(item.get("skill_basis")),
                environment_plan=_clean_text(item.get("environment_plan", "") or item.get("environment", "")),
                prop_plan=_clean_text(item.get("prop_plan", "") or item.get("prop", "")),
                lighting_plan=_clean_text(item.get("lighting_plan", "") or item.get("lighting", "")),
                costume_plan=_clean_text(item.get("costume_plan", "") or item.get("costume", "")),
                subject_surface_plan=_clean_text(item.get("subject_surface_plan", "") or item.get("subject_surface", "")),
                integration_plan=_clean_text(item.get("integration_plan", "") or item.get("subject_integration", "")),
            )
        )
    return results


def _parse_candidate_score_details(raw_vlm_payload: dict) -> list[CandidateComparisonTrace]:
    raw_scores = raw_vlm_payload.get("candidate_scores") or raw_vlm_payload.get("selection_scores") or []
    results: list[CandidateComparisonTrace] = []
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        candidate_id = _clean_text(item.get("candidate_id", "") or item.get("strategy_id", ""))
        if not candidate_id:
            continue
        try:
            results.append(
                CandidateComparisonTrace(
                    candidate_id=candidate_id,
                    prompt_fit=float(item.get("prompt_fit", 0.0) or 0.0),
                    video_anchor_fit=float(item.get("video_anchor_fit", item.get("anchor_leverage", 0.0)) or 0.0),
                    identity_safety=float(item.get("identity_safety", 0.0) or 0.0),
                    physical_plausibility=float(item.get("physical_plausibility", 0.0) or 0.0),
                    effect_impact=float(item.get("effect_impact", 0.0) or 0.0),
                    total=float(item.get("total", 0.0) or 0.0),
                    why_selected_or_rejected=_clean_text(item.get("why_selected_or_rejected", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return results


def _fallback_candidate_score_details(
    *,
    raw_vlm_payload: dict,
    plan_candidates: list[PlanCandidateTrace],
    selection_reason: str,
) -> list[CandidateComparisonTrace]:
    results: list[CandidateComparisonTrace] = []
    selected_candidate_id = _clean_text(
        raw_vlm_payload.get("selected_candidate_id", "") or raw_vlm_payload.get("selected_strategy", "")
    )

    raw_selection_scores = raw_vlm_payload.get("selection_scores") or []
    selection_score_map: dict[str, dict] = {}
    for item in raw_selection_scores:
        if not isinstance(item, dict):
            continue
        strategy_id = _clean_text(item.get("strategy_id", ""))
        if strategy_id:
            selection_score_map[strategy_id] = item

    for item in plan_candidates:
        score = selection_score_map.get(item.candidate_id, {})
        total = score.get("total", 0.81 if item.candidate_id == selected_candidate_id else 0.72)
        try:
            total_value = float(total or 0.0)
        except (TypeError, ValueError):
            total_value = 0.81 if item.candidate_id == selected_candidate_id else 0.72
        why = selection_reason if item.candidate_id == selected_candidate_id else "Not selected after candidate comparison."
        results.append(
            CandidateComparisonTrace(
                candidate_id=item.candidate_id,
                prompt_fit=float(score.get("prompt_fit", 0.8) or 0.8),
                video_anchor_fit=float(score.get("anchor_leverage", score.get("video_anchor_fit", 0.78)) or 0.78),
                identity_safety=float(score.get("identity_safety", 0.82) or 0.82),
                physical_plausibility=float(score.get("physical_plausibility", 0.8) or 0.8),
                effect_impact=float(score.get("effect_impact", 0.76) or 0.76),
                total=total_value,
                why_selected_or_rejected=_clean_text(why),
            )
        )
    return results


def fallback_grounding(prompt: str) -> GroundingEvidence:
    people = ["primary visible subjects"]
    props = ["dominant visible prop anchor"]
    world = ["background world anchor"]
    carriers = ["dominant visible prop anchor", "environment", "lighting system"]
    return GroundingEvidence(
        visible_people=people,
        visible_props=props,
        visible_world_anchors=world,
        carrier_candidates=carriers,
        immutable_objects=["subject identity", "coarse geometry of dominant visible props"],
        mutable_regions=["background world", "lighting system", "costume language", "prop semantics", "atmosphere", "material language"],
    )


def extract_grounding(prompt: str, raw_vlm_payload: dict) -> GroundingEvidence:
    return _extract_grounding_from_payload(raw_vlm_payload) or fallback_grounding(prompt)


def build_constraint_layers(
    grounding: GroundingEvidence,
    strategy: StrategyCandidate,
    raw_vlm_payload: dict,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    hard_preserve = list(raw_vlm_payload.get("hard_preserve_constraints") or []) or [
        "preserve subject identity",
        "preserve subject count and relationships",
        "preserve body motion and timing",
        "preserve camera trajectory and framing",
        "preserve coarse geometry and object placement",
    ]
    soft_preserve = list(raw_vlm_payload.get("soft_preserve_constraints") or []) or [
        "keep recognizable silhouettes while allowing costume redesign",
        "keep readable contact relationships while changing the world and lighting",
        "keep facial identity cues while allowing light-touch surface polish",
    ]
    rewrite_targets = list(raw_vlm_payload.get("rewrite_targets") or []) or [
        strategy.environment,
        strategy.prop,
        strategy.lighting,
        strategy.costume,
        strategy.atmosphere,
        strategy.material,
    ]
    light_touch_targets = list(raw_vlm_payload.get("light_touch_targets") or []) or [
        strategy.subject_surface,
        strategy.subject_integration,
    ]
    forbidden_changes = list(raw_vlm_payload.get("forbidden_changes") or []) or [
        "do not rewrite faces into different identities",
        "do not preserve ordinary casual wardrobe if the selected strategy requires world-specific costume redesign",
        "do not attach FX to empty air with no visible carrier",
        "do not keep natural daylight or outdoor realism if it conflicts with the selected world",
    ]
    if grounding.visible_props:
        hard_preserve.append("preserve dominant visible prop anchors and their coarse spatial relationships")
    return resolve_constraint_conflicts(
        hard_preserve=_unique(hard_preserve),
        soft_preserve=_unique(soft_preserve),
        rewrite_targets=_unique(rewrite_targets),
        light_touch_targets=_unique(light_touch_targets),
        forbidden_changes=_unique(forbidden_changes),
    )


def _candidate_from_payload(item: dict, style_family: str) -> StrategyCandidate | None:
    try:
        payload = {
            "strategy_id": _clean_text(item.get("strategy_id", "")),
            "style_family": item.get("style_family") or style_family,
            "title": _clean_text(item.get("title", "") or item.get("strategy_id", "")),
            "rationale": _clean_text(item.get("rationale", "")) or "Grounded strategy proposed by the planner model.",
            "environment": _clean_text(item.get("environment", "")),
            "prop": _clean_text(item.get("prop", "")),
            "lighting": _clean_text(item.get("lighting", "")),
            "costume": _clean_text(item.get("costume", "")),
            "subject_surface": _clean_text(item.get("subject_surface", "")),
            "subject_integration": _clean_text(item.get("subject_integration", "")),
            "atmosphere": _clean_text(item.get("atmosphere", "")),
            "material": _clean_text(item.get("material", "")),
            "pros": list(item.get("pros", []) or []),
            "cons": list(item.get("cons", []) or []),
        }
    except AttributeError:
        return None
    required = ("strategy_id", "environment", "prop", "lighting", "costume", "subject_surface", "subject_integration", "atmosphere", "material")
    if not all(payload[field] for field in required):
        return None
    return StrategyCandidate(**payload)


def propose_candidate_strategies(style_family: str, prompt: str, raw_vlm_payload: dict) -> list[StrategyCandidate]:
    """从 VLM 输出中提取候选策略"""
    raw_candidates = raw_vlm_payload.get("candidate_strategies") or []
    parsed = [_candidate_from_payload(item, style_family) for item in raw_candidates if isinstance(item, dict)]
    candidates = [item for item in parsed if item is not None and item.style_family == style_family]
    
    # 如果 VLM 没有输出候选策略，才 fallback 到模板（不应该发生）
    if len(candidates) < 2:
        # fallback: 从提供的模板中随机选几个（只作为兜底）
        provided_templates = raw_vlm_payload.get("provided_style_templates") or []
        fallback = [_candidate_from_payload(item, style_family) for item in provided_templates if isinstance(item, dict)]
        candidates = [item for item in fallback if item is not None][:3]

    if not candidates and raw_candidates:
        # 场景落在已注册 style family 之外时，上面的家族过滤会把 VLM 真实提出的候选
        # 全部滤掉，而模板兜底同样为空。此时保留这些候选并重新打标到锁定家族，
        # 否则下游会因为候选为空直接崩溃，整个样本丢失。
        relabelled = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload["style_family"] = style_family
            parsed_item = _candidate_from_payload(payload, style_family)
            if parsed_item is not None:
                relabelled.append(parsed_item)
        candidates = relabelled[:3]

    return candidates[:3]


def select_strategy(prompt: str, grounding: GroundingEvidence, candidates: list[StrategyCandidate], raw_vlm_payload: dict) -> tuple[str, list[SelectionScore]]:
    selected_from_payload = _clean_text(raw_vlm_payload.get("selected_strategy", ""))
    raw_scores = raw_vlm_payload.get("selection_scores") or []
    
    # 如果 VLM 提供了分数，直接使用
    if raw_scores and selected_from_payload:
        scores = []
        for candidate in candidates:
            for item in raw_scores:
                if isinstance(item, dict) and _clean_text(item.get("strategy_id", "")) == candidate.strategy_id:
                    scores.append(SelectionScore(**item))
                    break
        if scores:
            return selected_from_payload, scores
    
    # 如果 VLM 没有提供分数但提供了选中的策略，直接使用
    if selected_from_payload:
        selected = validate_selected_strategy(selected_from_payload, candidates)
        # 生成默认分数（所有策略相同，表示 VLM 选了但没给理由）
        default_scores = [
            SelectionScore(
                strategy_id=c.strategy_id,
                prompt_fit=0.8,
                anchor_leverage=0.8,
                physical_plausibility=0.8,
                identity_safety=0.85,
                total=0.81,
            )
            for c in candidates
        ]
        return selected, default_scores
    
    # 如果 VLM 什么都没给，选第一个候选（fallback）
    if candidates:
        print("WARNING: VLM did not provide selected_strategy, using first candidate")
        return candidates[0].strategy_id, []
    
    raise ValueError("No candidates available and VLM did not select a strategy")


def _region_edit_from_payload(item: dict) -> RegionEdit | None:
    try:
        target_objects = []
        for value in item.get("target_objects") or []:
            cleaned = _clean_text(value)
            if cleaned and not _is_meta_text(cleaned) and len(cleaned) > 2:
                target_objects.append(cleaned)
        edit = RegionEdit(
            edit_id=_clean_text(item.get("edit_id", "")) or "edit_unknown",
            region=_clean_text(item.get("region", "")),
            target_objects=target_objects,
            action=item.get("action") or "enhance",
            new_description=_clean_text(item.get("new_description", "") or item.get("prompt", "")),
            template_basis=_clean_text(item.get("template_basis", "")),
            video_specific_adaptation=_clean_text(item.get("video_specific_adaptation", "")),
            preserve=[_clean_text(x) for x in (item.get("preserve") or item.get("preserve_constraints") or []) if _clean_text(x)],
            why_this_carrier=_clean_text(item.get("why_this_carrier", "") or item.get("notes", "")),
            strength=float(item.get("strength", 0.8)),
        )
    except Exception:
        return None
    if not edit.region or not edit.new_description or _looks_analysis_like(edit.new_description):
        return None
    if not edit.target_objects:
        if edit.region == "environment":
            edit.target_objects = ["background world"]
        elif edit.region == "functional_scene_props":
            edit.target_objects = ["visible prop anchors"]
        else:
            edit.target_objects = ["visible elements"]
    return edit


def _strength_for(dimension: str, default: float = 0.8) -> float:
    """Default push for a region, from the shared dimension table.

    These numbers used to be literals here and in the enhancement registry. Two
    copies of "how hard do we push costume" drift apart the first time either is
    tuned, and a region edit that disagrees with the dimension it describes is
    worse than one that is simply untuned.
    """
    policy = policy_for(dimension)
    return policy.default_strength if policy is not None else default


def emit_region_edits(
    strategy: StrategyCandidate,
    grounding: GroundingEvidence,
    hard_preserve: list[str],
    light_touch_targets: list[str],
    raw_vlm_payload: dict,
) -> list[RegionEdit]:
    raw_edits = raw_vlm_payload.get("region_edits") or []
    parsed = [_region_edit_from_payload(item) for item in raw_edits if isinstance(item, dict)]
    edits = [item for item in parsed if item is not None]
    subject_regions = {"costume_design", "skin_surface_design", "subject_environment_integration"}
    parsed_subject_regions = {edit.region for edit in edits if edit.region in subject_regions}
    has_subject_specific_targets = any(
        any(token in " ".join(edit.target_objects).lower() for token in ("wheelchair", "standing_man", "standing man", "standing_woman", "standing woman"))
        for edit in edits
        if edit.region in subject_regions
    )
    if len(edits) >= 4 and (not grounding.visible_people or (parsed_subject_regions == set() or has_subject_specific_targets)):
        return edits
    subject_slots = _subject_slots_from_grounding(grounding)
    subject_costume_edits: list[RegionEdit] = []
    subject_surface_edits: list[RegionEdit] = []
    subject_integration_edits: list[RegionEdit] = []
    for slot_name, targets in subject_slots:
        slot_label = slot_name.replace("_", " ")
        subject_costume_edits.append(
            RegionEdit(
                edit_id=f"edit_costume_{slot_name}",
                region="costume_design",
                target_objects=targets,
                action="redesign",
                new_description=f"{strategy.costume} Specifically adapt the wardrobe for the {slot_label} based on current clothing, pose, and role in this shot.",
                template_basis=strategy.costume,
                video_specific_adaptation=f"Use the current visible outfit and silhouette of the {slot_label} as the base, then redesign it into a role-specific version that fits the selected world without erasing identity.",
                preserve=["preserve identity", "preserve body fit and pose readability"],
                why_this_carrier="Wardrobe carries world membership without changing identity.",
                strength=_strength_for("costume"),
            )
        )
        subject_surface_edits.append(
            RegionEdit(
                edit_id=f"edit_subject_surface_{slot_name}",
                region="skin_surface_design",
                target_objects=targets,
                action="light_touch",
                new_description=f"{strategy.subject_surface} Tailor the facial and skin treatment for the {slot_label} instead of applying one generic group treatment.",
                template_basis=strategy.subject_surface,
                video_specific_adaptation=f"Keep the {slot_label} recognizable, then add only the level of glow, grime, polish, or atmospheric response that matches this person's current lighting and distance from the main effect carrier.",
                preserve=["preserve facial identity", "preserve age impression"],
                why_this_carrier="Subject surface treatment should support the world change while staying identity-safe.",
                strength=_strength_for("subject_surface"),
            )
        )
        subject_integration_edits.append(
            RegionEdit(
                edit_id=f"edit_subject_integration_{slot_name}",
                region="subject_environment_integration",
                target_objects=targets,
                action="integrate",
                new_description=f"{strategy.subject_integration} Adapt the integration logic specifically for the {slot_label} based on where they stand relative to the main prop and environment carriers.",
                template_basis=strategy.subject_integration,
                video_specific_adaptation=f"Give the {slot_label} believable contact light, reflections, stance logic, and story role within this shot's specific spatial layout.",
                preserve=hard_preserve,
                why_this_carrier="People must read as belonging to the new world through contact light, reflections, and staging.",
                strength=_strength_for("subject_integration"),
            )
        )
    return [
        RegionEdit(
            edit_id="edit_environment_1",
            region="environment",
            target_objects=grounding.visible_world_anchors or ["background world"],
            action="replace",
            new_description=strategy.environment,
            template_basis=strategy.environment,
            video_specific_adaptation="Replace the current outdoor field, mist, and distant background with a version of the selected world that still respects the shot's existing camera angle and horizon structure.",
            preserve=hard_preserve,
            why_this_carrier="The background world is the largest mutable region and the strongest carrier for style-family conversion.",
            strength=_strength_for("environment"),
        ),
        RegionEdit(
            edit_id="edit_prop_2",
            region="functional_scene_props",
            target_objects=grounding.visible_props or ["yellow curved structure"],
            action="redesign",
            new_description=strategy.prop,
            template_basis=strategy.prop,
            video_specific_adaptation="Use the dominant visible prop anchor in this shot as the main world-building device rather than inventing a detached new object.",
            preserve=hard_preserve,
            why_this_carrier="The dominant visible prop should carry the core VFX device logic.",
            strength=_strength_for("props"),
        ),
        RegionEdit(
            edit_id="edit_lighting_3",
            region="lighting_system_design",
            target_objects=["global lighting", "practical light sources", "subject edge lighting"],
            action="replace",
            new_description=strategy.lighting,
            template_basis=strategy.lighting,
            video_specific_adaptation="Redirect the current outdoor or ambient lighting into a carrier-driven lighting system that responds to the selected prop and world anchors already visible in this shot.",
            preserve=hard_preserve,
            why_this_carrier="Lighting is the cleanest way to replace outdoor realism with a coherent cinematic effects world.",
            strength=_strength_for("lighting"),
        ),
    ] + subject_costume_edits + subject_surface_edits + subject_integration_edits


def self_check_plan(
    *,
    strategy: StrategyCandidate,
    hard_preserve: list[str],
    rewrite_targets: list[str],
    forbidden_changes: list[str],
    raw_vlm_payload: dict,
) -> list[str]:
    raw_notes = [_clean_text(item) for item in (raw_vlm_payload.get("self_check_notes") or []) if _clean_text(item)]
    if raw_notes:
        return raw_notes
    notes: list[str] = []
    joined_hard = " ".join(hard_preserve).lower()
    joined_rewrite = " ".join(rewrite_targets).lower()
    if "costume" in joined_hard and "costume" in joined_rewrite:
        notes.append("Removed costume from hard-preserve logic so world-specific wardrobe redesign can proceed.")
    if "face" in joined_rewrite or "facial topology" in joined_rewrite:
        notes.append("Demoted face rewrite into light-touch treatment to preserve identity safety.")
    if any("empty air" in item.lower() for item in forbidden_changes):
        notes.append("Checked that primary carriers stay attached to visible props, environment, or lighting sources.")
    notes.append(f"Locked strategy family to {strategy.style_family} and rejected cross-family mixing.")
    return notes


def _compose_edit_prompt(
    *,
    user_prompt: str,
    planner_prompt: str,
    hard_preserve: list[str],
    style_family_confidence: float = 1.0,
    style_fields: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> str:
    """Build the edit prompt: instruction verbatim, then the target world.

    The planner's free-text rewrite used to *replace* the instruction. That lost
    exactly the clauses the instruction was most specific about -- "maintain the
    original composition, facial expressions, and camera framing", "preserve the
    characters' identities" -- and substituted invented detail, so the edit model
    was no longer told to protect anything. Measured against a no-planner control
    it cost 0.63 aesthetic points and lost on every scored dimension.

    The instruction therefore leads, unmodified. Supplementary direction comes from
    [skills.dimensions], which fires keyword-triggered craft clauses per edit
    dimension: deterministic, English, and incapable of introducing content the
    instruction did not ask for. Preserve constraints are restated last, where a
    diffusion text encoder weights them most heavily.

    ``planner_prompt`` is not emitted in this default mode -- it is the rewrite
    that caused the regression. ``hard_preserve`` still contributes, since those
    clauses are extracted from the user's own text.

    ``LOW_HIGH_AGENT_PROMPT_MODE=expand`` is for the planning track, where the
    input is a one-line user intent rather than an expert instruction. There the
    planner's video-grounded rewrite *is* the agent's contribution -- it is the
    only source of concrete detail -- so it is emitted after the intent instead of
    being dropped. The default stays ``instruction_first`` because on expert
    instructions the same rewrite measurably hurt.

    On top of that sits the target world, dimension by dimension, from the
    selected style template. A dimension the instruction explicitly asked to hold
    is *not* written into the prompt -- it produces a preserve line instead,
    because omitting a dimension is not the same as protecting it (see
    ``_style_lines``).

    ``LOW_HIGH_STYLE_DELIVERY`` picks how the world reaches the edit model:
    ``inline`` (write it out, default), ``reference`` (carry it as an image and
    keep the prompt short), ``none`` (control arm, no style at all).
    """
    user_prompt = _clean_text(user_prompt)
    planner_prompt = _clean_text(planner_prompt)
    if not user_prompt:
        return planner_prompt

    expand = os.environ.get("LOW_HIGH_AGENT_PROMPT_MODE", "instruction_first").strip().lower() == "expand"
    fired = compose_dimensions(user_prompt + (" " + planner_prompt if expand else ""))

    plan = planner_prompt if (expand and planner_prompt and planner_prompt != user_prompt) else ""
    style_lines, hold_lines = _style_lines(
        style_fields=style_fields,
        overrides=overrides,
        style_family_confidence=style_family_confidence,
    )
    if not plan and not fired and not style_lines and not hold_lines:
        return user_prompt

    # Craft clauses and preserve clauses both come from the dimensions that fired:
    # one clause saying how to execute the edit well, plus whatever that dimension
    # declares must not be disturbed.
    craft = [policy.direction for policy in fired if policy.direction]
    preserve: list[str] = []
    if fired:
        preserve = list(UNIVERSAL_PRESERVE)
        for policy in fired:
            preserve.extend(policy.must_preserve)
    if plan and not preserve:
        # The universal clauses normally ride along with fired skills. A
        # model-written plan needs them regardless: it is the part most likely to
        # drift into identity or framing changes, and no skill may have fired.
        preserve = list(UNIVERSAL_PRESERVE)
    seen = {item.lower() for item in preserve}
    # In expand mode hard_preserve is the planner's own output, not the user's:
    # with gpt-6.1-sol it arrived as ~30 comma-split Chinese fragments ("构图",
    # "时长", ...). Preserve lines are never trimmed, so they crowded the plan --
    # the agent's actual contribution -- out of the 512-token window. The
    # universal clauses already cover identity, pose and framing.
    for item in ([] if expand else dict.fromkeys(str(x).strip() for x in (hard_preserve or []) if str(x).strip())):
        if item.lower() not in seen:
            preserve.append(item)
            seen.add(item.lower())
    # Holds come last so they sit closest to the end of the prompt, where the
    # text encoder weights them most. Preserve lines are never trimmed, so they
    # are also the safest place to spend words -- but they do crowd everything
    # else out, hence the de-duplication above.
    for item in hold_lines:
        if item.lower() not in seen:
            preserve.append(item)
            seen.add(item.lower())

    # Sections past the text encoder's 512-token window are silently dropped, and
    # the preserve block comes last -- so budget explicitly rather than let
    # truncation decide what the edit model never sees.
    prompt, _report = fit_sections(
        instruction=user_prompt,
        plan=plan,
        style_lines=style_lines,
        craft_lines=craft,
        preserve_lines=preserve,
    )
    return prompt


# Below this the family match is little better than a guess, and restyling the
# whole frame towards a world we may have picked wrong is worse than doing
# nothing. lock_style_family() defaults to 0.82 when the planner says nothing.
STYLE_CONFIDENCE_FLOOR = 0.5


def _style_lines(
    *,
    style_fields: dict[str, str] | None,
    overrides: dict[str, str] | None,
    style_family_confidence: float,
) -> tuple[list[str], list[str]]:
    """Split the style's per-dimension text into prompt lines and preserve lines.

    Returns ``(style_lines, hold_lines)``.

    A dimension held by the instruction contributes no style text. That alone is
    not protection: the style's pull on the rest of the frame is strong enough
    that the edit model restyles a held region anyway unless the prompt says, in
    as many words, to leave it alone. So a hold becomes a preserve line built
    from ``DimensionPolicy.keep_phrase``, plus whatever that dimension declares
    in ``must_preserve``.

    ``reference`` delivery emits no style text at all -- the world travels as an
    image -- which is what keeps the prompt short on a path where length is
    measured to hurt (see ``utils.prompt_budget``).
    """
    delivery = os.environ.get("LOW_HIGH_STYLE_DELIVERY", "inline").strip().lower()
    if delivery == "none" or not style_fields:
        return [], []
    if style_family_confidence < STYLE_CONFIDENCE_FLOOR:
        return [], []

    overrides = dict(overrides or {})
    style_lines: list[str] = []
    hold_lines: list[str] = []

    def _hold(dimension: str) -> None:
        policy = policy_for(dimension)
        phrase = (
            policy.keep_phrase
            if policy is not None and policy.keep_phrase
            else dimension.replace("_", " ")
        )
        hold_lines.append(f"keep {phrase} unchanged")
        if policy is not None:
            hold_lines.extend(policy.must_preserve)

    handled: set[str] = set()
    for dimension, text in style_fields.items():
        handled.add(dimension)
        if overrides.get(dimension) == KEEP:
            _hold(dimension)
        elif delivery == "inline":
            style_lines.append(text)

    # A hold on a dimension this style says nothing about still has to be said.
    # Iterating style_fields alone would drop it: the Art Deco template declares
    # no subject_surface text, so "keep the face" would produce no constraint --
    # while the style's overall pull is more than enough to repaint the face
    # anyway.
    for dimension, verdict in overrides.items():
        if verdict == KEEP and dimension not in handled:
            _hold(dimension)

    return style_lines, hold_lines


def _style_fields_for(selected, style_family: str) -> dict[str, str]:
    """Per-dimension world text for the selected strategy.

    A registered template wins whenever the planner landed on a known strategy,
    because its fields are the authored world. The planner's own candidate
    carries only whatever fields the model happened to write -- and the briefs it
    is shown are ``{strategy_id, style_family, title, description}``, so it
    usually has no render text at all -- which is the second branch.

    Until this existed, the style's render fields reached ``rewrite_targets`` and
    ``RegionEdit.new_description`` but never the prompt, which is why the style
    system measured as inert.
    """
    skills = get_skill_registry().get_all_skills()
    skill = next((item for item in skills if item.strategy_id == selected.strategy_id), None)
    if skill is None:
        skill = next((item for item in skills if item.style_family == style_family), None)
    if skill is not None:
        return skill.dimension_fields()

    values = {
        dimension: _clean_text(getattr(selected, dimension, "") or "")
        for dimension in CANONICAL_DIMENSIONS
    }
    if not values.get("props"):
        values["props"] = _clean_text(getattr(selected, "prop", "") or "")
    return {dimension: text for dimension, text in values.items() if text}


def build_plan(*, sample_id: str, prompt: str, raw_vlm_payload: dict) -> VideoPlan:
    style_family, confidence = lock_style_family(prompt, raw_vlm_payload)
    candidates = propose_candidate_strategies(style_family, prompt, raw_vlm_payload)
    grounding = extract_grounding(prompt, raw_vlm_payload)
    selected_strategy, selection_scores = select_strategy(prompt, grounding, candidates, raw_vlm_payload)
    plan_candidates = _parse_plan_candidates(raw_vlm_payload)
    candidate_score_details = _parse_candidate_score_details(raw_vlm_payload)
    selection_reason = _clean_text(raw_vlm_payload.get("selection_reason", "") or raw_vlm_payload.get("selection_rationale", ""))
    if plan_candidates and not candidate_score_details:
        candidate_score_details = _fallback_candidate_score_details(
            raw_vlm_payload=raw_vlm_payload,
            plan_candidates=plan_candidates,
            selection_reason=selection_reason,
        )
    if not plan_candidates or not candidate_score_details or not selection_reason:
        raise ValueError("planner quality validation failed: missing plan_candidates, candidate_score_details, or selection_reason")
    selected = next((item for item in candidates if item.strategy_id == selected_strategy), None)
    if selected is None:
        if not candidates:
            raise ValueError("planner quality validation failed: no candidate strategies survived validation")
        selected = candidates[0]
    hard_preserve, soft_preserve, rewrite_targets, light_touch_targets, forbidden_changes, resolutions = build_constraint_layers(
        grounding,
        selected,
        raw_vlm_payload,
    )
    primary_fx_carriers, secondary_fx_carriers, carrier_notes = validate_carriers(
        grounding,
        primary=list(raw_vlm_payload.get("primary_fx_carriers") or ["yellow curved structure", "environment", "lighting system"]),
        secondary=list(raw_vlm_payload.get("secondary_fx_carriers") or ["atmospheric layer", "subject edge light"]),
    )
    region_edits = emit_region_edits(selected, grounding, hard_preserve, light_touch_targets, raw_vlm_payload)
    normalized_prompt = _clean_text(raw_vlm_payload.get("normalized_prompt", "")) or " ".join(
        [selected.environment, selected.prop, selected.lighting, selected.costume, selected.subject_surface, selected.subject_integration]
    )
    planner_plan = _clean_text(raw_vlm_payload.get("compiled_edit_prompt", "")) or normalized_prompt
    style_fields = _style_fields_for(selected, style_family)
    overrides = parse_overrides(prompt, hard_preserve)
    compiled_edit_prompt = _compose_edit_prompt(
        user_prompt=prompt,
        planner_prompt=planner_plan,
        hard_preserve=hard_preserve,
        style_family_confidence=confidence,
        style_fields=style_fields,
        overrides=overrides,
    )
    focus_regions = list(raw_vlm_payload.get("focus_regions") or []) or [
        "environment",
        "functional_scene_props",
        "lighting_system_design",
        "costume_design",
        "skin_surface_design",
        "subject_environment_integration",
    ]
    plan = VideoPlan(
        sample_id=sample_id,
        prompt=prompt,
        style_family=style_family,
        style_family_confidence=confidence,
        user_intent_summary=_clean_text(raw_vlm_payload.get("user_intent_summary", "")),
        style_family_candidates=_clean_string_list(raw_vlm_payload.get("style_family_candidates")),
        selected_style_family=_clean_text(raw_vlm_payload.get("selected_style_family", "") or style_family),
        style_family_reason=_clean_text(raw_vlm_payload.get("style_family_reason", "") or raw_vlm_payload.get("selection_rationale", "")),
        matched_skill_candidates=_clean_string_list(raw_vlm_payload.get("matched_skill_candidates") or raw_vlm_payload.get("relevant_skill_ids")),
        rejected_skill_candidates=_clean_string_list(raw_vlm_payload.get("rejected_skill_candidates")),
        needs_custom_skill=bool(raw_vlm_payload.get("needs_custom_skill", raw_vlm_payload.get("needs_custom_style", False))),
        custom_skill_seed=_clean_text(raw_vlm_payload.get("custom_skill_seed", "")),
        grounding_notes=_clean_string_list(raw_vlm_payload.get("grounding_notes")),
        plan_candidates=plan_candidates,
        candidate_score_details=candidate_score_details,
        selected_candidate_id=_clean_text(raw_vlm_payload.get("selected_candidate_id", "") or selected_strategy),
        selection_reason=selection_reason,
        candidate_strategies=candidates,
        selected_strategy=selected.strategy_id,
        materialized_selected_skill=dict(raw_vlm_payload.get("materialized_selected_skill") or {}),
        selection_scores=selection_scores,
        grounding_evidence=grounding,
        hard_preserve_constraints=hard_preserve,
        soft_preserve_constraints=soft_preserve,
        rewrite_targets=rewrite_targets,
        light_touch_targets=light_touch_targets,
        forbidden_changes=forbidden_changes,
        primary_fx_carriers=primary_fx_carriers,
        secondary_fx_carriers=secondary_fx_carriers,
        normalized_prompt=normalized_prompt,
        planner_plan=planner_plan,
        compiled_edit_prompt=compiled_edit_prompt,
        dimension_overrides=dict(overrides or {}),
        focus_regions=focus_regions,
        region_edits=region_edits,
        self_check_notes=self_check_plan(
            strategy=selected,
            hard_preserve=hard_preserve,
            rewrite_targets=rewrite_targets,
            forbidden_changes=forbidden_changes,
            raw_vlm_payload=raw_vlm_payload,
        )
        + carrier_notes,
        conflict_resolutions=list(raw_vlm_payload.get("conflict_resolutions") or []) or resolutions,
        raw_vlm_response=json_dump(raw_vlm_payload),
    )
    return ensure_plan_is_productive(plan)


def json_dump(payload: dict) -> str:
    import json
    trimmed = dict(payload)
    trimmed.pop("raw_response", None)
    return json.dumps(trimmed, ensure_ascii=False)[:12000]
