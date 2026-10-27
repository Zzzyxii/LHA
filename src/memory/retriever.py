from __future__ import annotations

from copy import deepcopy

from memory.config import load_memory_policy
from memory.store import ensure_memory_schema


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or "")).split() if len(token) > 2]


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


def _unique_preserve(values: list[str]) -> list[str]:
    cleaned = [_clean_text(value) for value in values if _clean_text(value)]
    return list(dict.fromkeys(cleaned))


def _nested_get(item: dict, field_path: str) -> object:
    current: object = item
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _field_values(item: dict, fields: list[str]) -> list[str]:
    values: list[str] = []
    for field in fields:
        value = _nested_get(item, field)
        if isinstance(value, list):
            values.extend(_clean_list(value))
        elif isinstance(value, str):
            values.extend(_clean_list(value))
        elif value is not None:
            text = _clean_text(value)
            if text:
                values.append(text)
    return list(dict.fromkeys(values))


def _item_text(item: dict, fields: list[str]) -> str:
    return " ".join(_field_values(item, fields))


def _memory_policy(config_path: str | None = None) -> dict:
    return load_memory_policy(config_path)


def _retrieval_policy(policy: dict | None = None) -> dict:
    memory_policy = policy or _memory_policy()
    return (memory_policy.get("retrieval", {}) or {})


def _config_float(section: dict, key: str, default: float) -> float:
    return float(section.get(key, default) or default)


def _collection_base_weight(name: str, default: float, *, policy: dict | None = None) -> float:
    retrieval = _retrieval_policy(policy)
    weights = (retrieval.get("collection_base_weights", {}) or {})
    return float(weights.get(name, default) or default)


def _collection_limits(name: str, *, policy: dict | None = None) -> dict:
    retrieval = _retrieval_policy(policy)
    return ((retrieval.get("collection_limits", {}) or {}).get(name, {}) or {})


def _collection_top_k(name: str, default: int, *, policy: dict | None = None) -> int:
    limits = _collection_limits(name, policy=policy)
    return max(int(limits.get("top_k", default) or default), 1)


def _collection_min_score(name: str, default: float = 0.0, *, policy: dict | None = None) -> float:
    limits = _collection_limits(name, policy=policy)
    return float(limits.get("min_score", default) or default)


def _retrieval_scoring(policy: dict | None = None) -> dict:
    return (_retrieval_policy(policy).get("scoring", {}) or {})


def _structured_scoring(policy: dict | None = None) -> dict:
    return (_retrieval_policy(policy).get("structured_bonus", {}) or {})


def _runtime_scoring(policy: dict | None = None) -> dict:
    return (_retrieval_policy(policy).get("runtime_scoring", {}) or {})


def _execution_bias_scoring(policy: dict | None = None) -> dict:
    return (_retrieval_policy(policy).get("execution_bias", {}) or {})


def _runtime_limit(default: int, *, policy: dict | None = None) -> int:
    retrieval = _retrieval_policy(policy)
    runtime_selection = (retrieval.get("runtime_selection", {}) or {})
    return max(int(runtime_selection.get("limit", default) or default), 1)


def build_memory_query(
    *,
    prompt: str,
    variant: str = "",
    selected_family: str = "",
    scene_archetype: dict | str | None = None,
    visible_people: list[str] | None = None,
    visible_props: list[str] | None = None,
    visible_world_anchors: list[str] | None = None,
    carrier_candidates: list[str] | None = None,
    primary_fx_carriers: list[str] | None = None,
) -> dict:
    archetype_id = ""
    if isinstance(scene_archetype, dict):
        archetype_id = _clean_text(scene_archetype.get("archetype_id", ""))
    else:
        archetype_id = _clean_text(scene_archetype or "")
    anchor_terms = _unique_preserve(
        list(visible_people or [])
        + list(visible_props or [])
        + list(visible_world_anchors or [])
    )
    carrier_terms = _unique_preserve(
        list(carrier_candidates or [])
        + list(primary_fx_carriers or [])
    )
    return {
        "prompt": str(prompt or "").strip(),
        "variant": _clean_text(variant),
        "selected_family": _clean_text(selected_family),
        "archetype_id": archetype_id,
        "anchor_terms": anchor_terms,
        "carrier_candidates": carrier_terms,
    }


def _normalize_query(query: dict | None = None, *, prompt: str = "", variant: str = "") -> dict:
    payload = dict(query or {})
    normalized = build_memory_query(
        prompt=str(payload.get("prompt", prompt) or ""),
        variant=str(payload.get("variant", variant) or ""),
        selected_family=str(payload.get("selected_family", "") or ""),
        scene_archetype=payload.get("scene_archetype") or payload.get("archetype_id", ""),
        visible_people=payload.get("visible_people") or [],
        visible_props=payload.get("visible_props") or [],
        visible_world_anchors=payload.get("visible_world_anchors") or [],
        carrier_candidates=payload.get("carrier_candidates") or [],
        primary_fx_carriers=payload.get("primary_fx_carriers") or [],
    )
    normalized["anchor_terms"] = _unique_preserve(
        list(normalized.get("anchor_terms", []) or [])
        + list(payload.get("anchor_terms") or [])
    )
    normalized["carrier_candidates"] = _unique_preserve(
        list(normalized.get("carrier_candidates", []) or [])
        + list(payload.get("carrier_candidates") or [])
    )
    prompt_terms = _unique_preserve(
        [normalized["prompt"], normalized["selected_family"], normalized["archetype_id"]]
        + normalized["anchor_terms"]
        + normalized["carrier_candidates"]
    )
    normalized["_prompt_tokens"] = set(_tokenize(" ".join(prompt_terms)))
    normalized["_anchor_terms_set"] = set(_clean_list(normalized["anchor_terms"]))
    normalized["_carrier_terms_set"] = set(_clean_list(normalized["carrier_candidates"]))
    return normalized


def _variant_match(item: dict, variant: str) -> bool:
    item_variant = _clean_text(item.get("variant", ""))
    return not variant or item_variant in {"", _clean_text(variant)}


def _score_text_match(prompt_tokens: set[str], text: str) -> float:
    item_tokens = set(_tokenize(text))
    if not prompt_tokens or not item_tokens:
        return 0.0
    overlap = prompt_tokens.intersection(item_tokens)
    return len(overlap) / max(len(prompt_tokens), 1)


def _structured_bonus(
    item: dict,
    query: dict,
    *,
    family_fields: list[str] | None = None,
    archetype_fields: list[str] | None = None,
    preferred_archetype_fields: list[str] | None = None,
    risk_archetype_fields: list[str] | None = None,
    carrier_fields: list[str] | None = None,
    anchor_fields: list[str] | None = None,
    policy: dict | None = None,
) -> float:
    bonus = 0.0
    structured = _structured_scoring(policy)
    selected_family = _clean_text(query.get("selected_family", ""))
    archetype_id = _clean_text(query.get("archetype_id", ""))
    anchor_terms = query.get("_anchor_terms_set", set()) or set()
    carrier_terms = query.get("_carrier_terms_set", set()) or set()

    item_families = set(_field_values(item, family_fields or ["style_family"]))
    if selected_family and item_families:
        if selected_family in item_families:
            bonus += _config_float(structured, "family_match", 0.35)
        elif not item_families.intersection({"", "default", "general", "all"}):
            bonus -= _config_float(structured, "family_mismatch_penalty", 0.18)

    exact_archetypes = set(_field_values(item, archetype_fields or ["archetype_id"]))
    preferred_archetypes = set(_field_values(item, preferred_archetype_fields or ["best_for_archetypes"]))
    risky_archetypes = set(_field_values(item, risk_archetype_fields or ["high_risk_archetypes"]))
    if archetype_id:
        if archetype_id in exact_archetypes or archetype_id in preferred_archetypes:
            bonus += _config_float(structured, "archetype_match", 0.4)
        elif archetype_id in risky_archetypes:
            bonus -= _config_float(structured, "archetype_risk_penalty", 0.35)
        elif exact_archetypes and archetype_id not in exact_archetypes and "general_edit_scene" not in exact_archetypes:
            bonus -= _config_float(structured, "archetype_specific_mismatch_penalty", 0.08)

    item_carriers = set(_field_values(item, carrier_fields or ["recommended_carriers", "preferred_carriers", "anchor_requirements"]))
    if carrier_terms and item_carriers:
        carrier_overlap = carrier_terms.intersection(item_carriers)
        bonus += min(
            _config_float(structured, "carrier_overlap_cap", 0.45),
            _config_float(structured, "carrier_overlap_step", 0.15) * len(carrier_overlap),
        )

    item_anchors = set(_field_values(item, anchor_fields or carrier_fields or ["recommended_carriers", "anchor_requirements", "scene_hint"]))
    if anchor_terms and item_anchors:
        anchor_overlap = anchor_terms.intersection(item_anchors)
        bonus += min(
            _config_float(structured, "anchor_overlap_cap", 0.35),
            _config_float(structured, "anchor_overlap_step", 0.1) * len(anchor_overlap),
        )

    return bonus


def _score_item(
    *,
    item: dict,
    query: dict,
    text_fields: list[str],
    base_weight: float,
    family_fields: list[str] | None = None,
    archetype_fields: list[str] | None = None,
    preferred_archetype_fields: list[str] | None = None,
    risk_archetype_fields: list[str] | None = None,
    carrier_fields: list[str] | None = None,
    anchor_fields: list[str] | None = None,
    policy: dict | None = None,
) -> float:
    scoring = _retrieval_scoring(policy)
    lexical = _score_text_match(query.get("_prompt_tokens", set()) or set(), _item_text(item, text_fields))
    sample_count = int(item.get("sample_count", item.get("count", 0)) or 0)
    count_divisor = max(_config_float(scoring, "count_divisor", 15.0), 1e-6)
    count_bonus = min(sample_count / count_divisor, _config_float(scoring, "count_cap", 0.3))
    evidence_bonus = min(
        float(item.get("evidence_score", 0.0) or 0.0) * _config_float(scoring, "evidence_multiplier", 0.25),
        _config_float(scoring, "evidence_cap", 0.25),
    )
    success_bonus = min(
        float(item.get("success_rate", 0.0) or 0.0) * _config_float(scoring, "success_multiplier", 0.2),
        _config_float(scoring, "success_cap", 0.2),
    )
    recency_divisor = max(_config_float(scoring, "recency_divisor", 4.0), 1e-6)
    recency_bonus = min(
        float(item.get("recency_weight", 0.0) or 0.0) / recency_divisor,
        _config_float(scoring, "recency_input_cap", 1.0),
    ) * _config_float(scoring, "recency_multiplier", 0.15)
    source_quality_bonus = min(
        float(item.get("source_quality", 0.0) or 0.0) * _config_float(scoring, "source_quality_multiplier", 0.12),
        _config_float(scoring, "source_quality_cap", 0.12),
    )
    structure_bonus = _structured_bonus(
        item,
        query,
        family_fields=family_fields,
        archetype_fields=archetype_fields,
        preferred_archetype_fields=preferred_archetype_fields,
        risk_archetype_fields=risk_archetype_fields,
        carrier_fields=carrier_fields,
        anchor_fields=anchor_fields,
        policy=policy,
    )
    return lexical * base_weight + count_bonus + evidence_bonus + success_bonus + recency_bonus + source_quality_bonus + structure_bonus


def _rank_items(
    *,
    items: list[dict],
    query: dict,
    text_fields: list[str],
    base_weight: float,
    min_score: float = 0.0,
    top_k: int = 5,
    family_fields: list[str] | None = None,
    archetype_fields: list[str] | None = None,
    preferred_archetype_fields: list[str] | None = None,
    risk_archetype_fields: list[str] | None = None,
    carrier_fields: list[str] | None = None,
    anchor_fields: list[str] | None = None,
    policy: dict | None = None,
) -> list[dict]:
    scored: list[tuple[float, dict]] = []
    variant = _clean_text(query.get("variant", ""))
    for item in items:
        if not isinstance(item, dict):
            continue
        if not _variant_match(item, variant):
            continue
        score = _score_item(
            item=item,
            query=query,
            text_fields=text_fields,
            base_weight=base_weight,
            family_fields=family_fields,
            archetype_fields=archetype_fields,
            preferred_archetype_fields=preferred_archetype_fields,
            risk_archetype_fields=risk_archetype_fields,
            carrier_fields=carrier_fields,
            anchor_fields=anchor_fields,
            policy=policy,
        )
        if score < min_score and not item.get("count") and not item.get("sample_count"):
            continue
        enriched = deepcopy(item)
        enriched["_retrieval_score"] = round(score, 4)
        scored.append((score, enriched))
    scored.sort(key=lambda pair: (pair[0], pair[1].get("sample_count", pair[1].get("count", 0))), reverse=True)
    return [item for _, item in scored[:top_k]]


def _merge_ranked_records(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = (
                _clean_text(item.get("record_id", ""))
                or _clean_text(item.get("skill_id", ""))
                or _clean_text((item.get("skill_payload") or {}).get("strategy_id", ""))
            )
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _skill_carriers(skill_payload: dict) -> list[str]:
    carriers = _clean_list(skill_payload.get("recommended_carriers") or [])
    carriers.extend(_clean_list(skill_payload.get("anchor_requirements") or []))
    carriers.extend(_clean_list(skill_payload.get("prop") or ""))
    carriers.extend(_clean_list(skill_payload.get("title") or ""))
    carriers.extend(_clean_list(skill_payload.get("strategy_id") or ""))
    return list(dict.fromkeys(carriers))


def _carrier_compatibility_bonus(
    skill_payload: dict,
    *,
    good_carriers: list[str],
    bad_carriers: list[str],
    policy: dict | None = None,
) -> tuple[float, str]:
    if not isinstance(skill_payload, dict):
        return 0.0, ""

    skill_carriers = _skill_carriers(skill_payload)
    if not skill_carriers:
        return 0.0, ""

    good = set(_clean_list(good_carriers))
    bad = set(_clean_list(bad_carriers))

    positive_hits = [carrier for carrier in skill_carriers if any(tag in carrier or carrier in tag for tag in good)]
    negative_hits = [carrier for carrier in skill_carriers if any(tag in carrier or carrier in tag for tag in bad)]
    runtime = _runtime_scoring(policy)

    bonus = 0.0
    reason_parts: list[str] = []
    if positive_hits:
        bonus += min(
            _config_float(runtime, "good_carrier_cap", 0.6),
            _config_float(runtime, "good_carrier_base", 0.25)
            + _config_float(runtime, "good_carrier_step", 0.12) * len(set(positive_hits)),
        )
        reason_parts.append(f"命中上一轮有效 carrier: {', '.join(list(dict.fromkeys(positive_hits))[:3])}")
    if negative_hits:
        bonus -= min(
            _config_float(runtime, "bad_carrier_cap", 0.8),
            _config_float(runtime, "bad_carrier_base", 0.35)
            + _config_float(runtime, "bad_carrier_step", 0.15) * len(set(negative_hits)),
        )
        reason_parts.append(f"命中上一轮失败 carrier: {', '.join(list(dict.fromkeys(negative_hits))[:3])}")
    return bonus, "；".join(reason_parts)


def _execution_bias_summary(records: list[dict], *, selected_family: str, archetype_id: str = "", policy: dict | None = None) -> dict:
    family = _clean_text(selected_family)
    target_archetype = _clean_text(archetype_id)
    execution_bias = _execution_bias_scoring(policy)
    carrier_scores: dict[str, float] = {}
    carrier_counts: dict[str, int] = {}
    skill_scores: dict[str, float] = {}
    skill_counts: dict[str, int] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        if family and _clean_text(record.get("style_family", "")) != family:
            continue
        record_archetype = _clean_text(record.get("archetype_id", ""))
        if target_archetype and record_archetype and record_archetype != target_archetype:
            continue
        judgment = record.get("judgment", {}) if isinstance(record.get("judgment", {}), dict) else {}
        carriers = _clean_list(record.get("recommended_carriers") or [])
        skill_id = _clean_text(record.get("skill_id", ""))
        carrier_grounding = float(judgment.get("carrier_grounding", 0.0) or 0.0)
        transformation = float(judgment.get("transformation_strength", 0.0) or 0.0)
        world_realization = float(judgment.get("world_realization", 0.0) or 0.0)
        accepted = 1.0 if judgment.get("accepted") or record.get("accepted") else 0.0
        quality_score = float(record.get("quality_score", 0.0) or 0.0)
        failure_tags = {str(tag).strip() for tag in (judgment.get("failure_tags") or record.get("failure_tags") or []) if str(tag).strip()}

        signal = (
            accepted * 0.9
            + float(record.get("overall_score", 0.0) or 0.0) * 0.6
            + carrier_grounding * 0.7
            + transformation * 0.2
            + world_realization * 0.2
            + quality_score * 0.4
        )
        if "detached_fx" in failure_tags:
            signal -= 0.9
        if carrier_grounding < 0.45:
            signal -= 0.4

        if skill_id:
            skill_scores[skill_id] = skill_scores.get(skill_id, 0.0) + signal
            skill_counts[skill_id] = skill_counts.get(skill_id, 0) + 1
        for carrier in carriers:
            carrier_scores[carrier] = carrier_scores.get(carrier, 0.0) + signal
            carrier_counts[carrier] = carrier_counts.get(carrier, 0) + 1

    averaged_skill_scores = {
        key: (value / max(skill_counts.get(key, 1), 1))
        for key, value in skill_scores.items()
    }
    averaged_carrier_scores = {
        key: (value / max(carrier_counts.get(key, 1), 1))
        for key, value in carrier_scores.items()
    }
    preferred_threshold = _config_float(execution_bias, "preferred_carrier_threshold", 0.35)
    forbidden_threshold = _config_float(execution_bias, "forbidden_carrier_threshold", -0.1)
    preferred_carriers = [
        key
        for key, value in sorted(averaged_carrier_scores.items(), key=lambda pair: pair[1], reverse=True)
        if value > preferred_threshold
    ][:4]
    forbidden_carriers = [
        key
        for key, value in sorted(averaged_carrier_scores.items(), key=lambda pair: pair[1])
        if value < forbidden_threshold
    ][:4]
    return {
        "preferred_carriers": preferred_carriers,
        "forbidden_carriers": forbidden_carriers,
        "skill_scores": averaged_skill_scores,
        "prefer_environment_first": bool(preferred_carriers) and target_archetype == "world_anchor_driven_scene",
        "avoid_face_heavy_rewrite": target_archetype == "single_person_indoor_closeup" and any(
            score < 0 for key, score in averaged_skill_scores.items() if "face" in key or "portrait" in key
        ),
    }


def retrieve_memory_context(
    memory: dict,
    *,
    prompt: str = "",
    variant: str = "",
    query: dict | None = None,
    config_path: str | None = None,
) -> dict:
    policy = _memory_policy(config_path)
    payload = ensure_memory_schema(memory)
    query_payload = _normalize_query(query, prompt=prompt, variant=variant)

    scene_archetypes = _rank_items(
        items=payload.get("scene_archetypes", []),
        query=query_payload,
        text_fields=["archetype_id", "scene_hint"],
        archetype_fields=["archetype_id"],
        base_weight=_collection_base_weight("scene_archetypes", 1.0, policy=policy),
        min_score=_collection_min_score("scene_archetypes", policy=policy),
        top_k=_collection_top_k("scene_archetypes", 3, policy=policy),
        policy=policy,
    )
    failure_patterns = _rank_items(
        items=payload.get("failure_patterns", []),
        query=query_payload,
        text_fields=["pattern", "scene_hint", "when_prompt_contains", "notes"],
        archetype_fields=["archetype_id"],
        carrier_fields=["recommended_carriers"],
        base_weight=_collection_base_weight("failure_patterns", 1.25, policy=policy),
        min_score=_collection_min_score("failure_patterns", policy=policy),
        top_k=_collection_top_k("failure_patterns", 5, policy=policy),
        policy=policy,
    )
    style_convertibility_priors = _rank_items(
        items=payload.get("style_convertibility_priors", []),
        query=query_payload,
        text_fields=["style_family", "archetype_id", "scene_hint", "observation", "failure_tags"],
        family_fields=["style_family"],
        archetype_fields=["archetype_id"],
        base_weight=_collection_base_weight("style_convertibility_priors", 1.1, policy=policy),
        min_score=_collection_min_score("style_convertibility_priors", policy=policy),
        top_k=_collection_top_k("style_convertibility_priors", 5, policy=policy),
        policy=policy,
    )
    model_behavior_priors = _rank_items(
        items=payload.get("model_behavior_priors", []),
        query=query_payload,
        text_fields=["variant", "pattern", "observation"],
        base_weight=_collection_base_weight("model_behavior_priors", 0.9, policy=policy),
        min_score=_collection_min_score("model_behavior_priors", policy=policy),
        top_k=_collection_top_k("model_behavior_priors", 5, policy=policy),
        policy=policy,
    )
    prompt_compilation_recipes = _rank_items(
        items=payload.get("prompt_compilation_recipes", []),
        query=query_payload,
        text_fields=["recipe_id", "archetype_id", "when_prompt_contains", "patches", "notes"],
        archetype_fields=["archetype_id"],
        carrier_fields=["primary_carriers"],
        base_weight=_collection_base_weight("prompt_compilation_recipes", 1.35, policy=policy),
        min_score=_collection_min_score("prompt_compilation_recipes", policy=policy),
        top_k=_collection_top_k("prompt_compilation_recipes", 5, policy=policy),
        policy=policy,
    )
    promotable_skill_records = _rank_items(
        items=payload.get("promotable_skill_records", []),
        query=query_payload,
        text_fields=["skill_id", "style_family", "archetype_id", "prompt", "source", "skill_payload.title", "skill_payload.description"],
        family_fields=["style_family"],
        archetype_fields=["archetype_id"],
        preferred_archetype_fields=["best_for_archetypes", "skill_payload.best_for_archetypes"],
        risk_archetype_fields=["high_risk_archetypes", "skill_payload.high_risk_archetypes"],
        carrier_fields=["recommended_carriers", "anchor_requirements", "skill_payload.recommended_carriers", "skill_payload.anchor_requirements"],
        base_weight=_collection_base_weight("promotable_skill_records", 1.25, policy=policy),
        min_score=_collection_min_score("promotable_skill_records", policy=policy),
        top_k=_collection_top_k("promotable_skill_records", 8, policy=policy),
        policy=policy,
    )
    custom_skill_source = _rank_items(
        items=payload.get("custom_skill_records", []),
        query=query_payload,
        text_fields=["skill_id", "style_family", "archetype_id", "prompt", "source", "skill_payload.title", "skill_payload.description"],
        family_fields=["style_family"],
        archetype_fields=["archetype_id"],
        preferred_archetype_fields=["best_for_archetypes", "skill_payload.best_for_archetypes"],
        risk_archetype_fields=["high_risk_archetypes", "skill_payload.high_risk_archetypes"],
        carrier_fields=["recommended_carriers", "anchor_requirements", "skill_payload.recommended_carriers", "skill_payload.anchor_requirements"],
        base_weight=_collection_base_weight("custom_skill_records", 1.1, policy=policy),
        min_score=_collection_min_score("custom_skill_records", policy=policy),
        top_k=_collection_top_k("custom_skill_records", 8, policy=policy),
        policy=policy,
    )
    runtime_skill_records = _rank_items(
        items=payload.get("runtime_skill_records", []),
        query=query_payload,
        text_fields=["skill_id", "style_family", "archetype_id", "prompt", "source", "skill_payload.title", "skill_payload.description"],
        family_fields=["style_family"],
        archetype_fields=["archetype_id"],
        preferred_archetype_fields=["best_for_archetypes", "skill_payload.best_for_archetypes"],
        risk_archetype_fields=["high_risk_archetypes", "skill_payload.high_risk_archetypes"],
        carrier_fields=["recommended_carriers", "anchor_requirements", "skill_payload.recommended_carriers", "skill_payload.anchor_requirements"],
        base_weight=_collection_base_weight("runtime_skill_records", 1.2, policy=policy),
        min_score=_collection_min_score("runtime_skill_records", policy=policy),
        top_k=_collection_top_k("runtime_skill_records", 8, policy=policy),
        policy=policy,
    )
    quarantined_skill_records = _rank_items(
        items=payload.get("quarantined_skill_records", []),
        query=query_payload,
        text_fields=["skill_id", "style_family", "archetype_id", "prompt", "source", "failure_tags", "skill_payload.title"],
        family_fields=["style_family"],
        archetype_fields=["archetype_id"],
        preferred_archetype_fields=["best_for_archetypes", "skill_payload.best_for_archetypes"],
        risk_archetype_fields=["high_risk_archetypes", "skill_payload.high_risk_archetypes"],
        carrier_fields=["recommended_carriers", "anchor_requirements", "skill_payload.recommended_carriers", "skill_payload.anchor_requirements"],
        base_weight=_collection_base_weight("quarantined_skill_records", 1.15, policy=policy),
        min_score=_collection_min_score("quarantined_skill_records", policy=policy),
        top_k=_collection_top_k("quarantined_skill_records", 12, policy=policy),
        policy=policy,
    )
    skill_execution_records = _rank_items(
        items=payload.get("skill_execution_records", []),
        query=query_payload,
        text_fields=["skill_id", "style_family", "archetype_id", "source", "failure_tags"],
        family_fields=["style_family"],
        archetype_fields=["archetype_id"],
        carrier_fields=["recommended_carriers"],
        base_weight=_collection_base_weight("skill_execution_records", 1.0, policy=policy),
        min_score=_collection_min_score("skill_execution_records", policy=policy),
        top_k=_collection_top_k("skill_execution_records", 24, policy=policy),
        policy=policy,
    )
    skill_execution_summaries = _rank_items(
        items=payload.get("skill_execution_summaries", []),
        query=query_payload,
        text_fields=["style_family", "archetype_id", "preferred_carriers", "forbidden_carriers", "boosted_skill_ids", "penalized_skill_ids"],
        family_fields=["style_family"],
        archetype_fields=["archetype_id"],
        carrier_fields=["preferred_carriers", "forbidden_carriers"],
        base_weight=_collection_base_weight("skill_execution_summaries", 1.0, policy=policy),
        min_score=_collection_min_score("skill_execution_summaries", policy=policy),
        top_k=_collection_top_k("skill_execution_summaries", 16, policy=policy),
        policy=policy,
    )

    merged_custom_skill_records = _merge_ranked_records(
        promotable_skill_records,
        [item for item in custom_skill_source if bool(item.get("accepted", False))],
    )

    public_query = {
        key: value
        for key, value in query_payload.items()
        if not str(key).startswith("_")
    }
    return {
        "memory_query": public_query,
        "scene_archetype_hints": scene_archetypes,
        "failure_patterns": failure_patterns,
        "style_convertibility_priors": style_convertibility_priors,
        "model_behavior_priors": model_behavior_priors,
        "prompt_compilation_recipes": prompt_compilation_recipes,
        "custom_skill_records": merged_custom_skill_records,
        "promotable_skill_records": promotable_skill_records,
        "runtime_skill_records": runtime_skill_records,
        "quarantined_skill_records": quarantined_skill_records,
        "skill_execution_records": skill_execution_records,
        "skill_execution_summaries": skill_execution_summaries,
    }


def _quarantined_strategy_ids(memory_context: dict, *, selected_family: str, archetype_id: str = "") -> set[str]:
    family = _clean_text(selected_family)
    target_archetype = _clean_text(archetype_id)
    blocked: set[str] = set()
    for record in memory_context.get("quarantined_skill_records") or []:
        if not isinstance(record, dict):
            continue
        record_family = _clean_text(record.get("style_family", ""))
        if family and record_family and record_family != family:
            continue
        record_archetype = _clean_text(record.get("archetype_id", ""))
        if target_archetype and record_archetype and record_archetype != target_archetype:
            continue
        skill_id = _clean_text(record.get("skill_id", "")) or _clean_text((record.get("skill_payload") or {}).get("strategy_id", ""))
        if skill_id:
            blocked.add(skill_id)
    return blocked


def retrieve_runtime_skills(
    memory_context: dict,
    *,
    selected_family: str,
    archetype_id: str = "",
    limit: int | None = None,
    config_path: str | None = None,
) -> list[dict]:
    policy = _memory_policy(config_path)
    runtime = _runtime_scoring(policy)
    effective_limit = max(int(limit or _runtime_limit(5, policy=policy)), 1)
    family = _clean_text(selected_family)
    target_archetype = _clean_text(archetype_id)
    candidates: list[tuple[float, dict]] = []
    seen: set[str] = set()
    attempt_state = memory_context.get("attempt_adaptation_state") or {}
    execution_bias = _execution_bias_summary(
        memory_context.get("skill_execution_records") or [],
        selected_family=family,
        archetype_id=target_archetype,
        policy=policy,
    )
    quarantined_ids = _quarantined_strategy_ids(
        memory_context,
        selected_family=family,
        archetype_id=target_archetype,
    )
    good_carriers = []
    bad_carriers = []
    if isinstance(attempt_state, dict):
        good_carriers = [str(item).strip() for item in (attempt_state.get("last_good_carriers") or []) if str(item).strip()]
        bad_carriers = [str(item).strip() for item in (attempt_state.get("last_bad_carriers") or []) if str(item).strip()]
    good_carriers = list(dict.fromkeys(good_carriers + execution_bias.get("preferred_carriers", [])))
    bad_carriers = list(dict.fromkeys(bad_carriers + execution_bias.get("forbidden_carriers", [])))

    def _push(
        skill_payload: dict,
        *,
        score: float,
        source: str,
        lineage: dict | None = None,
        retrieval_reason: str = "",
    ) -> None:
        if not isinstance(skill_payload, dict):
            return
        strategy_id = str(skill_payload.get("strategy_id", "")).strip()
        skill_family = _clean_text(skill_payload.get("style_family", "")) or family
        if not strategy_id or strategy_id in seen:
            return
        if family and skill_family != family:
            return
        if source.startswith("memory_") and _clean_text(strategy_id) in quarantined_ids:
            return
        historical_skill_bias = float((execution_bias.get("skill_scores", {}) or {}).get(_clean_text(strategy_id), 0.0) or 0.0)
        carrier_bonus, carrier_reason = _carrier_compatibility_bonus(
            skill_payload,
            good_carriers=good_carriers,
            bad_carriers=bad_carriers,
            policy=policy,
        )
        skill_bias_cap = max(_config_float(runtime, "historical_skill_bias_cap", 0.8), 0.0)
        skill_bonus = max(
            -skill_bias_cap,
            min(skill_bias_cap, historical_skill_bias * _config_float(runtime, "historical_skill_bias_scale", 0.35)),
        )
        total_score = score + carrier_bonus + skill_bonus
        enriched = deepcopy(skill_payload)
        enriched["style_family"] = skill_family
        enriched["_runtime_score"] = round(total_score, 4)
        enriched["_runtime_source"] = source
        enriched["_lineage"] = deepcopy(lineage or {})
        enriched["_retrieval_reason"] = "；".join(
            part
            for part in [
                retrieval_reason.strip(),
                carrier_reason.strip(),
                (
                    f"历史执行统计对该 skill {'升权' if skill_bonus > 0 else '降权'}"
                    if abs(skill_bonus) >= 0.12
                    else ""
                ),
            ]
            if part
        )
        seen.add(strategy_id)
        candidates.append((total_score, enriched))

    if isinstance(attempt_state, dict):
        materialized = attempt_state.get("last_materialized_skill") or {}
        if isinstance(materialized, dict):
            _push(
                materialized,
                score=_config_float(runtime, "attempt_materialized_base", 3.0),
                source="attempt_materialized",
                lineage={
                    "origin": "attempt_materialized",
                    "parent_skill_ids": [],
                    "generation": 0,
                },
                retrieval_reason="上一轮 final 直接使用过的 materialized skill，最接近当前 sample 的可执行模板。",
            )
        for item in attempt_state.get("last_accepted_custom_skills") or []:
            if isinstance(item, dict):
                _push(
                    item,
                    score=_config_float(runtime, "attempt_custom_base", 2.6),
                    source="attempt_custom",
                    lineage={
                        "origin": "attempt_custom",
                        "parent_skill_ids": [],
                        "generation": 0,
                    },
                    retrieval_reason="上一轮已通过或半通过的 custom skill，可直接作为当前 sample 的高优先级复用候选。",
                )

    for record in memory_context.get("runtime_skill_records") or []:
        if not isinstance(record, dict):
            continue
        skill_payload = record.get("skill_payload") if isinstance(record.get("skill_payload"), dict) else record
        lineage = record.get("lineage", {}) if isinstance(record.get("lineage", {}), dict) else {}
        base_score = float(record.get("_retrieval_score", 0.0) or 0.0)
        accepted_bonus = _config_float(runtime, "runtime_accepted_bonus", 0.35) if record.get("accepted") else 0.0
        overall_bonus = float(record.get("overall_score", 0.0) or 0.0) * _config_float(runtime, "runtime_overall_multiplier", 0.45)
        quality_bonus = float(record.get("quality_score", 0.0) or 0.0) * _config_float(runtime, "runtime_quality_multiplier", 0.4)
        _push(
            skill_payload,
            score=_config_float(runtime, "runtime_record_base", 1.2) + base_score + accepted_bonus + overall_bonus + quality_bonus,
            source=str(record.get("source", "memory_runtime")).strip() or "memory_runtime",
            lineage={
                "origin": str(lineage.get("origin", "")).strip() or str(record.get("source", "memory_runtime")).strip() or "memory_runtime",
                "parent_skill_ids": list(lineage.get("parent_skill_ids") or []),
                "generation": int(lineage.get("generation", 0) or 0),
                "change_summary": str(lineage.get("change_summary", "")).strip(),
            },
            retrieval_reason=(
                "历史 runtime memory 中与当前 family/archetype/anchor 接近"
                + ("，且已验证通过" if record.get("accepted") else "")
            ),
        )

    promotable_records = _merge_ranked_records(
        memory_context.get("promotable_skill_records") or [],
        [item for item in (memory_context.get("custom_skill_records") or []) if bool(item.get("accepted", False))],
    )
    for record in promotable_records:
        if not isinstance(record, dict):
            continue
        skill_payload = record.get("skill_payload") or {}
        lineage = record.get("lineage", {}) if isinstance(record.get("lineage", {}), dict) else {}
        base_score = float(record.get("_retrieval_score", 0.0) or 0.0)
        accepted_bonus = _config_float(runtime, "promotable_accepted_bonus", 0.5) if record.get("accepted") else 0.0
        overall_bonus = float(record.get("overall_score", 0.0) or 0.0) * _config_float(runtime, "promotable_overall_multiplier", 0.5)
        quality_bonus = float(record.get("quality_score", 0.0) or 0.0) * _config_float(runtime, "promotable_quality_multiplier", 0.45)
        source_bonus = (
            _config_float(runtime, "analysis_stage_source_bonus", 0.2)
            if str(record.get("source", "")).strip() == "analysis_stage_custom"
            else 0.0
        )
        _push(
            skill_payload,
            score=_config_float(runtime, "promotable_base", 1.0) + base_score + accepted_bonus + overall_bonus + quality_bonus + source_bonus,
            source=str(record.get("source", "memory_custom")).strip() or "memory_custom",
            lineage={
                "origin": str(lineage.get("origin", "")).strip() or str(record.get("source", "memory_custom")).strip() or "memory_custom",
                "parent_skill_ids": list(lineage.get("parent_skill_ids") or []),
                "generation": int(lineage.get("generation", 0) or 0),
                "change_summary": str(lineage.get("change_summary", "")).strip(),
            },
            retrieval_reason=(
                "历史 memory 中与当前 prompt/family/archetype 接近"
                + ("，且已被接受" if record.get("accepted") else "")
                + ("，可作为稳定的 runtime 补技能" if float(record.get("overall_score", 0.0) or 0.0) >= 0.7 else "")
            ),
        )

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in candidates[:effective_limit]]


def retrieve_execution_adaptive_bias(
    memory_context: dict,
    *,
    selected_family: str,
    archetype_id: str = "",
    config_path: str | None = None,
) -> dict:
    policy = _memory_policy(config_path)
    execution_bias = _execution_bias_scoring(policy)
    family = _clean_text(selected_family)
    target_archetype = _clean_text(archetype_id)
    summaries = [
        item for item in (memory_context.get("skill_execution_summaries") or []) if isinstance(item, dict)
    ]
    matched_summary = next(
        (
            item
            for item in summaries
            if _clean_text(item.get("style_family", "")) == family
            and _clean_text(item.get("archetype_id", "")) == target_archetype
        ),
        None,
    )
    quarantined_ids = sorted(
        _quarantined_strategy_ids(
            memory_context,
            selected_family=family,
            archetype_id=target_archetype,
        )
    )
    if matched_summary is not None:
        penalized = list(matched_summary.get("penalized_skill_ids", [])[:4])
        penalized = list(dict.fromkeys(penalized + quarantined_ids))[:4]
        return {
            "preferred_carriers": list(matched_summary.get("preferred_carriers", [])[:4]),
            "forbidden_carriers": list(matched_summary.get("forbidden_carriers", [])[:4]),
            "prefer_environment_first": bool(matched_summary.get("prefer_environment_first", False)),
            "avoid_face_heavy_rewrite": bool(matched_summary.get("avoid_face_heavy_rewrite", False)),
            "boosted_skill_ids": list(matched_summary.get("boosted_skill_ids", [])[:4]),
            "penalized_skill_ids": penalized,
        }

    records = memory_context.get("skill_execution_records") or []
    summary = _execution_bias_summary(
        records,
        selected_family=family,
        archetype_id=target_archetype,
        policy=policy,
    )
    skill_scores = summary.get("skill_scores", {}) or {}
    boosted_skill_ids = [
        key
        for key, value in sorted(skill_scores.items(), key=lambda pair: pair[1], reverse=True)
        if value > _config_float(execution_bias, "boosted_skill_threshold", 0.25)
    ][:4]
    penalized_skill_ids = [
        key
        for key, value in sorted(skill_scores.items(), key=lambda pair: pair[1])
        if value < _config_float(execution_bias, "penalized_skill_threshold", -0.05)
    ][:4]
    penalized_skill_ids = list(dict.fromkeys(penalized_skill_ids + quarantined_ids))[:4]
    return {
        "preferred_carriers": summary.get("preferred_carriers", [])[:4],
        "forbidden_carriers": summary.get("forbidden_carriers", [])[:4],
        "prefer_environment_first": bool(summary.get("prefer_environment_first", False)),
        "avoid_face_heavy_rewrite": bool(summary.get("avoid_face_heavy_rewrite", False)),
        "boosted_skill_ids": boosted_skill_ids,
        "penalized_skill_ids": penalized_skill_ids,
    }
