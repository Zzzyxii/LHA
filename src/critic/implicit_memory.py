from __future__ import annotations

import json
from typing import Any

from memory.contracts import (
    extract_contract_critic_evaluation,
    extract_contract_critic_reflection,
    extract_contract_planner_payload,
)
from memory.runtime_modes import mode_enables_explicit, mode_enables_implicit, normalize_memory_mode


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: object, *, limit: int = 0) -> list[str]:
    if isinstance(values, list):
        raw_values = values
    elif isinstance(values, str):
        raw_values = [part.strip() for part in values.replace("；", ";").replace("，", ",").replace("、", ",").split(",")]
    else:
        raw_values = []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw_values:
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _safe_dict(value: object) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _safe_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compact_explicit_context(explicit_context: dict | None) -> dict:
    context = _safe_dict(explicit_context)
    return {
        "scene_archetypes": list(context.get("scene_archetypes", []) or [])[:3],
        "failure_patterns": list(context.get("failure_patterns", []) or [])[:5],
        "style_convertibility_priors": list(context.get("style_priors", []) or context.get("style_convertibility_priors", []) or [])[:4],
        "prompt_compilation_recipes": list(context.get("prompt_recipes", []) or context.get("prompt_compilation_recipes", []) or [])[:4],
        "runtime_skill_records": list(context.get("runtime_skill_records", []) or [])[:6],
        "promotable_skill_records": list(context.get("promotable_skill_records", []) or [])[:6],
        "quarantined_skill_records": list(context.get("quarantined_skill_records", []) or [])[:6],
    }


def build_critic_memory_query(
    *,
    sample_id: str,
    variant: str,
    prompt: str,
    planner_payload: dict | None,
    attempt_adaptation_state: dict | None = None,
) -> dict:
    planner_payload = _safe_dict(planner_payload)
    adaptation = _safe_dict(attempt_adaptation_state)
    grounding = _safe_dict(planner_payload.get("grounding_evidence"))
    scene_archetype = _safe_dict(planner_payload.get("scene_archetype"))
    return {
        "sample_id": _clean_text(sample_id),
        "variant": _clean_text(variant),
        "prompt": _clean_text(prompt)[:500],
        "style_family": _clean_text(planner_payload.get("style_family", "")),
        "scene_archetype": _clean_text(scene_archetype.get("archetype_id", "")),
        "selected_strategy": _clean_text(planner_payload.get("selected_strategy", "")),
        "primary_fx_carriers": _clean_list(planner_payload.get("primary_fx_carriers"), limit=6),
        "visible_world_anchors": _clean_list(grounding.get("visible_world_anchors"), limit=6),
        "last_failure_hints": _clean_list(adaptation.get("last_failure_hints"), limit=8),
        "last_prompt_patch": _clean_list(adaptation.get("last_prompt_patch"), limit=6),
        "last_good_carriers": _clean_list(adaptation.get("last_good_carriers"), limit=4),
        "last_bad_carriers": _clean_list(adaptation.get("last_bad_carriers"), limit=4),
    }


def pack_critic_implicit_state(
    *,
    previous_state: dict | None,
    sample_id: str = "",
    attempt_index: int = 0,
    planner_payload: dict | None = None,
    evaluation_payload: dict | None = None,
    reflection_payload: dict | None = None,
    memory_contract: dict | None = None,
) -> dict:
    state = _safe_dict(previous_state)
    contract_payload = _safe_dict(memory_contract)
    contract_planner = extract_contract_planner_payload(contract_payload)
    contract_evaluation = extract_contract_critic_evaluation(contract_payload)
    contract_reflection = extract_contract_critic_reflection(contract_payload)

    planner_payload = _safe_dict(contract_planner or planner_payload)
    evaluation_payload = _safe_dict(contract_evaluation or evaluation_payload)
    reflection_payload = _safe_dict(contract_reflection or reflection_payload)
    scores = _safe_dict(evaluation_payload.get("scores"))
    failure_tags = _clean_list(evaluation_payload.get("failure_tags"), limit=10)
    action = _clean_text(reflection_payload.get("action", ""))
    overall_score = float(scores.get("overall_score", 0.0) or 0.0)
    resolved_sample_id = _clean_text(sample_id or contract_payload.get("sample_id", ""))
    resolved_attempt_index = int(attempt_index or contract_payload.get("attempt_index", 0) or 0)

    history = list(state.get("recent_history", []) or [])
    history.append(
        {
            "attempt_index": resolved_attempt_index,
            "overall_score": overall_score,
            "action": action,
            "style_family": _clean_text(planner_payload.get("style_family", "")),
            "selected_strategy": _clean_text(planner_payload.get("selected_strategy", "")),
            "failure_tags": failure_tags[:8],
        }
    )
    history = history[-4:]

    failure_counter = _safe_dict(state.get("failure_counter"))
    strategy_counter = _safe_dict(state.get("strategy_counter"))
    for tag in failure_tags:
        failure_counter[tag] = int(failure_counter.get(tag, 0)) + 1
    strategy = _clean_text(planner_payload.get("selected_strategy", ""))
    if strategy:
        strategy_counter[strategy] = int(strategy_counter.get(strategy, 0)) + 1

    return {
        "sample_id": resolved_sample_id,
        "last_attempt_index": resolved_attempt_index,
        "last_action": action,
        "last_overall_score": overall_score,
        "recent_history": history,
        "failure_counter": failure_counter,
        "strategy_counter": strategy_counter,
        "last_failure_tags": failure_tags[:8],
        "last_prompt_patch": _clean_list(reflection_payload.get("prompt_patch"), limit=6),
        "last_replan_directives": _clean_list(reflection_payload.get("replan_directives"), limit=6),
        "last_good_carriers": _clean_list(_safe_dict(state).get("last_good_carriers"), limit=4),
        "last_bad_carriers": _clean_list(_safe_dict(state).get("last_bad_carriers"), limit=4),
    }


def apply_critic_memory_mode(
    user_prompt: str,
    *,
    memory_mode: str,
    explicit_context: dict | None = None,
    implicit_state: dict | None = None,
) -> str:
    mode = normalize_memory_mode(memory_mode)
    if not mode_enables_explicit(mode) and not mode_enables_implicit(mode):
        return user_prompt

    payload: dict[str, Any] = {
        "mode": mode.value,
    }
    if mode_enables_explicit(mode):
        payload["explicit_memory"] = _compact_explicit_context(explicit_context)
    if mode_enables_implicit(mode):
        payload["implicit_memory_state"] = _safe_dict(implicit_state)
    return user_prompt.rstrip() + "\n\ncritic_memory_context:\n" + _safe_json(payload)


def decode_routing_bias(
    *,
    implicit_state: dict | None,
    evaluation_payload: dict | None,
) -> dict:
    state = _safe_dict(implicit_state)
    evaluation_payload = _safe_dict(evaluation_payload)
    scores = _safe_dict(evaluation_payload.get("scores"))
    overall = float(scores.get("overall_score", 0.0) or 0.0)
    recent_history = list(state.get("recent_history", []) or [])
    recent_actions = [_clean_text(item.get("action", "")) for item in recent_history[-3:] if isinstance(item, dict)]
    repeated_replan = sum(1 for action in recent_actions if action == "global_replan")
    repeated_local_refine = sum(1 for action in recent_actions if action == "local_refine_prompt")
    return {
        "prefer_global_replan": bool(repeated_replan >= 2 and overall < 0.6),
        "prefer_local_refine": bool(repeated_local_refine >= 2 and overall >= 0.55),
        "recent_actions": recent_actions,
        "failure_focus": _clean_list(state.get("last_failure_tags"), limit=8),
    }
