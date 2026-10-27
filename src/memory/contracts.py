from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MEMORY_CONTRACT_VERSION = "memory_contract_v1"

_KNOWN_SCORE_KEYS = [
    "prompt_alignment",
    "structure_preservation",
    "transformation_strength",
    "carrier_grounding",
    "world_realization",
    "temporal_coherence",
    "artifact_penalty",
    "quality_score",
    "overall_score",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_list(values: Any, *, limit: int = 0) -> list[str]:
    if isinstance(values, list):
        raw_values = values
    elif isinstance(values, str):
        raw_values = [part.strip() for part in values.replace("；", ";").replace("，", ",").replace("、", ",").split(",")]
    else:
        raw_values = []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _safe_dict(payload: Any) -> dict[str, Any]:
    return dict(payload) if isinstance(payload, dict) else {}


def _normalize_scores(payload: Any) -> dict[str, float]:
    source = _safe_dict(payload)
    scores: dict[str, float] = {}
    for key in _KNOWN_SCORE_KEYS:
        if key in source:
            scores[key] = _safe_float(source.get(key), 0.0)
    for key, value in source.items():
        normalized_key = _clean_text(key)
        if not normalized_key or normalized_key in scores:
            continue
        try:
            scores[normalized_key] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def load_attempt_memory_contract(attempt_dir: str | Path) -> dict[str, Any]:
    path = Path(attempt_dir) / "memory.contract.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_contract_critic_evaluation(contract_payload: dict | None) -> dict[str, Any]:
    payload = _safe_dict(contract_payload)
    critic = _safe_dict(payload.get("critic"))
    return _safe_dict(critic.get("evaluation"))


def extract_contract_critic_reflection(contract_payload: dict | None) -> dict[str, Any]:
    payload = _safe_dict(contract_payload)
    critic = _safe_dict(payload.get("critic"))
    return _safe_dict(critic.get("reflection"))


def extract_contract_planner_payload(contract_payload: dict | None) -> dict[str, Any]:
    payload = _safe_dict(contract_payload)
    planner = _safe_dict(payload.get("planner"))
    return _safe_dict(planner.get("payload"))


def extract_contract_edit_hidden_report(contract_payload: dict | None) -> dict[str, Any]:
    payload = _safe_dict(contract_payload)
    edit = _safe_dict(payload.get("edit"))
    if isinstance(edit.get("implicit_hidden_report"), dict):
        return _safe_dict(edit.get("implicit_hidden_report"))
    signal = _safe_dict(edit.get("signal"))
    return _safe_dict(signal.get("implicit_hidden_report"))


def _clean_hold_checks(value: Any) -> list[dict]:
    """Per-dimension hold verdicts, kept as structured rows.

    Not flattened into failure_tags: the tags tell the promotion loop *that*
    something failed, these tell it *which dimension* -- which is the only way to
    say "this template's costume field is too strong" rather than "this template
    is bad".
    """
    rows: list[dict] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if not isinstance(item, dict):
            continue
        dimension = _clean_text(item.get("dimension", ""))
        if not dimension:
            continue
        rows.append(
            {
                "dimension": dimension,
                "held": bool(item.get("held", True)),
                "reported": bool(item.get("reported", True)),
                "note": _clean_text(item.get("note", ""))[:200],
            }
        )
        if len(rows) >= 16:
            break
    return rows


def build_critic_evaluation_signal(
    payload: dict | None,
    *,
    sample_id: str = "",
    variant: str = "",
    memory_mode: str = "",
    routing_bias: dict | None = None,
) -> dict[str, Any]:
    source = _safe_dict(payload)
    return {
        "sample_id": _clean_text(sample_id or source.get("sample_id", "")),
        "variant": _clean_text(variant or source.get("variant", "")),
        "memory_mode": _clean_text(memory_mode or source.get("memory_mode", "")),
        "scores": _normalize_scores(source.get("scores")),
        "failure_tags": _clean_list(source.get("failure_tags"), limit=16),
        "reflection_hints": _clean_list(source.get("reflection_hints"), limit=10),
        "notes": _clean_list(source.get("notes"), limit=10),
        "metric_hints": _safe_dict(source.get("metric_hints")),
        "routing_bias": _safe_dict(routing_bias if routing_bias is not None else source.get("routing_bias")),
        "evaluation_incomplete": bool(source.get("evaluation_incomplete", False)),
        "missing_headings": _clean_list(source.get("missing_headings"), limit=16),
        "hold_checks": _clean_hold_checks(source.get("hold_checks")),
    }


def build_critic_reflection_signal(
    payload: dict | None,
    *,
    memory_mode: str = "",
    routing_bias: dict | None = None,
) -> dict[str, Any]:
    source = _safe_dict(payload)
    action = _clean_text(source.get("action", ""))
    if action not in {"accept", "local_refine_prompt", "global_replan"}:
        action = ""
    return {
        "action": action,
        "reason": _clean_text(source.get("reason", "")),
        "prompt_patch": _clean_list(source.get("prompt_patch"), limit=10),
        "replan_directives": _clean_list(source.get("replan_directives"), limit=10),
        "replan_scope": _clean_text(source.get("replan_scope", "")),
        "memory_mode": _clean_text(memory_mode or source.get("memory_mode", "")),
        "routing_bias": _safe_dict(routing_bias if routing_bias is not None else source.get("routing_bias")),
    }


def build_edit_memory_signal(
    *,
    edit_payload: dict | None,
    memory_mode: str = "",
    controls: dict | None = None,
    hidden_report: dict | None = None,
) -> dict[str, Any]:
    source = _safe_dict(edit_payload)
    report = _safe_dict(hidden_report if hidden_report is not None else source.get("implicit_hidden_memory"))
    signal_controls = _safe_dict(controls if controls is not None else source.get("implicit_controls"))
    summary_vector = report.get("summary_vector") if isinstance(report.get("summary_vector"), list) else []
    captures = [item for item in (report.get("captures") or []) if isinstance(item, dict)]
    return {
        "memory_mode": _clean_text(memory_mode),
        "variant": _clean_text(source.get("variant", "")),
        "backend": _clean_text(source.get("backend", "")),
        "output_video": _clean_text(source.get("output_video", "")),
        "controls": signal_controls,
        "implicit_hidden_report": report,
        "implicit_hidden_summary": {
            "applied_events": _safe_int(report.get("applied_events", 0), 0),
            "positive_hits": _safe_int(report.get("positive_hits", 0), 0),
            "negative_hits": _safe_int(report.get("negative_hits", 0), 0),
            "capture_count": len(captures),
            "summary_vector_dim": len(summary_vector),
        },
    }


def _planner_summary(planner_payload: dict | None) -> dict[str, Any]:
    payload = _safe_dict(planner_payload)
    scene_archetype = _safe_dict(payload.get("scene_archetype"))
    grounding = _safe_dict(payload.get("grounding_evidence"))
    return {
        "style_family": _clean_text(payload.get("style_family", "")),
        "scene_archetype": _clean_text(scene_archetype.get("archetype_id", "")),
        "selected_strategy": _clean_text(payload.get("selected_strategy", "")),
        "primary_fx_carriers": _clean_list(payload.get("primary_fx_carriers"), limit=8),
        "visible_world_anchors": _clean_list(grounding.get("visible_world_anchors"), limit=8),
        "region_edits": _clean_list(payload.get("region_edits"), limit=8),
    }


def build_attempt_memory_contract(
    *,
    sample_id: str,
    attempt_index: int,
    variant: str,
    prompt: str,
    planner_payload: dict | None,
    critic_query: dict | None,
    critic_memory_mode: str,
    critic_evaluation_payload: dict | None,
    critic_reflection_payload: dict | None,
    critic_implicit_state_before: dict | None = None,
    critic_implicit_state_after: dict | None = None,
    edit_memory_mode: str = "",
    edit_controls: dict | None = None,
    edit_payload: dict | None = None,
    edit_hidden_report: dict | None = None,
    edit_hidden_state_before: dict | None = None,
    edit_hidden_state_after: dict | None = None,
) -> dict[str, Any]:
    planner = _safe_dict(planner_payload)
    query = _safe_dict(critic_query)
    evaluation_signal = build_critic_evaluation_signal(
        critic_evaluation_payload,
        sample_id=sample_id,
        variant=variant,
        memory_mode=critic_memory_mode,
    )
    reflection_signal = build_critic_reflection_signal(
        critic_reflection_payload,
        memory_mode=critic_memory_mode,
    )
    edit_signal = build_edit_memory_signal(
        edit_payload=edit_payload,
        memory_mode=edit_memory_mode,
        controls=edit_controls,
        hidden_report=edit_hidden_report,
    )
    routing_bias = _safe_dict(reflection_signal.get("routing_bias") or evaluation_signal.get("routing_bias"))
    resolved_variant = _clean_text(
        variant
        or evaluation_signal.get("variant", "")
        or edit_signal.get("variant", "")
        or planner.get("variant", "")
        or query.get("variant", "")
    )
    return {
        "version": MEMORY_CONTRACT_VERSION,
        "sample_id": _clean_text(sample_id or evaluation_signal.get("sample_id", "") or query.get("sample_id", "")),
        "attempt_index": max(0, _safe_int(attempt_index, 0)),
        "variant": resolved_variant,
        "prompt": _clean_text(prompt),
        "planner": {
            "payload": planner,
            "summary": _planner_summary(planner),
        },
        "critic": {
            "memory_mode": _clean_text(critic_memory_mode or evaluation_signal.get("memory_mode", "") or reflection_signal.get("memory_mode", "")),
            "query": query,
            "evaluation": evaluation_signal,
            "reflection": reflection_signal,
            "implicit_state_before": _safe_dict(critic_implicit_state_before),
            "implicit_state_after": _safe_dict(critic_implicit_state_after),
        },
        "edit": {
            "memory_mode": _clean_text(edit_memory_mode or edit_signal.get("memory_mode", "")),
            "signal": edit_signal,
            "controls": _safe_dict(edit_signal.get("controls")),
            "implicit_hidden_report": _safe_dict(edit_signal.get("implicit_hidden_report")),
            "hidden_state_before": _safe_dict(edit_hidden_state_before),
            "hidden_state_after": _safe_dict(edit_hidden_state_after),
        },
        "routing": {
            "action": _clean_text(reflection_signal.get("action", "")),
            "replan_scope": _clean_text(reflection_signal.get("replan_scope", "")),
            "prefer_global_replan": bool(routing_bias.get("prefer_global_replan", False)),
            "prefer_local_refine": bool(routing_bias.get("prefer_local_refine", False)),
            "recent_actions": _clean_list(routing_bias.get("recent_actions"), limit=6),
            "failure_focus": _clean_list(routing_bias.get("failure_focus"), limit=10),
        },
        "metadata": {
            "contract_version": MEMORY_CONTRACT_VERSION,
        },
    }
