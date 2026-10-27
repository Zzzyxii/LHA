from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from memory.contracts import (
    extract_contract_critic_evaluation,
    extract_contract_critic_reflection,
    extract_contract_edit_hidden_report,
    extract_contract_planner_payload,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


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


def _clean_vector(values: Any, *, max_dim: int = 0) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        out.append(number)
        if max_dim > 0 and len(out) >= max_dim:
            break
    return out


def _ema_vectors(previous: list[float], current: list[float], *, decay: float) -> list[float]:
    if not previous:
        return list(current)
    if len(previous) != len(current):
        return list(current)
    alpha = max(0.0, min(1.0, float(decay)))
    return [alpha * previous[index] + (1.0 - alpha) * current[index] for index in range(len(previous))]


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    if dim <= 0:
        return []
    total = [0.0 for _ in range(dim)]
    count = 0
    for vector in vectors:
        if len(vector) != dim:
            continue
        count += 1
        for index, value in enumerate(vector):
            total[index] += value
    if count <= 0:
        return []
    return [value / count for value in total]


def _default_hidden_config() -> dict:
    return {
        "enabled": True,
        "block_indices": [],
        "capture_block_index": -1,
        "capture_ema_decay": 0.8,
        "residual_scale": 0.04,
        "positive_weight": 1.0,
        "negative_weight": 1.0,
        "style_weight": 1.0,
        "positive_topk": 2,
        "negative_topk": 2,
        "similarity_floor": 0.05,
        "max_entries_per_bank": 8,
        "max_vector_dim": 0,
        "include_global_positive_fallback": True,
        "positive_score_threshold": 0.62,
        "scale_weights": {
            "short": 1.0,
            "mid": 0.7,
            "long": 0.5,
        },
    }


def _resolve_hidden_config(runtime_memory_config: dict | None) -> tuple[dict, dict, dict]:
    runtime = dict(runtime_memory_config or {})
    implicit_cfg = dict(runtime.get("implicit", {}) or {})
    edit_cfg = dict(implicit_cfg.get("edit", {}) or {})
    hidden_cfg = dict(_default_hidden_config())
    source_hidden = dict(edit_cfg.get("hidden_state", {}) or edit_cfg.get("hidden", {}) or {})
    hidden_cfg.update(source_hidden)
    scale_weights = dict(_default_hidden_config().get("scale_weights", {}))
    scale_weights.update(dict(hidden_cfg.get("scale_weights", {}) or {}))
    hidden_cfg["scale_weights"] = scale_weights
    return implicit_cfg, edit_cfg, hidden_cfg


def init_edit_hidden_state() -> dict:
    return {
        "version": "edit_hidden_v1",
        "local": {
            "positive": {"short": [], "mid": [], "long": []},
            "negative": {"short": [], "mid": [], "long": []},
        },
        "global": {
            "positive": {"short": [], "mid": [], "long": []},
            "negative": {"short": [], "mid": [], "long": []},
        },
        "metadata": {
            "last_style_family": "",
            "last_archetype_id": "",
            "last_action": "",
            "updates": 0,
        },
    }


def _ensure_state(state: dict | None) -> dict:
    payload = deepcopy(state) if isinstance(state, dict) else init_edit_hidden_state()
    for scope in ("local", "global"):
        scope_payload = payload.setdefault(scope, {})
        for polarity in ("positive", "negative"):
            polarity_payload = scope_payload.setdefault(polarity, {})
            for scale in ("short", "mid", "long"):
                polarity_payload.setdefault(scale, [])
    payload.setdefault("metadata", {})
    return payload


def _clear_local_short_mid(state: dict) -> None:
    for polarity in ("positive", "negative"):
        bank = state["local"][polarity]
        bank["short"] = []
        bank["mid"] = []


def _append_short_entry(
    *,
    bank: dict,
    entry: dict,
    short_window: int,
    mid_decay: float,
    long_max: int,
) -> None:
    short = list(bank.get("short", []) or [])
    short.append(entry)
    short = short[-max(1, int(short_window)) :]
    bank["short"] = short

    vectors = [_clean_vector(item.get("vector")) for item in short if isinstance(item, dict)]
    vectors = [vector for vector in vectors if vector]
    if vectors:
        ema = vectors[0]
        for vector in vectors[1:]:
            ema = _ema_vectors(ema, vector, decay=mid_decay)
        mid_entry = dict(entry)
        mid_entry["scale"] = "mid"
        mid_entry["vector"] = ema
        mid_entry["sample_count"] = len(vectors)
        bank["mid"] = [mid_entry]

    long_entries = [dict(item) for item in list(bank.get("long", []) or []) if isinstance(item, dict)]
    long_entry = dict(entry)
    long_entry["scale"] = "long"
    long_entries.append(long_entry)
    bank["long"] = long_entries[-max(1, int(long_max)) :]


def _extract_summary_vector(memory_report: dict | None, *, max_dim: int = 0) -> list[float]:
    report = dict(memory_report or {})
    vector = _clean_vector(report.get("summary_vector"), max_dim=max_dim)
    if vector:
        return vector
    captures = [item for item in (report.get("captures") or []) if isinstance(item, dict)]
    vectors = [_clean_vector(item.get("vector"), max_dim=max_dim) for item in captures]
    vectors = [item for item in vectors if item]
    return _average_vectors(vectors)


def _entry_payload(
    *,
    vector: list[float],
    sample_id: str,
    variant: str,
    attempt_index: int,
    planner_payload: dict,
    evaluation_payload: dict,
    reflection_payload: dict,
    scope: str,
    polarity: str,
) -> dict:
    scores = dict(evaluation_payload.get("scores", {}) or {})
    scene_archetype = dict(planner_payload.get("scene_archetype", {}) or {})
    overall_score = _safe_float(scores.get("overall_score", 0.0), 0.0)
    quality_score = _safe_float(scores.get("quality_score", 0.0), 0.0)
    failure_tags = _clean_list(evaluation_payload.get("failure_tags"), limit=8)
    base_weight = max(0.05, min(2.0, overall_score + quality_score))
    return {
        "vector": list(vector),
        "weight": base_weight,
        "score": overall_score,
        "quality_score": quality_score,
        "scope": scope,
        "polarity": polarity,
        "scale": "short",
        "sample_id": _clean_text(sample_id),
        "variant": _clean_text(variant),
        "attempt_index": int(attempt_index),
        "style_family": _clean_text(planner_payload.get("style_family", "")),
        "archetype_id": _clean_text(scene_archetype.get("archetype_id", "")),
        "selected_strategy": _clean_text(planner_payload.get("selected_strategy", "")),
        "action": _clean_text(reflection_payload.get("action", "")),
        "failure_tags": failure_tags,
    }


def update_edit_hidden_state(
    *,
    edit_hidden_state: dict | None,
    runtime_memory_config: dict | None,
    sample_id: str = "",
    variant: str = "",
    attempt_index: int = 0,
    planner_payload: dict | None = None,
    evaluation_payload: dict | None = None,
    reflection_payload: dict | None = None,
    memory_report: dict | None = None,
    memory_contract: dict | None = None,
) -> dict:
    state = _ensure_state(edit_hidden_state)
    contract_payload = dict(memory_contract or {})
    contract_planner = extract_contract_planner_payload(contract_payload)
    contract_evaluation = extract_contract_critic_evaluation(contract_payload)
    contract_reflection = extract_contract_critic_reflection(contract_payload)
    contract_report = extract_contract_edit_hidden_report(contract_payload)

    planner_payload = dict(contract_planner or planner_payload or {})
    evaluation_payload = dict(contract_evaluation or evaluation_payload or {})
    reflection_payload = dict(contract_reflection or reflection_payload or {})
    memory_report = dict(contract_report or memory_report or {})
    if not sample_id:
        sample_id = _clean_text(contract_payload.get("sample_id", ""))
    if not variant:
        variant = _clean_text(contract_payload.get("variant", ""))
    if not attempt_index:
        attempt_index = _safe_int(contract_payload.get("attempt_index", 0), 0)

    implicit_cfg, _edit_cfg, hidden_cfg = _resolve_hidden_config(runtime_memory_config)
    state_cfg = dict(implicit_cfg.get("state", {}) or {})
    reset_cfg = dict(implicit_cfg.get("reset", {}) or {})
    short_window = max(1, _safe_int(state_cfg.get("short_window", 3), 3))
    mid_decay = max(0.0, min(1.0, _safe_float(state_cfg.get("mid_ema_decay", 0.7), 0.7)))
    long_max = max(4, _safe_int(state_cfg.get("long_max_prototypes", 64), 64))

    vector = _extract_summary_vector(memory_report, max_dim=max(0, _safe_int(hidden_cfg.get("max_vector_dim", 0), 0)))
    if not vector:
        return state

    metadata = state.setdefault("metadata", {})
    style_family = _clean_text(planner_payload.get("style_family", ""))
    scene_archetype = dict(planner_payload.get("scene_archetype", {}) or {})
    archetype_id = _clean_text(scene_archetype.get("archetype_id", ""))
    action = _clean_text(reflection_payload.get("action", ""))
    scores = dict(evaluation_payload.get("scores", {}) or {})
    overall_score = _safe_float(scores.get("overall_score", 0.0), 0.0)
    world_realization = _safe_float(scores.get("world_realization", 0.0), 0.0)
    failure_tags = set(_clean_list(evaluation_payload.get("failure_tags"), limit=16))

    if bool(reset_cfg.get("hard_reset_on_global_replan", True)) and action == "global_replan":
        _clear_local_short_mid(state)
    if bool(reset_cfg.get("hard_reset_on_family_change", True)):
        previous_family = _clean_text(metadata.get("last_style_family", ""))
        if previous_family and style_family and previous_family != style_family:
            _clear_local_short_mid(state)
    if bool(reset_cfg.get("hard_reset_on_archetype_change", True)):
        previous_archetype = _clean_text(metadata.get("last_archetype_id", ""))
        if previous_archetype and archetype_id and previous_archetype != archetype_id:
            _clear_local_short_mid(state)

    positive_threshold = _safe_float(hidden_cfg.get("positive_score_threshold", 0.62), 0.62)
    is_positive = bool(action == "accept" or overall_score >= positive_threshold)
    prefer_global_negative = bool(
        action == "global_replan"
        or world_realization < 0.55
        or "weak_world_realization" in failure_tags
    )

    targets: list[tuple[str, str]] = []
    if is_positive:
        targets.append(("local", "positive"))
        targets.append(("global", "positive"))
    else:
        if prefer_global_negative:
            targets.append(("global", "negative"))
        else:
            targets.append(("local", "negative"))

    for scope, polarity in targets:
        entry = _entry_payload(
            vector=vector,
            sample_id=sample_id,
            variant=variant,
            attempt_index=attempt_index,
            planner_payload=planner_payload,
            evaluation_payload=evaluation_payload,
            reflection_payload=reflection_payload,
            scope=scope,
            polarity=polarity,
        )
        _append_short_entry(
            bank=state[scope][polarity],
            entry=entry,
            short_window=short_window,
            mid_decay=mid_decay,
            long_max=long_max,
        )

    metadata["last_style_family"] = style_family
    metadata["last_archetype_id"] = archetype_id
    metadata["last_action"] = action
    metadata["updates"] = _safe_int(metadata.get("updates", 0), 0) + 1
    return state


def _collect_entries(
    *,
    state: dict,
    scope: str,
    polarity: str,
    scales: list[str],
    scale_weights: dict,
    max_entries: int,
    max_vector_dim: int,
) -> list[dict]:
    payload = state.get(scope, {}).get(polarity, {})
    out: list[dict] = []
    for scale in scales:
        entries = [dict(item) for item in list(payload.get(scale, []) or []) if isinstance(item, dict)]
        for entry in reversed(entries):
            vector = _clean_vector(entry.get("vector"), max_dim=max_vector_dim)
            if not vector:
                continue
            weight = max(0.0, _safe_float(entry.get("weight", 1.0), 1.0))
            scale_weight = max(0.0, _safe_float(scale_weights.get(scale, 1.0), 1.0))
            merged = dict(entry)
            merged["vector"] = vector
            merged["weight"] = weight * scale_weight
            merged["scope"] = scope
            merged["polarity"] = polarity
            merged["scale"] = scale
            out.append(merged)
            if len(out) >= max_entries:
                return out
    return out


def build_edit_hidden_memory_controls(
    *,
    edit_hidden_state: dict | None,
    runtime_memory_config: dict | None,
    critic_implicit_state: dict | None,
    planner_payload: dict | None,
    attempt_adaptation_state: dict | None,
) -> dict:
    state = _ensure_state(edit_hidden_state)
    critic_state = dict(critic_implicit_state or {})
    planner_payload = dict(planner_payload or {})
    adaptation = dict(attempt_adaptation_state or {})

    _implicit_cfg, edit_cfg, hidden_cfg = _resolve_hidden_config(runtime_memory_config)
    if not bool(hidden_cfg.get("enabled", True)):
        return {}

    last_action = _clean_text(critic_state.get("last_action", ""))
    route_scope = "local"
    if last_action == "global_replan":
        route_scope = "global"
    elif last_action == "local_refine_prompt":
        route_scope = "local"
    elif "weak_world_realization" in set(_clean_list(adaptation.get("last_failure_hints"), limit=10)):
        route_scope = "global"

    max_entries = max(1, _safe_int(hidden_cfg.get("max_entries_per_bank", 8), 8))
    max_vector_dim = max(0, _safe_int(hidden_cfg.get("max_vector_dim", 0), 0))
    scale_weights = dict(hidden_cfg.get("scale_weights", {}) or {})
    include_global_fallback = bool(hidden_cfg.get("include_global_positive_fallback", True))

    if route_scope == "global":
        positive_entries = _collect_entries(
            state=state,
            scope="global",
            polarity="positive",
            scales=["short", "mid", "long"],
            scale_weights=scale_weights,
            max_entries=max_entries,
            max_vector_dim=max_vector_dim,
        )
        negative_entries = _collect_entries(
            state=state,
            scope="global",
            polarity="negative",
            scales=["short", "mid", "long"],
            scale_weights=scale_weights,
            max_entries=max_entries,
            max_vector_dim=max_vector_dim,
        )
    else:
        positive_entries = _collect_entries(
            state=state,
            scope="local",
            polarity="positive",
            scales=["short", "mid"],
            scale_weights=scale_weights,
            max_entries=max_entries,
            max_vector_dim=max_vector_dim,
        )
        negative_entries = _collect_entries(
            state=state,
            scope="local",
            polarity="negative",
            scales=["short", "mid"],
            scale_weights=scale_weights,
            max_entries=max_entries,
            max_vector_dim=max_vector_dim,
        )
        if include_global_fallback and len(positive_entries) < max_entries:
            fallback = _collect_entries(
                state=state,
                scope="global",
                polarity="positive",
                scales=["mid", "long"],
                scale_weights=scale_weights,
                max_entries=max_entries - len(positive_entries),
                max_vector_dim=max_vector_dim,
            )
            positive_entries.extend(fallback)

    if not positive_entries and not negative_entries:
        return {}

    controls = {
        "latent_memory": {
            "enabled": True,
            "method": "prototype_residual",
            "scope": route_scope,
            "block_indices": list(hidden_cfg.get("block_indices", []) or []),
            "capture_block_index": _safe_int(hidden_cfg.get("capture_block_index", -1), -1),
            "capture_ema_decay": _safe_float(hidden_cfg.get("capture_ema_decay", 0.8), 0.8),
            "residual_scale": _safe_float(hidden_cfg.get("residual_scale", 0.04), 0.04),
            "positive_weight": _safe_float(hidden_cfg.get("positive_weight", edit_cfg.get("preserve_weight", 1.0)), 1.0),
            "negative_weight": _safe_float(hidden_cfg.get("negative_weight", edit_cfg.get("repair_weight", 1.0)), 1.0),
            "style_weight": _safe_float(hidden_cfg.get("style_weight", edit_cfg.get("style_weight", 1.0)), 1.0),
            "positive_topk": max(1, _safe_int(hidden_cfg.get("positive_topk", 2), 2)),
            "negative_topk": max(1, _safe_int(hidden_cfg.get("negative_topk", 2), 2)),
            "similarity_floor": _safe_float(hidden_cfg.get("similarity_floor", 0.05), 0.05),
            "max_vector_dim": max_vector_dim,
            "banks": {
                "positive": positive_entries,
                "negative": negative_entries,
            },
            "routing": {
                "last_action": last_action,
                "style_family": _clean_text(planner_payload.get("style_family", "")),
                "selected_strategy": _clean_text(planner_payload.get("selected_strategy", "")),
            },
        }
    }
    return controls
