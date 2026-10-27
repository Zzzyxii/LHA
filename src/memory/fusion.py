from __future__ import annotations

from copy import deepcopy

from memory.implicit_state import ImplicitBias
from memory.runtime_modes import MemoryMode, normalize_memory_mode


_NUMERIC_BIAS_FIELDS = (
    "preferred_skills",
    "penalized_skills",
    "preferred_carriers",
    "penalized_carriers",
    "candidate_bias",
    "variant_bias",
    "routing_bias",
)


def _coerce_bias_payload(payload: dict | ImplicitBias | None) -> dict:
    if isinstance(payload, ImplicitBias):
        return payload.to_dict()
    if isinstance(payload, dict):
        return deepcopy(payload)
    return {}


def _weighted_merge_maps(
    explicit_map: dict | None,
    implicit_map: dict | None,
    *,
    explicit_weight: float,
    implicit_weight: float,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for source_map, weight in ((explicit_map or {}, explicit_weight), (implicit_map or {}, implicit_weight)):
        for key, value in dict(source_map).items():
            merged[str(key)] = float(merged.get(str(key), 0.0)) + float(value) * float(weight)
    return merged


def _merge_controls(
    explicit_controls: dict | None,
    implicit_controls: dict | None,
    *,
    conflict_policy: str,
) -> dict:
    result = dict(explicit_controls or {})
    if conflict_policy == "explicit_prefer":
        for key, value in dict(implicit_controls or {}).items():
            result.setdefault(str(key), value)
        return result
    for key, value in dict(implicit_controls or {}).items():
        if conflict_policy == "sum" and isinstance(result.get(key), (int, float)) and isinstance(value, (int, float)):
            result[str(key)] = float(result.get(key, 0.0)) + float(value)
        else:
            result[str(key)] = value
    return result


def fuse_memory_biases(
    explicit_bias: dict | ImplicitBias | None,
    implicit_bias: dict | ImplicitBias | None,
    *,
    mode: MemoryMode | str = MemoryMode.HYBRID,
    explicit_weight: float = 1.0,
    implicit_weight: float = 1.0,
    conflict_policy: str = "sum",
) -> dict:
    normalized_mode = normalize_memory_mode(mode, default=MemoryMode.HYBRID)
    explicit_payload = _coerce_bias_payload(explicit_bias)
    implicit_payload = _coerce_bias_payload(implicit_bias)
    if normalized_mode == MemoryMode.NONE:
        return {"mode": normalized_mode.value}
    if normalized_mode == MemoryMode.EXPLICIT:
        explicit_payload.setdefault("mode", normalized_mode.value)
        return explicit_payload
    if normalized_mode == MemoryMode.IMPLICIT:
        implicit_payload.setdefault("mode", normalized_mode.value)
        return implicit_payload

    fused = {"mode": normalized_mode.value}
    for field in _NUMERIC_BIAS_FIELDS:
        fused[field] = _weighted_merge_maps(
            explicit_payload.get(field, {}),
            implicit_payload.get(field, {}),
            explicit_weight=explicit_weight,
            implicit_weight=implicit_weight,
        )
    fused["edit_controls"] = _merge_controls(
        explicit_payload.get("edit_controls", {}),
        implicit_payload.get("edit_controls", {}),
        conflict_policy=str(conflict_policy or "sum").strip().lower(),
    )
    fused["scope"] = str(implicit_payload.get("scope") or explicit_payload.get("scope") or "hybrid")
    fused["metadata"] = {
        "explicit_weight": float(explicit_weight),
        "implicit_weight": float(implicit_weight),
        "conflict_policy": str(conflict_policy or "sum"),
        "explicit_metadata": dict(explicit_payload.get("metadata", {}) or {}),
        "implicit_metadata": dict(implicit_payload.get("metadata", {}) or {}),
    }
    return fused
