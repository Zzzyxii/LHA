from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from memory.config import load_memory_policy
from memory.evolver import _execution_summary_records
from memory.store import ensure_memory_schema


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_days(item: dict, *, now: datetime) -> float:
    for key in ["last_seen", "first_seen"]:
        parsed = _parse_datetime(item.get(key, ""))
        if parsed is not None:
            return max((now - parsed).total_seconds() / 86400.0, 0.0)
    return 0.0


def _metric_float(item: dict, key: str, default: float = 0.0) -> float:
    return float(item.get(key, default) or default)


def _metric_int(item: dict, key: str, fallback_key: str = "", default: int = 0) -> int:
    if fallback_key:
        return int(item.get(key, item.get(fallback_key, default)) or default)
    return int(item.get(key, default) or default)


def _cleanup_policy(policy: dict | None = None) -> dict:
    memory_policy = policy or load_memory_policy()
    return (memory_policy.get("cleanup", {}) or {})


def _retention_policy(policy: dict | None = None) -> dict:
    return (_cleanup_policy(policy).get("retention_score", {}) or {})


def _retention_score(item: dict, *, policy: dict | None = None) -> float:
    retention = _retention_policy(policy)
    sample_count = _metric_int(item, "sample_count", fallback_key="count", default=0)
    recency_weight = _metric_float(item, "recency_weight", 0.0)
    success_rate = _metric_float(item, "success_rate", 0.0)
    quality_score = _metric_float(item, "quality_score", 0.0)
    evidence_score = _metric_float(item, "evidence_score", 0.0)
    source_quality = _metric_float(item, "source_quality", 0.0)
    accepted_bonus = float(retention.get("accepted_bonus", 0.05) or 0.05) if bool(item.get("accepted", False)) else 0.0
    recency_weight_divisor = max(float(retention.get("recency_weight_divisor", 4.0) or 4.0), 1e-6)
    recency_weight_cap = float(retention.get("recency_weight_cap", 1.0) or 1.0)
    sample_count_divisor = max(float(retention.get("sample_count_divisor", 10.0) or 10.0), 1e-6)
    sample_count_cap = float(retention.get("sample_count_cap", 1.0) or 1.0)
    return round(
        quality_score * float(retention.get("quality_score_weight", 0.35) or 0.35)
        + success_rate * float(retention.get("success_rate_weight", 0.2) or 0.2)
        + evidence_score * float(retention.get("evidence_score_weight", 0.15) or 0.15)
        + min(recency_weight / recency_weight_divisor, recency_weight_cap) * float(retention.get("recency_weight_weight", 0.15) or 0.15)
        + min(sample_count / sample_count_divisor, sample_count_cap) * float(retention.get("sample_count_weight", 0.1) or 0.1)
        + source_quality * float(retention.get("source_quality_weight", 0.05) or 0.05)
        + accepted_bonus,
        6,
    )


def _passes_thresholds(item: dict, rule: dict, *, age_days: float) -> bool:
    stale_after_days = float(rule.get("stale_after_days", 0.0) or 0.0)
    if stale_after_days > 0 and age_days > stale_after_days:
        return False
    if _metric_float(item, "recency_weight", 0.0) < float(rule.get("min_recency_weight", 0.0) or 0.0):
        return False
    if _metric_int(item, "sample_count", fallback_key="count", default=0) < int(rule.get("min_sample_count", 0) or 0):
        return False
    if _metric_float(item, "success_rate", 0.0) < float(rule.get("min_success_rate", 0.0) or 0.0):
        return False
    if _metric_float(item, "quality_score", 0.0) < float(rule.get("min_quality_score", 0.0) or 0.0):
        return False
    if _metric_float(item, "evidence_score", 0.0) < float(rule.get("min_evidence_score", 0.0) or 0.0):
        return False
    return True


def _cleanup_collection(items: list[dict], *, name: str, rules: dict, now: datetime, policy: dict | None = None) -> list[dict]:
    rule = (rules.get(name, {}) or {})
    if not rule:
        return [deepcopy(item) for item in items if isinstance(item, dict)]

    scored: list[tuple[float, float, int, dict]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        age_days = _age_days(item, now=now)
        scored.append((_retention_score(item, policy=policy), -age_days, -index, deepcopy(item)))
    scored.sort(reverse=True)

    keep_at_least = max(int(rule.get("keep_at_least", 0) or 0), 0)
    kept: list[tuple[float, float, int, dict]] = []
    for idx, entry in enumerate(scored):
        score, negative_age, negative_index, item = entry
        age_days = -negative_age
        if idx < keep_at_least or _passes_thresholds(item, rule, age_days=age_days):
            kept.append((score, negative_age, negative_index, item))

    max_items = max(int(rule.get("max_items", 0) or 0), 0)
    if max_items > 0:
        kept = kept[:max_items]

    kept.sort(key=lambda entry: entry[2], reverse=True)
    return [item for _, _, _, item in kept]


def cleanup_memory(
    memory: dict,
    *,
    config_path: str | None = None,
    now: datetime | None = None,
) -> dict:
    payload = ensure_memory_schema(memory)
    policy = load_memory_policy(config_path)
    cleanup_rules = (((policy.get("cleanup", {}) or {}).get("collections", {}) or {}))
    cleanup_now = now or _utc_now()

    cleaned = {
        "schema_version": payload.get("schema_version", 1),
        "scene_archetypes": _cleanup_collection(payload.get("scene_archetypes", []), name="scene_archetypes", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "style_convertibility_priors": _cleanup_collection(payload.get("style_convertibility_priors", []), name="style_convertibility_priors", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "model_behavior_priors": _cleanup_collection(payload.get("model_behavior_priors", []), name="model_behavior_priors", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "failure_patterns": _cleanup_collection(payload.get("failure_patterns", []), name="failure_patterns", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "prompt_compilation_recipes": _cleanup_collection(payload.get("prompt_compilation_recipes", []), name="prompt_compilation_recipes", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "custom_skill_records": _cleanup_collection(payload.get("custom_skill_records", []), name="custom_skill_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "runtime_skill_records": _cleanup_collection(payload.get("runtime_skill_records", []), name="runtime_skill_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "promotable_skill_records": _cleanup_collection(payload.get("promotable_skill_records", []), name="promotable_skill_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "quarantined_skill_records": _cleanup_collection(payload.get("quarantined_skill_records", []), name="quarantined_skill_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "skill_execution_records": _cleanup_collection(payload.get("skill_execution_records", []), name="skill_execution_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "run_records": _cleanup_collection(payload.get("run_records", []), name="run_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "evaluation_records": _cleanup_collection(payload.get("evaluation_records", []), name="evaluation_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
        "evolution_records": _cleanup_collection(payload.get("evolution_records", []), name="evolution_records", rules=cleanup_rules, now=cleanup_now, policy=policy),
    }
    regenerated_summaries = _execution_summary_records(cleaned.get("skill_execution_records", []), policy=policy)
    cleaned["skill_execution_summaries"] = _cleanup_collection(
        regenerated_summaries,
        name="skill_execution_summaries",
        rules=cleanup_rules,
        now=cleanup_now,
        policy=policy,
    )
    return ensure_memory_schema(cleaned)
