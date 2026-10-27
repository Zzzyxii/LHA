from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from memory.analyzer import MemoryAnalysisResult
from memory.config import load_memory_policy
from memory.store import (
    append_evaluation_record,
    append_evolution_record,
    append_run_record,
    default_global_memory,
    ensure_memory_schema,
)


def _upsert_by_key(items: list[dict], record: dict, *, key: str, policy: dict | None = None) -> list[dict]:
    updated = []
    matched = False
    target = str(record.get(key, ""))
    seen_at = _utc_now()
    memory_policy = policy or _memory_policy()
    recency_decay = _recency_decay(memory_policy)
    for item in items:
        if str(item.get(key, "")) == target and target:
            merged = dict(item)
            merged.update(record)
            count = int(item.get("count", 1) or 1)
            merged["count"] = count + 1
            merged["sample_count"] = int(item.get("sample_count", count) or count) + 1
            merged["last_seen"] = seen_at
            merged["first_seen"] = str(item.get("first_seen", "")).strip() or seen_at
            merged["recency_weight"] = round(float(item.get("recency_weight", 0.0) or 0.0) * recency_decay + 1.0, 4)
            previous_evidence = float(item.get("evidence_score", 0.0) or 0.0)
            current_evidence = float(record.get("evidence_score", previous_evidence) or previous_evidence)
            merged["evidence_score"] = round(((previous_evidence * count) + current_evidence) / max(count + 1, 1), 4)
            updated.append(merged)
            matched = True
        else:
            updated.append(item)
    if not matched:
        fresh = dict(record)
        fresh.setdefault("count", 1)
        fresh.setdefault("sample_count", int(fresh.get("count", 1) or 1))
        fresh.setdefault("first_seen", seen_at)
        fresh.setdefault("last_seen", seen_at)
        fresh.setdefault("recency_weight", 1.0)
        updated.append(fresh)
    return updated


def _remove_by_key(items: list[dict], target_value: str, *, key: str) -> list[dict]:
    target = str(target_value or "").strip()
    if not target:
        return list(items)
    return [item for item in items if str(item.get(key, "")).strip() != target]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _memory_policy(config_path: str | None = None) -> dict:
    return load_memory_policy(config_path)


def _recency_decay(policy: dict) -> float:
    return float(((policy.get("stats", {}) or {}).get("recency_decay", 0.85)) or 0.85)


def _source_quality(source: str, *, policy: dict | None = None) -> float:
    memory_policy = policy or _memory_policy()
    lookup = (memory_policy.get("source_quality", {}) or {})
    return float(lookup.get(str(source or "").strip(), lookup.get("default", 0.75)) or 0.75)


def _execution_bias_policy(policy: dict | None = None) -> dict:
    memory_policy = policy or _memory_policy()
    return ((memory_policy.get("retrieval", {}) or {}).get("execution_bias", {}) or {})


def _skill_failure_tags(record: dict) -> list[str]:
    top_level = record.get("failure_tags")
    if isinstance(top_level, list):
        return [str(item).strip() for item in top_level if str(item).strip()]
    judgment = record.get("judgment", {}) if isinstance(record.get("judgment", {}), dict) else {}
    return [str(item).strip() for item in (judgment.get("failure_tags") or []) if str(item).strip()]


def _hold_violation_dims(record: dict) -> list[str]:
    """Dimensions the critic said were supposed to be kept but were changed.

    Read off the ``hold_violated:<dimension>`` tags. Unreported dimensions are
    absent by construction -- see ``critic.evaluator`` -- so this is never
    inflated by a terse reply.
    """
    tags = _skill_failure_tags(record)
    dims = [
        tag.split(":", 1)[1].strip()
        for tag in tags
        if tag.startswith("hold_violated:") and tag.split(":", 1)[1].strip()
    ]
    return sorted(set(dims))


def _hold_check_dims(record: dict) -> list[str]:
    """Dimensions the critic actually reported on (held or violated)."""
    checks = record.get("hold_checks")
    if not isinstance(checks, list):
        return []
    return sorted(
        {
            str(row.get("dimension", "")).strip()
            for row in checks
            if isinstance(row, dict)
            and row.get("reported", True)
            and str(row.get("dimension", "")).strip()
        }
    )


def _record_strategy(record: dict) -> str:
    for key in ("selected_strategy", "strategy_id", "selected_skill_ids", "source"):
        value = record.get(key)
        if isinstance(value, list):
            value = value[0] if value else ""
        text = str(value or "").strip()
        if text:
            return text
    return ""


def hold_violation_summaries(records: list[dict]) -> list[dict]:
    """Per-(strategy, dimension) hold-violation rates.

    The signal the skill loop had no way to produce before: not "template X is
    bad" but "template X's costume field overrides holds". A whole template can
    be right about lighting and atmosphere and still be unusable because one
    field is too strong -- without this that shows up as a mediocre overall
    score and the usual fix is to throw the template away.

    ``checks`` counts only dimensions the critic actually reported on, so the
    rate is not diluted by runs where nothing was held.
    """
    stats: dict[tuple[str, str], dict] = {}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        strategy = _record_strategy(record)
        if not strategy:
            continue
        for dimension in _hold_check_dims(record):
            entry = stats.setdefault(
                (strategy, dimension),
                {"strategy_id": strategy, "dimension": dimension, "violations": 0, "checks": 0},
            )
            entry["checks"] += 1
        for dimension in _hold_violation_dims(record):
            entry = stats.setdefault(
                (strategy, dimension),
                {"strategy_id": strategy, "dimension": dimension, "violations": 0, "checks": 0},
            )
            entry["violations"] += 1

    summaries = []
    for entry in stats.values():
        checks = max(entry["checks"], entry["violations"])
        entry["checks"] = checks
        entry["rate"] = round(entry["violations"] / checks, 4) if checks else 0.0
        summaries.append(entry)
    return sorted(
        summaries,
        key=lambda item: (-item["rate"], -item["violations"], item["strategy_id"], item["dimension"]),
    )


def _skill_quality_score(record: dict) -> float:
    judgment = record.get("judgment", {}) if isinstance(record.get("judgment", {}), dict) else {}
    overall_score = float(record.get("overall_score", 0.0) or 0.0)
    accepted = bool(record.get("accepted", False) or judgment.get("accepted"))
    carrier_grounding = float(judgment.get("carrier_grounding", 0.0) or 0.0)
    transformation_strength = float(judgment.get("transformation_strength", 0.0) or 0.0)
    world_realization = float(judgment.get("world_realization", 0.0) or 0.0)
    failure_tags = set(_skill_failure_tags(record))

    score = (
        overall_score * 0.55
        + (0.25 if accepted else 0.0)
        + carrier_grounding * 0.1
        + transformation_strength * 0.05
        + world_realization * 0.05
    )
    if "identity_drift" in failure_tags:
        score -= 0.2
    if "artifact_heavy" in failure_tags:
        score -= 0.15
    if "detached_fx" in failure_tags:
        score -= 0.12
    if "weak_world_realization" in failure_tags:
        score -= 0.06
    if "under_transformation" in failure_tags:
        score -= 0.04
    # A hold violation is the edit model changing something it was explicitly
    # told to leave alone. As expensive as heavy artifacts: the instruction was
    # understood and then overridden.
    if _hold_violation_dims(record):
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 4)


def _annotate_skill_record(
    record: dict,
    *,
    previous: dict | None = None,
    quarantined: bool = False,
    policy: dict | None = None,
) -> dict:
    seen_at = _utc_now()
    memory_policy = policy or _memory_policy()
    current_sample_count = int(previous.get("sample_count", previous.get("count", 0)) or 0) if previous else 0
    sample_count = current_sample_count + 1
    accepted = bool(record.get("accepted", False))
    accepted_count = int(previous.get("accepted_count", 0) or 0) + (1 if accepted else 0) if previous else (1 if accepted else 0)
    failure_tags = _skill_failure_tags(record)
    failure_count = int(previous.get("failure_count", 0) or 0) + (1 if failure_tags else 0) if previous else (1 if failure_tags else 0)
    quarantine_count = int(previous.get("quarantine_count", 0) or 0) + (1 if quarantined else 0) if previous else (1 if quarantined else 0)
    previous_recency = float(previous.get("recency_weight", 0.0) or 0.0) if previous else 0.0
    current_quality = _skill_quality_score(record)
    current_source_quality = _source_quality(str(record.get("source", "")).strip(), policy=memory_policy)
    previous_quality = float(previous.get("quality_score", 0.0) or 0.0) if previous else 0.0
    previous_source_quality = float(previous.get("source_quality", 0.0) or 0.0) if previous else 0.0
    previous_weight = max(sample_count - 1, 0)
    recency_decay = _recency_decay(memory_policy)

    first_seen = str(previous.get("first_seen", "")).strip() if previous else ""
    annotated = dict(previous or {})
    annotated.update(deepcopy(record))
    annotated.update(
        {
            "count": sample_count,
            "sample_count": sample_count,
            "accepted_count": accepted_count,
            "failure_count": failure_count,
            "quarantine_count": quarantine_count,
            "success_rate": round(accepted_count / max(sample_count, 1), 4),
            "last_seen": seen_at,
            "first_seen": first_seen or seen_at,
            "recency_weight": round(previous_recency * recency_decay + 1.0, 4),
            "source_quality": round(
                ((previous_source_quality * previous_weight) + current_source_quality) / max(sample_count, 1),
                4,
            ),
            "quality_score": round(
                ((previous_quality * previous_weight) + current_quality) / max(sample_count, 1),
                4,
            ),
            "evidence_score": round(
                min(
                    1.0,
                    (
                        ((previous_quality * previous_weight) + current_quality) / max(sample_count, 1)
                    )
                    * 0.7
                    + (accepted_count / max(sample_count, 1)) * 0.3,
                ),
                4,
            ),
            "failure_tags": failure_tags[:8],
        }
    )
    return annotated


def _upsert_skill_record_bucket(
    items: list[dict],
    record: dict,
    *,
    key: str,
    quarantined: bool = False,
    policy: dict | None = None,
) -> list[dict]:
    updated = []
    matched = False
    target = str(record.get(key, "")).strip()
    for item in items:
        if str(item.get(key, "")).strip() == target and target:
            updated.append(_annotate_skill_record(record, previous=item, quarantined=quarantined, policy=policy))
            matched = True
        else:
            updated.append(item)
    if not matched:
        updated.append(_annotate_skill_record(record, quarantined=quarantined, policy=policy))
    return updated


def _promotion_bucket(record: dict, *, policy: dict | None = None) -> str:
    memory_policy = policy or _memory_policy()
    if bool(record.get("memory_only", False)):
        return "runtime"

    judgment = record.get("judgment", {}) if isinstance(record.get("judgment", {}), dict) else {}
    promotable_policy = ((memory_policy.get("promotion", {}) or {}).get("promotable", {}) or {})
    overall_score = float(record.get("overall_score", 0.0) or 0.0)
    accepted = bool(record.get("accepted", False) or judgment.get("accepted"))
    carrier_grounding = float(judgment.get("carrier_grounding", 0.0) or 0.0)
    failure_tags = set(_skill_failure_tags(record))
    severe_failures = {str(item).strip() for item in (promotable_policy.get("blocked_failure_tags") or []) if str(item).strip()}
    quality_score = _skill_quality_score(record)
    require_accepted = bool(promotable_policy.get("require_accepted", True))
    min_overall_score = float(promotable_policy.get("min_overall_score", 0.72) or 0.72)
    min_carrier_grounding = float(promotable_policy.get("min_carrier_grounding", 0.5) or 0.5)
    min_quality_score = float(promotable_policy.get("min_quality_score", 0.62) or 0.62)

    if (
        (accepted if require_accepted else True)
        and overall_score >= min_overall_score
        and carrier_grounding >= min_carrier_grounding
        and not failure_tags.intersection(severe_failures)
        and quality_score >= min_quality_score
    ):
        return "promotable"
    return "quarantine"


def _normalize_aux_record(record: dict, *, policy: dict) -> dict:
    count = int(record.get("sample_count", record.get("count", 1)) or 1)
    first_seen = str(record.get("first_seen", "")).strip()
    last_seen = str(record.get("last_seen", "")).strip()
    normalized = dict(record)
    normalized["count"] = max(int(record.get("count", count) or count), count)
    normalized["sample_count"] = count
    normalized["first_seen"] = first_seen or _utc_now()
    normalized["last_seen"] = last_seen or normalized["first_seen"]
    normalized["recency_weight"] = float(record.get("recency_weight", 1.0) or 1.0)
    normalized["evidence_score"] = float(record.get("evidence_score", 0.0) or 0.0)
    return normalized


def _best_non_empty(current: object, incoming: object) -> object:
    if current in ("", [], {}, None):
        return deepcopy(incoming)
    if incoming in ("", [], {}, None):
        return deepcopy(current)
    if isinstance(current, list) and isinstance(incoming, list):
        return list(dict.fromkeys(list(current) + list(incoming)))
    return deepcopy(incoming)


def _merge_skill_seed(existing: dict, incoming: dict) -> dict:
    if not existing:
        return deepcopy(incoming)
    merged = dict(existing)
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = deepcopy(value)
            continue
        merged[key] = _best_non_empty(merged.get(key), value)
    return merged


def _normalize_skill_seed(record: dict, *, policy: dict, bucket_hint: str = "") -> dict:
    normalized = deepcopy(record)
    count = int(normalized.get("sample_count", normalized.get("count", 1)) or 1)
    accepted = bool(normalized.get("accepted", False))
    accepted_count = int(normalized.get("accepted_count", count if accepted else 0) or (count if accepted else 0))
    failure_tags = _skill_failure_tags(normalized)
    failure_count = int(normalized.get("failure_count", count if failure_tags else 0) or (count if failure_tags else 0))
    first_seen = str(normalized.get("first_seen", "")).strip() or _utc_now()
    last_seen = str(normalized.get("last_seen", "")).strip() or first_seen
    quality_score = float(normalized.get("quality_score", _skill_quality_score(normalized)) or 0.0)
    source_quality = float(normalized.get("source_quality", _source_quality(str(normalized.get("source", "")).strip(), policy=policy)) or 0.75)
    quarantine_count = int(
        normalized.get(
            "quarantine_count",
            count if bucket_hint == "quarantine" else 0,
        )
        or (count if bucket_hint == "quarantine" else 0)
    )
    normalized.update(
        {
            "count": count,
            "sample_count": count,
            "accepted_count": min(max(accepted_count, 0), count),
            "failure_count": min(max(failure_count, 0), count),
            "quarantine_count": min(max(quarantine_count, 0), count),
            "success_rate": round(min(max(accepted_count, 0), count) / max(count, 1), 4),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "recency_weight": float(normalized.get("recency_weight", 1.0) or 1.0),
            "source_quality": source_quality,
            "quality_score": quality_score,
            "evidence_score": float(
                normalized.get(
                    "evidence_score",
                    min(1.0, quality_score * 0.7 + (min(max(accepted_count, 0), count) / max(count, 1)) * 0.3),
                )
                or 0.0
            ),
            "failure_tags": failure_tags[:8],
        }
    )
    return normalized


def _normalize_execution_record(record: dict, *, policy: dict) -> dict:
    normalized = deepcopy(record)
    judgment = normalized.get("judgment", {}) if isinstance(normalized.get("judgment", {}), dict) else {}
    normalized["sample_count"] = int(normalized.get("sample_count", 1) or 1)
    normalized["count"] = int(normalized.get("count", normalized["sample_count"]) or normalized["sample_count"])
    normalized["accepted"] = bool(normalized.get("accepted", False) or judgment.get("accepted"))
    normalized["failure_tags"] = _skill_failure_tags(normalized)
    normalized["quality_score"] = float(normalized.get("quality_score", _skill_quality_score(normalized)) or 0.0)
    normalized["evidence_score"] = float(normalized.get("evidence_score", normalized["quality_score"]) or normalized["quality_score"])
    normalized["source_quality"] = float(
        normalized.get("source_quality", _source_quality(str(normalized.get("source", "")).strip(), policy=policy)) or 0.75
    )
    first_seen = str(normalized.get("first_seen", "")).strip()
    last_seen = str(normalized.get("last_seen", "")).strip()
    normalized["first_seen"] = first_seen or last_seen or _utc_now()
    normalized["last_seen"] = last_seen or normalized["first_seen"]
    normalized["recency_weight"] = float(normalized.get("recency_weight", 1.0) or 1.0)
    return normalized


def _execution_summary_records(records: list[dict], *, policy: dict | None = None) -> list[dict]:
    execution_bias = _execution_bias_policy(policy)
    buckets: dict[tuple[str, str], dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        style_family = str(record.get("style_family", "")).strip()
        archetype_id = str(record.get("archetype_id", "")).strip()
        if not style_family:
            continue
        key = (style_family, archetype_id)
        bucket = buckets.setdefault(
            key,
            {
                "style_family": style_family,
                "archetype_id": archetype_id,
                "carrier_scores": {},
                "carrier_counts": {},
                "skill_scores": {},
                "skill_counts": {},
                "execution_count": 0,
                "accepted_count": 0,
                "sample_ids": set(),
                "last_seen": "",
                "recency_weight_sum": 0.0,
                "source_quality_sum": 0.0,
                "quality_score_sum": 0.0,
            },
        )
        judgment = record.get("judgment", {}) if isinstance(record.get("judgment", {}), dict) else {}
        carriers = list(record.get("recommended_carriers") or [])
        skill_id = str(record.get("skill_id", "")).strip()
        accepted = 1.0 if judgment.get("accepted") else 0.0
        carrier_grounding = float(judgment.get("carrier_grounding", 0.0) or 0.0)
        transformation = float(judgment.get("transformation_strength", 0.0) or 0.0)
        world_realization = float(judgment.get("world_realization", 0.0) or 0.0)
        failure_tags = {str(tag).strip() for tag in (judgment.get("failure_tags") or []) if str(tag).strip()}
        signal = (
            accepted * 0.9
            + float(record.get("overall_score", 0.0) or 0.0) * 0.6
            + carrier_grounding * 0.7
            + transformation * 0.2
            + world_realization * 0.2
        )
        if "detached_fx" in failure_tags:
            signal -= 0.9
        if carrier_grounding < 0.45:
            signal -= 0.4
        if skill_id:
            bucket["skill_scores"][skill_id] = bucket["skill_scores"].get(skill_id, 0.0) + signal
            bucket["skill_counts"][skill_id] = bucket["skill_counts"].get(skill_id, 0) + 1
        for carrier in carriers[:6]:
            carrier = str(carrier).strip().lower()
            if not carrier:
                continue
            bucket["carrier_scores"][carrier] = bucket["carrier_scores"].get(carrier, 0.0) + signal
            bucket["carrier_counts"][carrier] = bucket["carrier_counts"].get(carrier, 0) + 1
        bucket["execution_count"] += 1
        bucket["accepted_count"] += int(accepted)
        sample_id = str(record.get("sample_id", "")).strip()
        if sample_id:
            bucket["sample_ids"].add(sample_id)
        last_seen = str(record.get("last_seen", "")).strip()
        if last_seen and last_seen > str(bucket.get("last_seen", "")):
            bucket["last_seen"] = last_seen
        bucket["recency_weight_sum"] += float(record.get("recency_weight", 1.0) or 1.0)
        bucket["source_quality_sum"] += float(record.get("source_quality", 0.75) or 0.75)
        bucket["quality_score_sum"] += float(record.get("quality_score", 0.0) or 0.0)

    summaries: list[dict] = []
    for (style_family, archetype_id), bucket in buckets.items():
        avg_skill_scores = {
            key: value / max(bucket["skill_counts"].get(key, 1), 1)
            for key, value in bucket["skill_scores"].items()
        }
        avg_carrier_scores = {
            key: value / max(bucket["carrier_counts"].get(key, 1), 1)
            for key, value in bucket["carrier_scores"].items()
        }
        preferred_threshold = float(execution_bias.get("preferred_carrier_threshold", 0.35) or 0.35)
        forbidden_threshold = float(execution_bias.get("forbidden_carrier_threshold", -0.1) or -0.1)
        boosted_threshold = float(execution_bias.get("boosted_skill_threshold", 0.25) or 0.25)
        penalized_threshold = float(execution_bias.get("penalized_skill_threshold", -0.05) or -0.05)
        preferred_carriers = [key for key, value in sorted(avg_carrier_scores.items(), key=lambda pair: pair[1], reverse=True) if value > preferred_threshold][:5]
        forbidden_carriers = [key for key, value in sorted(avg_carrier_scores.items(), key=lambda pair: pair[1]) if value < forbidden_threshold][:5]
        boosted_skill_ids = [key for key, value in sorted(avg_skill_scores.items(), key=lambda pair: pair[1], reverse=True) if value > boosted_threshold][:5]
        penalized_skill_ids = [key for key, value in sorted(avg_skill_scores.items(), key=lambda pair: pair[1]) if value < penalized_threshold][:5]
        summaries.append(
            {
                "summary_key": f"{style_family}::{archetype_id}",
                "style_family": style_family,
                "archetype_id": archetype_id,
                "execution_count": bucket["execution_count"],
                "sample_count": len(bucket["sample_ids"]),
                "success_rate": round(bucket["accepted_count"] / max(bucket["execution_count"], 1), 4),
                "last_seen": str(bucket.get("last_seen", "")).strip(),
                "recency_weight": round(bucket["recency_weight_sum"] / max(bucket["execution_count"], 1), 4),
                "source_quality": round(bucket["source_quality_sum"] / max(bucket["execution_count"], 1), 4),
                "quality_score": round(bucket["quality_score_sum"] / max(bucket["execution_count"], 1), 4),
                "evidence_score": round(bucket["quality_score_sum"] / max(bucket["execution_count"], 1), 4),
                "preferred_carriers": preferred_carriers,
                "forbidden_carriers": forbidden_carriers,
                "boosted_skill_ids": boosted_skill_ids,
                "penalized_skill_ids": penalized_skill_ids,
                "prefer_environment_first": bool(preferred_carriers) and archetype_id == "world_anchor_driven_scene",
                "avoid_face_heavy_rewrite": archetype_id == "single_person_indoor_closeup" and any(
                    value < 0 for key, value in avg_skill_scores.items() if "face" in key or "portrait" in key
                ),
            }
        )
    return summaries


def rebuild_memory(memory: dict, *, config_path: str | None = None) -> dict:
    policy = _memory_policy(config_path)
    source = ensure_memory_schema(memory)
    rebuilt = default_global_memory()

    for key in ["run_records", "evaluation_records", "evolution_records"]:
        rebuilt[key] = [deepcopy(item) for item in (source.get(key) or []) if isinstance(item, dict)]

    for item in source.get("scene_archetypes", []):
        if isinstance(item, dict):
            rebuilt["scene_archetypes"] = _upsert_by_key(
                rebuilt.get("scene_archetypes", []),
                _normalize_aux_record(item, policy=policy),
                key="archetype_id",
                policy=policy,
            )

    for item in source.get("failure_patterns", []):
        if isinstance(item, dict):
            failure_key = str(item.get("failure_key", "")).strip() or f"{item.get('archetype_id', '')}::{item.get('pattern', '')}"
            rebuilt["failure_patterns"] = _upsert_by_key(
                rebuilt.get("failure_patterns", []),
                _normalize_aux_record({**item, "failure_key": failure_key}, policy=policy),
                key="failure_key",
                policy=policy,
            )

    for item in source.get("style_convertibility_priors", []):
        if isinstance(item, dict):
            prior_key = str(item.get("prior_key", "")).strip() or f"{item.get('style_family', '')}::{item.get('archetype_id', '')}"
            rebuilt["style_convertibility_priors"] = _upsert_by_key(
                rebuilt.get("style_convertibility_priors", []),
                _normalize_aux_record({**item, "prior_key": prior_key}, policy=policy),
                key="prior_key",
                policy=policy,
            )

    for item in source.get("model_behavior_priors", []):
        if isinstance(item, dict):
            prior_key = str(item.get("prior_key", "")).strip() or f"{item.get('variant', '')}::{item.get('pattern', '')}"
            rebuilt["model_behavior_priors"] = _upsert_by_key(
                rebuilt.get("model_behavior_priors", []),
                _normalize_aux_record({**item, "prior_key": prior_key}, policy=policy),
                key="prior_key",
                policy=policy,
            )

    for item in source.get("prompt_compilation_recipes", []):
        if isinstance(item, dict):
            rebuilt["prompt_compilation_recipes"] = _upsert_by_key(
                rebuilt.get("prompt_compilation_recipes", []),
                _normalize_aux_record(item, policy=policy),
                key="recipe_id",
                policy=policy,
            )

    skill_seed_sources = [
        ("custom_skill_records", ""),
        ("promotable_skill_records", "promotable"),
        ("runtime_skill_records", "runtime"),
        ("quarantined_skill_records", "quarantine"),
    ]
    skill_seeds: dict[str, dict] = {}
    skill_bucket_hints: dict[str, str] = {}
    for source_key, bucket_hint in skill_seed_sources:
        for item in source.get(source_key, []):
            if not isinstance(item, dict):
                continue
            record_id = str(item.get("record_id", "")).strip()
            if not record_id:
                continue
            skill_seeds[record_id] = _merge_skill_seed(skill_seeds.get(record_id, {}), item)
            if bucket_hint:
                skill_bucket_hints[record_id] = bucket_hint

    for record_id, seed in skill_seeds.items():
        normalized = _normalize_skill_seed(seed, policy=policy, bucket_hint=skill_bucket_hints.get(record_id, ""))
        rebuilt["custom_skill_records"].append(normalized)
        bucket = _promotion_bucket(normalized, policy=policy)
        if bucket == "runtime":
            rebuilt["runtime_skill_records"].append(deepcopy(normalized))
        elif bucket == "promotable":
            rebuilt["promotable_skill_records"].append(deepcopy(normalized))
        else:
            quarantine_record = deepcopy(normalized)
            quarantine_record["quarantine_count"] = int(
                quarantine_record.get("quarantine_count", quarantine_record.get("sample_count", 1)) or quarantine_record.get("sample_count", 1)
            )
            rebuilt["quarantined_skill_records"].append(quarantine_record)

    seen_execution_keys: set[str] = set()
    for item in source.get("skill_execution_records", []):
        if not isinstance(item, dict):
            continue
        execution_key = str(item.get("execution_key", "")).strip()
        if not execution_key or execution_key in seen_execution_keys:
            continue
        seen_execution_keys.add(execution_key)
        rebuilt["skill_execution_records"].append(_normalize_execution_record(item, policy=policy))

    rebuilt["skill_execution_summaries"] = _execution_summary_records(rebuilt.get("skill_execution_records", []), policy=policy)
    rebuilt["hold_violation_summaries"] = hold_violation_summaries(
        list(rebuilt.get("evaluation_records", [])) + list(rebuilt.get("skill_execution_records", []))
    )
    return ensure_memory_schema(rebuilt)


def evolve_memory(memory: dict, analysis: MemoryAnalysisResult, *, config_path: str | None = None) -> dict:
    policy = _memory_policy(config_path)
    payload = ensure_memory_schema(memory)
    payload = append_run_record(payload, analysis.run_record)
    payload = append_evaluation_record(payload, analysis.evaluation_record)

    payload["scene_archetypes"] = _upsert_by_key(
        payload.get("scene_archetypes", []),
        analysis.scene_archetype,
        key="archetype_id",
        policy=policy,
    )

    for item in analysis.failure_patterns:
        failure_key = f"{item.get('archetype_id', '')}::{item.get('pattern', '')}"
        payload["failure_patterns"] = _upsert_by_key(
            payload.get("failure_patterns", []),
            {**item, "failure_key": failure_key},
            key="failure_key",
            policy=policy,
        )

    for item in analysis.style_convertibility_priors:
        prior_key = f"{item.get('style_family', '')}::{item.get('archetype_id', '')}"
        payload["style_convertibility_priors"] = _upsert_by_key(
            payload.get("style_convertibility_priors", []),
            {**item, "prior_key": prior_key},
            key="prior_key",
            policy=policy,
        )

    for item in analysis.model_behavior_priors:
        prior_key = f"{item.get('variant', '')}::{item.get('pattern', '')}"
        payload["model_behavior_priors"] = _upsert_by_key(
            payload.get("model_behavior_priors", []),
            {**item, "prior_key": prior_key},
            key="prior_key",
            policy=policy,
        )

    for item in analysis.prompt_compilation_recipe_candidates:
        payload["prompt_compilation_recipes"] = _upsert_by_key(
            payload.get("prompt_compilation_recipes", []),
            deepcopy(item),
            key="recipe_id",
            policy=policy,
        )

    for item in analysis.custom_skill_records:
        payload["custom_skill_records"] = _upsert_skill_record_bucket(
            payload.get("custom_skill_records", []),
            deepcopy(item),
            key="record_id",
            policy=policy,
        )
        bucket = _promotion_bucket(item, policy=policy)
        record_id = str(item.get("record_id", "")).strip()
        if bucket != "promotable":
            payload["promotable_skill_records"] = _remove_by_key(payload.get("promotable_skill_records", []), record_id, key="record_id")
        if bucket != "quarantine":
            payload["quarantined_skill_records"] = _remove_by_key(payload.get("quarantined_skill_records", []), record_id, key="record_id")
        if bucket != "runtime":
            payload["runtime_skill_records"] = _remove_by_key(payload.get("runtime_skill_records", []), record_id, key="record_id")
        judgment = deepcopy(item.get("judgment", {})) if isinstance(item.get("judgment", {}), dict) else {}
        lineage = deepcopy(item.get("lineage", {})) if isinstance(item.get("lineage", {}), dict) else {}
        execution_record = {
            "execution_key": f"{item.get('record_id', '')}::{analysis.sample_id}::{analysis.attempt_index}",
            "record_id": item.get("record_id", ""),
            "skill_id": item.get("skill_id", ""),
            "style_family": item.get("style_family", ""),
            "archetype_id": item.get("archetype_id", ""),
            "sample_id": analysis.sample_id,
            "attempt_index": analysis.attempt_index,
            "variant": analysis.variant,
            "source": item.get("source", ""),
            "memory_only": bool(item.get("memory_only", False)),
            "recommended_carriers": list((item.get("recommended_carriers") or ((item.get("skill_payload") or {}).get("recommended_carriers") or []))[:6]),
            "lineage": lineage,
            "judgment": judgment,
            "overall_score": item.get("overall_score", 0.0),
            "accepted": bool(item.get("accepted", False)),
            "failure_tags": list(item.get("failure_tags") or []),
            "quality_score": float(item.get("quality_score", _skill_quality_score(item)) or 0.0),
            "evidence_score": float(item.get("quality_score", _skill_quality_score(item)) or 0.0),
            "last_seen": str(item.get("last_seen", "")).strip() or _utc_now(),
            "recency_weight": float(item.get("recency_weight", 1.0) or 1.0),
            "source_quality": float(item.get("source_quality", _source_quality(str(item.get("source", "")).strip(), policy=policy)) or 0.75),
        }
        payload["skill_execution_records"] = _upsert_by_key(
            payload.get("skill_execution_records", []),
            execution_record,
            key="execution_key",
            policy=policy,
        )
        if bucket == "runtime":
            payload["runtime_skill_records"] = _upsert_skill_record_bucket(
                payload.get("runtime_skill_records", []),
                deepcopy(item),
                key="record_id",
                policy=policy,
            )
        elif bucket == "promotable":
            payload["promotable_skill_records"] = _upsert_skill_record_bucket(
                payload.get("promotable_skill_records", []),
                deepcopy(item),
                key="record_id",
                policy=policy,
            )
        else:
            payload["quarantined_skill_records"] = _upsert_skill_record_bucket(
                payload.get("quarantined_skill_records", []),
                deepcopy(item),
                key="record_id",
                quarantined=True,
                policy=policy,
            )

    evolution_record = {
        "sample_id": analysis.sample_id,
        "attempt_index": analysis.attempt_index,
        "variant": analysis.variant,
        "evolution_type": "FIX" if analysis.prompt_compilation_recipe_candidates else "CAPTURED",
        "archetype_id": analysis.scene_archetype.get("archetype_id", ""),
        "failure_tags": [item.get("pattern", "") for item in analysis.failure_patterns],
        "recipe_ids": [item.get("recipe_id", "") for item in analysis.prompt_compilation_recipe_candidates],
        "custom_skill_ids": [item.get("skill_id", "") for item in analysis.custom_skill_records],
    }
    payload = append_evolution_record(payload, evolution_record)
    payload["skill_execution_summaries"] = _execution_summary_records(payload.get("skill_execution_records", []), policy=policy)
    payload["hold_violation_summaries"] = hold_violation_summaries(
        list(payload.get("evaluation_records", [])) + list(payload.get("skill_execution_records", []))
    )
    return payload
