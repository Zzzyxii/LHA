from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

MEMORY_SCHEMA_VERSION = 2


def default_global_memory() -> dict:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "scene_archetypes": [],
        "style_convertibility_priors": [],
        "model_behavior_priors": [],
        "failure_patterns": [],
        "prompt_compilation_recipes": [],
        "custom_skill_records": [],
        "runtime_skill_records": [],
        "promotable_skill_records": [],
        "quarantined_skill_records": [],
        "critic_implicit_prototypes": [],
        "edit_implicit_prototypes": [],
        "implicit_run_records": [],
        "skill_execution_records": [],
        "skill_execution_summaries": [],
        # Per-(strategy, dimension) rates at which an explicit "keep this"
        # instruction was overridden. See memory.evolver.hold_violation_summaries.
        "hold_violation_summaries": [],
        "run_records": [],
        "evaluation_records": [],
        "evolution_records": [],
    }


def _coerce_legacy_memory(payload: dict) -> dict:
    memory = default_global_memory()
    source = dict(payload or {})

    if isinstance(source.get("run_history"), list) and not source.get("run_records"):
        source["run_records"] = source["run_history"]

    for key in memory:
        if key == "schema_version":
            continue
        value = source.get(key)
        if isinstance(value, list):
            memory[key] = deepcopy(value)

    memory["schema_version"] = MEMORY_SCHEMA_VERSION
    return memory


def ensure_memory_schema(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return default_global_memory()
    if payload.get("schema_version") == MEMORY_SCHEMA_VERSION and all(key in payload for key in default_global_memory()):
        return deepcopy(payload)
    return _coerce_legacy_memory(payload)


def load_global_memory(path: str | Path) -> dict:
    memory_path = Path(path)
    if not memory_path.exists():
        return default_global_memory()
    return ensure_memory_schema(json.loads(memory_path.read_text(encoding="utf-8")))


def save_global_memory(path: str | Path, payload: dict) -> None:
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = ensure_memory_schema(payload)
    memory_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_bounded(items: list[dict], record: dict, *, max_items: int) -> list[dict]:
    updated = list(items)
    updated.append(deepcopy(record))
    return updated[-max_items:]


def append_run_record(memory: dict, run_record: dict, *, max_items: int = 200) -> dict:
    payload = ensure_memory_schema(memory)
    payload["run_records"] = _append_bounded(payload.get("run_records", []), run_record, max_items=max_items)
    return payload


def append_evaluation_record(memory: dict, evaluation_record: dict, *, max_items: int = 400) -> dict:
    payload = ensure_memory_schema(memory)
    payload["evaluation_records"] = _append_bounded(payload.get("evaluation_records", []), evaluation_record, max_items=max_items)
    return payload


def append_evolution_record(memory: dict, evolution_record: dict, *, max_items: int = 400) -> dict:
    payload = ensure_memory_schema(memory)
    payload["evolution_records"] = _append_bounded(payload.get("evolution_records", []), evolution_record, max_items=max_items)
    return payload


def append_custom_skill_record(memory: dict, custom_skill_record: dict, *, max_items: int = 1000) -> dict:
    payload = ensure_memory_schema(memory)
    payload["custom_skill_records"] = _append_bounded(
        payload.get("custom_skill_records", []),
        custom_skill_record,
        max_items=max_items,
    )
    return payload


def append_runtime_skill_record(memory: dict, runtime_skill_record: dict, *, max_items: int = 1000) -> dict:
    payload = ensure_memory_schema(memory)
    payload["runtime_skill_records"] = _append_bounded(
        payload.get("runtime_skill_records", []),
        runtime_skill_record,
        max_items=max_items,
    )
    return payload


def append_promotable_skill_record(memory: dict, promotable_skill_record: dict, *, max_items: int = 1000) -> dict:
    payload = ensure_memory_schema(memory)
    payload["promotable_skill_records"] = _append_bounded(
        payload.get("promotable_skill_records", []),
        promotable_skill_record,
        max_items=max_items,
    )
    return payload


def append_quarantined_skill_record(memory: dict, quarantined_skill_record: dict, *, max_items: int = 1000) -> dict:
    payload = ensure_memory_schema(memory)
    payload["quarantined_skill_records"] = _append_bounded(
        payload.get("quarantined_skill_records", []),
        quarantined_skill_record,
        max_items=max_items,
    )
    return payload


def append_skill_execution_record(memory: dict, skill_execution_record: dict, *, max_items: int = 2000) -> dict:
    payload = ensure_memory_schema(memory)
    payload["skill_execution_records"] = _append_bounded(
        payload.get("skill_execution_records", []),
        skill_execution_record,
        max_items=max_items,
    )
    return payload


def append_run_history(memory: dict, run_record: dict, *, max_items: int = 200) -> dict:
    return append_run_record(memory, run_record, max_items=max_items)
