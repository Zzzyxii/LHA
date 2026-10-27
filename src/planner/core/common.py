from __future__ import annotations

import json
import re
from pathlib import Path


def planner_state_template() -> dict:
    return {
        "request": {
            "sample_id": "",
            "prompt": "",
            "prompt_file": "",
            "low_video": "",
            "memory_context": {},
            "created_at": "",
            "available_style_families": [],
        },
        "raw_model_outputs": {},
        "skill_style_family": {
            "status": "pending",
            "updated_at": "",
            "validation_errors": [],
            "available_style_families": [],
            "user_intent_summary": "",
            "style_family_candidates": [],
            "selected_style_family": "",
            "style_family_confidence": 0.0,
            "style_family_reason": "",
            "style_intensity": "",
            "is_prompt_vague": False,
            "prompt_safety_notes": [],
            "suggested_prompt_rewrites": [],
            "matched_skill_candidates": [],
            "rejected_skill_candidates": [],
            "needs_custom_skill": False,
            "custom_skill_seed": "",
        },
        "prompt_video_analysis": {
            "status": "pending",
            "updated_at": "",
            "validation_errors": [],
            "scene_archetype": {},
            "video_brief": "",
            "visible_people": [],
            "visible_props": [],
            "visible_world_anchors": [],
            "primary_fx_carriers": [],
            "grounding_notes": [],
            "custom_skill_candidates": [],
        },
        "candidate_planning": {
            "status": "pending",
            "updated_at": "",
            "validation_errors": [],
            "candidate_strategies": [],
            "plan_candidates": [],
            "candidate_scores": [],
            "selected_candidate_id": "",
            "selected_strategy": "",
            "selection_reason": "",
            "focus_regions": [],
        },
        "final_plan": {
            "status": "pending",
            "updated_at": "",
            "validation_errors": [],
            "style_family": "",
            "style_family_confidence": 0.0,
            "selected_strategy": "",
            "grounding_evidence": {
                "visible_people": [],
                "visible_props": [],
                "visible_world_anchors": [],
                "carrier_candidates": [],
                "immutable_objects": [],
                "mutable_regions": [],
            },
            "grounding_notes": [],
            "hard_preserve_constraints": [],
            "soft_preserve_constraints": [],
            "rewrite_targets": [],
            "light_touch_targets": [],
            "forbidden_changes": [],
            "primary_fx_carriers": [],
            "secondary_fx_carriers": [],
            "normalized_prompt": "",
            "region_edits": [],
            "self_check_notes": [],
            "conflict_resolutions": [],
        },
        "compiled_edit_prompt": "",
        "validation_diagnostics": {
            "is_satisfactory": False,
            "failed_sections": [],
            "missing_fields": [],
        },
    }


def write_state(path: str | Path | None, state: dict) -> None:
    if not path:
        return
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def write_call_artifacts(
    *,
    state_path: str | Path | None,
    prefix: str,
    system_prompt: str,
    user_text: str,
    payload: object,
) -> None:
    if not state_path:
        return
    out_dir = Path(state_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}.system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (out_dir / f"{prefix}.user_prompt.txt").write_text(user_text, encoding="utf-8")
    if isinstance(payload, dict):
        (out_dir / f"{prefix}.response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raw_response = payload.get("raw_response")
        if isinstance(raw_response, str) and raw_response.strip():
            (out_dir / f"{prefix}.raw_response.txt").write_text(raw_response, encoding="utf-8")
        repair_raw_response = payload.get("repair_raw_response")
        if isinstance(repair_raw_response, str) and repair_raw_response.strip():
            (out_dir / f"{prefix}.repair_response.txt").write_text(repair_raw_response, encoding="utf-8")
    else:
        (out_dir / f"{prefix}.response.txt").write_text(str(payload), encoding="utf-8")


def planner_log(message: str) -> None:
    print(f"[planner] {message}", flush=True)


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def split_items(value: str) -> list[str]:
    if not value:
        return []
    text = value.replace("；", ";").replace("，", ",").replace("、", ",").replace("|", ",")
    parts = re.split(r"[\n,;]", text)
    return [item.strip(" -*") for item in parts if item.strip(" -*")]


def clean_string_list(items) -> list[str]:
    values = [str(item).strip() for item in (items or []) if str(item).strip()]
    return list(dict.fromkeys(values))


def parse_bool(value: str) -> bool:
    lowered = clean_text(value).lower()
    return lowered in {"true", "yes", "1", "是", "需要"}


def parse_float(value: str) -> float:
    try:
        return float(clean_text(value))
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
        return float(match.group(0)) if match else 0.0


def parse_section_fields(text: str, headings: list[str]) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^([A-Z_]+|[\u4e00-\u9fffA-Za-z_]+):\s*(.*)$", text))
    results: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key not in headings:
            continue
        inline = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        results[key] = inline if inline and not body else "\n".join(part for part in [inline, body] if part).strip()
    return results


def parse_tagged_blocks(text: str, tag: str) -> list[dict[str, str]]:
    pattern = rf"\[{tag}\]\s*(.*?)\s*\[/{tag}\]"
    blocks = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    parsed: list[dict[str, str]] = []
    for block in blocks:
        item: dict[str, str] = {}
        current_key = ""
        current_lines: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.rstrip()
            match = re.match(r"^([a-z_]+):\s*(.*)$", line.strip())
            if match:
                if current_key:
                    item[current_key] = "\n".join(current_lines).strip()
                current_key = match.group(1)
                current_lines = [match.group(2)] if match.group(2) else []
            elif current_key:
                current_lines.append(line.strip())
        if current_key:
            item[current_key] = "\n".join(current_lines).strip()
        if item:
            parsed.append(item)
    return parsed


def update_section(state: dict, section: str, payload: dict, required_fields: list[str]) -> None:
    merged = dict(state[section])
    for key, value in payload.items():
        if key in merged:
            merged[key] = value
    merged["status"] = "completed"
    missing = [field for field in required_fields if merged.get(field) in ("", [], {}, None)]
    merged["validation_errors"] = [f"missing {field}" for field in missing]
    if missing:
        merged["status"] = "failed"
    state[section] = merged


def missing_required_fields(payload: dict, required_fields: list[str]) -> list[str]:
    return [field for field in required_fields if payload.get(field) in ("", [], {}, None)]


def merge_payload_dict(base_payload: dict, repair_payload: dict) -> dict:
    merged = dict(base_payload)
    for key, value in repair_payload.items():
        if value not in ("", [], {}, None):
            merged[key] = value
    return merged


def state_diagnostics(state: dict) -> dict:
    diagnostics = {
        "is_satisfactory": True,
        "failed_sections": [],
        "missing_fields": [],
    }
    required_checks = {
        "skill_style_family": ["selected_style_family", "style_family_reason"],
        "prompt_video_analysis": ["video_brief", "visible_people", "visible_props"],
        "candidate_planning": ["candidate_strategies", "plan_candidates", "candidate_scores", "selection_reason"],
        "final_plan": ["region_edits", "normalized_prompt"],
    }
    for section, fields in required_checks.items():
        data = state.get(section) or {}
        missing = [field for field in fields if data.get(field) in ("", [], {}, None)]
        if data.get("status") == "failed" or missing:
            diagnostics["is_satisfactory"] = False
            diagnostics["failed_sections"].append(section)
            diagnostics["missing_fields"].extend([f"{section}.{field}" for field in missing])
    return diagnostics
