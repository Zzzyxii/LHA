from __future__ import annotations

import json
from pathlib import Path

from planner.core.common import (
    merge_payload_dict,
    missing_required_fields,
    planner_log,
    write_call_artifacts,
)


def _strip_think_tags(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "")


def _trim_to_structured_marker(prefix: str, response_text: str) -> tuple[str, bool]:
    marker = ""
    if prefix.startswith("planner.call_002a"):
        marker = "VIDEO_BRIEF:"
    elif prefix.startswith("planner.call_002b") or prefix.startswith("planner.call_002c"):
        marker = "MATCHED_SKILL_CANDIDATES:"
    elif prefix.startswith("planner.call_003"):
        marker = "[CANDIDATE_STRATEGY]"
    elif prefix.startswith("planner.call_004"):
        marker = "STYLE_FAMILY:"
    if not marker:
        return response_text, False
    index = response_text.find(marker)
    if index < 0:
        return response_text, False
    return response_text[index:], index > 0


def call_step(
    *,
    model,
    state_path: str | Path | None,
    prefix: str,
    system_prompt: str,
    user_text: str,
    image_paths: list[str],
    parser,
    max_new_tokens: int = 4096,
) -> tuple[dict, str]:
    planner_log(f"{prefix}.start max_new_tokens={max_new_tokens} frames={len(image_paths)}")
    response_text = model.predict_text(
        system_prompt=system_prompt,
        user_text=user_text,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
    )
    original_response_text = response_text
    response_text = _strip_think_tags(response_text)
    response_text, truncation_applied = _trim_to_structured_marker(prefix, response_text)
    payload = parser(response_text)
    if isinstance(payload, dict):
        payload["_call_meta"] = {
            "truncation_applied": truncation_applied,
            "pretrim_response_available": bool(truncation_applied),
        }
    if isinstance(payload, dict):
        keys = ",".join(sorted(payload.keys())[:8])
        planner_log(f"{prefix}.done parsed_keys={keys}")
    else:
        planner_log(f"{prefix}.done")
    write_call_artifacts(
        state_path=state_path,
        prefix=prefix,
        system_prompt=system_prompt,
        user_text=user_text,
        payload=response_text,
    )
    if truncation_applied and state_path:
        out_dir = Path(state_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{prefix}.pretrim_response.txt").write_text(original_response_text, encoding="utf-8")
    return payload, response_text


def call_repair_step(
    *,
    model,
    state_path: str | Path | None,
    prefix: str,
    section_name: str,
    current_payload: dict,
    required_fields: list[str],
    system_prompt: str,
    user_text: str,
    image_paths: list[str],
    parser,
    max_new_tokens: int,
) -> tuple[dict, str] | tuple[None, None]:
    missing_fields = missing_required_fields(current_payload, required_fields)
    if not missing_fields:
        planner_log(f"{prefix}.skip no_missing_fields")
        return None, None
    planner_log(f"{prefix}.repair_start missing_fields={','.join(missing_fields)}")
    repair_system_prompt = (
        system_prompt.rstrip()
        + "\n\n修复模式：只补齐缺失字段，不要重写整个 section，不要改动已经正确的字段。"
    )
    repair_user_text = (
        user_text.rstrip()
        + f"\n\n当前 section: {section_name}\n"
        + f"missing_fields: {json.dumps(missing_fields, ensure_ascii=False)}\n"
        + f"current_parsed_payload: {json.dumps(current_payload, ensure_ascii=False)}\n"
        + "只返回缺失字段及其必要的关联字段。"
    )
    repair_payload, repair_raw = call_step(
        model=model,
        state_path=state_path,
        prefix=prefix,
        system_prompt=repair_system_prompt,
        user_text=repair_user_text,
        image_paths=image_paths,
        parser=parser,
        max_new_tokens=max_new_tokens,
    )
    planner_log(f"{prefix}.repair_done")
    return repair_payload, repair_raw


def merge_repair_payload(base_payload: dict, repair_payload: dict | None) -> dict:
    if not repair_payload:
        return base_payload
    return merge_payload_dict(base_payload, repair_payload)
