#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from critic import evaluate_edit_result, reflect_edit_result
from memory import (
    analyze_loop_attempt,
    evolve_memory,
    extract_contract_critic_evaluation,
    extract_contract_critic_reflection,
    load_attempt_memory_contract,
    load_global_memory,
    save_global_memory,
)
from skills.registry import get_skill_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay memory analyzer/evolver from existing planner/edit/critic artifacts.")
    parser.add_argument("--sample-id", default="test")
    parser.add_argument("--attempt-index", type=int, default=1)
    parser.add_argument("--attempt-dir", default="")
    parser.add_argument("--planner-response", default="")
    parser.add_argument("--planner-state", default="")
    parser.add_argument("--edit-response", default="")
    parser.add_argument("--evaluation-response", default="")
    parser.add_argument("--reflection-response", default="")
    parser.add_argument("--edited-video", default="")
    parser.add_argument("--low-video", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--memory-path", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_attempt_path(base: Path | None, filename: str) -> str:
    if not base:
        return ""
    path = base / filename
    return str(path) if path.exists() else ""


def _resolve_prompt(args: argparse.Namespace, planner_payload: dict | None) -> str:
    if args.prompt:
        return args.prompt.strip()
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if planner_payload:
        return str(planner_payload.get("normalized_prompt", "") or planner_payload.get("prompt", "")).strip()
    raise ValueError("Provide --prompt/--prompt-file or make planner payload resolvable.")


def _resolve_contract_signals(attempt_dir: Path | None) -> tuple[dict, dict]:
    if attempt_dir is None:
        return {}, {}
    contract_payload = load_attempt_memory_contract(attempt_dir)
    if not contract_payload:
        return {}, {}
    return (
        extract_contract_critic_evaluation(contract_payload),
        extract_contract_critic_reflection(contract_payload),
    )


def _flatten_state_payload_local(state: dict, style_family_options: dict[str, list[dict]]) -> dict:
    theme = dict(state.get("skill_style_family") or {})
    prompt_video_analysis = dict(state.get("prompt_video_analysis") or {})
    candidate_planning = dict(state.get("candidate_planning") or {})
    final_plan = dict(state.get("final_plan") or {})

    selected_family = str(final_plan.get("style_family") or theme.get("selected_style_family") or "").strip()
    matched_ids = [str(item).strip() for item in (theme.get("matched_skill_candidates") or []) if str(item).strip()]
    provided_style_templates = [
        item
        for item in style_family_options.get(selected_family, [])
        if item.get("strategy_id") in matched_ids
    ] or style_family_options.get(selected_family, [])[:3]
    raw_outputs = state.get("raw_model_outputs") or {}

    def _call_meta_for(key: str) -> dict:
        item = raw_outputs.get(key) or {}
        parsed = item.get("parsed") or {}
        if isinstance(parsed, dict):
            meta = parsed.get("_call_meta") or {}
            if isinstance(meta, dict):
                return {
                    "truncation_applied": bool(meta.get("truncation_applied", False)),
                    "pretrim_response_available": bool(meta.get("pretrim_response_available", False)),
                }
        return {
            "truncation_applied": False,
            "pretrim_response_available": False,
        }

    section_diagnostics = {
        "analysis_structure_repair": {
            "invoked": "call_002c_analysis_structure_repair" in raw_outputs,
            "structure_errors": list((raw_outputs.get("call_002c_analysis_structure_repair", {}) or {}).get("structure_errors") or []),
            "remaining_validation_errors": list((((raw_outputs.get("call_002c_analysis_structure_repair", {}) or {}).get("parsed") or {}).get("remaining_validation_errors") or [])),
        },
        "custom_skill_repair": {
            "invoked": "call_002c_custom_skill_repair" in raw_outputs,
            "remaining_validation_errors": list((((raw_outputs.get("call_002c_custom_skill_repair", {}) or {}).get("parsed") or {}).get("remaining_validation_errors") or [])),
        },
        "candidate_generation": _call_meta_for("call_003_candidates"),
        "candidate_scoring": _call_meta_for("call_003b_scoring"),
        "final_plan": _call_meta_for("call_004_final"),
        "final_repair": _call_meta_for("call_004b_final_repair"),
    }

    return {
        "user_intent_summary": theme.get("user_intent_summary", ""),
        "style_family_candidates": theme.get("style_family_candidates", []),
        "selected_style_family": theme.get("selected_style_family", selected_family),
        "style_family_reason": theme.get("style_family_reason", ""),
        "matched_skill_candidates": matched_ids,
        "rejected_skill_candidates": theme.get("rejected_skill_candidates", []),
        "needs_custom_skill": bool(theme.get("needs_custom_skill", False)),
        "custom_skill_seed": theme.get("custom_skill_seed", ""),
        "custom_skill_candidates": prompt_video_analysis.get("custom_skill_candidates", []),
        "scene_archetype": prompt_video_analysis.get("scene_archetype", {}),
        "style_family": selected_family,
        "style_family_confidence": final_plan.get("style_family_confidence", theme.get("style_family_confidence", 0.0)),
        "grounding_evidence": final_plan.get("grounding_evidence", {}),
        "grounding_notes": final_plan.get("grounding_notes", prompt_video_analysis.get("grounding_notes", [])),
        "hard_preserve_constraints": final_plan.get("hard_preserve_constraints", []),
        "soft_preserve_constraints": final_plan.get("soft_preserve_constraints", []),
        "rewrite_targets": final_plan.get("rewrite_targets", []),
        "light_touch_targets": final_plan.get("light_touch_targets", []),
        "forbidden_changes": final_plan.get("forbidden_changes", []),
        "primary_fx_carriers": final_plan.get("primary_fx_carriers", prompt_video_analysis.get("primary_fx_carriers", [])),
        "secondary_fx_carriers": final_plan.get("secondary_fx_carriers", []),
        "plan_candidates": candidate_planning.get("plan_candidates", []),
        "candidate_strategies": candidate_planning.get("candidate_strategies", []),
        "selected_strategy": final_plan.get("selected_strategy", candidate_planning.get("selected_strategy", "")),
        "candidate_scores": candidate_planning.get("candidate_scores", []),
        "selected_candidate_id": candidate_planning.get("selected_candidate_id", ""),
        "selection_reason": candidate_planning.get("selection_reason", ""),
        "materialized_selected_skill": candidate_planning.get("materialized_selected_skill", {}),
        "normalized_prompt": final_plan.get("normalized_prompt", state.get("compiled_edit_prompt", "")),
        "compiled_edit_prompt": final_plan.get("compiled_edit_prompt", state.get("compiled_edit_prompt", "")),
        "focus_regions": candidate_planning.get("focus_regions", []),
        "region_edits": final_plan.get("region_edits", []),
        "self_check_notes": final_plan.get("self_check_notes", []),
        "conflict_resolutions": final_plan.get("conflict_resolutions", []),
        "provided_style_templates": provided_style_templates,
        "section_diagnostics": section_diagnostics,
        "raw_state": state,
    }


def _planner_payload_from_state(state_path: str | Path) -> dict:
    state = _read_json(state_path)
    registry = get_skill_registry()
    families = sorted({skill.style_family for skill in registry.get_all_skills()})
    family_skills = {family: [skill.to_dict() for skill in registry.get_skills_by_style(family)] for family in families}
    return _flatten_state_payload_local(state, family_skills)


def _resolve_planner_payload(args: argparse.Namespace, attempt_dir: Path | None) -> dict | None:
    planner_response = args.planner_response or _resolve_attempt_path(attempt_dir, "planner.response.json")
    if planner_response:
        return _read_json(planner_response)
    planner_state = args.planner_state or _resolve_attempt_path(attempt_dir, "planner.state.json")
    if planner_state:
        return _planner_payload_from_state(planner_state)
    return None


def _resolve_edit_payload(args: argparse.Namespace, attempt_dir: Path | None) -> dict | None:
    edit_response = args.edit_response or _resolve_attempt_path(attempt_dir, "edit.response.json")
    return _read_json(edit_response) if edit_response else None


def _resolve_evaluation_payload(
    args: argparse.Namespace,
    *,
    prompt: str,
    planner_payload: dict | None,
    edit_payload: dict | None,
    attempt_dir: Path | None,
) -> dict:
    evaluation_response = args.evaluation_response or _resolve_attempt_path(attempt_dir, "evaluation.response.json")
    if evaluation_response:
        return _read_json(evaluation_response)
    contract_evaluation, _ = _resolve_contract_signals(attempt_dir)
    if contract_evaluation:
        return contract_evaluation
    if not args.low_video or not args.edited_video:
        raise ValueError("Need --evaluation-response or both --low-video and --edited-video to compute evaluation.")
    result = evaluate_edit_result(
        sample_id=args.sample_id,
        prompt=prompt,
        low_video=args.low_video,
        edited_video=args.edited_video,
        planner_payload=planner_payload,
        edit_payload=edit_payload,
    )
    return result.to_dict()


def _resolve_reflection_payload(
    args: argparse.Namespace,
    *,
    evaluation_payload: dict,
    attempt_dir: Path | None,
) -> dict:
    reflection_response = args.reflection_response or _resolve_attempt_path(attempt_dir, "reflection.response.json")
    if reflection_response:
        return _read_json(reflection_response)
    _, contract_reflection = _resolve_contract_signals(attempt_dir)
    if contract_reflection:
        return contract_reflection
    from critic.evaluator import EvaluationResult, EvaluationScores

    scores = EvaluationScores(**dict(evaluation_payload.get("scores", {}) or {}))
    evaluation_obj = EvaluationResult(
        sample_id=evaluation_payload.get("sample_id", args.sample_id),
        variant=evaluation_payload.get("variant", args.variant),
        scores=scores,
        failure_tags=list(evaluation_payload.get("failure_tags", []) or []),
        reflection_hints=list(evaluation_payload.get("reflection_hints", []) or []),
        notes=list(evaluation_payload.get("notes", []) or []),
        metric_hints=dict(evaluation_payload.get("metric_hints", {}) or {}),
        raw_vlm_response=str(evaluation_payload.get("raw_vlm_response", "")),
    )
    return reflect_edit_result(evaluation_obj).to_dict()


def main() -> None:
    args = parse_args()
    attempt_dir = Path(args.attempt_dir) if args.attempt_dir else None

    planner_payload = _resolve_planner_payload(args, attempt_dir)
    prompt = _resolve_prompt(args, planner_payload)
    edit_payload = _resolve_edit_payload(args, attempt_dir)
    evaluation_payload = _resolve_evaluation_payload(
        args,
        prompt=prompt,
        planner_payload=planner_payload,
        edit_payload=edit_payload,
        attempt_dir=attempt_dir,
    )
    reflection_payload = _resolve_reflection_payload(
        args,
        evaluation_payload=evaluation_payload,
        attempt_dir=attempt_dir,
    )

    variant = (
        args.variant
        or str((edit_payload or {}).get("variant", ""))
        or str(evaluation_payload.get("variant", ""))
    )
    analysis = analyze_loop_attempt(
        sample_id=args.sample_id,
        attempt_index=args.attempt_index,
        variant=variant,
        prompt=prompt,
        planner_payload=planner_payload or {},
        edit_payload=edit_payload or {"output_video": args.edited_video, "variant": variant},
        evaluation_payload=evaluation_payload,
        reflection_payload=reflection_payload,
    )

    result = {
        "memory_analysis": analysis.to_dict(),
    }

    if args.memory_path:
        memory = load_global_memory(args.memory_path)
        updated_memory = evolve_memory(memory, analysis)
        save_global_memory(args.memory_path, updated_memory)
        result["memory_path"] = args.memory_path
        result["updated_memory"] = updated_memory

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "memory.analysis.json").write_text(json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        if "updated_memory" in result:
            (out_dir / "memory.updated.json").write_text(json.dumps(result["updated_memory"], ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
