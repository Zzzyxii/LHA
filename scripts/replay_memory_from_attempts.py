#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memory import (
    analyze_loop_attempt,
    cleanup_memory,
    default_global_memory,
    evolve_memory,
    extract_contract_critic_evaluation,
    extract_contract_critic_reflection,
    load_global_memory,
    load_attempt_memory_contract,
    load_memory_policy,
    save_global_memory,
)
from skills.registry import get_skill_registry


DEFAULT_OUTPUT_MEMORY_PATH = "out/memory/replayed_memory.json"
_STYLE_FAMILY_OPTIONS_CACHE: dict[str, list[dict]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay memory from closed-loop attempt_* artifacts.")
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root directory to recursively scan for attempt_* directories. Defaults to ./out when omitted.",
    )
    parser.add_argument(
        "--attempt-dir",
        action="append",
        default=[],
        help="Explicit attempt_* directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--base-memory-path",
        default="",
        help="Optional existing memory JSON to merge into. Defaults to rebuilding from an empty memory store.",
    )
    parser.add_argument(
        "--output-memory-path",
        default=DEFAULT_OUTPUT_MEMORY_PATH,
        help=f"Output memory JSON path. Defaults to {DEFAULT_OUTPUT_MEMORY_PATH}.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default="",
        help="Optional directory for per-attempt memory.analysis.json outputs.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional memory config path. Defaults to configs/memory.json or LOW_HIGH_MEMORY_CONFIG.",
    )
    parser.add_argument("--cleanup-after", action="store_true", help="Run memory cleanup after replay using the same config.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print summary without writing output files.")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_critic_payloads(attempt_dir: Path) -> tuple[dict, dict]:
    contract_payload = load_attempt_memory_contract(attempt_dir)
    if contract_payload:
        evaluation_payload = extract_contract_critic_evaluation(contract_payload)
        reflection_payload = extract_contract_critic_reflection(contract_payload)
        if evaluation_payload and reflection_payload:
            return evaluation_payload, reflection_payload

    evaluation_path = attempt_dir / "evaluation.response.json"
    reflection_path = attempt_dir / "reflection.response.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Missing {evaluation_path.name}")
    if not reflection_path.exists():
        raise FileNotFoundError(f"Missing {reflection_path.name}")
    return _read_json(evaluation_path), _read_json(reflection_path)


def _attempt_index(path: Path) -> int:
    match = re.search(r"attempt_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _style_family_options() -> dict[str, list[dict]]:
    global _STYLE_FAMILY_OPTIONS_CACHE
    if _STYLE_FAMILY_OPTIONS_CACHE is None:
        registry = get_skill_registry()
        families = sorted({skill.style_family for skill in registry.get_all_skills()})
        _STYLE_FAMILY_OPTIONS_CACHE = {
            family: [skill.to_dict() for skill in registry.get_skills_by_style(family)]
            for family in families
        }
    return _STYLE_FAMILY_OPTIONS_CACHE


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


def _discover_attempt_dirs(roots: list[Path], explicit_attempt_dirs: list[Path]) -> list[Path]:
    discovered: dict[str, Path] = {}
    for attempt_dir in explicit_attempt_dirs:
        if attempt_dir.is_dir():
            discovered[str(attempt_dir.resolve())] = attempt_dir.resolve()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("attempt_*"):
            if path.is_dir():
                discovered[str(path.resolve())] = path.resolve()
    return sorted(
        discovered.values(),
        key=lambda path: (str(path.parent), _attempt_index(path), str(path)),
    )


def _resolve_relative_attempt_path(attempt_dir: Path, roots: list[Path]) -> Path:
    for root in roots:
        try:
            return attempt_dir.relative_to(root)
        except ValueError:
            continue
    return Path(attempt_dir.name)


def _load_planner_payload(attempt_dir: Path) -> tuple[dict, dict]:
    planner_response_path = attempt_dir / "planner.response.json"
    if planner_response_path.exists():
        return _read_json(planner_response_path), {}

    planner_state_path = attempt_dir / "planner.state.json"
    if planner_state_path.exists():
        state_payload = _read_json(planner_state_path)
        planner_payload = _flatten_state_payload_local(state_payload, _style_family_options())
        return planner_payload, state_payload

    raise FileNotFoundError(f"Missing planner.response.json and planner.state.json in {attempt_dir}")


def _load_edit_payload(attempt_dir: Path) -> dict:
    edit_response_path = attempt_dir / "edit.response.json"
    if edit_response_path.exists():
        return _read_json(edit_response_path)
    edited_video_path = attempt_dir / "edited.mp4"
    return {
        "output_video": str(edited_video_path.resolve()) if edited_video_path.exists() else "",
    }


def _resolve_prompt(attempt_dir: Path, planner_payload: dict, planner_state: dict) -> str:
    prompt_path = attempt_dir / "edit.prompt.txt"
    if prompt_path.exists():
        prompt = _read_text(prompt_path)
        if prompt:
            return prompt

    request = planner_state.get("request", {}) if isinstance(planner_state.get("request", {}), dict) else {}
    if str(request.get("prompt", "")).strip():
        return str(request.get("prompt", "")).strip()

    contract_payload = load_attempt_memory_contract(attempt_dir)
    contract_prompt = str(contract_payload.get("prompt", "")).strip()
    if contract_prompt:
        return contract_prompt

    for key in ["compiled_edit_prompt", "normalized_prompt", "prompt"]:
        prompt = str(planner_payload.get(key, "")).strip()
        if prompt:
            return prompt

    raise ValueError(f"Unable to resolve prompt for {attempt_dir}")


def _resolve_sample_id(attempt_dir: Path, planner_payload: dict, planner_state: dict, evaluation_payload: dict) -> str:
    request = planner_state.get("request", {}) if isinstance(planner_state.get("request", {}), dict) else {}
    for value in [
        evaluation_payload.get("sample_id", ""),
        planner_payload.get("sample_id", ""),
        request.get("sample_id", ""),
    ]:
        sample_id = str(value or "").strip()
        if sample_id:
            return sample_id
    return f"replay::{attempt_dir.parent.name}"


def _resolve_variant(planner_payload: dict, planner_state: dict, edit_payload: dict, evaluation_payload: dict) -> str:
    request = planner_state.get("request", {}) if isinstance(planner_state.get("request", {}), dict) else {}
    for value in [
        edit_payload.get("variant", ""),
        evaluation_payload.get("variant", ""),
        planner_payload.get("variant", ""),
        request.get("variant", ""),
    ]:
        variant = str(value or "").strip()
        if variant:
            return variant
    return ""


def _write_analysis(analysis_output_dir: Path, attempt_dir: Path, roots: list[Path], analysis: dict) -> str:
    output_path = (analysis_output_dir / _resolve_relative_attempt_path(attempt_dir, roots) / "memory.analysis.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def _bank_counts(memory: dict) -> dict:
    return {
        "custom_skill_records": len(memory.get("custom_skill_records") or []),
        "promotable_skill_records": len(memory.get("promotable_skill_records") or []),
        "runtime_skill_records": len(memory.get("runtime_skill_records") or []),
        "quarantined_skill_records": len(memory.get("quarantined_skill_records") or []),
        "skill_execution_records": len(memory.get("skill_execution_records") or []),
        "skill_execution_summaries": len(memory.get("skill_execution_summaries") or []),
        "run_records": len(memory.get("run_records") or []),
        "evaluation_records": len(memory.get("evaluation_records") or []),
        "evolution_records": len(memory.get("evolution_records") or []),
    }


def main() -> None:
    args = parse_args()
    roots = [Path(path).expanduser().resolve() for path in (args.root or ["out"])]
    explicit_attempt_dirs = [Path(path).expanduser().resolve() for path in (args.attempt_dir or [])]
    output_memory_path = Path(args.output_memory_path).expanduser().resolve()
    analysis_output_dir = Path(args.analysis_output_dir).expanduser().resolve() if args.analysis_output_dir else None
    policy = load_memory_policy(args.config or None)

    attempt_dirs = _discover_attempt_dirs(roots, explicit_attempt_dirs)
    base_memory = (
        load_global_memory(args.base_memory_path)
        if args.base_memory_path
        else default_global_memory()
    )
    memory = base_memory
    processed: list[dict] = []
    skipped: list[dict] = []

    for attempt_dir in attempt_dirs:
        try:
            planner_payload, planner_state = _load_planner_payload(attempt_dir)
            edit_payload = _load_edit_payload(attempt_dir)
            evaluation_payload, reflection_payload = _load_critic_payloads(attempt_dir)
            prompt = _resolve_prompt(attempt_dir, planner_payload, planner_state)
            sample_id = _resolve_sample_id(attempt_dir, planner_payload, planner_state, evaluation_payload)
            variant = _resolve_variant(planner_payload, planner_state, edit_payload, evaluation_payload)
            attempt_index = _attempt_index(attempt_dir)

            analysis = analyze_loop_attempt(
                sample_id=sample_id,
                attempt_index=attempt_index,
                variant=variant,
                prompt=prompt,
                planner_payload=planner_payload,
                edit_payload=edit_payload,
                evaluation_payload=evaluation_payload,
                reflection_payload=reflection_payload,
            )
            memory = evolve_memory(memory, analysis, config_path=args.config or None)
            analysis_path = ""
            if analysis_output_dir is not None and not args.dry_run:
                analysis_path = _write_analysis(analysis_output_dir, attempt_dir, roots, analysis.to_dict())

            processed.append(
                {
                    "attempt_dir": str(attempt_dir),
                    "sample_id": sample_id,
                    "attempt_index": attempt_index,
                    "variant": variant,
                    "action": str(reflection_payload.get("action", "") or ""),
                    "analysis_output": analysis_path,
                }
            )
        except Exception as exc:
            skipped.append(
                {
                    "attempt_dir": str(attempt_dir),
                    "reason": str(exc),
                }
            )

    replayed_memory = memory
    final_memory = cleanup_memory(replayed_memory, config_path=args.config or None) if args.cleanup_after else replayed_memory
    summary = {
        "roots": [str(path) for path in roots],
        "explicit_attempt_dirs": [str(path) for path in explicit_attempt_dirs],
        "discovered_attempt_dirs": len(attempt_dirs),
        "processed_attempts": len(processed),
        "skipped_attempts": len(skipped),
        "base_memory_path": str(Path(args.base_memory_path).expanduser().resolve()) if args.base_memory_path else "",
        "output_memory_path": str(output_memory_path),
        "analysis_output_dir": str(analysis_output_dir) if analysis_output_dir is not None else "",
        "config": policy,
        "cleanup_applied": bool(args.cleanup_after),
        "before": _bank_counts(base_memory),
        "after_replay": _bank_counts(replayed_memory),
        "final": _bank_counts(final_memory),
        "processed_examples": processed[:20],
        "skipped_examples": skipped[:20],
    }
    if args.cleanup_after:
        summary["after_cleanup"] = dict(summary["final"])

    if args.dry_run:
        print(json.dumps({**summary, "written": False}, ensure_ascii=False, indent=2))
        return

    save_global_memory(output_memory_path, final_memory)
    print(json.dumps({**summary, "written": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
