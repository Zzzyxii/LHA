from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from critic import evaluate_edit_result, reflect_edit_result
from critic.implicit_memory import build_critic_memory_query, pack_critic_implicit_state
from edit import EditRequest, render_edit_video
from memory import (
    analyze_loop_attempt,
    build_attempt_memory_contract,
    build_edit_hidden_memory_controls,
    build_memory_query,
    evolve_memory,
    fuse_memory_biases,
    init_edit_hidden_state,
    load_global_memory,
    load_memory_runtime_config,
    resolve_memory_mode,
    retrieve_execution_adaptive_bias,
    retrieve_memory_context,
    save_global_memory,
    update_edit_hidden_state,
)
from planner.core.video_planner import plan_video, replan_video_from_existing


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed face, extra limbs, identity drift, cartoon, anime, oversaturated neon, broken geometry"
)


def _log_stage(message: str) -> None:
    print(f"[loop] {message}", flush=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_prompt_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _refined_prompt(base_prompt: str, prompt_patch: list[str]) -> str:
    if not prompt_patch:
        return base_prompt
    return base_prompt.strip() + "\n\nRefinement priorities:\n- " + "\n- ".join(prompt_patch)


def _clean_string_list(items: object) -> list[str]:
    values = [str(item).strip() for item in (items or []) if str(item).strip()]
    return list(dict.fromkeys(values))


def _candidate_strategy_preference_summary(planner_payload: dict) -> tuple[list[str], list[str]]:
    candidate_scores = [item for item in (planner_payload.get("candidate_score_details") or planner_payload.get("candidate_scores") or []) if isinstance(item, dict)]
    candidate_strategies = [item for item in (planner_payload.get("candidate_strategies") or []) if isinstance(item, dict)]
    plan_candidates = [item for item in (planner_payload.get("plan_candidates") or []) if isinstance(item, dict)]
    candidate_to_strategy: dict[str, str] = {}
    pair_count = min(len(candidate_strategies), len(plan_candidates))
    for index in range(pair_count):
        candidate_id = str(plan_candidates[index].get("candidate_id", "")).strip()
        strategy_id = str(candidate_strategies[index].get("strategy_id", "")).strip()
        if candidate_id and strategy_id:
            candidate_to_strategy[candidate_id] = strategy_id

    ranked: list[tuple[float, str]] = []
    for item in candidate_scores:
        candidate_id = str(item.get("candidate_id", "")).strip()
        strategy_id = candidate_to_strategy.get(candidate_id, candidate_id)
        if not strategy_id:
            continue
        total = float(item.get("total", 0.0) or 0.0)
        ranked.append((total, strategy_id))

    if not ranked:
        selected_strategy = str(planner_payload.get("selected_strategy", "")).strip()
        return ([selected_strategy] if selected_strategy else []), []

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    preferred = [strategy_id for score, strategy_id in ranked if score >= 0.7][:3]
    disfavored = [strategy_id for score, strategy_id in sorted(ranked, key=lambda pair: pair[0]) if score <= 0.55][:3]
    if not preferred and ranked:
        preferred = [ranked[0][1]]
    return _clean_string_list(preferred), _clean_string_list(disfavored)


def _build_edit_implicit_controls(
    *,
    edit_memory_mode: str,
    runtime_memory_config: dict,
    critic_implicit_state: dict,
    edit_hidden_state: dict,
    planner_payload: dict,
    attempt_adaptation_state: dict,
) -> dict:
    if str(edit_memory_mode or "").strip().lower() not in {"implicit", "hybrid"}:
        return {}
    state = dict(critic_implicit_state or {})
    adaptation = dict(attempt_adaptation_state or {})
    controls: dict = {}

    last_action = str(state.get("last_action", "") or "").strip()
    last_score = float(state.get("last_overall_score", 0.0) or 0.0)
    failure_counter = dict(state.get("failure_counter", {}) or {})
    detached_count = int(failure_counter.get("detached_fx", 0) or 0)
    weak_world_count = int(failure_counter.get("weak_world_realization", 0) or 0)

    if last_action == "local_refine_prompt":
        controls["steps_delta"] = 1
        if last_score >= 0.6:
            controls["denoise_delta"] = -0.03
        if detached_count >= 1:
            controls["reference_from_low_video"] = True
    elif last_action == "global_replan":
        controls["steps_delta"] = 2
        controls["seed_delta"] = 17
        controls["denoise_delta"] = 0.05
        if weak_world_count >= 1:
            controls["cfg_delta"] = 0.1

    if not controls:
        preferred_strategies = list(adaptation.get("last_preferred_candidate_strategies", []) or [])
        disfavored_strategies = set(adaptation.get("last_disfavored_candidate_strategies", []) or [])
        preferred_strategies = [item for item in preferred_strategies if item and item not in disfavored_strategies]
        if preferred_strategies:
            controls["preferred_candidate_strategies"] = preferred_strategies[:3]

    hidden_controls = build_edit_hidden_memory_controls(
        edit_hidden_state=edit_hidden_state,
        runtime_memory_config=runtime_memory_config,
        critic_implicit_state=critic_implicit_state,
        planner_payload=planner_payload,
        attempt_adaptation_state=attempt_adaptation_state,
    )
    if hidden_controls:
        controls.update(hidden_controls)

    return controls


def _build_edit_explicit_controls(
    *,
    memory_context: dict,
    planner_payload: dict,
    attempt_adaptation_state: dict,
) -> dict:
    planner_payload = dict(planner_payload or {})
    adaptation = dict(attempt_adaptation_state or {})
    scene_archetype = planner_payload.get("scene_archetype", {}) if isinstance(planner_payload.get("scene_archetype", {}), dict) else {}
    selected_family = str(planner_payload.get("style_family", "") or "").strip()
    archetype_id = str(scene_archetype.get("archetype_id", "") or "").strip()
    if not selected_family:
        return {}
    execution_bias = dict(memory_context.get("execution_adaptive_bias") or {})
    if not execution_bias:
        execution_bias = retrieve_execution_adaptive_bias(
            memory_context,
            selected_family=selected_family,
            archetype_id=archetype_id,
        )
    controls: dict = {}
    if execution_bias.get("prefer_environment_first"):
        controls["steps_delta"] = controls.get("steps_delta", 0) + 1
        controls["cfg_delta"] = controls.get("cfg_delta", 0.0) + 0.05
    if execution_bias.get("avoid_face_heavy_rewrite"):
        controls["denoise_delta"] = controls.get("denoise_delta", 0.0) - 0.04
        controls["reference_from_low_video"] = True
    preferred_carriers = list(execution_bias.get("preferred_carriers", []) or [])[:4]
    forbidden_carriers = list(execution_bias.get("forbidden_carriers", []) or [])[:4]
    boosted_skill_ids = list(execution_bias.get("boosted_skill_ids", []) or [])[:4]
    penalized_skill_ids = list(execution_bias.get("penalized_skill_ids", []) or [])[:4]
    if preferred_carriers:
        controls["reference_from_low_video"] = True
    if boosted_skill_ids:
        controls["preferred_candidate_strategies"] = boosted_skill_ids
    if penalized_skill_ids:
        controls["forbidden_candidate_strategies"] = penalized_skill_ids
        selected_strategy = str(adaptation.get("last_selected_strategy", "") or "").strip()
        if selected_strategy and selected_strategy in set(penalized_skill_ids):
            controls["seed_delta"] = controls.get("seed_delta", 0) + 11
            controls["steps_delta"] = controls.get("steps_delta", 0) + 1
    if forbidden_carriers and not preferred_carriers:
        controls["denoise_delta"] = controls.get("denoise_delta", 0.0) - 0.02
    controls["preferred_carriers"] = preferred_carriers
    controls["forbidden_carriers"] = forbidden_carriers
    return controls


def _merge_edit_controls(explicit_controls: dict, implicit_controls: dict) -> dict:
    fused = fuse_memory_biases(
        {"edit_controls": dict(explicit_controls or {})},
        {"edit_controls": dict(implicit_controls or {})},
        mode="hybrid",
        explicit_weight=1.0,
        implicit_weight=1.0,
        conflict_policy="sum",
    )
    controls = dict(fused.get("edit_controls", {}) or {})

    for key in ("preferred_candidate_strategies", "forbidden_candidate_strategies", "preferred_carriers", "forbidden_carriers"):
        explicit_list = list((explicit_controls or {}).get(key, []) or [])
        implicit_list = list((implicit_controls or {}).get(key, []) or [])
        merged = list(dict.fromkeys(explicit_list + implicit_list))
        if merged:
            controls[key] = merged[:6]
    controls["_fusion"] = {
        "explicit": dict(explicit_controls or {}),
        "implicit": dict(implicit_controls or {}),
    }
    return controls


def _resolve_edit_controls(
    *,
    edit_memory_mode: str,
    runtime_memory_config: dict,
    memory_context: dict,
    planner_payload: dict,
    critic_implicit_state: dict,
    edit_hidden_state: dict,
    attempt_adaptation_state: dict,
) -> dict:
    mode = str(edit_memory_mode or "").strip().lower()
    explicit_controls = _build_edit_explicit_controls(
        memory_context=memory_context,
        planner_payload=planner_payload,
        attempt_adaptation_state=attempt_adaptation_state,
    )
    implicit_controls = _build_edit_implicit_controls(
        edit_memory_mode=edit_memory_mode,
        runtime_memory_config=runtime_memory_config,
        critic_implicit_state=critic_implicit_state,
        edit_hidden_state=edit_hidden_state,
        planner_payload=planner_payload,
        attempt_adaptation_state=attempt_adaptation_state,
    )
    if mode == "none":
        return {}
    if mode == "explicit":
        return explicit_controls
    if mode == "implicit":
        return implicit_controls
    if mode == "hybrid":
        return _merge_edit_controls(explicit_controls, implicit_controls)
    return {}


def _extract_attempt_adaptation_state(
    *,
    planner_payload: dict,
    evaluation_payload: dict,
    reflection_payload: dict,
) -> dict:
    scores = evaluation_payload.get("scores", {}) or {}
    overall_score = float(scores.get("overall_score", 0.0) or 0.0)
    carrier_grounding = float(scores.get("carrier_grounding", 0.0) or 0.0)
    failure_hints = _clean_string_list(evaluation_payload.get("failure_tags") or [])
    prompt_patch = _clean_string_list(reflection_payload.get("prompt_patch") or [])
    primary_carriers = _clean_string_list(planner_payload.get("primary_fx_carriers") or [])

    good_carriers = primary_carriers if carrier_grounding >= 0.6 else []
    bad_carriers = primary_carriers if "detached_fx" in failure_hints or carrier_grounding < 0.45 else []
    preferred_candidate_strategies, disfavored_candidate_strategies = _candidate_strategy_preference_summary(planner_payload)

    custom_skills = [
        item for item in (planner_payload.get("custom_skill_candidates") or []) if isinstance(item, dict)
    ]
    selected_custom_skills = [
        item
        for item in custom_skills
        if str(item.get("strategy_id", "")).strip() == str(planner_payload.get("selected_strategy", "")).strip()
    ]
    accepted_custom_skills = selected_custom_skills[:3] if overall_score >= 0.55 else []

    adaptive_bias = {
        "prefer_device_bound_fx": bool(good_carriers),
        "avoid_detached_background_fx": "detached_fx" in failure_hints,
        "avoid_face_heavy_rewrite": any(tag in failure_hints for tag in ["identity_drift", "artifact_heavy"]),
        "prefer_environment_first": "weak_world_realization" in failure_hints,
        "preferred_carriers": good_carriers[:4],
        "forbidden_carriers": bad_carriers[:4],
        "preferred_candidate_strategies": preferred_candidate_strategies[:3],
        "forbidden_candidate_strategies": disfavored_candidate_strategies[:3],
    }

    return {
        "last_prompt_patch": prompt_patch[:6],
        "last_selected_strategy": str(planner_payload.get("selected_strategy", "")).strip(),
        "last_good_carriers": good_carriers[:4],
        "last_bad_carriers": bad_carriers[:4],
        "last_preserve_constraints": _clean_string_list(planner_payload.get("hard_preserve_constraints") or [])[:8],
        "last_rewrite_targets": _clean_string_list(planner_payload.get("rewrite_targets") or [])[:8],
        "last_failure_hints": failure_hints[:8],
        "last_materialized_skill": dict(planner_payload.get("materialized_selected_skill") or {}),
        "last_accepted_custom_skills": accepted_custom_skills,
        "last_preferred_candidate_strategies": preferred_candidate_strategies[:3],
        "last_disfavored_candidate_strategies": disfavored_candidate_strategies[:3],
        "adaptive_bias": adaptive_bias,
    }


def _local_refine_replan_scope(evaluation_payload: dict) -> str:
    scores = evaluation_payload.get("scores", {}) or {}
    failure_hints = set(_clean_string_list(evaluation_payload.get("failure_tags") or []))
    carrier_grounding = float(scores.get("carrier_grounding", 0.0) or 0.0)
    world_realization = float(scores.get("world_realization", 0.0) or 0.0)
    transformation_strength = float(scores.get("transformation_strength", 0.0) or 0.0)
    localized_failure = (
        "detached_fx" in failure_hints
        or "weak_world_realization" in failure_hints
        or "under_transformation" in failure_hints
        or carrier_grounding < 0.6
        or world_realization < 0.6
        or transformation_strength < 0.6
    )
    return "candidate_and_final" if localized_failure else "final_only"


def _should_escalate_local_refine(
    *,
    loop_trace: dict,
    evaluation_payload: dict,
    reflection_payload: dict,
) -> bool:
    if reflection_payload.get("action") != "local_refine_prompt":
        return False
    attempts = list(loop_trace.get("attempts") or [])
    if len(attempts) < 2:
        return False

    recent = attempts[-2:]
    if any(item.get("action") != "local_refine_prompt" for item in recent):
        return False

    current_tags = set(_clean_string_list(evaluation_payload.get("failure_tags") or []))
    if not current_tags:
        return False

    previous_scores = []
    previous_tags = []
    for item in recent:
        eval_path = item.get("evaluation_response")
        if not eval_path:
            return False
        try:
            payload = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        except Exception:
            return False
        previous_scores.append(float(((payload.get("scores") or {}).get("overall_score", 0.0)) or 0.0))
        previous_tags.append(set(_clean_string_list(payload.get("failure_tags") or [])))

    current_score = float(((evaluation_payload.get("scores") or {}).get("overall_score", 0.0)) or 0.0)
    stagnant_score = all(current_score <= score + 0.05 for score in previous_scores)
    repeated_failure = all(current_tags == tags or current_tags.issubset(tags) or tags.issubset(current_tags) for tags in previous_tags)
    return repeated_failure and stagnant_score


def run_planner_and_edit_loop(
    *,
    sample_id: str,
    prompt: str,
    low_video: str | Path,
    output_dir: str | Path,
    variant: str = "wan14b",
    memory_path: str | Path | None = None,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    max_attempts: int = 3,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_stage(
        f"run.start sample_id={sample_id} variant={variant} max_attempts={max_attempts} output_dir={output_dir}"
    )
    memory = load_global_memory(memory_path) if memory_path else None
    runtime_memory_config = load_memory_runtime_config()
    critic_memory_mode = resolve_memory_mode(runtime_memory_config, module_name="critic").value
    edit_memory_mode = resolve_memory_mode(runtime_memory_config, module_name="edit").value
    memory_context = (
        retrieve_memory_context(
            memory,
            query=build_memory_query(prompt=prompt, variant=variant),
        )
        if memory
        else {}
    )
    loop_trace: dict = {"attempts": []}
    current_memory_context = deepcopy(memory_context)
    current_prompt = prompt
    reuse_planner_payload: dict | None = None
    partial_replan_scope: str | None = None
    attempt_adaptation_state: dict = {}
    critic_implicit_state: dict = {}
    edit_hidden_state: dict = init_edit_hidden_state()

    for attempt_index in range(1, max_attempts + 1):
        attempt_dir = output_dir / f"attempt_{attempt_index:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _log_stage(f"attempt_{attempt_index:03d}.start")

        if reuse_planner_payload is not None and partial_replan_scope:
            _log_stage(f"attempt_{attempt_index:03d}.planner.{partial_replan_scope}.start")
            plan = replan_video_from_existing(
                sample_id=sample_id,
                prompt=current_prompt,
                low_video=str(low_video),
                planner_payload=deepcopy(reuse_planner_payload),
                memory_context=current_memory_context,
                memory_store=memory,
                attempt_adaptation_state=attempt_adaptation_state,
                state_path=attempt_dir / "planner.state.json",
                scope=partial_replan_scope,
                variant=variant,
            )
            planner_payload = plan.model_dump()
            planner_response_path = str(attempt_dir / "planner.response.json")
            _write_json(attempt_dir / "planner.response.json", planner_payload)
            _log_stage(
                f"attempt_{attempt_index:03d}.planner.{partial_replan_scope}.done style_family={planner_payload.get('style_family', '')} selected_strategy={planner_payload.get('selected_strategy', '')}"
            )
            reuse_planner_payload = None
            partial_replan_scope = None
        elif reuse_planner_payload is not None:
            planner_payload = deepcopy(reuse_planner_payload)
            planner_response_path = str(attempt_dir / "planner.response.json")
            _write_json(attempt_dir / "planner.response.json", planner_payload)
            _log_stage(
                f"attempt_{attempt_index:03d}.planner.reused style_family={planner_payload.get('style_family', '')} selected_strategy={planner_payload.get('selected_strategy', '')}"
            )
            reuse_planner_payload = None
        else:
            _log_stage(f"attempt_{attempt_index:03d}.planner.start")
            plan = plan_video(
                sample_id=sample_id,
                prompt=current_prompt,
                low_video=str(low_video),
                memory_context=current_memory_context,
                memory_store=memory,
                attempt_adaptation_state=attempt_adaptation_state,
                state_path=attempt_dir / "planner.state.json",
                variant=variant,
            )
            planner_payload = plan.model_dump()
            planner_response_path = str(attempt_dir / "planner.response.json")
            _write_json(attempt_dir / "planner.response.json", planner_payload)
            _log_stage(
                f"attempt_{attempt_index:03d}.planner.done style_family={planner_payload.get('style_family', '')} selected_strategy={planner_payload.get('selected_strategy', '')}"
            )

        edit_prompt = (
            planner_payload.get("compiled_edit_prompt")
            or planner_payload.get("normalized_prompt")
            or current_prompt
        )
        _write_text(attempt_dir / "edit.prompt.txt", edit_prompt)
        _write_json(
            attempt_dir / "edit.prompt.meta.json",
            {
                "sample_id": sample_id,
                "attempt_index": attempt_index,
                "source_field": (
                    "compiled_edit_prompt"
                    if planner_payload.get("compiled_edit_prompt")
                    else ("normalized_prompt" if planner_payload.get("normalized_prompt") else "current_prompt")
                ),
                "selected_strategy": planner_payload.get("selected_strategy", ""),
                "style_family": planner_payload.get("style_family", ""),
                "prompt_length": len(edit_prompt),
                "compiled_edit_prompt_present": bool(planner_payload.get("compiled_edit_prompt")),
                "normalized_prompt_present": bool(planner_payload.get("normalized_prompt")),
            },
        )
        edit_request = EditRequest(
            low_video=Path(low_video),
            output_video=attempt_dir / "edited.mp4",
            positive_prompt=edit_prompt,
            negative_prompt=negative_prompt,
            variant=variant,
            memory_mode=edit_memory_mode,
            implicit_controls=_resolve_edit_controls(
                edit_memory_mode=edit_memory_mode,
                runtime_memory_config=runtime_memory_config,
                memory_context=current_memory_context,
                planner_payload=planner_payload,
                critic_implicit_state=critic_implicit_state,
                edit_hidden_state=edit_hidden_state,
                attempt_adaptation_state=attempt_adaptation_state,
            ),
        )
        _write_json(attempt_dir / "edit.controls.json", edit_request.implicit_controls or {})
        _log_stage(f"attempt_{attempt_index:03d}.edit.start")
        edit_result = render_edit_video(edit_request)
        _write_json(attempt_dir / "edit.response.json", edit_result)
        edit_hidden_report = dict(edit_result.get("implicit_hidden_memory", {}) or {})
        _write_json(attempt_dir / "edit.implicit_hidden_memory.json", edit_hidden_report)
        _log_stage(
            f"attempt_{attempt_index:03d}.edit.done output_video={edit_result.get('output_video', '')}"
        )

        _log_stage(f"attempt_{attempt_index:03d}.critic.evaluate.start")
        critic_query = build_critic_memory_query(
            sample_id=sample_id,
            variant=variant,
            prompt=edit_prompt,
            planner_payload=planner_payload,
            attempt_adaptation_state=attempt_adaptation_state,
        )
        critic_runtime_state = dict(critic_implicit_state)
        critic_runtime_state["query"] = critic_query
        evaluation = evaluate_edit_result(
            sample_id=sample_id,
            prompt=edit_prompt,
            low_video=low_video,
            edited_video=edit_result["output_video"],
            planner_payload=planner_payload,
            edit_payload=edit_result,
            memory_mode=critic_memory_mode,
            explicit_context=current_memory_context,
            implicit_state=critic_runtime_state,
        )
        evaluation_payload = evaluation.to_dict()
        _write_json(attempt_dir / "evaluation.response.json", evaluation_payload)
        _log_stage(
            f"attempt_{attempt_index:03d}.critic.evaluate.done overall_score={evaluation_payload.get('scores', {}).get('overall_score', 0.0)}"
        )

        _log_stage(f"attempt_{attempt_index:03d}.critic.reflect.start")
        reflection = reflect_edit_result(
            evaluation,
            memory_mode=critic_memory_mode,
            explicit_context=current_memory_context,
            implicit_state=critic_runtime_state,
        )
        reflection_payload = reflection.to_dict()
        if reflection.action == "local_refine_prompt":
            reflection_payload["replan_scope"] = _local_refine_replan_scope(evaluation_payload)
        elif reflection.action == "global_replan":
            reflection_payload["replan_scope"] = "full"
        if _should_escalate_local_refine(
            loop_trace=loop_trace,
            evaluation_payload=evaluation_payload,
            reflection_payload=reflection_payload,
        ):
            reflection.action = "global_replan"
            reflection.reason = (
                "Repeated local refinements with similar failure tags and no meaningful score improvement; escalating to global replanning."
            )
            reflection.replan_directives = _clean_string_list(
                list(reflection.replan_directives)
                + list(evaluation.failure_tags)
                + ["rebuild_prompt_with_stronger_world_realization", "force_stronger_visible_fx_and_carrier_binding"]
            )
            reflection_payload = reflection.to_dict()
            reflection_payload["replan_scope"] = "full"
        _write_json(attempt_dir / "reflection.response.json", reflection_payload)
        _log_stage(
            f"attempt_{attempt_index:03d}.critic.reflect.done action={reflection_payload.get('action', '')}"
        )

        memory_contract_payload = build_attempt_memory_contract(
            sample_id=sample_id,
            attempt_index=attempt_index,
            variant=variant,
            prompt=edit_prompt,
            planner_payload=planner_payload,
            critic_query=critic_query,
            critic_memory_mode=critic_memory_mode,
            critic_evaluation_payload=evaluation_payload,
            critic_reflection_payload=reflection_payload,
            critic_implicit_state_before=critic_implicit_state,
            edit_memory_mode=edit_memory_mode,
            edit_controls=edit_request.implicit_controls or {},
            edit_payload=edit_result,
            edit_hidden_report=edit_hidden_report,
            edit_hidden_state_before=edit_hidden_state,
        )

        edit_hidden_state = update_edit_hidden_state(
            edit_hidden_state=edit_hidden_state,
            runtime_memory_config=runtime_memory_config,
            sample_id=sample_id,
            variant=variant,
            attempt_index=attempt_index,
            planner_payload=planner_payload,
            evaluation_payload=evaluation_payload,
            reflection_payload=reflection_payload,
            memory_report=edit_hidden_report,
            memory_contract=memory_contract_payload,
        )
        _write_json(attempt_dir / "edit.hidden_state.json", edit_hidden_state)

        critic_implicit_state = pack_critic_implicit_state(
            previous_state=critic_implicit_state,
            sample_id=sample_id,
            attempt_index=attempt_index,
            planner_payload=planner_payload,
            evaluation_payload=evaluation_payload,
            reflection_payload=reflection_payload,
            memory_contract=memory_contract_payload,
        )
        _write_json(attempt_dir / "critic.implicit_state.json", critic_implicit_state)
        memory_contract_payload = build_attempt_memory_contract(
            sample_id=sample_id,
            attempt_index=attempt_index,
            variant=variant,
            prompt=edit_prompt,
            planner_payload=planner_payload,
            critic_query=critic_query,
            critic_memory_mode=critic_memory_mode,
            critic_evaluation_payload=evaluation_payload,
            critic_reflection_payload=reflection_payload,
            critic_implicit_state_before=memory_contract_payload.get("critic", {}).get("implicit_state_before", {}),
            critic_implicit_state_after=critic_implicit_state,
            edit_memory_mode=edit_memory_mode,
            edit_controls=edit_request.implicit_controls or {},
            edit_payload=edit_result,
            edit_hidden_report=edit_hidden_report,
            edit_hidden_state_before=memory_contract_payload.get("edit", {}).get("hidden_state_before", {}),
            edit_hidden_state_after=edit_hidden_state,
        )
        _write_json(attempt_dir / "memory.contract.json", memory_contract_payload)

        _log_stage(f"attempt_{attempt_index:03d}.memory.analyze.start")
        memory_analysis = analyze_loop_attempt(
            sample_id=sample_id,
            attempt_index=attempt_index,
            variant=variant,
            prompt=current_prompt,
            planner_payload=planner_payload,
            edit_payload=edit_result,
            evaluation_payload=evaluation_payload,
            reflection_payload=reflection_payload,
        )
        _write_json(attempt_dir / "memory.analysis.json", memory_analysis.to_dict())
        _log_stage(f"attempt_{attempt_index:03d}.memory.analyze.done")
        if memory is not None and memory_path:
            _log_stage(f"attempt_{attempt_index:03d}.memory.evolve.start")
            memory = evolve_memory(memory, memory_analysis)
            save_global_memory(memory_path, memory)
            _log_stage(f"attempt_{attempt_index:03d}.memory.evolve.done")

        attempt_adaptation_state = _extract_attempt_adaptation_state(
            planner_payload=planner_payload,
            evaluation_payload=evaluation_payload,
            reflection_payload=reflection_payload,
        )
        critic_implicit_state["last_good_carriers"] = list(attempt_adaptation_state.get("last_good_carriers", []) or [])[:4]
        critic_implicit_state["last_bad_carriers"] = list(attempt_adaptation_state.get("last_bad_carriers", []) or [])[:4]

        loop_trace["attempts"].append(
            {
                "attempt_index": attempt_index,
                "planner_response": str(attempt_dir / "planner.response.json"),
                "edit_response": str(attempt_dir / "edit.response.json"),
                "evaluation_response": str(attempt_dir / "evaluation.response.json"),
                "reflection_response": str(attempt_dir / "reflection.response.json"),
                "critic_implicit_state": str(attempt_dir / "critic.implicit_state.json"),
                "edit_hidden_state": str(attempt_dir / "edit.hidden_state.json"),
                "memory_contract": str(attempt_dir / "memory.contract.json"),
                "action": reflection.action,
                "replan_scope": reflection_payload.get("replan_scope", ""),
                "critic_memory_mode": critic_memory_mode,
            }
        )
        _write_json(output_dir / "loop.trace.json", loop_trace)

        if reflection.action == "accept":
            _log_stage(f"attempt_{attempt_index:03d}.accept")
            result = {
                "status": "accepted",
                "attempt_index": attempt_index,
                "planner_response": str(attempt_dir / "planner.response.json"),
                "edit_response": str(attempt_dir / "edit.response.json"),
                "evaluation_response": str(attempt_dir / "evaluation.response.json"),
                "reflection_response": str(attempt_dir / "reflection.response.json"),
                "memory_contract": str(attempt_dir / "memory.contract.json"),
            }
            _log_stage("run.done status=accepted")
            return result

        if reflection.action == "local_refine_prompt":
            _log_stage(f"attempt_{attempt_index:03d}.local_refine_prompt")
            refined_prompt = _refined_prompt(edit_prompt, reflection.prompt_patch)
            planner_payload = deepcopy(planner_payload)
            planner_payload["compiled_edit_prompt"] = refined_prompt
            planner_payload["normalized_prompt"] = refined_prompt
            reuse_planner_payload = planner_payload
            partial_replan_scope = _local_refine_replan_scope(evaluation_payload)
            current_prompt = refined_prompt
            current_memory_context = deepcopy(current_memory_context)
            current_memory_context["critic_prompt_patch"] = reflection.prompt_patch
            current_memory_context["attempt_adaptation_state"] = deepcopy(attempt_adaptation_state)
            continue

        _log_stage(f"attempt_{attempt_index:03d}.global_replan")
        reuse_planner_payload = None
        partial_replan_scope = None
        current_memory_context = deepcopy(current_memory_context)
        current_memory_context["critic_replan_directives"] = reflection.replan_directives
        current_memory_context["critic_failure_hints"] = evaluation.failure_tags
        current_memory_context["attempt_adaptation_state"] = deepcopy(attempt_adaptation_state)

    result = {"status": "max_attempts_reached", "attempts": loop_trace["attempts"]}
    _log_stage("run.done status=max_attempts_reached")
    return result
