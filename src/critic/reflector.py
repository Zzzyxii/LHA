from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from critic.evaluator import EvaluationResult
from critic.implicit_memory import apply_critic_memory_mode, decode_routing_bias
from critic.model_loader import CriticTextModel
from critic.prompts import build_critic_identity_prefix
from memory.contracts import build_critic_reflection_signal


@dataclass
class ReflectionResult:
    action: str
    reason: str
    prompt_patch: list[str] = field(default_factory=list)
    replan_directives: list[str] = field(default_factory=list)
    memory_mode: str = ""
    routing_bias: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_memory_signal(self) -> dict:
        return build_critic_reflection_signal(self.to_dict())


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def reflect_edit_result(
    evaluation: EvaluationResult,
    *,
    accept_threshold: float = 0.78,
    local_refine_floor: float = 0.52,
    memory_mode: str = "explicit",
    explicit_context: dict | None = None,
    implicit_state: dict | None = None,
) -> ReflectionResult:
    routing_bias = decode_routing_bias(
        implicit_state=implicit_state,
        evaluation_payload=evaluation.to_dict(),
    )

    def _annotate(result: ReflectionResult) -> ReflectionResult:
        result.memory_mode = str(memory_mode or "")
        result.routing_bias = routing_bias
        if routing_bias.get("prefer_global_replan") and result.action == "local_refine_prompt":
            result.action = "global_replan"
            result.reason = (
                result.reason
                + " Memory routing bias escalated this attempt to global_replan due to repeated local cycles."
            ).strip()
            result.replan_directives = _dedupe(
                list(result.replan_directives)
                + ["break_local_refine_loop", "rebuild_global_style_and_strategy"]
            )
        if routing_bias.get("prefer_local_refine") and result.action == "global_replan":
            result.action = "local_refine_prompt"
            result.reason = (
                result.reason
                + " Memory routing bias downgraded to local_refine_prompt because recent local refinements showed recoverable trend."
            ).strip()
            result.prompt_patch = _dedupe(list(result.prompt_patch) + list(evaluation.reflection_hints))
        return result

    model = CriticTextModel(kind="reflector")
    if model.available():
        evaluation_payload = evaluation.to_dict()
        system_prompt = (
            build_critic_identity_prefix()
            +
            "根据给定的结构化评估结果，输出一个 JSON object，字段必须是："
            "action, reason, prompt_patch, replan_directives。"
            "action 只能是 accept、local_refine_prompt、global_replan。"
        )
        user_prompt = (
            "evaluation_result:\n"
            + json.dumps(evaluation_payload, ensure_ascii=False)
            + "\n请严格输出 JSON，不要输出额外文本。"
        )
        user_prompt = apply_critic_memory_mode(
            user_prompt,
            memory_mode=memory_mode,
            explicit_context=explicit_context,
            implicit_state=implicit_state,
        )
        try:
            text = model.predict_text(system_prompt=system_prompt, user_text=user_prompt, image_paths=[], max_new_tokens=768)
            payload = json.loads(text[text.find("{") : text.rfind("}") + 1])
            action = str(payload.get("action", "")).strip()
            if action in {"accept", "local_refine_prompt", "global_replan"}:
                return _annotate(
                    ReflectionResult(
                    action=action,
                    reason=str(payload.get("reason", "")).strip(),
                    prompt_patch=_dedupe([str(x).strip() for x in (payload.get("prompt_patch") or []) if str(x).strip()]),
                    replan_directives=_dedupe(
                        [str(x).strip() for x in (payload.get("replan_directives") or []) if str(x).strip()]
                    ),
                    )
                )
        except Exception:
            pass

    scores = evaluation.scores
    failure_tags = set(evaluation.failure_tags)
    evaluation_incomplete = bool(getattr(evaluation, "evaluation_incomplete", False) or "evaluation_incomplete" in failure_tags)

    severe_prompt_failure = scores.prompt_alignment < 0.45 or "prompt_misalignment" in failure_tags
    weak_world = scores.world_realization < 0.45 or "weak_world_realization" in failure_tags
    weak_transform = scores.transformation_strength < 0.45 or "under_transformation" in failure_tags
    weak_grounding = scores.carrier_grounding < 0.45 or "detached_fx" in failure_tags
    heavy_artifact = (not evaluation_incomplete) and (scores.artifact_penalty < 0.35 or "artifact_heavy" in failure_tags)
    structure_strong = scores.structure_preservation >= 0.75
    artifacts_clean = scores.artifact_penalty >= 0.7
    directionally_recoverable = structure_strong and artifacts_clean

    if evaluation_incomplete:
        patches = [
            "Restate the target world and visible object changes more literally and compactly.",
            "Make the named prop transformation, world-building cues, and costume upgrade clearly visible.",
        ]
        patches.extend(evaluation.reflection_hints)
        return _annotate(
            ReflectionResult(
            action="local_refine_prompt",
            reason="Critic output was incomplete, so avoid treating missing tail fields as hard failures. Try a compact prompt refinement first.",
            prompt_patch=_dedupe(patches),
            )
        )

    if scores.overall_score >= accept_threshold and not failure_tags:
        return _annotate(ReflectionResult(action="accept", reason="Overall score is strong and no failure tags were raised."))

    if heavy_artifact:
        directives: list[str] = []
        if heavy_artifact:
            directives.extend(
                [
                    "rebuild_prompt_with_cleaner_visual_design_and_lower_artifact_risk",
                    "replace_generic_glow_or_overprocessed_surface_language_with_device_attached_or_environment_attached_fx",
                    "keep facial identity and coarse geometry stable while reducing unstable high-frequency texture changes",
                ]
            )
        return _annotate(
            ReflectionResult(
            action="global_replan",
            reason="Current result has artifact-level problems that should be fixed by replanning rather than prompt-only patching.",
            replan_directives=_dedupe(directives + evaluation.reflection_hints),
            )
        )

    if directionally_recoverable and (severe_prompt_failure or weak_world or weak_grounding or weak_transform):
        patches: list[str] = []
        if severe_prompt_failure:
            patches.append("Match the prompt more literally. Make the named object, world change, and costume upgrade visibly present on screen.")
        if weak_world:
            patches.append("Make the target world more legible through background design, materials, lighting, and set-level changes.")
        if weak_grounding:
            patches.append("Attach effects to visible carriers such as props, body surfaces, lighting, or environment anchors. Avoid floating unmotivated FX.")
        if weak_transform:
            patches.append("Increase visible transformation strength. Make the new world/style clearly readable, not just lightly polished.")
        patches.extend(evaluation.reflection_hints)
        return _annotate(
            ReflectionResult(
            action="local_refine_prompt",
            reason="The result preserves structure and stays visually clean, so it is better treated as a prompt-compilation problem than a full replanning failure.",
            prompt_patch=_dedupe(patches),
            )
        )

    if scores.overall_score >= local_refine_floor or weak_transform:
        patches: list[str] = []
        if weak_transform:
            patches.append("Increase visible transformation strength. Make the new world/style clearly readable, not just lightly polished.")
        if scores.structure_preservation < 0.55:
            patches.append("Preserve identity, pose, framing, and coarse layout more carefully while keeping the style change visible.")
        if scores.carrier_grounding < 0.6:
            patches.append("Attach effects to visible carriers such as props, body surfaces, lighting, or environment anchors. Avoid floating unmotivated FX.")
        if scores.world_realization < 0.6:
            patches.append("Make the target world more legible through background design, materials, lighting, and set-level changes.")
        patches.extend(evaluation.reflection_hints)
        return _annotate(
            ReflectionResult(
            action="local_refine_prompt",
            reason="The result is recoverable with a stronger edit prompt patch.",
            prompt_patch=_dedupe(patches),
            )
        )

    return _annotate(
        ReflectionResult(
        action="global_replan",
        reason="Overall quality is too weak to salvage with prompt-only refinement.",
        replan_directives=_dedupe(
            evaluation.reflection_hints
            or [
                "rebuild_prompt_with_stronger_world_realization",
                "make_primary_prop_geometry_and_material_language_explicit",
                "replace_generic_costume_language_with_visible_silhouette_and_structure_changes",
                "attach all visible fx to concrete carriers in props, subjects, or room surfaces",
            ]
        ),
        )
    )


def reflection_to_json(result: ReflectionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
