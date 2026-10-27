from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vlm.local_qwen import extract_frames
from memory.contracts import build_critic_evaluation_signal
from skills.dimension_parse import KEEP as _KEEP
from skills.dimensions import policy_for

from critic.prompts import (
    build_edit_evaluator_repair_system_prompt,
    build_edit_evaluator_repair_user_prompt,
    build_edit_evaluator_system_prompt,
    build_edit_evaluator_user_prompt,
)
from critic.model_loader import CriticScoreModel, CriticTextModel
from critic.implicit_memory import apply_critic_memory_mode, decode_routing_bias


@dataclass
class EvaluationScores:
    prompt_alignment: float = 0.0
    structure_preservation: float = 0.0
    transformation_strength: float = 0.0
    carrier_grounding: float = 0.0
    world_realization: float = 0.0
    temporal_coherence: float = 0.0
    artifact_penalty: float = 0.0
    overall_score: float = 0.0


@dataclass
class HoldCheck:
    """Whether one dimension the instruction asked to keep was actually kept.

    ``reported`` separates "the critic said it was held" from "the critic never
    mentioned it". Silence is not evidence either way, so an unreported dimension
    is recorded for the memory loop but does not raise a failure tag -- otherwise
    a terse reply would look like every hold had been violated.
    """

    dimension: str
    held: bool = True
    reported: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "held": self.held,
            "reported": self.reported,
            "note": self.note,
        }


@dataclass
class EvaluationResult:
    sample_id: str
    variant: str = ""
    scores: EvaluationScores = field(default_factory=EvaluationScores)
    failure_tags: list[str] = field(default_factory=list)
    hold_checks: list[HoldCheck] = field(default_factory=list)
    reflection_hints: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metric_hints: dict = field(default_factory=dict)
    raw_vlm_response: str = ""
    evaluation_incomplete: bool = False
    missing_headings: list[str] = field(default_factory=list)
    memory_mode: str = ""
    routing_bias: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scores"] = asdict(self.scores)
        return payload

    def to_memory_signal(self) -> dict:
        return build_critic_evaluation_signal(self.to_dict())


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _split_items(value: str) -> list[str]:
    if not value:
        return []
    text = value.replace("；", ";").replace("，", ",").replace("、", ",").replace("|", ",")
    parts = re.split(r"[\n,;]", text)
    return [item.strip(" -*") for item in parts if item.strip(" -*")]


def _normalize_failure_tag(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"\(.*?\)", "", lowered).strip()
    if not lowered:
        return ""
    match = re.search(r"[a-z]+(?:_[a-z]+)+", lowered)
    if match:
        return match.group(0)
    lowered = re.sub(r"[^a-z0-9\s_]+", " ", lowered)
    lowered = re.sub(r"\s+", "_", lowered).strip("_")
    if lowered in {"", "under", "weak", "if", "world", "not", "attached"}:
        return ""
    return lowered if "_" in lowered else ""


def _clean_failure_tags(value: str) -> list[str]:
    raw_items = _split_items(value)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        tag = _normalize_failure_tag(item)
        if tag and tag not in seen:
            cleaned.append(tag)
            seen.add(tag)
    return cleaned


def _clean_short_items(value: str, *, limit: int = 3) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in _split_items(value):
        text = _clean_text(item)
        if not text:
            continue
        text = re.sub(r"^(wait|now|maybe|let'?s check)\b[:, ]*", "", text, flags=re.IGNORECASE).strip()
        if not text:
            continue
        if len(text) > 160:
            text = text[:157].rstrip() + "..."
        if text not in seen:
            cleaned.append(text)
            seen.add(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _parse_float(value: str) -> float:
    try:
        return max(0.0, min(1.0, float(_clean_text(value))))
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
        if not match:
            return 0.0
        return max(0.0, min(1.0, float(match.group(0))))


def _parse_section_fields(text: str, headings: list[str]) -> dict[str, str]:
    matches = list(re.finditer(r"(?m)^([A-Z_]+):\s*(.*)$", text))
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


def _missing_heading_names(fields: dict[str, str], headings: list[str]) -> list[str]:
    missing = []
    for heading in headings:
        value = str(fields.get(heading, "") or "").strip()
        if not value:
            missing.append(heading)
    return missing


def _probe_ffmpeg_metric(command: list[str], pattern: str) -> float | None:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _ffmpeg_ssim(reference_video: Path, distorted_video: Path) -> float | None:
    command = [
        "ffmpeg",
        "-i",
        str(reference_video),
        "-i",
        str(distorted_video),
        "-lavfi",
        "[0:v][1:v]ssim",
        "-f",
        "null",
        "-",
    ]
    return _probe_ffmpeg_metric(command, r"All:([0-9.]+)")


def _ffmpeg_self_msad(video_path: Path) -> float | None:
    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-filter_complex",
        "[0:v]split[a][b];[a]trim=end_frame=80,setpts=PTS-STARTPTS[a1];[b]trim=start_frame=1,setpts=PTS-STARTPTS[b1];[a1][b1]msad",
        "-f",
        "null",
        "-",
    ]
    return _probe_ffmpeg_metric(command, r"average:([0-9.]+)")


def _metric_hints(low_video: Path, edited_video: Path) -> dict:
    ssim = _ffmpeg_ssim(low_video, edited_video)
    msad = _ffmpeg_self_msad(edited_video)
    temporal_proxy = None
    if msad is not None:
        temporal_proxy = max(0.0, min(1.0, 1.0 - min(msad / 12000.0, 1.0)))
    return {
        "ffmpeg_ssim": ssim,
        "edited_self_msad": msad,
        "temporal_proxy": temporal_proxy,
    }


def _planner_summary(planner_payload: dict | None) -> dict:
    planner_payload = planner_payload or {}
    return {
        "style_family": planner_payload.get("style_family", ""),
        "selected_strategy": planner_payload.get("selected_strategy", ""),
        "normalized_prompt": planner_payload.get("normalized_prompt", ""),
        "primary_fx_carriers": planner_payload.get("primary_fx_carriers", []),
        "rewrite_targets": planner_payload.get("rewrite_targets", []),
        "light_touch_targets": planner_payload.get("light_touch_targets", []),
        "forbidden_changes": planner_payload.get("forbidden_changes", []),
    }


def _hold_requirements_from(overrides: dict | None) -> dict[str, str]:
    """{dimension: keep phrase} for the dimensions the instruction told to hold."""
    requirements: dict[str, str] = {}
    for dimension, verdict in (overrides or {}).items():
        if verdict != _KEEP:
            continue
        policy = policy_for(dimension)
        phrase = (policy.keep_phrase if policy is not None else "") or dimension.replace("_", " ")
        requirements[str(dimension)] = f"keep {phrase} unchanged"
    return requirements


def _edit_summary(edit_payload: dict | None) -> dict:
    edit_payload = edit_payload or {}
    return {
        "variant": edit_payload.get("variant", ""),
        "backend": edit_payload.get("backend", ""),
        "resolution": edit_payload.get("resolution", []),
        "num_frames": edit_payload.get("num_frames"),
        "num_inference_steps": edit_payload.get("num_inference_steps"),
    }


def _merge_image_paths(low_video: Path, edited_video: Path, max_frames_per_video: int = 3) -> list[str]:
    low_frames = extract_frames(str(low_video), max_frames=max_frames_per_video)
    edited_frames = extract_frames(str(edited_video), max_frames=max_frames_per_video)
    return low_frames + edited_frames


_HOLD_LINE = re.compile(
    r"^-\s*([a-z_]+)\s*:\s*(HELD|VIOLATED)\b[:\s]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_hold_checks(text: str, dimensions: list[str]) -> list[HoldCheck]:
    """Parse the HOLD_CHECK section, one entry per required dimension."""
    found: dict[str, HoldCheck] = {}
    for match in _HOLD_LINE.finditer(text or ""):
        dimension = match.group(1).strip().lower()
        note = (match.group(3) or "").strip().strip("()").strip()
        found[dimension] = HoldCheck(
            dimension=dimension,
            held=match.group(2).upper() == "HELD",
            reported=True,
            note=note[:200],
        )
    checks: list[HoldCheck] = []
    for dimension in dimensions:
        if dimension in found:
            checks.append(found[dimension])
        else:
            checks.append(
                HoldCheck(dimension=dimension, held=True, reported=False, note="not reported")
            )
    return checks


def hold_failure_tags(checks: list[HoldCheck]) -> list[str]:
    """`hold_violated:<dimension>` for each hold the critic says was broken.

    Deliberately not run through ``_normalize_failure_tag``: that strips
    everything down to snake_case words and would turn the dimension suffix into
    noise. The dimension is the point -- it is what tells the skill loop which
    field of a style template to change.
    """
    return [
        f"hold_violated:{check.dimension}"
        for check in checks
        if check.reported and not check.held
    ]


def _scores_to_failure_tags(scores: EvaluationScores) -> list[str]:
    tags: list[str] = []
    if scores.prompt_alignment < 0.45:
        tags.append("prompt_misalignment")
    if scores.transformation_strength < 0.45:
        tags.append("under_transformation")
    if scores.carrier_grounding < 0.45:
        tags.append("detached_fx")
    if scores.world_realization < 0.45:
        tags.append("weak_world_realization")
    if scores.structure_preservation < 0.45:
        tags.append("identity_drift")
    if scores.artifact_penalty < 0.35:
        tags.append("artifact_heavy")
    return tags


def _scores_to_reflection_hints(scores: EvaluationScores) -> list[str]:
    hints: list[str] = []
    if scores.prompt_alignment < 0.5:
        hints.append("更直接匹配提示词要求的可见变化")
    if scores.world_realization < 0.5:
        hints.append("增强背景材质、光照和环境层的世界观变化")
    if scores.carrier_grounding < 0.5:
        hints.append("把特效更明确地绑定到可见载体上")
    if scores.transformation_strength < 0.5:
        hints.append("提高可见变换强度，不要只做轻微润色")
    return hints[:3]


def _parse_evaluation_text(
    text: str,
    sample_id: str,
    variant: str,
    metric_hints: dict,
    hold_dimensions: list[str] | None = None,
) -> EvaluationResult:
    headings = [
        "PROMPT_ALIGNMENT",
        "STRUCTURE_PRESERVATION",
        "TRANSFORMATION_STRENGTH",
        "CARRIER_GROUNDING",
        "WORLD_REALIZATION",
        "TEMPORAL_COHERENCE",
        "ARTIFACT_PENALTY",
        "OVERALL_SCORE",
        "FAILURE_TAGS",
        "REFLECTION_HINTS",
        "NOTES",
    ]
    # Only demanded when the instruction actually named something to keep, so a
    # run with no holds does not get flagged incomplete for a section it was
    # never asked for.
    if hold_dimensions:
        headings.append("HOLD_CHECK")
    fields = _parse_section_fields(text, headings)
    missing_headings = _missing_heading_names(fields, headings)
    evaluation_incomplete = bool(missing_headings)
    scores = EvaluationScores(
        prompt_alignment=_parse_float(fields.get("PROMPT_ALIGNMENT", "")),
        structure_preservation=_parse_float(fields.get("STRUCTURE_PRESERVATION", "")),
        transformation_strength=_parse_float(fields.get("TRANSFORMATION_STRENGTH", "")),
        carrier_grounding=_parse_float(fields.get("CARRIER_GROUNDING", "")),
        world_realization=_parse_float(fields.get("WORLD_REALIZATION", "")),
        temporal_coherence=_parse_float(fields.get("TEMPORAL_COHERENCE", "")),
        artifact_penalty=_parse_float(fields.get("ARTIFACT_PENALTY", "")),
        overall_score=_parse_float(fields.get("OVERALL_SCORE", "")),
    )
    if (not fields.get("TEMPORAL_COHERENCE")) and metric_hints.get("temporal_proxy") is not None:
        scores.temporal_coherence = float(metric_hints["temporal_proxy"])
    if (not fields.get("STRUCTURE_PRESERVATION")) and metric_hints.get("ffmpeg_ssim") is not None:
        scores.structure_preservation = max(0.0, min(1.0, float(metric_hints["ffmpeg_ssim"])))
    failure_tags = _clean_failure_tags(fields.get("FAILURE_TAGS", ""))
    hold_checks = (
        _parse_hold_checks(fields.get("HOLD_CHECK", ""), hold_dimensions)
        if hold_dimensions
        else []
    )
    for tag in hold_failure_tags(hold_checks):
        if tag not in failure_tags:
            failure_tags.append(tag)
    if evaluation_incomplete and "evaluation_incomplete" not in failure_tags:
        failure_tags.append("evaluation_incomplete")
    return EvaluationResult(
        sample_id=sample_id,
        variant=variant,
        scores=scores,
        failure_tags=failure_tags,
        hold_checks=hold_checks,
        reflection_hints=_clean_short_items(fields.get("REFLECTION_HINTS", ""), limit=3),
        notes=_clean_short_items(fields.get("NOTES", ""), limit=3),
        metric_hints=metric_hints,
        raw_vlm_response=text[:12000],
        evaluation_incomplete=evaluation_incomplete,
        missing_headings=missing_headings,
    )


def _merge_repair_text(original_text: str, repair_text: str) -> str:
    return original_text.rstrip() + "\n" + repair_text.strip()


def _repair_incomplete_evaluation(
    *,
    model: Any,
    raw_text: str,
    missing_headings: list[str],
) -> tuple[str, str] | tuple[None, None]:
    if not missing_headings:
        return None, None
    repair_text = model.predict_text(
        system_prompt=build_edit_evaluator_repair_system_prompt(),
        user_text=build_edit_evaluator_repair_user_prompt(raw_response=raw_text, missing_headings=missing_headings),
        image_paths=[],
        max_new_tokens=1024,
    )
    return _merge_repair_text(raw_text, repair_text), repair_text


def evaluate_edit_result(
    *,
    sample_id: str,
    prompt: str,
    low_video: str | Path,
    edited_video: str | Path,
    planner_payload: dict | None = None,
    edit_payload: dict | None = None,
    max_frames_per_video: int = 3,
    memory_mode: str = "explicit",
    explicit_context: dict | None = None,
    implicit_state: dict | None = None,
    hold_requirements: dict[str, str] | None = None,
) -> EvaluationResult:
    low_video = Path(low_video)
    edited_video = Path(edited_video)
    if hold_requirements is None:
        # Fall back to whatever the planner recorded, so a caller that already
        # threads planner_payload through gets hold checking without extra work.
        hold_requirements = _hold_requirements_from(
            (planner_payload or {}).get("dimension_overrides")
        )
    hold_dimensions = sorted(hold_requirements) or None
    metric_hints = _metric_hints(low_video, edited_video)
    image_paths = _merge_image_paths(low_video, edited_video, max_frames_per_video=max_frames_per_video)
    user_prompt = build_edit_evaluator_user_prompt(
        sample_id=sample_id,
        prompt=prompt,
        planner_summary=_planner_summary(planner_payload),
        edit_summary=_edit_summary(edit_payload),
        metric_hints=metric_hints,
        hold_requirements=hold_requirements or None,
    )
    user_prompt = apply_critic_memory_mode(
        user_prompt,
        memory_mode=memory_mode,
        explicit_context=explicit_context,
        implicit_state=implicit_state,
    )
    variant = str((edit_payload or {}).get("variant", ""))

    def _annotate_result(result: EvaluationResult) -> EvaluationResult:
        result.memory_mode = str(memory_mode or "")
        result.routing_bias = decode_routing_bias(
            implicit_state=implicit_state,
            evaluation_payload=result.to_dict(),
        )
        return result

    model = CriticTextModel(kind="evaluator")
    if model.available():
        system_prompt = build_edit_evaluator_system_prompt(with_hold_check=bool(hold_dimensions))
        text = model.predict_text(system_prompt=system_prompt, user_text=user_prompt, image_paths=image_paths, max_new_tokens=4096)
        result = _parse_evaluation_text(
            text, sample_id=sample_id, variant=variant, metric_hints=metric_hints,
            hold_dimensions=hold_dimensions,
        )
        if result.evaluation_incomplete:
            repaired_text, repair_raw = _repair_incomplete_evaluation(
                model=model,
                raw_text=text,
                missing_headings=result.missing_headings,
            )
            if repaired_text:
                result = _parse_evaluation_text(
                repaired_text, sample_id=sample_id, variant=variant, metric_hints=metric_hints,
                hold_dimensions=hold_dimensions,
            )
                if repair_raw:
                    result.raw_vlm_response = (repaired_text[:11000] + "\n\n[REPAIR_APPEND]\n" + repair_raw[:1000])[:12000]
        return _annotate_result(result)

    fast_model = CriticScoreModel(kind="evaluator")
    if fast_model.available():
        predicted = fast_model.predict_scores(user_text=user_prompt, image_paths=image_paths)
        scores = EvaluationScores(
            prompt_alignment=float(predicted.get("prompt_alignment", 0.0) or 0.0),
            structure_preservation=float(predicted.get("structure_preservation", 0.0) or 0.0),
            transformation_strength=float(predicted.get("transformation_strength", 0.0) or 0.0),
            carrier_grounding=float(predicted.get("carrier_grounding", 0.0) or 0.0),
            world_realization=float(predicted.get("world_realization", 0.0) or 0.0),
            temporal_coherence=float(predicted.get("temporal_coherence", 0.0) or 0.0),
            artifact_penalty=float(predicted.get("artifact_penalty", 0.0) or 0.0),
            overall_score=float(predicted.get("overall_score", 0.0) or 0.0),
        )
        result = EvaluationResult(
            sample_id=sample_id,
            variant=variant,
            scores=scores,
            failure_tags=_scores_to_failure_tags(scores),
            reflection_hints=_scores_to_reflection_hints(scores),
            notes=["fast_evaluator_head_prediction"],
            metric_hints=metric_hints,
            raw_vlm_response="FAST_EVALUATOR_HEAD",
            evaluation_incomplete=False,
            missing_headings=[],
        )
        return _annotate_result(result)

    system_prompt = build_edit_evaluator_system_prompt(with_hold_check=bool(hold_dimensions))
    text = model.predict_text(system_prompt=system_prompt, user_text=user_prompt, image_paths=image_paths, max_new_tokens=4096)
    result = _parse_evaluation_text(
            text, sample_id=sample_id, variant=variant, metric_hints=metric_hints,
            hold_dimensions=hold_dimensions,
        )
    if result.evaluation_incomplete:
        repaired_text, repair_raw = _repair_incomplete_evaluation(
            model=model,
            raw_text=text,
            missing_headings=result.missing_headings,
        )
        if repaired_text:
            result = _parse_evaluation_text(
                repaired_text, sample_id=sample_id, variant=variant, metric_hints=metric_hints,
                hold_dimensions=hold_dimensions,
            )
            if repair_raw:
                result.raw_vlm_response = (repaired_text[:11000] + "\n\n[REPAIR_APPEND]\n" + repair_raw[:1000])[:12000]
    return _annotate_result(result)


def evaluation_to_json(result: EvaluationResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
