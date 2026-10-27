#!/usr/bin/env python3
"""Low-high critic evaluator.

This is separate from ``scripts/eval/ditto_quant``. The Ditto evaluator is for
offline edit-model checkpoint comparison. This script computes the four
low-high critic dimensions used by the agent/reflection loop:

1. High-vs-low distribution alignment.
2. Temporal consistency.
3. High-cost aesthetic quality.
4. Edit semantic preservation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


METRIC_VERSION = "low_high_critic_v0.1"
CORE_METRICS = [
    "high_vs_low_distribution_alignment",
    "temporal_consistency",
    "high_cost_aesthetic_quality",
    "edit_semantic_preservation",
]
RESULT_FIELDS = [
    "sample_id",
    "method",
    "instruction",
    "source_video",
    "edited_video",
    "high_vs_low_distribution_alignment",
    "temporal_consistency",
    "high_cost_aesthetic_quality",
    "edit_semantic_preservation",
    "overall",
    "decision",
    "failure_reasons",
    "refinement_suggestions",
    "distribution_high_similarity",
    "distribution_low_similarity",
    "distribution_margin",
    "clip_f",
    "source_edited_clip_similarity",
    "high_cost_aesthetic_quality_source",
    "edit_semantic_preservation_source",
    "error",
]


@dataclass(frozen=True)
class CriticItem:
    sample_id: str
    method: str
    instruction: str
    source_video: Path | None
    edited_video: Path
    raw: dict[str, Any]


def _fail(message: str) -> None:
    raise SystemExit(f"[low_high_critic] {message}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            payload.setdefault("_line_no", line_no)
            rows.append(payload)
    return rows


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _path_from_row(row: dict[str, Any], keys: tuple[str, ...], *, required: bool, base_dir: Path) -> Path | None:
    raw = _first_value(row, keys)
    if raw in (None, ""):
        if required:
            raise ValueError(f"missing required path field; expected one of {', '.join(keys)}")
        return None
    return _resolve_path(Path(str(raw)).expanduser(), base_dir)


def _instruction_from_row(row: dict[str, Any]) -> str:
    raw = _first_value(row, ("instruction", "prompt", "edit_instruction", "compiled_edit_prompt"))
    return str(raw or "")


def load_manifest(path: Path) -> list[CriticItem]:
    rows = _read_jsonl(path)
    base_dir = path.parent.resolve()
    items: list[CriticItem] = []
    for index, row in enumerate(rows):
        edited_video = _path_from_row(
            row,
            ("edited_video", "output_video", "candidate_video", "generated_video", "video"),
            required=True,
            base_dir=base_dir,
        )
        source_video = _path_from_row(
            row,
            ("source_video", "low_video", "low_video_path", "input_video", "original_video"),
            required=False,
            base_dir=base_dir,
        )
        assert edited_video is not None
        items.append(
            CriticItem(
                sample_id=str(row.get("sample_id") or row.get("id") or f"row_{index:06d}"),
                method=str(row.get("method") or "default"),
                instruction=_instruction_from_row(row),
                source_video=source_video,
                edited_video=edited_video,
                raw=row,
            )
        )
    return items


def _load_reference_video_paths(path: Path | None) -> list[Path]:
    if path is None:
        return []
    rows = _read_jsonl(path)
    base_dir = path.parent.resolve()
    keys = (
        "video",
        "video_path",
        "path",
        "high_video",
        "high_video_path",
        "low_video",
        "low_video_path",
        "source_video",
        "edited_video",
        "output_video",
    )
    paths: list[Path] = []
    for row in rows:
        video_path = _path_from_row(row, keys, required=True, base_dir=base_dir)
        assert video_path is not None
        paths.append(video_path)
    return paths


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 0 or count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    indices: list[int] = []
    for index in range(count):
        value = round(index * (length - 1) / (count - 1))
        if not indices or value != indices[-1]:
            indices.append(value)
    return indices


def sample_video_frames(video_path: Path, frame_count: int) -> list[Image.Image]:
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    try:
        import imageio.v3 as iio
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("imageio is required for video frame loading") from exc

    frames: list[Image.Image] = []
    for frame in iio.imiter(str(video_path)):
        frames.append(Image.fromarray(frame).convert("RGB"))
    if not frames:
        raise RuntimeError(f"no decodable frames found in {video_path}")
    return [frames[index] for index in _uniform_indices(len(frames), frame_count)]


class ClipFeatureExtractor:
    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode_frames(self, images: list[Image.Image]):
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(features)
        return self.torch.cat(chunks, dim=0)

    def video_feature_from_frame_features(self, frame_features):
        feature = frame_features.mean(dim=0)
        return feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def video_feature(self, video_path: Path, frame_count: int):
        frames = sample_video_frames(video_path, frame_count)
        frame_features = self.encode_frames(frames)
        return self.video_feature_from_frame_features(frame_features), frame_features

    def clip_f_0_10(self, frame_features) -> tuple[float | None, float | None]:
        if frame_features.shape[0] < 2:
            return None, None
        adjacent = (frame_features[:-1] * frame_features[1:]).sum(dim=-1)
        clip_f = adjacent.mean().item() * 100.0
        return max(0.0, min(10.0, clip_f / 10.0)), clip_f

    def cosine_0_10(self, feature_a, feature_b) -> tuple[float, float]:
        cosine = float((feature_a * feature_b).sum().item())
        return max(0.0, min(10.0, cosine * 10.0)), cosine


class DistributionScorer:
    def __init__(
        self,
        *,
        clip: ClipFeatureExtractor,
        high_ref_paths: list[Path],
        low_ref_paths: list[Path],
        frame_count: int,
        top_k: int,
        temperature: float,
    ) -> None:
        self.clip = clip
        self.high_ref_paths = high_ref_paths
        self.low_ref_paths = low_ref_paths
        self.frame_count = frame_count
        self.top_k = max(1, top_k)
        self.temperature = max(1e-6, temperature)
        self.high_features = self._encode_reference_set("high", high_ref_paths)
        self.low_features = self._encode_reference_set("low", low_ref_paths)

    def available(self) -> bool:
        return bool(self.high_features is not None and self.low_features is not None)

    def _encode_reference_set(self, label: str, paths: list[Path]):
        if not paths:
            return None
        features = []
        for index, path in enumerate(paths, start=1):
            print(f"[low_high_critic] encoding {label} reference {index}/{len(paths)}: {path}", flush=True)
            feature, _ = self.clip.video_feature(path, self.frame_count)
            features.append(feature)
        return self.clip.torch.stack(features, dim=0)

    def _topk_mean_similarity(self, candidate_feature, ref_features) -> float:
        similarities = (ref_features * candidate_feature.unsqueeze(0)).sum(dim=-1)
        k = min(self.top_k, int(similarities.shape[0]))
        return float(self.clip.torch.topk(similarities, k=k).values.mean().item())

    def score(self, candidate_feature) -> dict[str, float | None]:
        if not self.available():
            return {
                "high_vs_low_distribution_alignment": None,
                "distribution_high_similarity": None,
                "distribution_low_similarity": None,
                "distribution_margin": None,
            }
        high_similarity = self._topk_mean_similarity(candidate_feature, self.high_features)
        low_similarity = self._topk_mean_similarity(candidate_feature, self.low_features)
        margin = high_similarity - low_similarity
        probability = 1.0 / (1.0 + math.exp(-margin / self.temperature))
        return {
            "high_vs_low_distribution_alignment": max(0.0, min(10.0, probability * 10.0)),
            "distribution_high_similarity": high_similarity,
            "distribution_low_similarity": low_similarity,
            "distribution_margin": margin,
        }


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = json.loads(stripped[start : end + 1])
        if isinstance(payload, dict):
            return payload
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("VLM response is not a JSON object")
    return payload


def _score_0_10(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return max(0.0, min(10.0, number))


LOW_HIGH_VLM_SYSTEM_PROMPT = """You are an impartial critic for a low-cost to high-cost video editing agent.
You will receive sampled source-video frames followed by sampled edited-video frames.

Judge only what is visible in the frames. Use a 0-10 scale.

Definitions:
- high_cost_aesthetic_quality: whether the edited video looks premium/high-cost, with stronger cinematic quality, refined lighting, polished effects, clean details, and fewer artifacts than low-cost footage.
- edit_semantic_preservation: whether the edited video preserves the source video's main subject identity, scene semantics, layout, and action while allowing the requested high-cost transformation.
- temporal_consistency: whether the sampled edited frames look temporally coherent, without flicker, style drift, or unstable effects.
- instruction_alignment: whether the edited video follows the supplied instruction, if an instruction is provided.

Return exactly one JSON object with keys:
high_cost_aesthetic_quality, edit_semantic_preservation, temporal_consistency, instruction_alignment, rationale, refinement_suggestions
"""


class LocalQwenLowHighJudge:
    def __init__(self, *, model_path: str = "", device: str = "", frames_per_video: int = 4) -> None:
        if model_path:
            os.environ["LOW_HIGH_AGENT_LOCAL_QWEN_MODEL"] = str(Path(model_path).expanduser())
        if device:
            os.environ["LOW_HIGH_AGENT_LOCAL_QWEN_DEVICE"] = device
        os.environ.setdefault("LOW_HIGH_AGENT_USE_LOCAL_QWEN", "1")
        self.frames_per_video = frames_per_video
        from vlm.local_qwen import LocalQwenPlannerModel

        self.model = LocalQwenPlannerModel()

    def _extract_frames(self, path: Path | None) -> list[str]:
        if path is None:
            return []
        from vlm.local_qwen import extract_frames

        return extract_frames(str(path), max_frames=self.frames_per_video)

    def score(self, item: CriticItem) -> dict[str, Any]:
        source_frames = self._extract_frames(item.source_video)
        edited_frames = self._extract_frames(item.edited_video)
        if not edited_frames:
            raise RuntimeError(f"could not extract edited frames from {item.edited_video}")

        if source_frames:
            user_text = (
                f"sample_id: {item.sample_id}\n"
                f"method: {item.method}\n"
                f"instruction: {item.instruction or '(none)'}\n\n"
                f"The first {len(source_frames)} images are source-video frames in temporal order. "
                f"The next {len(edited_frames)} images are edited-video frames in temporal order. "
                "Evaluate whether the edited video is a high-cost transformation that preserves the source semantics."
            )
            image_paths = source_frames + edited_frames
        else:
            user_text = (
                f"sample_id: {item.sample_id}\n"
                f"method: {item.method}\n"
                f"instruction: {item.instruction or '(none)'}\n\n"
                f"All {len(edited_frames)} images are edited-video frames in temporal order. "
                "Source frames are unavailable, so edit_semantic_preservation should reflect only visible continuity."
            )
            image_paths = edited_frames

        raw_text = self.model.predict_text(
            system_prompt=LOW_HIGH_VLM_SYSTEM_PROMPT,
            user_text=user_text,
            image_paths=image_paths,
            max_new_tokens=512,
        )
        payload = _extract_json_object(raw_text)
        suggestions = payload.get("refinement_suggestions") or []
        if isinstance(suggestions, str):
            suggestions = [suggestions]
        if not isinstance(suggestions, list):
            suggestions = []
        return {
            "high_cost_aesthetic_quality": _score_0_10(payload.get("high_cost_aesthetic_quality")),
            "edit_semantic_preservation": _score_0_10(payload.get("edit_semantic_preservation")),
            "temporal_consistency": _score_0_10(payload.get("temporal_consistency")),
            "instruction_alignment": _score_0_10(payload.get("instruction_alignment")),
            "rationale": str(payload.get("rationale") or ""),
            "refinement_suggestions": [str(item) for item in suggestions if str(item).strip()][:5],
            "raw_response": raw_text[:4000],
        }


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _empty_result(item: CriticItem) -> dict[str, Any]:
    return {
        "sample_id": item.sample_id,
        "method": item.method,
        "instruction": item.instruction,
        "source_video": str(item.source_video) if item.source_video else "",
        "edited_video": str(item.edited_video),
        "high_vs_low_distribution_alignment": None,
        "temporal_consistency": None,
        "high_cost_aesthetic_quality": None,
        "edit_semantic_preservation": None,
        "overall": None,
        "decision": "incomplete",
        "failure_reasons": [],
        "refinement_suggestions": [],
        "distribution_high_similarity": None,
        "distribution_low_similarity": None,
        "distribution_margin": None,
        "clip_f": None,
        "source_edited_clip_similarity": None,
        "high_cost_aesthetic_quality_source": "",
        "edit_semantic_preservation_source": "",
        "vlm_temporal_consistency": None,
        "vlm_instruction_alignment": None,
        "vlm_rationale": "",
        "error": "",
        "metric_version": METRIC_VERSION,
        "raw_manifest": item.raw,
    }


def _weighted_overall(result: dict[str, Any]) -> float | None:
    weights = {
        "high_vs_low_distribution_alignment": 0.30,
        "temporal_consistency": 0.20,
        "high_cost_aesthetic_quality": 0.30,
        "edit_semantic_preservation": 0.20,
    }
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in weights.items():
        value = _safe_float(result.get(key))
        if value is None:
            continue
        weighted_sum += value * weight
        weight_total += weight
    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


def _failure_reasons(result: dict[str, Any], *, min_core_score: float) -> list[str]:
    labels = {
        "high_vs_low_distribution_alignment": "not_close_to_high_cost_distribution",
        "temporal_consistency": "temporal_inconsistency",
        "high_cost_aesthetic_quality": "weak_high_cost_aesthetic_quality",
        "edit_semantic_preservation": "semantic_or_source_drift",
    }
    reasons: list[str] = []
    for key, label in labels.items():
        value = _safe_float(result.get(key))
        if value is not None and value < min_core_score:
            reasons.append(label)
    return reasons


def _decide(result: dict[str, Any], *, accept_threshold: float, min_core_score: float) -> str:
    overall = _safe_float(result.get("overall"))
    if overall is None:
        return "incomplete"
    failures = _failure_reasons(result, min_core_score=min_core_score)
    if overall >= accept_threshold and not failures:
        return "accept"
    return "refine"


def _merge_vlm_result(result: dict[str, Any], vlm_result: dict[str, Any]) -> None:
    if vlm_result.get("high_cost_aesthetic_quality") is not None:
        result["high_cost_aesthetic_quality"] = vlm_result["high_cost_aesthetic_quality"]
        result["high_cost_aesthetic_quality_source"] = "vlm"
    if vlm_result.get("edit_semantic_preservation") is not None:
        result["edit_semantic_preservation"] = vlm_result["edit_semantic_preservation"]
        result["edit_semantic_preservation_source"] = "vlm"
    result["vlm_temporal_consistency"] = vlm_result.get("temporal_consistency")
    result["vlm_instruction_alignment"] = vlm_result.get("instruction_alignment")
    result["vlm_rationale"] = vlm_result.get("rationale") or ""
    result["refinement_suggestions"] = vlm_result.get("refinement_suggestions") or []
    result["vlm_raw"] = vlm_result


def evaluate_items(args: argparse.Namespace, items: list[CriticItem]) -> list[dict[str, Any]]:
    clip = None
    if not args.skip_clip:
        clip = ClipFeatureExtractor(model_name=args.clip_model, device=args.device, batch_size=args.clip_batch_size)

    high_refs = _load_reference_video_paths(args.high_ref_manifest)
    low_refs = _load_reference_video_paths(args.low_ref_manifest)
    distribution_scorer = None
    if clip is not None and high_refs and low_refs and not args.skip_distribution:
        distribution_scorer = DistributionScorer(
            clip=clip,
            high_ref_paths=high_refs,
            low_ref_paths=low_refs,
            frame_count=args.ref_frames_per_video,
            top_k=args.distribution_top_k,
            temperature=args.distribution_temperature,
        )
    elif not args.skip_distribution:
        print(
            "[low_high_critic] distribution metric disabled; provide --high-ref-manifest and --low-ref-manifest with --skip-clip unset",
            file=sys.stderr,
            flush=True,
        )

    judge = None
    if args.vlm_backend != "none":
        judge = LocalQwenLowHighJudge(
            model_path=args.vlm_model,
            device=args.vlm_device,
            frames_per_video=args.vlm_frames_per_video,
        )

    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(f"[low_high_critic] evaluating {index}/{len(items)} sample_id={item.sample_id} method={item.method}", flush=True)
        result = _empty_result(item)
        try:
            edited_feature = None
            if clip is not None:
                edited_feature, edited_frame_features = clip.video_feature(item.edited_video, args.frames_per_video)
                temporal_0_10, clip_f = clip.clip_f_0_10(edited_frame_features)
                result["temporal_consistency"] = temporal_0_10
                result["clip_f"] = clip_f

                if item.source_video is not None:
                    source_feature, _ = clip.video_feature(item.source_video, args.frames_per_video)
                    semantic_0_10, semantic_cosine = clip.cosine_0_10(source_feature, edited_feature)
                    result["edit_semantic_preservation"] = semantic_0_10
                    result["edit_semantic_preservation_source"] = "clip_source_similarity"
                    result["source_edited_clip_similarity"] = semantic_cosine

                if distribution_scorer is not None:
                    distribution = distribution_scorer.score(edited_feature)
                    result.update(distribution)
                    if args.aesthetic_fallback == "distribution" and result["high_vs_low_distribution_alignment"] is not None:
                        result["high_cost_aesthetic_quality"] = result["high_vs_low_distribution_alignment"]
                        result["high_cost_aesthetic_quality_source"] = "distribution_proxy"

            if judge is not None:
                _merge_vlm_result(result, judge.score(item))

            result["overall"] = _weighted_overall(result)
            result["failure_reasons"] = _failure_reasons(result, min_core_score=args.min_core_score)
            result["decision"] = _decide(
                result,
                accept_threshold=args.accept_threshold,
                min_core_score=args.min_core_score,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = CORE_METRICS + ["overall", "clip_f", "source_edited_clip_similarity", "distribution_margin"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("method") or "default"), []).append(result)

    summary: dict[str, Any] = {
        "metric_version": METRIC_VERSION,
        "num_rows": len(results),
        "methods": {},
    }
    for method, rows in sorted(grouped.items()):
        method_summary: dict[str, Any] = {
            "num_rows": len(rows),
            "num_errors": sum(1 for row in rows if row.get("error")),
            "num_accept": sum(1 for row in rows if row.get("decision") == "accept"),
            "num_refine": sum(1 for row in rows if row.get("decision") == "refine"),
            "num_incomplete": sum(1 for row in rows if row.get("decision") == "incomplete"),
        }
        for metric in metrics:
            values = [_safe_float(row.get(metric)) for row in rows]
            numbers = [value for value in values if value is not None]
            if not numbers:
                method_summary[metric] = {"count": 0, "mean": None, "std": None, "stderr": None}
                continue
            std = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
            method_summary[metric] = {
                "count": len(numbers),
                "mean": statistics.mean(numbers),
                "std": std,
                "stderr": std / math.sqrt(len(numbers)) if numbers else None,
            }
        summary["methods"][method] = method_summary
    return summary


def write_outputs(output_dir: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "per_sample_critic.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    csv_path = output_dir / "per_sample_critic.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, "") for field in RESULT_FIELDS}
            row["failure_reasons"] = json.dumps(row["failure_reasons"], ensure_ascii=False)
            row["refinement_suggestions"] = json.dumps(row["refinement_suggestions"], ensure_ascii=False)
            writer.writerow(row)

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Low-High Critic Summary", "", f"metric_version: `{METRIC_VERSION}`", ""]
    lines.append("| method | n | errors | accept | refine | dist | temporal | aesthetic | semantic | overall |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method, payload in summary["methods"].items():
        def fmt(metric: str) -> str:
            value = payload[metric]["mean"]
            return "" if value is None else f"{value:.4f}"

        lines.append(
            f"| {method} | {payload['num_rows']} | {payload['num_errors']} | "
            f"{payload['num_accept']} | {payload['num_refine']} | "
            f"{fmt('high_vs_low_distribution_alignment')} | {fmt('temporal_consistency')} | "
            f"{fmt('high_cost_aesthetic_quality')} | {fmt('edit_semantic_preservation')} | {fmt('overall')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-high critic four-dimensional evaluation.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL with source/edited video paths.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--high-ref-manifest", type=Path, default=None, help="JSONL of high-cost reference videos for distribution scoring.")
    parser.add_argument("--low-ref-manifest", type=Path, default=None, help="JSONL of low-cost reference videos for distribution scoring.")
    parser.add_argument("--device", default=_default_device(), help="Torch device for CLIP metrics.")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14", help="Transformers CLIP model id/path.")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=16, help="Sampled frames for candidate/source CLIP metrics.")
    parser.add_argument("--ref-frames-per-video", type=int, default=8, help="Sampled frames per reference video.")
    parser.add_argument("--distribution-top-k", type=int, default=5, help="Top-k high/low reference similarities to average.")
    parser.add_argument("--distribution-temperature", type=float, default=0.02, help="Sigmoid temperature for high-low similarity margin.")
    parser.add_argument("--aesthetic-fallback", choices=["none", "distribution"], default="distribution", help="Fallback when VLM aesthetic score is unavailable.")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP-derived metrics.")
    parser.add_argument("--skip-distribution", action="store_true", help="Skip high-vs-low distribution scoring.")
    parser.add_argument("--vlm-backend", choices=["none", "local-qwen"], default="none", help="Optional VLM judge backend.")
    parser.add_argument("--vlm-model", default="", help="Local Qwen-VL model path.")
    parser.add_argument("--vlm-device", default="", help="Optional Qwen-VL device override, for example cuda:0 or cpu.")
    parser.add_argument("--vlm-frames-per-video", type=int, default=4, help="Frames sampled from source and edited videos for VLM judge.")
    parser.add_argument("--accept-threshold", type=float, default=7.0, help="Overall score threshold for accept.")
    parser.add_argument("--min-core-score", type=float, default=5.0, help="Minimum individual core score for accept.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on first sample-level error.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames_per_video < 1:
        _fail("--frames-per-video must be >= 1")
    if args.ref_frames_per_video < 1:
        _fail("--ref-frames-per-video must be >= 1")
    if args.vlm_frames_per_video < 1:
        _fail("--vlm-frames-per-video must be >= 1")
    items = load_manifest(args.manifest)
    if not items:
        _fail("manifest is empty")
    results = evaluate_items(args, items)
    summary = summarize(results)
    write_outputs(args.output_dir, results, summary)
    print(f"[low_high_critic] wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
