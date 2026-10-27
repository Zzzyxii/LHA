#!/usr/bin/env python3
"""Paper-style quantitative evaluator for Ditto/Editto experiments.

This is not the official Ditto paper evaluator. It implements reproducible
approximations of the automatic metrics described in the paper:

* CLIP-T: text-video CLIP similarity, averaged over sampled output frames.
* CLIP-F: adjacent-frame CLIP similarity, averaged over sampled output frames.
* VLM score: optional local Qwen-VL judge implemented in ``vlm_judge.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


METRIC_VERSION = "ditto_paper_style_v0.4"
RESULT_FIELDS = [
    "sample_id",
    "method",
    "instruction",
    "source_video",
    "edited_video",
    "clip_t",
    "clip_f",
    "vlm_instruction_fidelity",
    "vlm_source_preservation",
    "vlm_visual_quality",
    "vlm_safety",
    "vlm_overall",
    "vlm_rationale",
    "error",
]


@dataclass(frozen=True)
class EvalItem:
    sample_id: str
    method: str
    instruction: str
    edited_video: Path
    source_video: Path | None
    raw: dict[str, Any]


def _fail(message: str) -> None:
    raise SystemExit(f"[ditto_quant] {message}")


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
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            payload.setdefault("_line_no", line_no)
            rows.append(payload)
    return rows


def _path_from_row(row: dict[str, Any], keys: tuple[str, ...], *, required: bool) -> Path | None:
    for key in keys:
        raw = row.get(key)
        if raw:
            return Path(str(raw)).expanduser()
    if required:
        raise ValueError(f"missing required path field; expected one of {', '.join(keys)}")
    return None


def _instruction_from_row(row: dict[str, Any]) -> str:
    for key in ("instruction", "prompt", "edit_instruction"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError("missing instruction field; expected instruction, prompt, or edit_instruction")


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_manifest(path: Path) -> list[EvalItem]:
    rows = _read_jsonl(path)
    base_dir = path.parent.resolve()
    items: list[EvalItem] = []
    for index, row in enumerate(rows):
        edited = _path_from_row(row, ("edited_video", "output_video", "video"), required=True)
        assert edited is not None
        source = _path_from_row(row, ("source_video", "input_video", "low_video"), required=False)
        item = EvalItem(
            sample_id=str(row.get("sample_id") or row.get("id") or f"row_{index:06d}"),
            method=str(row.get("method") or "default"),
            instruction=_instruction_from_row(row),
            edited_video=_resolve_path(edited, base_dir),
            source_video=_resolve_path(source, base_dir) if source is not None else None,
            raw=row,
        )
        items.append(item)
    return items


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 0 or count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    indices = []
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
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("imageio is required for video frame loading") from exc

    frames: list[Image.Image] = []
    for frame in iio.imiter(str(video_path)):
        frames.append(Image.fromarray(frame).convert("RGB"))
    if not frames:
        raise RuntimeError(f"no decodable frames found in {video_path}")
    return [frames[index] for index in _uniform_indices(len(frames), frame_count)]


class ClipVideoMetrics:
    def __init__(self, *, model_name: str, device: str, batch_size: int) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def _encode_images(self, images: list[Image.Image]):
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                chunks.append(features)
        return self.torch.cat(chunks, dim=0)

    def _encode_text(self, text: str):
        with self.torch.inference_mode():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return features[0]

    def encode_frames(self, frames: list[Image.Image]):
        return self._encode_images(frames)

    def clip_t_from_features(self, image_features, instruction: str) -> float:
        text_feature = self._encode_text(instruction)
        return (image_features @ text_feature).mean().item() * 100.0

    def clip_f_from_features(self, image_features) -> float | None:
        if image_features.shape[0] < 2:
            return None
        adjacent = (image_features[:-1] * image_features[1:]).sum(dim=-1)
        return adjacent.mean().item() * 100.0

def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _empty_result(item: EvalItem) -> dict[str, Any]:
    return {
        "sample_id": item.sample_id,
        "method": item.method,
        "instruction": item.instruction,
        "source_video": str(item.source_video) if item.source_video else "",
        "edited_video": str(item.edited_video),
        "clip_t": None,
        "clip_f": None,
        "vlm_instruction_fidelity": None,
        "vlm_source_preservation": None,
        "vlm_visual_quality": None,
        "vlm_safety": None,
        "vlm_overall": None,
        "vlm_rationale": "",
        "error": "",
        "metric_version": METRIC_VERSION,
        "raw_manifest": item.raw,
    }


def _merge_vlm_result(result: dict[str, Any], vlm_result: dict[str, Any]) -> None:
    result["vlm_instruction_fidelity"] = vlm_result.get("instruction_fidelity")
    result["vlm_source_preservation"] = vlm_result.get("source_preservation")
    result["vlm_visual_quality"] = vlm_result.get("visual_quality")
    result["vlm_safety"] = vlm_result.get("safety")
    result["vlm_overall"] = vlm_result.get("overall")
    result["vlm_rationale"] = vlm_result.get("rationale") or ""
    result["vlm_raw"] = vlm_result


def evaluate_items(args: argparse.Namespace, items: list[EvalItem]) -> list[dict[str, Any]]:
    clip_metrics = None
    if not args.skip_clip:
        clip_metrics = ClipVideoMetrics(model_name=args.clip_model, device=args.device, batch_size=args.clip_batch_size)

    judge = None
    if args.vlm_backend != "none":
        from vlm_judge import LocalQwenVideoJudge

        judge = LocalQwenVideoJudge(
            model_path=args.vlm_model,
            device=args.vlm_device,
            frames_per_video=args.vlm_frames_per_video,
        )

    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(f"[ditto_quant] evaluating {index}/{len(items)} sample_id={item.sample_id} method={item.method}", flush=True)
        result = _empty_result(item)
        try:
            if clip_metrics is not None:
                edited_frames = sample_video_frames(item.edited_video, args.frames_per_video)
                edited_features = clip_metrics.encode_frames(edited_frames)
                result["clip_t"] = clip_metrics.clip_t_from_features(edited_features, item.instruction)
                result["clip_f"] = clip_metrics.clip_f_from_features(edited_features)
            if judge is not None:
                _merge_vlm_result(result, judge.score(item))
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
    metrics = [
        "clip_t",
        "clip_f",
        "vlm_instruction_fidelity",
        "vlm_source_preservation",
        "vlm_visual_quality",
        "vlm_safety",
        "vlm_overall",
    ]
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
    jsonl_path = output_dir / "per_sample_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    csv_path = output_dir / "per_sample_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Ditto Paper-Style Quantitative Summary", "", f"metric_version: `{METRIC_VERSION}`", ""]
    lines.append("| method | n | errors | CLIP-T | CLIP-F | VLM Overall | VLM Instr | VLM Source | VLM Quality | VLM Safety |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method, payload in summary["methods"].items():
        def fmt(metric: str) -> str:
            value = payload[metric]["mean"]
            return "" if value is None else f"{value:.4f}"

        lines.append(
            f"| {method} | {payload['num_rows']} | {payload['num_errors']} | "
            f"{fmt('clip_t')} | {fmt('clip_f')} | {fmt('vlm_overall')} | "
            f"{fmt('vlm_instruction_fidelity')} | {fmt('vlm_source_preservation')} | "
            f"{fmt('vlm_visual_quality')} | {fmt('vlm_safety')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style Ditto/Editto quantitative metrics.")
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL manifest with source/edited video paths and instructions.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for metric outputs.")
    parser.add_argument("--device", default=_default_device(), help="Torch device for CLIP metrics.")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14", help="Transformers CLIP model id/path.")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--frames-per-video", type=int, default=16, help="Uniformly sampled edited frames for CLIP metrics.")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP-T and CLIP-F.")
    parser.add_argument("--vlm-backend", choices=["none", "local-qwen"], default="none", help="Optional VLM judge backend.")
    parser.add_argument("--vlm-model", default="", help="Local Qwen-VL model path. Also sets LOW_HIGH_AGENT_LOCAL_QWEN_MODEL.")
    parser.add_argument("--vlm-device", default="", help="Optional Qwen-VL device override, for example cuda:0 or cpu.")
    parser.add_argument("--vlm-frames-per-video", type=int, default=4, help="Frames sampled from source and edited videos for VLM judge.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first sample-level error.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames_per_video < 1:
        _fail("--frames-per-video must be >= 1")
    if args.vlm_frames_per_video < 1:
        _fail("--vlm-frames-per-video must be >= 1")
    items = load_manifest(args.manifest)
    if not items:
        _fail("manifest is empty")
    results = evaluate_items(args, items)
    summary = summarize(results)
    write_outputs(args.output_dir, results, summary)
    print(f"[ditto_quant] wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
