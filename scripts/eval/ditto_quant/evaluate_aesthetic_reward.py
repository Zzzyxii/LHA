#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm.openrouter import extract_frames as extract_frames_jpg

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("qwen-vl-utils is required for aesthetic reward evaluation") from exc


DIMENSIONS = [
    "narrative_emotional_fit",
    "style_world_consistency",
    "composition_lighting_design",
    "color_texture_refinement",
    "visual_hierarchy_readability",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON payload is not an object")
    return parsed


def _generation_path(root: Path, method: str, sample_id: str) -> Path:
    candidates = [
        root / method / "generations" / method / f"{sample_id}.mp4",
        root / "generations" / method / f"{sample_id}.mp4",
        root / method / f"{sample_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"generation not found for method={method} sample_id={sample_id}: {candidates[0]}")


def _build_aesthetic_evaluator_system_prompt() -> str:
    return """You are a strict cinematic/VFX aesthetic rater.
You are doing pointwise standalone aesthetic scoring: you see sampled frames
from the edited video, but you do not see the source video. Use the editing
instruction only as weak context for the intended visual direction. Do not judge
whether the edit accurately followed the instruction, because the source video is
not provided.

Scoring scale for every dimension:
4 = excellent / strongly successful
3 = good with minor issues
2 = weak with clear issues
1 = failed or harms the aesthetic goal

Rules:
- Score each dimension independently.
- There is no neutral middle score. Choose 2 or 3 when uncertain between weak and good.
- If a video matches multiple descriptions, assign the lowest applicable score.
- For object removal, cleanup, denoising, de-watermarking, or other utility edits,
  a seamless and visually natural result can be aesthetically successful even if
  it is not dramatic or cinematic.
- Do not penalize a candidate because the requested edit removes an interesting
  object or makes the scene simpler.
- Do not give high artistic scores just because the image is sharp or expensive-looking.
- Do not give high color scores just because colors are saturated.
- Penalize visible inpainting seams, visual clutter, incoherent style mixing,
  cheap texture/filter look, and unclear focal hierarchy when they are visible in
  the edited frames.
- Return valid JSON only, with no markdown.
"""


def _infer_aesthetic_task_type(instruction: str) -> str:
    text = instruction.lower()
    if any(word in text for word in ["remove", "erase", "delete", "hide", "clean up", "de-watermark", "watermark", "logo"]):
        return "removal_or_cleanup"
    if any(word in text for word in ["add", "insert", "place", "put ", "introduce", "include"]):
        return "addition_or_insertion"
    if any(word in text for word in ["replace", "swap", "change into", "turn into", "transform", "convert"]):
        return "replacement_or_transformation"
    if any(word in text for word in ["style", "aesthetic", "cinematic", "film", "color", "lighting", "tone", "grain", "texture"]):
        return "style_or_look_change"
    if any(word in text for word in ["enhance", "restore", "sharpen", "denoise", "improve", "refine"]):
        return "quality_refinement"
    return "general_edit"


def _build_aesthetic_evaluator_user_prompt(instruction: str) -> str:
    task_type = _infer_aesthetic_task_type(instruction)
    return f"""Editing instruction / intended effect:
{instruction}

Inferred task type:
{task_type}

Important: you do not see the source video. Do not evaluate whether the edit was
completed relative to the source. Evaluate the final edited video as a standalone
visual result. For removal or cleanup tasks, invisible/seamless blending is a
positive aesthetic outcome.

Evaluate the edited video frames on these five dimensions:
1. narrative_emotional_fit: whether the final edited result naturally integrates with the scene mood, emotional tone, and visual context without artificial or distracting anomalies.
2. style_world_consistency: whether the final visual style fits the world, era, genre, and style language such as classical, wuxia, sci-fi, realistic, fantasy, or cinematic.
3. composition_lighting_design: composition, contrast, lighting hierarchy, lens/cinematic design, and shot-level visual arrangement.
4. color_texture_refinement: color harmony, saturation control, material/texture/filter refinement, and whether it avoids cheap or generic looks.
5. visual_hierarchy_readability: whether the main visual intent is clear, focal hierarchy is readable, and important content is not obscured.

"overall_aesthetic_score" should be a holistic assessment of final visual
quality from 1.0 to 4.0, not a simple mathematical average of the five
dimensions.

Return this exact JSON schema:
{{
  "scores": {{
    "narrative_emotional_fit": 1,
    "style_world_consistency": 1,
    "composition_lighting_design": 1,
    "color_texture_refinement": 1,
    "visual_hierarchy_readability": 1
  }},
  "overall_aesthetic_score": 1.0,
  "uncertain": false,
  "reason": "one concise sentence"
}}
"""


def _build_messages(instruction: str, image_paths: list[str]) -> list[dict[str, Any]]:
    user_text = _build_aesthetic_evaluator_user_prompt(instruction)
    system_text = _build_aesthetic_evaluator_system_prompt()
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for image_path in image_paths:
        user_content.append({"type": "image", "image": image_path})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]


def _score_video(
    *,
    processor: Any,
    model: Any,
    video_path: Path,
    instruction: str,
    frames: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], str]:
    image_paths = [str(path) for path in extract_frames_jpg(str(video_path), frame_count=frames)]
    messages = _build_messages(instruction, image_paths)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return _extract_json_object(decoded), decoded


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score eval videos with the trained aesthetic reward LoRA.")
    parser.add_argument("--eval-inputs", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(args.eval_inputs)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(str(args.adapter), trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(args.base_model),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(args.device)
    model.eval()

    summary: dict[str, Any] = {}
    for method in args.methods:
        per_sample_path = args.output_dir / f"{method}.aesthetic_reward.jsonl"
        per_sample_path.write_text("", encoding="utf-8")
        aggregates: dict[str, list[float]] = {name: [] for name in DIMENSIONS}
        aggregates["overall_aesthetic_score"] = []
        errors = 0
        for index, row in enumerate(rows, start=1):
            sample_id = str(row.get("sample_id") or f"eval50_{index - 1:03d}")
            instruction = str(row.get("instruction") or row.get("prompt") or "")
            try:
                video_path = _generation_path(args.generation_root, method, sample_id)
                payload, raw_text = _score_video(
                    processor=processor,
                    model=model,
                    video_path=video_path,
                    instruction=instruction,
                    frames=args.frames,
                    max_new_tokens=args.max_new_tokens,
                )
                scores = payload.get("scores") or {}
                result = {
                    "sample_id": sample_id,
                    "method": method,
                    "video": str(video_path),
                    "scores": scores,
                    "overall_aesthetic_score": payload.get("overall_aesthetic_score"),
                    "uncertain": payload.get("uncertain"),
                    "reason": payload.get("reason"),
                    "raw_output": raw_text,
                    "error": "",
                }
                for name in DIMENSIONS:
                    value = _float_or_none(scores.get(name))
                    if value is not None:
                        aggregates[name].append(value)
                overall = _float_or_none(payload.get("overall_aesthetic_score"))
                if overall is not None:
                    aggregates["overall_aesthetic_score"].append(overall)
            except Exception as exc:
                errors += 1
                result = {
                    "sample_id": sample_id,
                    "method": method,
                    "video": "",
                    "scores": {},
                    "overall_aesthetic_score": None,
                    "uncertain": None,
                    "reason": "",
                    "raw_output": "",
                    "error": repr(exc),
                }
            with per_sample_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[aesthetic_reward] {method} {index}/{len(rows)} {sample_id} error={bool(result['error'])}", flush=True)

        summary[method] = {
            "n": len(rows),
            "errors": errors,
            **{name: _mean(values) for name, values in aggregates.items()},
        }

    (args.output_dir / "summary_aesthetic_reward.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
