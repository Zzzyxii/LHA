#!/usr/bin/env python3
"""Build strong-VLM teacher request JSONL for low-high critic labels.

This script prepares labels for the last two critic dimensions:

1. high_cost_aesthetic_quality
2. edit_semantic_preservation

It can read either a generic JSONL manifest or Ditto-1M training metadata
(`source_path`, `edited_path`, `instruction`) and emits a small-batch JSONL
that a strong VLM caller can run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


DEFAULT_SYSTEM_PROMPT = """You are a strict video-editing quality evaluator for a low-cost to high-cost video editing task.
You will judge sampled source-video frames and sampled edited-video frames.

Score only the two requested dimensions on a 0-10 scale:
1. high_cost_aesthetic_quality:
   Whether the edited video looks premium/high-cost, cinematic, visually refined, and technically polished.
   Reward stronger lighting, composition, texture/detail, coherent special effects, natural color grading, and low artifact/noise/flicker.
   Penalize low-budget look, blur, visible artifacts, poor lighting, cheap effects, distorted subjects, or unpolished frames.
2. edit_semantic_preservation:
   Whether the edited video preserves the source video's main subject identity, scene semantics, layout, action, and important content.
   Reward keeping the same main objects/people, spatial layout, action trajectory, and scene meaning while allowing the requested edit.
   Penalize subject replacement, identity drift, scene/layout changes that are not requested, missing objects, or motion/action mismatch.

Use the full source-to-edited comparison. Do not score based on prompt expectations alone; score what is actually visible.
Return JSON that matches the provided schema."""


TEACHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "high_cost_aesthetic_quality": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "0-10 high-cost aesthetic and technical quality score for the edited video.",
        },
        "edit_semantic_preservation": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "0-10 source semantic preservation score comparing source and edited videos.",
        },
        "decision": {
            "type": "string",
            "enum": ["accept", "refine", "reject"],
            "description": "accept if both dimensions are strong, refine if fixable, reject if fundamentally bad.",
        },
        "failure_reasons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short machine-readable or concise natural-language failure reasons.",
        },
        "refinement_suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete suggestions for improving the edited video or prompt.",
        },
        "rationale": {
            "type": "string",
            "description": "Brief visual evidence for the two scores.",
        },
    },
    "required": [
        "high_cost_aesthetic_quality",
        "edit_semantic_preservation",
        "decision",
        "failure_reasons",
        "refinement_suggestions",
        "rationale",
    ],
}


@dataclass(frozen=True)
class TeacherItem:
    sample_id: str
    source_video: Path
    edited_video: Path
    instruction: str
    raw: dict[str, Any]


def _fail(message: str) -> None:
    raise SystemExit(f"[strong_vlm_teacher_jsonl] {message}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            payload.setdefault("_line_no", line_no)
            rows.append(payload)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "items", "records", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a JSON list, or an object with data/items/records/examples")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        item.setdefault("_line_no", index + 1)
        rows.append(item)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{**row, "_line_no": index + 1} for index, row in enumerate(reader)]


def _read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".csv":
        return _read_csv(path)
    raise ValueError(f"unsupported metadata/manifest extension: {path}")


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve_path(raw: Any, *, base_dir: Path, ditto_root: Path | None) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path
    if ditto_root is not None:
        candidate = (ditto_root / path).resolve()
        if candidate.exists():
            return candidate
    return (base_dir / path).resolve()


def _row_to_item(row: dict[str, Any], *, index: int, base_dir: Path, ditto_root: Path | None) -> TeacherItem | None:
    source_raw = _first_value(row, ("source_video", "source_path", "input_video", "low_video", "low_video_path", "video"))
    edited_raw = _first_value(row, ("edited_video", "edited_path", "output_video", "target_video", "high_video", "high_video_path"))
    instruction = str(_first_value(row, ("instruction", "prompt", "edit_instruction", "compiled_edit_prompt")) or "").strip()
    if source_raw in (None, "") or edited_raw in (None, "") or not instruction:
        return None
    sample_id = str(row.get("sample_id") or row.get("id") or f"sample_{index:06d}")
    return TeacherItem(
        sample_id=sample_id,
        source_video=_resolve_path(source_raw, base_dir=base_dir, ditto_root=ditto_root),
        edited_video=_resolve_path(edited_raw, base_dir=base_dir, ditto_root=ditto_root),
        instruction=instruction,
        raw=row,
    )


def load_items(args: argparse.Namespace) -> list[TeacherItem]:
    if args.manifest is None and args.metadata is None:
        _fail("provide --manifest or --metadata")
    path = args.manifest or args.metadata
    assert path is not None
    path = path.expanduser().resolve()
    rows = _read_table(path)
    ditto_root = args.ditto_root.expanduser().resolve() if args.ditto_root else None
    base_dir = ditto_root or path.parent
    items: list[TeacherItem] = []
    for index, row in enumerate(rows):
        item = _row_to_item(row, index=index, base_dir=base_dir, ditto_root=ditto_root)
        if item is not None:
            items.append(item)
    if args.require_existing:
        items = [item for item in items if item.source_video.exists() and item.edited_video.exists()]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(items)
    if args.start_index > 0:
        items = items[args.start_index :]
    if args.sample_size is not None:
        items = items[: args.sample_size]
    return items


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
    try:
        import imageio.v3 as iio
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("imageio is required for frame extraction") from exc
    frames: list[Image.Image] = []
    for frame in iio.imiter(str(video_path)):
        frames.append(Image.fromarray(frame).convert("RGB"))
    if not frames:
        raise RuntimeError(f"no decodable frames found in {video_path}")
    return [frames[index] for index in _uniform_indices(len(frames), frame_count)]


def image_to_data_url(image: Image.Image, *, max_side: int, jpeg_quality: int) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_user_prompt(item: TeacherItem, *, frames_per_video: int, media_mode: str) -> str:
    if media_mode == "sampled-frames":
        media_description = (
            f"The attached images contain {frames_per_video} uniformly sampled source frames followed by "
            f"{frames_per_video} uniformly sampled edited frames."
        )
    else:
        media_description = (
            f"Please load the two videos and uniformly sample {frames_per_video} frames from each video. "
            "Evaluate the source frames first, then the edited frames."
        )
    return (
        f"sample_id: {item.sample_id}\n"
        f"instruction: {item.instruction}\n"
        f"source_video: {item.source_video}\n"
        f"edited_video: {item.edited_video}\n\n"
        f"{media_description}\n\n"
        "Return only the requested JSON object. The scores should be calibrated as:\n"
        "- 0-2: severe failure\n"
        "- 3-4: weak / many issues\n"
        "- 5-6: acceptable but noticeably flawed\n"
        "- 7-8: good quality\n"
        "- 9-10: excellent, close to professional high-cost output\n"
    )


def _content_with_frames(item: TeacherItem, args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": build_user_prompt(item, frames_per_video=args.frames_per_video, media_mode="sampled-frames")}]
    for label, video_path in (("SOURCE", item.source_video), ("EDITED", item.edited_video)):
        frames = sample_video_frames(video_path, args.frames_per_video)
        for index, frame in enumerate(frames, start=1):
            content.append({"type": "input_text", "text": f"{label} FRAME {index}/{len(frames)}"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(frame, max_side=args.max_image_side, jpeg_quality=args.jpeg_quality),
                }
            )
    return content


def _openai_body(item: TeacherItem, args: argparse.Namespace) -> dict[str, Any]:
    if args.media_mode == "sampled-frames":
        content = _content_with_frames(item, args)
    else:
        content = [{"type": "input_text", "text": build_user_prompt(item, frames_per_video=args.frames_per_video, media_mode="paths")}]
    body: dict[str, Any] = {
        "model": args.model,
        "instructions": args.system_prompt,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": args.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "low_high_critic_teacher_scores",
                "strict": True,
                "schema": TEACHER_SCHEMA,
            }
        },
    }
    if args.reasoning_effort:
        body["reasoning"] = {"effort": args.reasoning_effort}
    if args.temperature is not None:
        body["temperature"] = args.temperature
    return body


def build_openai_batch_row(item: TeacherItem, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "custom_id": item.sample_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": _openai_body(item, args),
    }


def build_generic_row(item: TeacherItem, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {
        "custom_id": item.sample_id,
        "model": args.model,
        "recommended_endpoint": "/v1/responses",
        "recommended_hyperparameters": {
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "frames_per_video": args.frames_per_video,
            "media_mode": args.media_mode,
        },
        "system_prompt": args.system_prompt,
        "user_prompt": build_user_prompt(item, frames_per_video=args.frames_per_video, media_mode=args.media_mode),
        "source_video": str(item.source_video),
        "edited_video": str(item.edited_video),
        "instruction": item.instruction,
        "expected_output_schema": TEACHER_SCHEMA,
        "raw_item": item.raw,
    }
    if args.media_mode == "sampled-frames":
        # Keep the generic format portable but self-contained for small batches.
        row["source_frames"] = [
            image_to_data_url(frame, max_side=args.max_image_side, jpeg_quality=args.jpeg_quality)
            for frame in sample_video_frames(item.source_video, args.frames_per_video)
        ]
        row["edited_frames"] = [
            image_to_data_url(frame, max_side=args.max_image_side, jpeg_quality=args.jpeg_quality)
            for frame in sample_video_frames(item.edited_video, args.frames_per_video)
        ]
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sidecar(path: Path, args: argparse.Namespace, items: list[TeacherItem]) -> None:
    payload = {
        "output_jsonl": str(path),
        "format": args.output_format,
        "media_mode": args.media_mode,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "frames_per_video": args.frames_per_video,
        "max_image_side": args.max_image_side if args.media_mode == "sampled-frames" else None,
        "num_items": len(items),
        "source": str(args.manifest or args.metadata),
        "ditto_root": str(args.ditto_root or ""),
        "schema": TEACHER_SCHEMA,
        "openai_docs": {
            "batch_api": "https://platform.openai.com/docs/guides/batch",
            "responses_api": "https://platform.openai.com/docs/api-reference/responses",
            "structured_outputs": "https://platform.openai.com/docs/guides/structured-outputs",
            "images_vision": "https://platform.openai.com/docs/guides/images-vision",
        },
    }
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strong-VLM teacher JSONL for low-high critic labels.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Generic JSONL/JSON/CSV with source_video, edited_video, instruction.")
    source.add_argument("--metadata", type=Path, help="Ditto-1M metadata JSON/JSONL/CSV with source_path, edited_path, instruction.")
    parser.add_argument("--ditto-root", type=Path, default=None, help="Local Ditto-1M root, e.g. /home/featurize/data/Ditto-1M.")
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-format", choices=["teacher-generic", "openai-batch-responses"], default="teacher-generic")
    parser.add_argument("--media-mode", choices=["paths", "sampled-frames"], default="paths")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="medium")
    parser.add_argument("--temperature", type=float, default=None, help="Optional sampling temperature. Omit if the caller/model does not support it.")
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--frames-per-video", type=int, default=4)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--sample-size", type=int, default=20, help="Small batch size. Use 0 to keep all after start-index.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--require-existing", action="store_true", help="Drop rows whose source/edited videos are missing locally.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames_per_video < 1:
        _fail("--frames-per-video must be >= 1")
    if args.output_format == "openai-batch-responses" and args.media_mode == "paths":
        print(
            "[strong_vlm_teacher_jsonl] warning: OpenAI Responses cannot read local filesystem video paths directly. "
            "Use --media-mode sampled-frames for directly callable OpenAI batch files, or keep teacher-generic for a custom caller.",
            file=sys.stderr,
            flush=True,
        )
    if args.sample_size == 0:
        args.sample_size = None
    items = load_items(args)
    if not items:
        _fail("no usable items found")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(f"[strong_vlm_teacher_jsonl] building {index}/{len(items)} sample_id={item.sample_id}", flush=True)
        if args.output_format == "openai-batch-responses":
            rows.append(build_openai_batch_row(item, args))
        else:
            rows.append(build_generic_row(item, args))
    output_path = args.output_jsonl.expanduser().resolve()
    write_jsonl(output_path, rows)
    write_sidecar(output_path, args, items)
    print(f"[strong_vlm_teacher_jsonl] wrote {len(rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
