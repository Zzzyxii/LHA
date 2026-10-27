from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from qwen_vl_utils import process_vision_info


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _resolve_default_local_model() -> Path:
    project_model = PROJECT_ROOT / "models" / "Qwen3-VL-8B-Thinking"
    workspace_model = WORKSPACE_ROOT / "models" / "Qwen3-VL-8B-Thinking"
    if project_model.exists():
        return project_model
    return workspace_model


DEFAULT_LOCAL_MODEL = _resolve_default_local_model()


class LocalQwenResponseError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str = "", repair_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.repair_response = repair_response


def _log_stage(stage: str, start_time: float | None = None) -> float:
    now = time.time()
    if start_time is None:
        print(f"[planner.local_qwen] {stage}", file=sys.stderr, flush=True)
    else:
        print(f"[planner.local_qwen] {stage} ({now - start_time:.2f}s)", file=sys.stderr, flush=True)
    return now


def use_local_qwen() -> bool:
    flag = os.environ.get("LOW_HIGH_AGENT_USE_LOCAL_QWEN", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    return local_model_path().exists()


def local_model_path() -> Path:
    raw = os.environ.get("LOW_HIGH_AGENT_LOCAL_QWEN_MODEL", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_LOCAL_MODEL


def requested_local_device() -> str:
    requested = os.environ.get("LOW_HIGH_AGENT_LOCAL_QWEN_DEVICE", "").strip()
    return requested or "auto"


def local_device() -> str:
    requested = os.environ.get("LOW_HIGH_AGENT_LOCAL_QWEN_DEVICE", "").strip()
    if requested:
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_device_label(device: str) -> str:
    lowered = str(device).strip().lower()
    if lowered.startswith("cuda"):
        return "cuda"
    if lowered.startswith("cpu"):
        return "cpu"
    return lowered


def keep_vlm_on_gpu() -> bool:
    raw = os.environ.get("LOW_HIGH_AGENT_KEEP_VLM_ON_GPU", "").strip().lower()
    if raw in {"0", "false", "no"}:
        return False
    if raw in {"1", "true", "yes"}:
        return True
    return local_device().startswith("cuda")


def _preferred_attention_impl() -> str:
    requested = os.environ.get("LOW_HIGH_AGENT_QWEN_ATTENTION", "").strip().lower()
    if requested in {"flash_attention_2", "flash", "fa2"}:
        return "flash_attention_2"
    if requested in {"sdpa", "torch_sdpa"}:
        return "sdpa"
    return "flash_attention_2"


def _actual_model_device(model) -> str:
    try:
        first_param = next(model.parameters())
        return str(first_param.device)
    except StopIteration:
        pass
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        return str(next(iter(hf_device_map.values())))
    return "unknown"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        payload = json.loads(stripped[start : end + 1])
        if isinstance(payload, dict):
            return payload
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            payload, _ = decoder.raw_decode(stripped[match.start() :])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("local qwen response did not contain a valid JSON object")


def _message_content(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append({"type": "image", "image": str(image_path)})
    return content


@lru_cache(maxsize=1)
def _load_processor_and_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = str(local_model_path())
    requested_device = requested_local_device()
    resolved_device = local_device()
    if _normalize_device_label(requested_device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("local qwen was required on cuda but torch.cuda.is_available() is false")
    dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if resolved_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        attention_impl_candidates: list[str] = []
        preferred_impl = _preferred_attention_impl()
        if preferred_impl == "flash_attention_2":
            attention_impl_candidates = ["flash_attention_2", "sdpa"]
        else:
            attention_impl_candidates = [preferred_impl]
        last_error: Exception | None = None
        model = None
        for attention_impl in attention_impl_candidates:
            common_kwargs = {
                "dtype": dtype,
                "low_cpu_mem_usage": True,
                "attn_implementation": attention_impl,
                "trust_remote_code": True,
            }
            try:
                if keep_vlm_on_gpu():
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        **common_kwargs,
                    )
                    model = model.to(resolved_device)
                else:
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        device_map="auto",
                        **common_kwargs,
                    )
                if attention_impl != preferred_impl:
                    print(
                        f"[planner.local_qwen] attention_fallback preferred={preferred_impl} actual={attention_impl}",
                        file=sys.stderr,
                        flush=True,
                    )
                break
            except ImportError as exc:
                last_error = exc
                if attention_impl != attention_impl_candidates[-1]:
                    print(
                        f"[planner.local_qwen] attention_unavailable impl={attention_impl}, falling back",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                raise
        if model is None:
            assert last_error is not None
            raise last_error
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model = model.to(resolved_device)
    model.eval()
    return processor, model, requested_device, resolved_device


def _generate_text(*, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    import torch

    processor, model, requested_device, resolved_device = _load_processor_and_model()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    actual_model_device = _actual_model_device(model)
    if _normalize_device_label(requested_device) == "cuda" and _normalize_device_label(actual_model_device) != "cuda":
        raise RuntimeError("local qwen was required on cuda but loaded model is not on CUDA")
    if _normalize_device_label(resolved_device) == "cuda":
        inputs = inputs.to(next(model.parameters()).device)
    else:
        inputs = inputs.to(resolved_device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0]


def _repair_json_response(raw_text: str) -> tuple[dict[str, Any], str]:
    repair_messages = [
        {
            "role": "system",
            "content": _message_content(
                "Convert the user-provided planner output into exactly one valid JSON object. "
                "Preserve concrete fields. Do not add markdown fences or extra text.",
                [],
            ),
        },
        {
            "role": "user",
            "content": _message_content("Convert the following planner output into one valid JSON object only:\n\n" + raw_text[:12000], []),
        },
    ]
    repaired = _generate_text(messages=repair_messages, max_new_tokens=768)
    payload = _extract_json_object(repaired)
    payload.setdefault("repair_method", "local_qwen_json_repair")
    payload.setdefault("repair_raw_response", repaired[:2000])
    return payload, repaired


def _predict_json_strict_local(*, system_prompt: str, user_text: str, image_paths: list[str]) -> dict:
    strict_system_prompt = (
        system_prompt.strip()
        + "\n\nStrict response requirements:\n"
        + "- Return exactly one JSON object.\n"
        + "- Follow the provided JSON template keys exactly.\n"
        + "- Do not output free-form analysis before or after the JSON.\n"
        + "- If you need reasoning, encode it into the requested trace fields inside the JSON.\n"
    )
    start = _log_stage("loading_model")
    processor, model = _load_processor_and_model()[:2]
    _log_stage("loading_model_done", start)
    messages = [
        {"role": "system", "content": _message_content(strict_system_prompt, [])},
        {"role": "user", "content": _message_content(user_text, image_paths)},
    ]
    build_start = _log_stage("building_inputs")
    _ = processor
    _ = model
    _log_stage("building_inputs_done", build_start)
    generate_start = _log_stage("generating_primary_json")
    raw_response = _generate_text(messages=messages, max_new_tokens=1024)
    _log_stage("generating_primary_json_done", generate_start)
    try:
        payload = _extract_json_object(raw_response)
    except Exception as exc:
        repair_start = _log_stage("repairing_json")
        try:
            payload, repaired = _repair_json_response(raw_response[:12000])
            _log_stage("repairing_json_done", repair_start)
            payload.setdefault("repair_raw_response", repaired[:2000])
        except Exception as repair_exc:
            _log_stage("repairing_json_failed", repair_start)
            raise LocalQwenResponseError(
                f"local qwen response could not be structured into a valid JSON object: {exc}",
                raw_response=raw_response[:12000],
                repair_response=getattr(repair_exc, "repair_response", "")[:12000] if hasattr(repair_exc, "repair_response") else "",
            ) from repair_exc
    payload.setdefault("method", "local_qwen3_vl")
    payload.setdefault("model", str(local_model_path()))
    payload.setdefault("raw_response", raw_response[:4000])
    payload.setdefault("requested_device", requested_local_device())
    payload.setdefault("resolved_device", local_device())
    payload.setdefault("actual_model_device", _actual_model_device(_load_processor_and_model()[1]))
    return payload


class LocalQwenPlannerModel:
    def predict_text(self, *, system_prompt: str, user_text: str, image_paths: list[str], max_new_tokens: int = 1024) -> str:
        if use_local_qwen():
            start = _log_stage("loading_model")
            processor, model = _load_processor_and_model()[:2]
            _log_stage("loading_model_done", start)
            messages = [
                {"role": "system", "content": _message_content(system_prompt, [])},
                {"role": "user", "content": _message_content(user_text, image_paths)},
            ]
            build_start = _log_stage("building_inputs")
            _ = processor
            _ = model
            _log_stage("building_inputs_done", build_start)
            generate_start = _log_stage("generating_primary_text")
            text = _generate_text(messages=messages, max_new_tokens=max_new_tokens)
            _log_stage("generating_primary_text_done", generate_start)
            return text
        raise RuntimeError("local_qwen planner currently requires a local qwen model path")

    def predict_json(self, *, system_prompt: str, user_text: str, image_paths: list[str]) -> dict:
        if use_local_qwen():
            payload = _predict_json_strict_local(system_prompt=system_prompt, user_text=user_text, image_paths=image_paths)
            payload["backend_used"] = "local_qwen_vl"
            return payload
        raise RuntimeError("local_qwen planner currently requires a local qwen model path")


def _probe_video_duration(video_path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    text = completed.stdout.strip()
    if not text:
        return None
    try:
        duration = float(text)
    except ValueError:
        return None
    return duration if duration > 0 else None


def _uniform_sample_timestamps(duration: float, frame_count: int) -> list[float]:
    if frame_count <= 1:
        return [max(0.0, duration * 0.5)]
    epsilon = min(0.05, duration / 100.0)
    timestamps: list[float] = []
    for index in range(frame_count):
        timestamp = (index + 0.5) * duration / frame_count
        timestamp = min(max(0.0, timestamp), max(0.0, duration - epsilon))
        timestamps.append(timestamp)
    return timestamps


def _extract_frame_at_timestamp(video_path: Path, timestamp: float, output_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is not available")
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def extract_frames(low_video: str, max_frames: int = 6) -> list[str]:
    video_path = Path(low_video)
    if not video_path.exists():
        return []
    tmp_dir = Path(tempfile.mkdtemp(prefix="low_high_new_frames_"))
    duration = _probe_video_duration(video_path)
    if duration is None:
        output_path = tmp_dir / "frame_000.png"
        try:
            _extract_frame_at_timestamp(video_path, 0.0, output_path)
        except (subprocess.SubprocessError, FileNotFoundError):
            return []
        return [str(output_path)]
    timestamps = _uniform_sample_timestamps(duration, max_frames)
    frames: list[str] = []
    for index, timestamp in enumerate(timestamps):
        output_path = tmp_dir / f"frame_{index:03d}.png"
        try:
            _extract_frame_at_timestamp(video_path, timestamp, output_path)
        except (subprocess.SubprocessError, FileNotFoundError):
            return frames
        frames.append(str(output_path))
    return frames
