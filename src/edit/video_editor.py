from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys

from edit.latent_memory import build_edit_latent_memory_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DIFFSYNTH_ROOT = PROJECT_ROOT / "DiffSynth-Studio"
DITTO_ROOT = PROJECT_ROOT / "Ditto"


def _resolve_models_root() -> Path:
    project_models = PROJECT_ROOT / "models"
    workspace_models = WORKSPACE_ROOT / "models"
    if project_models.exists():
        return project_models
    return workspace_models


MODELS_ROOT = _resolve_models_root()

WAN14B_DIR = MODELS_ROOT / "Wan2.1-VACE-14B"
BERNINI_DIR = MODELS_ROOT / "Bernini-R-1.3B-Diffusers"
BERNINI_REPO = WORKSPACE_ROOT / "third_party" / "Bernini"
if not BERNINI_REPO.exists():
    BERNINI_REPO = PROJECT_ROOT / "third_party" / "Bernini"
WAN13B_DIR = MODELS_ROOT / "Wan2.1-VACE-1.3B"
DISTILL_4STEP_LORA = MODELS_ROOT / "Wan2.1-Distill-Loras" / "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors"
DITTO_GLOBAL_LORA = MODELS_ROOT / "Ditto_models" / "models" / "ditto_global.safetensors"
EDIT_MODEL_ROOT = MODELS_ROOT / "edit-model"
EDIT_BASE_SFT_LORA = EDIT_MODEL_ROOT / "base_sft" / "latest.safetensors"
EDIT_PLANNER_CONDITIONED_LORA = EDIT_MODEL_ROOT / "planner_conditioned" / "latest.safetensors"

# "Ours": Ditto-architecture LoRA plus the trained latent-memory hint module. The
# single safetensors file carries both the LoRA tensors and the
# latent_memory_hint.* / pipe.latent_memory_text.* parameters.
OURS_EXP5_CHECKPOINT = (
    PROJECT_ROOT
    / "out"
    / "hf_checkpoints"
    / "checkpoints"
    / "exp5_ours_auxall_all_data_4gpu_save1k"
    / "real_train_output"
    / "step-6000.safetensors"
)


@dataclass(frozen=True)
class EditVariant:
    variant_id: str
    backend: str
    checkpoint_path: Path
    lora_paths: tuple[Path, ...] = ()
    lora_alphas: tuple[float, ...] = ()
    num_inference_steps: int = 20
    cfg_scale: float = 1.0
    denoising_strength: float = 0.7
    resolution: tuple[int, int] = (832, 480)
    inference_mode: str = "vt2v"
    description: str = ""
    # Trained latent-memory hint module. When the mode is not "none" the loader
    # attaches the adapter and restores its weights from latent_memory_checkpoint
    # (defaulting to the first LoRA path, which is where training writes them).
    latent_memory_mode: str = "none"
    latent_memory_tokens: int = 4
    latent_memory_hidden_dim: int = 1024
    latent_memory_scale: float = 1.0
    latent_memory_init_std: float = 0.02
    latent_memory_checkpoint: Path | None = None


EDIT_VARIANTS: dict[str, EditVariant] = {
    "wan1.3b": EditVariant(
        variant_id="wan1.3b",
        backend="diffsynth",
        checkpoint_path=WAN13B_DIR,
        description="Wan VACE 1.3B base model without LoRA.",
    ),
    "wan14b": EditVariant(
        variant_id="wan14b",
        backend="diffsynth",
        checkpoint_path=WAN14B_DIR,
        description="Wan VACE 14B base model without LoRA.",
    ),
    "wan14b-vace-base": EditVariant(
        variant_id="wan14b-vace-base",
        backend="ditto",
        checkpoint_path=WAN14B_DIR,
        num_inference_steps=50,
        cfg_scale=5.0,
        description=(
            "Wan2.1-VACE-14B base model without LoRA, using the local WanVideo "
            "pipeline and its documented 50-step / CFG-5 defaults."
        ),
    ),
    "wan14b+dittolora": EditVariant(
        variant_id="wan14b+dittolora",
        backend="ditto",
        checkpoint_path=WAN14B_DIR,
        lora_paths=(DITTO_GLOBAL_LORA,),
        lora_alphas=(1.0,),
        description="Wan VACE 14B with Ditto global LoRA.",
    ),
    "wan14b+4steplora": EditVariant(
        variant_id="wan14b+4steplora",
        backend="diffsynth",
        checkpoint_path=WAN14B_DIR,
        lora_paths=(DISTILL_4STEP_LORA,),
        lora_alphas=(1.0,),
        num_inference_steps=4,
        description="Wan VACE 14B with the LightX2V 4-step LoRA.",
    ),
    "wan14b+dittolora+4steplora": EditVariant(
        variant_id="wan14b+dittolora+4steplora",
        backend="ditto",
        checkpoint_path=WAN14B_DIR,
        lora_paths=(DITTO_GLOBAL_LORA, DISTILL_4STEP_LORA),
        lora_alphas=(1.0, 1.0),
        num_inference_steps=4,
        description="Wan VACE 14B with Ditto global LoRA plus the LightX2V 4-step LoRA.",
    ),
    "wan14b+editbasesft": EditVariant(
        variant_id="wan14b+editbasesft",
        backend="ditto",
        checkpoint_path=WAN14B_DIR,
        lora_paths=(EDIT_BASE_SFT_LORA,),
        lora_alphas=(1.0,),
        description="Wan VACE 14B with the locally trained base_sft edit-model LoRA.",
    ),
    "wan14b+editplanner": EditVariant(
        variant_id="wan14b+editplanner",
        backend="ditto",
        checkpoint_path=WAN14B_DIR,
        lora_paths=(EDIT_PLANNER_CONDITIONED_LORA,),
        lora_alphas=(1.0,),
        description="Wan VACE 14B with the locally trained planner-conditioned edit-model LoRA.",
    ),
    "wan14b+editplanner+4steplora": EditVariant(
        variant_id="wan14b+editplanner+4steplora",
        backend="ditto",
        checkpoint_path=WAN14B_DIR,
        lora_paths=(EDIT_PLANNER_CONDITIONED_LORA, DISTILL_4STEP_LORA),
        lora_alphas=(1.0, 1.0),
        num_inference_steps=4,
        description="Wan VACE 14B with planner-conditioned edit-model LoRA plus the LightX2V 4-step LoRA.",
    ),
    "bernini1.3b": EditVariant(
        variant_id="bernini1.3b",
        backend="bernini",
        checkpoint_path=BERNINI_DIR,
        num_inference_steps=40,
        description="ByteDance Bernini-R-1.3B via its own Diffusers inference entry point.",
    ),
    "wan14b+ours": EditVariant(
        variant_id="wan14b+ours",
        backend="ditto",
        checkpoint_path=WAN14B_DIR,
        lora_paths=(OURS_EXP5_CHECKPOINT,),
        lora_alphas=(1.0,),
        latent_memory_mode="text+vace_hint",
        latent_memory_tokens=4,
        latent_memory_hidden_dim=1024,
        description="Wan VACE 14B with the trained exp5 LoRA plus its latent-memory hint/text adapters (paper 'Ours').",
    ),
}


@dataclass
class EditRequest:
    low_video: Path
    output_video: Path
    positive_prompt: str
    negative_prompt: str
    variant: str = "wan14b"
    control_video: Path | None = None
    reference_image: Path | None = None
    num_frames: int | None = None
    fps: float | None = None
    seed: int = 42
    device: str = "cuda"
    memory_mode: str = "explicit"
    implicit_controls: dict | None = None


def _keep_edit_model_on_gpu(device: str) -> bool:
    raw = str(__import__("os").environ.get("LOW_HIGH_AGENT_KEEP_EDIT_ON_GPU", "")).strip().lower()
    if raw in {"0", "false", "no"}:
        return False
    if raw in {"1", "true", "yes"}:
        return True
    return str(device).startswith("cuda")


def _existing_weight(path: Path) -> bool:
    return path.exists() and path.is_file() and not path.name.endswith((".fdmdownload", ".crdownload"))


def _validate_local_model(model_dir: Path) -> dict:
    transformer_shards = sorted(model_dir.glob("diffusion_pytorch_model*.safetensors"))
    text_encoder_path = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = model_dir / "Wan2.1_VAE.pth"
    tokenizer_dir = model_dir / "google" / "umt5-xxl"
    required = [
        bool(transformer_shards),
        _existing_weight(text_encoder_path),
        _existing_weight(vae_path),
        tokenizer_dir.exists(),
    ]
    if not all(required):
        raise RuntimeError(
            f"Incomplete local checkpoint at {model_dir}. Expected diffusion shards, text encoder, VAE, and google/umt5-xxl tokenizer files."
        )
    return {
        "transformer_paths": [str(path) for path in transformer_shards],
        "text_encoder_path": str(text_encoder_path),
        "vae_path": str(vae_path),
        "tokenizer_dir": str(tokenizer_dir),
    }


def _probe_video_metadata(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    import json

    info = json.loads(result.stdout)
    stream = info["streams"][0]
    avg_rate = stream.get("avg_frame_rate", "0/1")
    fps = float(Fraction(avg_rate)) if avg_rate not in ("0/0", "0/1", "N/A") else 16.0
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": int(stream["nb_frames"]),
    }


def _normalize_num_frames(requested_num_frames: int | None, available_frame_count: int) -> int:
    if requested_num_frames is None:
        requested_num_frames = available_frame_count
    valid_num_frames = requested_num_frames - ((requested_num_frames - 1) % 4)
    if valid_num_frames > available_frame_count:
        valid_num_frames = available_frame_count - ((available_frame_count - 1) % 4)
    if valid_num_frames < 1:
        raise ValueError("The control video is too short for Wan's 4n+1 frame requirement.")
    return valid_num_frames


def _extract_reference_image(low_video: Path) -> Path:
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="low_high_new_ref_"))
    image_path = tmp_dir / "reference.png"
    cmd = ["ffmpeg", "-y", "-i", str(low_video), "-frames:v", "1", str(image_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return image_path


def _ensure_diffsynth_runtime() -> None:
    if find_spec("diffsynth") is not None:
        return
    if not DIFFSYNTH_ROOT.exists():
        raise RuntimeError(f"DiffSynth-Studio not found at {DIFFSYNTH_ROOT}.")
    if str(DIFFSYNTH_ROOT) not in sys.path:
        sys.path.insert(0, str(DIFFSYNTH_ROOT))


def _ensure_ditto_runtime() -> None:
    if not DITTO_ROOT.exists():
        raise RuntimeError(f"Ditto not found at {DITTO_ROOT}.")
    if str(DITTO_ROOT) not in sys.path:
        sys.path.insert(0, str(DITTO_ROOT))


def _variant_availability(variant: EditVariant) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if variant.backend == "bernini":
        # Diffusers layout, not Wan sharded weights, so validate its own entry points.
        if not variant.checkpoint_path.exists():
            missing.append(str(variant.checkpoint_path))
        if not (BERNINI_REPO / "infer_single_gpu.py").exists():
            missing.append(str(BERNINI_REPO / "infer_single_gpu.py"))
        return (not missing), missing
    if not variant.checkpoint_path.exists():
        missing.append(str(variant.checkpoint_path))
    else:
        try:
            _validate_local_model(variant.checkpoint_path)
        except RuntimeError:
            missing.append(str(variant.checkpoint_path))
    for path in variant.lora_paths:
        if not _existing_weight(path):
            missing.append(str(path))
    return (not missing), missing


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _effective_edit_controls(request: EditRequest, variant: EditVariant) -> tuple[EditRequest, EditVariant, dict]:
    controls = dict(request.implicit_controls or {})
    if not controls:
        return request, variant, {"applied": False}

    effective_request = replace(request)
    effective_variant = variant
    applied: dict = {"applied": True}

    variant_override = str(controls.get("variant_override", "") or "").strip()
    if variant_override and variant_override in EDIT_VARIANTS:
        effective_variant = EDIT_VARIANTS[variant_override]
        applied["variant_override"] = variant_override

    steps_delta = _safe_int(controls.get("steps_delta", controls.get("num_inference_steps_delta", 0)))
    cfg_delta = _safe_float(controls.get("cfg_delta", controls.get("cfg_scale_delta", 0.0)))
    denoise_delta = _safe_float(controls.get("denoise_delta", controls.get("denoising_strength_delta", 0.0)))
    if steps_delta or cfg_delta or denoise_delta:
        effective_variant = replace(
            effective_variant,
            num_inference_steps=_clamp_int(effective_variant.num_inference_steps + steps_delta, 1, 64),
            cfg_scale=_clamp_float(effective_variant.cfg_scale + cfg_delta, 0.1, 20.0),
            denoising_strength=_clamp_float(effective_variant.denoising_strength + denoise_delta, 0.0, 1.0),
        )
        applied["steps_delta"] = steps_delta
        applied["cfg_delta"] = cfg_delta
        applied["denoise_delta"] = denoise_delta

    if "seed_override" in controls:
        effective_request.seed = _safe_int(controls.get("seed_override"), request.seed)
        applied["seed_override"] = int(effective_request.seed)
    elif "seed_delta" in controls:
        effective_request.seed = _safe_int(request.seed, 42) + _safe_int(controls.get("seed_delta"), 0)
        applied["seed_delta"] = _safe_int(controls.get("seed_delta"), 0)

    control_video = str(controls.get("control_video", "") or "").strip()
    if control_video:
        control_video_path = Path(control_video).expanduser().resolve()
        if control_video_path.exists():
            effective_request.control_video = control_video_path
            applied["control_video"] = str(control_video_path)

    reference_image = str(controls.get("reference_image", "") or "").strip()
    if reference_image:
        reference_path = Path(reference_image).expanduser().resolve()
        if reference_path.exists():
            effective_request.reference_image = reference_path
            applied["reference_image"] = str(reference_path)

    if bool(controls.get("reference_from_low_video", False)) and effective_request.reference_image is None:
        try:
            effective_request.reference_image = _extract_reference_image(effective_request.low_video)
            applied["reference_from_low_video"] = True
        except Exception:
            pass

    return effective_request, effective_variant, applied


def list_variant_status() -> list[dict]:
    results = []
    for variant in EDIT_VARIANTS.values():
        available, missing = _variant_availability(variant)
        results.append(
            {
                "variant": variant.variant_id,
                "backend": variant.backend,
                "available": available,
                "missing": missing,
                "description": variant.description,
            }
        )
    return results


def _render_with_diffsynth(request: EditRequest, variant: EditVariant) -> dict:
    _ensure_diffsynth_runtime()
    import torch
    from diffsynth.utils.data import VideoData, save_video
    from PIL import Image

    if str(request.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the assigned GPU machine.")

    metadata = _probe_video_metadata(request.low_video)
    num_frames = _normalize_num_frames(request.num_frames, metadata["frame_count"])
    fps = request.fps or metadata["fps"]
    rendered_fps = int(round(float(fps)))
    width, height = variant.resolution
    pipe = _load_diffsynth_pipeline(variant.variant_id, request.device)
    latent_session = build_edit_latent_memory_session(
        pipe=pipe,
        controls=request.implicit_controls,
        backend=variant.backend,
        variant_id=variant.variant_id,
    )

    request.output_video.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "cfg_scale": variant.cfg_scale,
        "seed": request.seed,
        "tiled": True,
        "num_inference_steps": variant.num_inference_steps,
        "height": height,
        "width": width,
        "num_frames": num_frames,
    }
    if variant.inference_mode == "vt2v":
        kwargs["input_video"] = VideoData(str(request.low_video.resolve()), height=height, width=width)
        kwargs["input_video"].set_length(num_frames)
        kwargs["denoising_strength"] = variant.denoising_strength
        if request.control_video is not None:
            kwargs["control_video"] = VideoData(str(request.control_video.resolve()), height=height, width=width)
            kwargs["control_video"].set_length(num_frames)
        if request.reference_image is not None:
            ref_path = request.reference_image.resolve()
            kwargs["reference_image"] = Image.open(ref_path).convert("RGB").resize((width, height))
    else:
        kwargs["vace_video"] = VideoData(str((request.control_video or request.low_video).resolve()), height=height, width=width)
        kwargs["vace_video"].set_length(num_frames)

    try:
        video = pipe(**kwargs)
    finally:
        latent_session.close()
    save_video(video, str(request.output_video), fps=fps, quality=5)
    return {
        "variant": variant.variant_id,
        "backend": variant.backend,
        "checkpoint_path": str(variant.checkpoint_path),
        "output_video": str(request.output_video),
        "resolution": [width, height],
        "fps": fps,
        "num_frames": num_frames,
        "num_inference_steps": variant.num_inference_steps,
        "cfg_scale": variant.cfg_scale,
        "lora_paths": [str(path) for path in variant.lora_paths],
        "lora_alphas": [float(alpha) for alpha in variant.lora_alphas],
        "implicit_hidden_memory": latent_session.report(),
    }


def _render_with_ditto(request: EditRequest, variant: EditVariant) -> dict:
    _ensure_ditto_runtime()
    import torch
    from diffsynth import VideoData, save_video
    from PIL import Image

    if str(request.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the assigned GPU machine.")

    metadata = _probe_video_metadata(request.low_video)
    num_frames = _normalize_num_frames(request.num_frames, metadata["frame_count"])
    fps = request.fps or metadata["fps"]
    width, height = variant.resolution
    pipe = _load_ditto_pipeline(variant.variant_id, request.device)
    latent_session = build_edit_latent_memory_session(
        pipe=pipe,
        controls=request.implicit_controls,
        backend=variant.backend,
        variant_id=variant.variant_id,
    )

    video_path = request.control_video or request.low_video
    video = VideoData(str(video_path.resolve()), height=height, width=width)
    frames = [video[i] for i in range(num_frames)]

    reference_image = None
    if request.reference_image is not None:
        reference_image = Image.open(request.reference_image.resolve()).convert("RGB").resize((width, height))

    try:
        result = pipe(
            prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            vace_video=frames,
            vace_reference_image=reference_image,
            num_frames=num_frames,
            num_inference_steps=variant.num_inference_steps,
            cfg_scale=variant.cfg_scale,
            seed=request.seed,
            tiled=True,
        )
    finally:
        latent_session.close()

    request.output_video.parent.mkdir(parents=True, exist_ok=True)
    save_video(result, str(request.output_video), fps=fps, quality=5)
    return {
        "variant": variant.variant_id,
        "backend": variant.backend,
        "checkpoint_path": str(variant.checkpoint_path),
        "output_video": str(request.output_video),
        "resolution": [width, height],
        "fps": fps,
        "num_frames": num_frames,
        "num_inference_steps": variant.num_inference_steps,
        "cfg_scale": variant.cfg_scale,
        "lora_paths": [str(path) for path in variant.lora_paths],
        "lora_alphas": [float(alpha) for alpha in variant.lora_alphas],
        "implicit_hidden_memory": latent_session.report(),
    }


def _render_with_bernini(request: EditRequest, variant: EditVariant) -> dict:
    """Render through Bernini's own single-GPU entry point.

    Bernini is a separate Diffusers stack rather than a Wan/VACE LoRA, so it runs
    as a subprocess instead of sharing the resident pipeline. Generation settings
    mirror the published top50 baseline (40 steps, v2v_apg) so results stay
    comparable with the existing Bernini-R-1.3B videos.
    """
    if not BERNINI_REPO.exists():
        raise RuntimeError(f"Bernini repo not found at {BERNINI_REPO}")

    metadata = _probe_video_metadata(request.low_video)
    num_frames = _normalize_num_frames(request.num_frames, metadata["frame_count"])
    fps = request.fps or metadata["fps"]
    width, height = variant.resolution
    request.output_video.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "infer_single_gpu.py",
        "--config", str(variant.checkpoint_path),
        "--task_type", "v2v",
        "--video", str(request.low_video.resolve()),
        "--output", str(request.output_video.resolve()),
        "--prompt", request.positive_prompt,
        "--neg_prompt", request.negative_prompt,
        "--height", str(height),
        "--width", str(width),
        "--num_frames", str(num_frames),
        "--num_inference_steps", str(variant.num_inference_steps),
        "--guidance_mode", "v2v_apg",
        "--seed", str(request.seed),
        "--fps", str(rendered_fps),
    ]
    completed = subprocess.run(command, cwd=str(BERNINI_REPO), capture_output=True, text=True)
    if completed.returncode != 0 or not request.output_video.exists():
        tail = (completed.stderr or completed.stdout or "").strip()[-1500:]
        raise RuntimeError(f"Bernini inference failed (rc={completed.returncode}): {tail}")

    return {
        "variant": variant.variant_id,
        "backend": variant.backend,
        "checkpoint_path": str(variant.checkpoint_path),
        "output_video": str(request.output_video),
        "resolution": [width, height],
        "fps": rendered_fps,
        "requested_fps": fps,
        "num_frames": num_frames,
        "num_inference_steps": variant.num_inference_steps,
        "guidance_mode": "v2v_apg",
        "lora_paths": [],
        "lora_alphas": [],
        "implicit_hidden_memory": {},
    }


@lru_cache(maxsize=8)
def _load_diffsynth_pipeline(variant_id: str, device: str):
    _ensure_diffsynth_runtime()
    import torch
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    variant = EDIT_VARIANTS[variant_id]
    model_info = _validate_local_model(variant.checkpoint_path)
    keep_on_gpu = _keep_edit_model_on_gpu(device)
    if keep_on_gpu:
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cuda",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cuda",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
    else:
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }

    model_configs = [
        ModelConfig(path=model_info["transformer_paths"], **vram_config),
        ModelConfig(path=[model_info["text_encoder_path"]], **vram_config),
        ModelConfig(path=[model_info["vae_path"]], **vram_config),
    ]
    tokenizer_config = ModelConfig(path=[model_info["tokenizer_dir"]])
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        vram_limit=None if keep_on_gpu else (torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - 2 if str(device).startswith("cuda") else None),
    )
    target_module = pipe.dit if variant.inference_mode == "vt2v" else pipe.vace
    for path, alpha in zip(variant.lora_paths, variant.lora_alphas or ()):
        pipe.load_lora(target_module, str(path), alpha=float(alpha))
    return pipe


@lru_cache(maxsize=8)
def _load_ditto_pipeline(variant_id: str, device: str):
    _ensure_ditto_runtime()
    import torch
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    variant = EDIT_VARIANTS[variant_id]
    model_info = _validate_local_model(variant.checkpoint_path)
    keep_on_gpu = _keep_edit_model_on_gpu(device)
    offload_device = "cuda" if keep_on_gpu else "cpu"
    model_configs = [
        ModelConfig(path=model_info["transformer_paths"], offload_device=offload_device),
        ModelConfig(path=[model_info["text_encoder_path"]], offload_device=offload_device),
        ModelConfig(path=[model_info["vae_path"]], offload_device=offload_device),
    ]
    tokenizer_config = ModelConfig(path=str(model_info["tokenizer_dir"]))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )
    for path, alpha in zip(variant.lora_paths, variant.lora_alphas or ()):
        pipe.load_lora(pipe.vace, str(path), alpha=float(alpha))
    _attach_trained_latent_memory(pipe, variant)
    if not keep_on_gpu:
        pipe.enable_vram_management()
    return pipe


def _attach_trained_latent_memory(pipe, variant: EditVariant) -> None:
    """Restore the trained latent-memory adapters onto a loaded pipeline.

    ``load_lora`` only consumes the LoRA tensors, so without this the
    latent_memory_* parameters stored in the same checkpoint are silently
    dropped and the variant degrades to a plain LoRA run.
    """
    mode = str(variant.latent_memory_mode or "none").strip()
    if not mode or mode == "none":
        return

    from diffsynth.models.latent_memory_runtime import attach_latent_memory, load_latent_memory_checkpoint

    checkpoint = variant.latent_memory_checkpoint
    if checkpoint is None:
        if not variant.lora_paths:
            raise RuntimeError(
                f"variant {variant.variant_id} sets latent_memory_mode={mode} but has no checkpoint to restore from"
            )
        checkpoint = variant.lora_paths[0]

    adapter, prefixes = attach_latent_memory(
        pipe,
        mode=mode,
        num_tokens=variant.latent_memory_tokens,
        hidden_dim=variant.latent_memory_hidden_dim,
        scale=variant.latent_memory_scale,
        init_std=variant.latent_memory_init_std,
        log_prefix=f"[edit:{variant.variant_id}] latent_memory",
    )
    load_latent_memory_checkpoint(
        adapter,
        str(checkpoint),
        prefixes,
        required=True,
        log_prefix=f"[edit:{variant.variant_id}] latent_memory",
    )


def _apply_bench_protocol(request: EditRequest) -> EditRequest:
    """Pin export geometry to the bench protocol when it is configured.

    The closed loop builds its own EditRequest and does not thread frame/fps
    through, so benchmarking needs an out-of-band way to force every arm onto the
    same 17-frame / 20fps export. Explicit per-request values always win.
    """
    import os

    raw_frames = str(os.environ.get("LOW_HIGH_BENCH_NUM_FRAMES", "")).strip()
    raw_fps = str(os.environ.get("LOW_HIGH_BENCH_FPS", "")).strip()
    if not raw_frames and not raw_fps:
        return request
    if raw_frames and request.num_frames is None:
        try:
            request.num_frames = int(raw_frames)
        except ValueError:
            pass
    if raw_fps and request.fps is None:
        try:
            request.fps = float(raw_fps)
        except ValueError:
            pass
    return request


def render_edit_video(request: EditRequest) -> dict:
    if request.variant not in EDIT_VARIANTS:
        raise ValueError(f"Unsupported variant: {request.variant}. Available: {', '.join(sorted(EDIT_VARIANTS))}")
    request = _apply_bench_protocol(request)
    variant = EDIT_VARIANTS[request.variant]
    effective_request, effective_variant, control_meta = _effective_edit_controls(request, variant)
    request = effective_request
    variant = effective_variant
    available, missing = _variant_availability(variant)
    if not available:
        raise RuntimeError(f"Variant {variant.variant_id} is not locally runnable. Missing: {missing}")
    if variant.backend == "diffsynth":
        result = _render_with_diffsynth(request, variant)
    elif variant.backend == "ditto":
        result = _render_with_ditto(request, variant)
    elif variant.backend == "bernini":
        result = _render_with_bernini(request, variant)
    else:
        raise ValueError(f"Unsupported backend: {variant.backend}")
    result["memory_mode"] = str(request.memory_mode or "")
    result["implicit_controls_applied"] = control_meta
    return result
