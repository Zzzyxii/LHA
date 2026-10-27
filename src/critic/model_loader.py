from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from vlm import resolve_vision_model
from vlm.local_qwen import _actual_model_device, local_model_path
from vlm.openrouter import extract_frames as extract_frames_jpg

from critic.prompts import build_critic_identity_prefix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CRITIC_MODELS_ROOT = PROJECT_ROOT / "models" / "critic"


def critic_models_root() -> Path:
    raw = os.environ.get("LOW_HIGH_AGENT_CRITIC_MODELS_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_CRITIC_MODELS_ROOT


def _candidate_root_specs(kind: str) -> list[tuple[str, Path]]:
    env_specific = os.environ.get(f"LOW_HIGH_AGENT_CRITIC_{kind.upper()}_MODEL", "").strip()
    env_generic = os.environ.get("LOW_HIGH_AGENT_CRITIC_MODEL", "").strip()
    env_pairwise = os.environ.get("LOW_HIGH_AGENT_CRITIC_PAIRWISE_MODEL", "").strip() if kind == "evaluator" else ""
    candidates: list[tuple[str, Path]] = []
    if env_specific:
        candidates.append((f"{kind}_env_override", Path(env_specific).expanduser().resolve()))
    if env_generic:
        candidates.append(("generic_env_override", Path(env_generic).expanduser().resolve()))
    if env_pairwise:
        candidates.append(("pairwise_env_override", Path(env_pairwise).expanduser().resolve()))
    root = critic_models_root()
    if kind == "evaluator":
        # After pairwise RM, the sharpened critic should be treated as the final evaluator checkpoint.
        candidates.extend(
            [
                ("models_root_pairwise", root / "pairwise"),
                ("models_root_evaluator", root / "evaluator"),
            ]
        )
    else:
        candidates.append((f"models_root_{kind}", root / kind))

    deduped: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, candidate in candidates:
        if candidate in seen:
            continue
        deduped.append((label, candidate))
        seen.add(candidate)
    return deduped


def _candidate_roots(kind: str) -> list[Path]:
    return [path for _, path in _candidate_root_specs(kind)]


def _resolve_checkpoint_dir(kind: str) -> Path | None:
    for root in _candidate_roots(kind):
        if not root.exists():
            continue
        if (root / "adapter_config.json").exists():
            return root
        checkpoint_dir = root / "real_sft_checkpoint"
        if (checkpoint_dir / "adapter_config.json").exists():
            return checkpoint_dir
    return None


def _resolve_checkpoint_dir_with_file(kind: str, filename: str, preferred_subdir: str = "fast_checkpoint") -> Path | None:
    for root in _candidate_roots(kind):
        if not root.exists():
            continue
        candidates = [root / preferred_subdir, root / "real_sft_checkpoint", root]
        for candidate in candidates:
            if (candidate / filename).exists():
                return candidate
    return None


def critic_checkpoint_dir(kind: str) -> Path | None:
    return _resolve_checkpoint_dir(kind)


def critic_has_checkpoint(kind: str) -> bool:
    return critic_checkpoint_dir(kind) is not None


def _checkpoint_source_label(kind: str, checkpoint_dir: Path | None) -> str:
    if checkpoint_dir is None:
        return ""
    for label, root in _candidate_root_specs(kind):
        candidate_paths = [root, root / "real_sft_checkpoint", root / "fast_checkpoint"]
        if any(candidate == checkpoint_dir for candidate in candidate_paths):
            return label
    return "unknown"


def _message_content(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append({"type": "image", "image": str(image_path)})
    return content


def _infer_hidden_size(model) -> int:
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Unable to infer hidden_size for critic head")
    return int(hidden_size)


@lru_cache(maxsize=4)
def _load_adapted_model(kind: str):
    checkpoint_dir = critic_checkpoint_dir(kind)
    if checkpoint_dir is None:
        raise RuntimeError(f"No critic checkpoint configured for kind={kind}")
    base_model_path = str(local_model_path())
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return processor, model, checkpoint_dir


@lru_cache(maxsize=2)
def _load_fast_score_model(kind: str):
    checkpoint_dir = _resolve_checkpoint_dir_with_file(kind, "evaluator_head.pt")
    if checkpoint_dir is None:
        raise RuntimeError(f"No fast evaluator checkpoint configured for kind={kind}")
    processor = AutoProcessor.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    base_model_path = str(local_model_path())
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if (checkpoint_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    else:
        model = base_model
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    hidden_size = _infer_hidden_size(model)
    evaluator_head = torch.nn.Linear(hidden_size, len(CriticScoreModel.SCORE_NAMES))
    evaluator_head.load_state_dict(torch.load(checkpoint_dir / "evaluator_head.pt", map_location="cpu"))
    model_param = next(model.parameters())
    evaluator_head = evaluator_head.to(device=model_param.device, dtype=model_param.dtype)
    evaluator_head.eval()
    return processor, model, evaluator_head, checkpoint_dir


@lru_cache(maxsize=2)
def _load_pairwise_reward_model():
    checkpoint_dir = _resolve_checkpoint_dir_with_file("pairwise", "reward_head.pt")
    if checkpoint_dir is None:
        raise RuntimeError("No pairwise reward checkpoint configured")
    processor, model, _ = _load_adapted_model("pairwise")
    reward_head = torch.nn.Linear(_infer_hidden_size(model), 1)
    reward_head.load_state_dict(torch.load(checkpoint_dir / "reward_head.pt", map_location="cpu"))
    model_param = next(model.parameters())
    reward_head = reward_head.to(device=model_param.device, dtype=model_param.dtype)
    reward_head.eval()
    return processor, model, reward_head, checkpoint_dir


class CriticTextModel:
    def __init__(self, *, kind: str) -> None:
        self.kind = kind
        self._fallback_model = None

    @property
    def _fallback(self):
        # Resolved lazily: instantiating the hosted backend validates credentials and
        # the local backend loads weights, neither of which is wanted at import time.
        if self._fallback_model is None:
            self._fallback_model = resolve_vision_model(role="critic")
        return self._fallback_model

    def available(self) -> bool:
        return critic_has_checkpoint(self.kind)

    def warmup(self) -> bool:
        if not self.available():
            return False
        _load_adapted_model(self.kind)
        return True

    def predict_text(self, *, system_prompt: str, user_text: str, image_paths: list[str], max_new_tokens: int = 1024) -> str:
        if not self.available():
            return self._fallback.predict_text(
                system_prompt=system_prompt,
                user_text=user_text,
                image_paths=image_paths,
                max_new_tokens=max_new_tokens,
            )

        processor, model, _ = _load_adapted_model(self.kind)
        messages = [
            {"role": "system", "content": _message_content(system_prompt, [])},
            {"role": "user", "content": _message_content(user_text, image_paths)},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        decoded = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0]

    def describe(self) -> dict[str, Any]:
        checkpoint_dir = critic_checkpoint_dir(self.kind)
        if checkpoint_dir is None:
            return {"kind": self.kind, "backend": "local_qwen_fallback"}
        _, model, _ = _load_adapted_model(self.kind)
        return {
            "kind": self.kind,
            "backend": "adapted_qwen_vl",
            "checkpoint_dir": str(checkpoint_dir),
            "actual_model_device": _actual_model_device(model),
        }


class CriticScoreModel:
    SCORE_NAMES = [
        "prompt_alignment",
        "structure_preservation",
        "transformation_strength",
        "carrier_grounding",
        "world_realization",
        "temporal_coherence",
        "artifact_penalty",
        "overall_score",
    ]

    def __init__(self, *, kind: str) -> None:
        self.kind = kind

    def checkpoint_dir(self) -> Path | None:
        return _resolve_checkpoint_dir_with_file(self.kind, "evaluator_head.pt")

    def available(self) -> bool:
        return self.checkpoint_dir() is not None

    def warmup(self) -> bool:
        if not self.available():
            return False
        _load_fast_score_model(self.kind)
        return True

    def predict_scores(self, *, user_text: str, image_paths: list[str]) -> dict[str, float]:
        checkpoint_dir = self.checkpoint_dir()
        if checkpoint_dir is None:
            raise RuntimeError(f"No fast evaluator checkpoint configured for kind={self.kind}")
        processor, model, evaluator_head, _ = _load_fast_score_model(self.kind)

        messages = [{"role": "user", "content": _message_content(user_text, image_paths)}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            model_inputs = model_inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            outputs = model(**model_inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            last_token_index = model_inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_token_index]
            preds = evaluator_head(pooled)[0]
            preds = torch.clamp(preds, 0.0, 1.0)
        return {name: float(preds[idx]) for idx, name in enumerate(self.SCORE_NAMES)}


class CriticPairwiseModel:
    def __init__(self) -> None:
        self.kind = "pairwise"

    def available(self) -> bool:
        return (
            _resolve_checkpoint_dir_with_file(self.kind, "reward_head.pt") is not None
            or _resolve_checkpoint_dir_with_file(self.kind, "pairwise_head.pt") is not None
        )

    def compare(
        self,
        *,
        prompt: str,
        low_video: str | Path,
        high_video: str | Path,
        candidate_a_video: str | Path,
        candidate_b_video: str | Path,
        candidate_a_prompt: str = "",
        candidate_b_prompt: str = "",
        frames_per_video: int = 1,
    ) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError("pairwise critic checkpoint is not available")

        reward_checkpoint_dir = _resolve_checkpoint_dir_with_file(self.kind, "reward_head.pt")
        if reward_checkpoint_dir is not None:
            processor, model, reward_head, checkpoint_dir = _load_pairwise_reward_model()

            def _score_candidate(*, candidate_video: str | Path, candidate_prompt: str) -> float:
                image_paths: list[str] = []
                for path in [low_video, high_video, candidate_video]:
                    image_paths.extend(str(p) for p in extract_frames_jpg(str(path), frame_count=frames_per_video))
                user_text = (
                    build_critic_identity_prefix()
                    +
                    f"prompt: {prompt}\n"
                    f"candidate_prompt: {candidate_prompt or prompt}\n"
                    "图片顺序如下：先是 low video 抽帧，再是 high video 抽帧，最后是该 edited candidate 抽帧。"
                )
                messages = [{"role": "user", "content": _message_content(user_text, image_paths)}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                model_inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                if torch.cuda.is_available():
                    model_inputs = model_inputs.to(next(model.parameters()).device)
                with torch.inference_mode():
                    outputs = model(**model_inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    last_token_index = model_inputs["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_token_index]
                    pooled = pooled.to(device=reward_head.weight.device, dtype=reward_head.weight.dtype)
                    score = reward_head(pooled).view(-1)[0]
                return float(score)

            score_a = _score_candidate(candidate_video=candidate_a_video, candidate_prompt=candidate_a_prompt)
            score_b = _score_candidate(candidate_video=candidate_b_video, candidate_prompt=candidate_b_prompt)
            prob_a = float(torch.sigmoid(torch.tensor(score_a - score_b)).item())
            prob_b = 1.0 - prob_a
            winner = "a" if score_a >= score_b else "b"
            return {
                "winner": winner,
                "prob_a": prob_a,
                "prob_b": prob_b,
                "score_a": score_a,
                "score_b": score_b,
                "checkpoint_dir": str(checkpoint_dir),
                "head_type": "reward_head",
            }

        processor, model, checkpoint_dir = _load_adapted_model(self.kind)
        pairwise_head_path = _resolve_checkpoint_dir_with_file(self.kind, "pairwise_head.pt")
        if pairwise_head_path is None:
            raise RuntimeError("Neither reward_head.pt nor legacy pairwise_head.pt is available")
        pairwise_head = torch.nn.Linear(_infer_hidden_size(model), 2)
        pairwise_head.load_state_dict(torch.load(pairwise_head_path / "pairwise_head.pt", map_location="cpu"))
        model_param = next(model.parameters())
        pairwise_head = pairwise_head.to(device=model_param.device, dtype=model_param.dtype)
        pairwise_head.eval()

        image_paths: list[str] = []
        for path in [low_video, high_video, candidate_a_video, candidate_b_video]:
            image_paths.extend(str(p) for p in extract_frames_jpg(str(path), frame_count=frames_per_video))
        user_text = (
            build_critic_identity_prefix()
            +
            f"prompt: {prompt}\n"
            f"candidate_a_prompt: {candidate_a_prompt}\n"
            f"candidate_b_prompt: {candidate_b_prompt}\n"
            "图片顺序如下：先是 low video 抽帧，再是 high video 抽帧，然后是 candidate A 抽帧，最后是 candidate B 抽帧。"
        )
        messages = [{"role": "user", "content": _message_content(user_text, image_paths)}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        model_inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            model_inputs = model_inputs.to(next(model.parameters()).device)
        with torch.inference_mode():
            outputs = model(**model_inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            last_token_index = model_inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_token_index]
            logits = pairwise_head(pooled)
            probs = torch.softmax(logits, dim=-1)[0]
        winner = "a" if float(probs[0]) >= float(probs[1]) else "b"
        return {
            "winner": winner,
            "prob_a": float(probs[0]),
            "prob_b": float(probs[1]),
            "checkpoint_dir": str(checkpoint_dir),
            "head_type": "legacy_pairwise_head",
        }


def load_critic_artifact_summary() -> dict[str, Any]:
    summary = {}
    for kind in ["evaluator", "reflector", "pairwise"]:
        checkpoint_dir = critic_checkpoint_dir(kind)
        summary[kind] = {
            "available": checkpoint_dir is not None,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else "",
            "source": _checkpoint_source_label(kind, checkpoint_dir),
            "candidate_roots": [
                {"label": label, "path": str(path)}
                for label, path in _candidate_root_specs(kind)
            ],
        }
    evaluator_fast = _resolve_checkpoint_dir_with_file("evaluator", "evaluator_head.pt")
    summary["evaluator_fast"] = {
        "available": evaluator_fast is not None,
        "checkpoint_dir": str(evaluator_fast) if evaluator_fast else "",
        "source": _checkpoint_source_label("evaluator", evaluator_fast),
    }
    evaluator_reward = _resolve_checkpoint_dir_with_file("evaluator", "reward_head.pt")
    summary["evaluator_reward"] = {
        "available": evaluator_reward is not None,
        "checkpoint_dir": str(evaluator_reward) if evaluator_reward else "",
        "source": _checkpoint_source_label("evaluator", evaluator_reward),
    }
    pairwise_reward = _resolve_checkpoint_dir_with_file("pairwise", "reward_head.pt")
    summary["pairwise_reward"] = {
        "available": pairwise_reward is not None,
        "checkpoint_dir": str(pairwise_reward) if pairwise_reward else "",
        "source": _checkpoint_source_label("pairwise", pairwise_reward),
    }
    pairwise_legacy = _resolve_checkpoint_dir_with_file("pairwise", "pairwise_head.pt")
    summary["pairwise_legacy"] = {
        "available": pairwise_legacy is not None,
        "checkpoint_dir": str(pairwise_legacy) if pairwise_legacy else "",
        "source": _checkpoint_source_label("pairwise", pairwise_legacy),
    }
    summary["evaluator_runtime"] = {
        "mode": "text_critic"
        if summary["evaluator"]["available"]
        else ("fast_score_head" if summary["evaluator_fast"]["available"] else "fallback"),
        "text_checkpoint_dir": summary["evaluator"]["checkpoint_dir"],
        "text_source": summary["evaluator"]["source"],
        "fast_checkpoint_dir": summary["evaluator_fast"]["checkpoint_dir"],
        "fast_source": summary["evaluator_fast"]["source"],
        "reward_checkpoint_dir": summary["evaluator_reward"]["checkpoint_dir"],
        "reward_source": summary["evaluator_reward"]["source"],
    }
    return
