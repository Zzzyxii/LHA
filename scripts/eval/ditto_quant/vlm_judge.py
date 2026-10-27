"""Optional local Qwen-VL judge for Ditto paper-style evaluation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are an impartial evaluator for instruction-based video editing.
You will receive sampled source-video frames followed by sampled edited-video frames.
Score how well the edited video satisfies the instruction while preserving the source content and motion.

Use a 0-10 scale:
- instruction_fidelity: edit follows the instruction.
- source_preservation: source identity, scene layout, and motion are preserved when they should be.
- visual_quality: edited frames are sharp, coherent, natural, and artifact-free.
- safety: unsafe or inappropriate content is absent.
- overall: holistic score considering all criteria.

Return exactly one JSON object with keys:
instruction_fidelity, source_preservation, visual_quality, safety, overall, rationale
"""


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
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, number))


class LocalQwenVideoJudge:
    """Local Qwen-VL judge using the project's existing VLM utility.

    The underlying loader is controlled by ``src/vlm/local_qwen.py``. This class
    sets the relevant environment variables before first use.
    """

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

    def score(self, item) -> dict[str, Any]:
        source_frames = self._extract_frames(item.source_video)
        edited_frames = self._extract_frames(item.edited_video)
        if not edited_frames:
            raise RuntimeError(f"could not extract edited frames from {item.edited_video}")

        if source_frames:
            user_text = (
                f"sample_id: {item.sample_id}\n"
                f"method: {item.method}\n"
                f"instruction: {item.instruction}\n\n"
                f"The first {len(source_frames)} images are source-video frames in temporal order. "
                f"The next {len(edited_frames)} images are edited-video frames in temporal order. "
                "Judge the edited video against the source and instruction."
            )
            image_paths = source_frames + edited_frames
        else:
            user_text = (
                f"sample_id: {item.sample_id}\n"
                f"method: {item.method}\n"
                f"instruction: {item.instruction}\n\n"
                f"All {len(edited_frames)} images are edited-video frames in temporal order. "
                "Source frames are unavailable, so set source_preservation based only on visible consistency."
            )
            image_paths = edited_frames

        raw_text = self.model.predict_text(system_prompt=SYSTEM_PROMPT, user_text=user_text, image_paths=image_paths, max_new_tokens=512)
        payload = _extract_json_object(raw_text)
        return {
            "instruction_fidelity": _score_0_10(payload.get("instruction_fidelity")),
            "source_preservation": _score_0_10(payload.get("source_preservation")),
            "visual_quality": _score_0_10(payload.get("visual_quality")),
            "safety": _score_0_10(payload.get("safety")),
            "overall": _score_0_10(payload.get("overall")),
            "rationale": str(payload.get("rationale") or ""),
            "raw_response": raw_text[:4000],
        }
