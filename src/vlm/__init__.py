import os

from vlm.litellm_vlm import LiteLLMVisionModel, use_litellm_vlm
from vlm.local_qwen import LocalQwenPlannerModel, extract_frames
from vlm.openrouter import OpenRouterVisionModel

__all__ = [
    "LiteLLMVisionModel",
    "LocalQwenPlannerModel",
    "OpenRouterVisionModel",
    "extract_frames",
    "resolve_vision_model",
    "use_litellm_vlm",
]


def resolve_vision_model(role: str = "planner"):
    """Return the VLM backend selected by environment.

    ``LOW_HIGH_AGENT_USE_LITELLM_VLM=1`` routes reasoning to the hosted litellm
    gateway; otherwise the local Qwen3-VL checkpoint is used.

    ``role`` lets the critic run on a different hosted model from the planner
    without touching the planner: ``LOW_HIGH_AGENT_LITELLM_<ROLE>_MODEL`` and
    ``LOW_HIGH_AGENT_LITELLM_<ROLE>_REASONING_EFFORT`` take precedence over the
    shared ``LOW_HIGH_AGENT_LITELLM_MODEL`` / ``..._REASONING_EFFORT``, so an
    unset role variable leaves behaviour exactly as before.
    """
    if not use_litellm_vlm():
        return LocalQwenPlannerModel()
    prefix = f"LOW_HIGH_AGENT_LITELLM_{role.strip().upper()}_"
    model = os.environ.get(prefix + "MODEL", "").strip()
    effort = os.environ.get(prefix + "REASONING_EFFORT", "").strip()
    return LiteLLMVisionModel(model=model, reasoning_effort=effort)
