from __future__ import annotations

from typing import Protocol


class VisionPlannerModel(Protocol):
    def predict_json(self, *, system_prompt: str, user_text: str, image_paths: list[str]) -> dict:
        ...
