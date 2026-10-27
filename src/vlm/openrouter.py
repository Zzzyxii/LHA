from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests


DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_SITE_URL = "http://localhost"
DEFAULT_OPENROUTER_APP_NAME = "low-high-new"


def extract_frames(video_path: str, *, frame_count: int = 4, width: int = 640) -> list[Path]:
    tmpdir = Path(tempfile.mkdtemp(prefix="openrouter_frames_"))
    out_pattern = tmpdir / "frame_%02d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps=1,scale={width}:-1",
        "-frames:v",
        str(frame_count),
        str(out_pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(tmpdir.glob("frame_*.jpg"))


def _image_to_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(stripped[start : end + 1])


class OpenRouterVisionModel:
    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        api_url: str = DEFAULT_OPENROUTER_URL,
        api_key_env: str = "OPENROUTER_API_KEY",
        timeout: int = 300,
        max_tokens: int = 1500,
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self.model = model
        self.api_url = api_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.site_url = (site_url or os.environ.get("OPENROUTER_SITE_URL") or DEFAULT_OPENROUTER_SITE_URL).strip()
        self.app_name = (app_name or os.environ.get("OPENROUTER_APP_NAME") or DEFAULT_OPENROUTER_APP_NAME).strip()
        self.api_key = os.environ.get(api_key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"Missing OpenRouter API key in env var: {api_key_env}")

    def predict_json(self, *, system_prompt: str, user_text: str, image_paths: list[str]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(Path(image_path))}})
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.2,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )
        if not response.ok:
            body = response.text.strip()
            try:
                payload = response.json()
                body = json.dumps(payload, ensure_ascii=False)
            except Exception:
                pass
            raise RuntimeError(
                f"OpenRouter request failed: status={response.status_code}, model={self.model}, body={body}"
            )
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        return _extract_json_object(text)
