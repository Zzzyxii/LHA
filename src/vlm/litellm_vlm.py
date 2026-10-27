"""Hosted VLM backend speaking the OpenAI-compatible litellm gateway protocol.

Drop-in replacement for ``vlm.local_qwen.LocalQwenPlannerModel``: same
``predict_text`` / ``predict_json`` surface, so planner and critic can run against
a hosted judge (e.g. ``gemini-3.1-pro-preview``) instead of the local Qwen3-VL.

Activated with ``LOW_HIGH_AGENT_USE_LITELLM_VLM=1``; see ``vlm.resolve_planner_model``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://litellm-sgp.woa.com"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

# Hard ceiling on max_completion_tokens; gpt-6.1-sol and the gpt-5.6 family
# advertise 128k output tokens on this gateway.
MAX_COMPLETION_TOKENS = 128000
# Extra output tokens reserved for reasoning, by reasoning_effort. Measured on
# gpt-6.1-sol: an xhigh planner call spent >8k tokens thinking before answering.
REASONING_HEADROOM = {"minimal": 2048, "low": 4096, "medium": 8192, "high": 16384, "xhigh": 32768}


def use_litellm_vlm() -> bool:
    raw = str(os.environ.get("LOW_HIGH_AGENT_USE_LITELLM_VLM", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL | re.IGNORECASE).strip()
    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE):
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
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            payload, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"No JSON object found in model output: {stripped[:400]}")


_TRANSIENT_GATEWAY_MARKERS = (
    "下游服务暂不可用",
    "请稍后重试",
    "service_offline",
    "temporarily unavailable",
    "try again later",
    "overloaded",
    "upstream connect error",
)


def _is_transient_gateway_error(status_code: int, body: str) -> bool:
    """Whether a 4xx body actually describes a retryable upstream outage.

    The gateway wraps transient downstream failures in HTTP 400 rather than 5xx,
    so a literal reading of the status code would abandon a sample that would
    have succeeded moments later. Request-size rejections (413) stay fatal.
    """
    if status_code != 400:
        return False
    lowered = (body or "").lower()
    return any(marker.lower() in lowered for marker in _TRANSIENT_GATEWAY_MARKERS)


def _max_image_width() -> int:
    try:
        return max(64, int(_env("LOW_HIGH_AGENT_LITELLM_MAX_IMAGE_WIDTH") or 768))
    except ValueError:
        return 768


def _image_to_data_url(path: str) -> str:
    """Encode one frame as a JPEG data URL, downscaling wide frames.

    The planner hands over PNG frames; six of them base64-encoded at full width
    is enough to trip the gateway's request-size limit (HTTP 413), so re-encode
    to bounded-width JPEG instead of forwarding the raw bytes.
    """
    max_width = _max_image_width()
    try:
        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height), Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=85)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
    except Exception:
        # Fall back to the original bytes rather than losing the frame entirely.
        payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        mime = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{payload}"


class LiteLLMVisionModel:
    """OpenAI-compatible hosted VLM with retry and usage accounting."""

    _usage_lock = threading.Lock()
    total_cost = 0.0
    total_tokens = 0
    total_calls = 0

    def __init__(
        self,
        *,
        base_url: str = "",
        model: str = "",
        api_key: str = "",
        reasoning_effort: str = "",
        temperature: float | None = None,
        timeout: int = 600,
        retries: int = 8,
    ) -> None:
        self.base_url = (base_url or _env("LOW_HIGH_JUDGE_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or _env("LOW_HIGH_AGENT_LITELLM_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or _env("LOW_HIGH_JUDGE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing judge API key: set LOW_HIGH_JUDGE_API_KEY")
        self.reasoning_effort = reasoning_effort or _env("LOW_HIGH_AGENT_LITELLM_REASONING_EFFORT")
        self.temperature = temperature
        self.timeout = int(_env("LOW_HIGH_AGENT_LITELLM_TIMEOUT") or timeout)
        self.retries = int(_env("LOW_HIGH_AGENT_LITELLM_RETRIES") or retries)

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _completion_budget(self, max_new_tokens: int) -> int:
        """Output-token budget: the caller's answer length plus room to reason.

        ``max_completion_tokens`` covers reasoning *and* the visible answer, but
        callers size ``max_new_tokens`` for the answer alone (they were written for
        a local model that does not think out loud). Without headroom a reasoning
        model can spend the entire budget thinking and return nothing.
        """
        headroom = REASONING_HEADROOM.get(self.reasoning_effort.lower(), 0) if self.reasoning_effort else 0
        return min(max(int(max_new_tokens), 2048) + headroom, MAX_COMPLETION_TOKENS)

    def _record_usage(self, usage: dict[str, Any]) -> None:
        with LiteLLMVisionModel._usage_lock:
            LiteLLMVisionModel.total_cost += float(usage.get("cost") or 0.0)
            LiteLLMVisionModel.total_tokens += int(usage.get("total_tokens") or 0)
            LiteLLMVisionModel.total_calls += 1

    @classmethod
    def usage_summary(cls) -> dict[str, Any]:
        with cls._usage_lock:
            return {
                "calls": cls.total_calls,
                "tokens": cls.total_tokens,
                "cost_usd": round(cls.total_cost, 6),
            }

    def predict_text(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_paths: list[str],
        max_new_tokens: int = 1024,
    ) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for image_path in image_paths or []:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self._completion_budget(max_new_tokens),
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            body["temperature"] = self.temperature

        last_error: Exception | None = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                request = urllib.request.Request(
                    self.url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                self._record_usage(payload.get("usage") or {})
                choice = payload["choices"][0]
                text = choice["message"].get("content") or ""
                if not text.strip():
                    if choice.get("finish_reason") == "length":
                        # Reasoning consumed the whole budget before any answer was
                        # written. Retrying at the same budget fails identically, so
                        # widen it -- this is what kept the planner's final call
                        # (8192 tokens at xhigh) looping for half an hour.
                        widened = min(body["max_completion_tokens"] * 2, MAX_COMPLETION_TOKENS)
                        if widened > body["max_completion_tokens"]:
                            body["max_completion_tokens"] = widened
                        raise RuntimeError(
                            f"empty completion: hit length limit, budget widened to {body['max_completion_tokens']}"
                        )
                    raise RuntimeError("empty completion content")
                return text
            except Exception as exc:  # noqa: BLE001 - retry transport and parse failures alike
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in {400, 413}:
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:600]
                    except Exception:
                        detail = ""
                    if not _is_transient_gateway_error(exc.code, detail):
                        raise RuntimeError(
                            f"litellm request rejected ({exc.code}) model={self.model}: {detail}"
                        ) from exc
                    last_error = RuntimeError(f"transient gateway {exc.code}: {detail}")
                if attempt >= self.retries - 1:
                    break
                delay = self._retry_delay(exc, attempt)
                # Retries used to be silent, so a request stuck in a timeout/retry
                # cycle was indistinguishable from a merely slow one.
                print(
                    f"[litellm_vlm] model={self.model} attempt {attempt + 1}/{self.retries} failed after "
                    f"{time.time() - started:.0f}s: {type(exc).__name__}: {str(exc)[:160]} -- retrying in {delay:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(f"litellm request failed after {self.retries} attempts: {last_error!r}")

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Backoff for one retry, honouring Retry-After and widening for 429s.

        Several bench shards share one gateway quota, so throttling is expected
        rather than exceptional: 429 needs a much longer, jittered wait than a
        transient socket error, or every shard retries in lockstep and re-trips
        the limit together.
        """
        if isinstance(exc, urllib.error.HTTPError):
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after:
                try:
                    return min(float(retry_after) + random.uniform(0, 5), 180.0)
                except (TypeError, ValueError):
                    pass
            if exc.code == 429:
                return min(15.0 * (2 ** attempt), 180.0) + random.uniform(0, 10)
        return min(2.0 * (2 ** attempt), 30.0) + random.uniform(0, 2)

    def predict_json(self, *, system_prompt: str, user_text: str, image_paths: list[str]) -> dict:
        text = self.predict_text(
            system_prompt=system_prompt,
            user_text=user_text,
            image_paths=image_paths,
            max_new_tokens=4096,
        )
        return _extract_json_object(text)
