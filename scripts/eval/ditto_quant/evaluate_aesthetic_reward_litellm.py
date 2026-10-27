#!/usr/bin/env python3
"""Score eval videos with a hosted judge model over an OpenAI-compatible endpoint.

This mirrors ``evaluate_aesthetic_reward.py`` exactly in prompt, dimensions and
1-4 scale, so the resulting ``overall_aesthetic_score`` stays comparable with the
``Reward overall`` column in ``reports/.../scores.csv``. The only difference is
the rater: instead of the locally trained aesthetic reward LoRA, requests go to a
litellm gateway (e.g. ``gpt-5.6-luna``).

Frames are decoded with PyAV so no ffmpeg binary is required.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import av

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


def _read_eval_inputs(path: Path) -> list[dict[str, Any]]:
    """Accept either a jsonl eval-inputs file or a sample_index.json array."""
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported eval-inputs format: {path}")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL | re.IGNORECASE).strip()
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
        root / f"{sample_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"generation not found for method={method} sample_id={sample_id}: {candidates[0]}")


def _extract_frames_b64(video_path: Path, *, frames: int, width: int = 640, quality: int = 85) -> list[str]:
    """Decode ``frames`` evenly spaced frames and return base64 JPEG payloads."""
    container = av.open(str(video_path))
    try:
        images = [frame.to_image() for frame in container.decode(video=0)]
    finally:
        container.close()
    if not images:
        raise RuntimeError(f"no decodable frames in {video_path}")

    if frames >= len(images):
        picked = images
    else:
        # Evenly spaced sample across the clip, matching fps=1-style coverage.
        step = (len(images) - 1) / max(frames - 1, 1) if frames > 1 else 0
        picked = [images[min(len(images) - 1, round(index * step))] for index in range(frames)]

    payloads: list[str] = []
    for image in picked:
        if image.width > width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=quality)
        payloads.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return payloads


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


class LiteLLMJudge:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str = "",
        temperature: float | None = None,
        max_completion_tokens: int = 3000,
        timeout: int = 600,
        retries: int = 4,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.retries = retries
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0

    def complete(self, *, system_prompt: str, user_text: str, images_b64: list[str]) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for payload in images_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            body["temperature"] = self.temperature

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    self.url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                usage = payload.get("usage") or {}
                with self._cost_lock:
                    self.total_cost += float(usage.get("cost") or 0.0)
                    self.total_tokens += int(usage.get("total_tokens") or 0)
                choice = payload["choices"][0]
                text = choice["message"].get("content") or ""
                if not text.strip():
                    if choice.get("finish_reason") == "length":
                        # Reasoning used up the budget before the JSON was written;
                        # an identical retry would fail the same way, so widen it.
                        body["max_completion_tokens"] = min(body["max_completion_tokens"] * 2, 128000)
                    raise RuntimeError("empty completion content")
                return text
            except Exception as exc:  # noqa: BLE001 - retry on any transport/parse failure
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 400:
                    try:
                        detail = exc.read().decode("utf-8", "replace")[:800]
                    except Exception:
                        detail = ""
                    raise RuntimeError(f"judge request rejected (400): {detail}") from exc
                if attempt < self.retries - 1:
                    time.sleep(min(2 ** attempt * 2, 30))
        raise RuntimeError(f"judge request failed after {self.retries} attempts: {last_error!r}")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score_one(
    *,
    judge: LiteLLMJudge,
    row: dict[str, Any],
    method: str,
    generation_root: Path,
    frames: int,
    index: int,
    total: int,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or f"eval50_{index - 1:03d}")
    instruction = str(row.get("instruction") or row.get("prompt") or "")
    try:
        video_path = _generation_path(generation_root, method, sample_id)
        images = _extract_frames_b64(video_path, frames=frames)
        raw_text = judge.complete(
            system_prompt=_build_aesthetic_evaluator_system_prompt(),
            user_text=_build_aesthetic_evaluator_user_prompt(instruction),
            images_b64=images,
        )
        payload = _extract_json_object(raw_text)
        scores = payload.get("scores") or {}
        result = {
            "sample_id": sample_id,
            "method": method,
            "video": str(video_path),
            "frames": len(images),
            "scores": scores,
            "overall_aesthetic_score": payload.get("overall_aesthetic_score"),
            "uncertain": payload.get("uncertain"),
            "reason": payload.get("reason"),
            "raw_output": raw_text,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - record and continue
        result = {
            "sample_id": sample_id,
            "method": method,
            "video": "",
            "frames": 0,
            "scores": {},
            "overall_aesthetic_score": None,
            "uncertain": None,
            "reason": "",
            "raw_output": "",
            "error": repr(exc),
        }
    print(f"[litellm_judge] {method} {index}/{total} {sample_id} error={bool(result['error'])}", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score eval videos with a hosted judge over an OpenAI-compatible endpoint.")
    parser.add_argument("--eval-inputs", type=Path, required=True, help="eval-inputs jsonl or sample_index.json")
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("LOW_HIGH_JUDGE_BASE_URL", "http://litellm-sgp.woa.com"))
    parser.add_argument("--model", default=os.environ.get("LOW_HIGH_JUDGE_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--api-key-env", default="LOW_HIGH_JUDGE_API_KEY")
    parser.add_argument("--reasoning-effort", default=os.environ.get("LOW_HIGH_JUDGE_REASONING_EFFORT", "xhigh"), help="none/low/medium/high/xhigh, or empty to omit")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-sample-ids", type=Path, default=None, help="optional txt file of sample_ids to keep")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing judge API key in env var: {args.api_key_env}")

    rows = _read_eval_inputs(args.eval_inputs)
    if args.only_sample_ids:
        keep = {line.strip() for line in args.only_sample_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        rows = [row for row in rows if str(row.get("sample_id")) in keep]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no eval rows selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    judge = LiteLLMJudge(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort.strip(),
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.timeout,
        retries=args.retries,
    )

    summary: dict[str, Any] = {}
    for method in args.methods:
        total = len(rows)
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [
                pool.submit(
                    _score_one,
                    judge=judge,
                    row=row,
                    method=method,
                    generation_root=args.generation_root,
                    frames=args.frames,
                    index=index,
                    total=total,
                )
                for index, row in enumerate(rows, start=1)
            ]
            results = [future.result() for future in futures]

        results.sort(key=lambda item: item["sample_id"])
        per_sample_path = args.output_dir / f"{method}.aesthetic_reward.jsonl"
        with per_sample_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

        aggregates: dict[str, list[float]] = {name: [] for name in DIMENSIONS}
        aggregates["overall_aesthetic_score"] = []
        errors = 0
        for result in results:
            if result["error"]:
                errors += 1
                continue
            for name in DIMENSIONS:
                value = _float_or_none((result["scores"] or {}).get(name))
                if value is not None:
                    aggregates[name].append(value)
            overall = _float_or_none(result["overall_aesthetic_score"])
            if overall is not None:
                aggregates["overall_aesthetic_score"].append(overall)

        summary[method] = {
            "n": total,
            "errors": errors,
            "scored": len(aggregates["overall_aesthetic_score"]),
            **{name: _mean(values) for name, values in aggregates.items()},
        }
        print(f"[litellm_judge] {method} done: {json.dumps(summary[method], ensure_ascii=False)}", flush=True)

    summary["_meta"] = {
        "judge_model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "frames": args.frames,
        "n_samples": len(rows),
        "sample_ids": [str(row.get("sample_id")) for row in rows],
        "total_cost_usd": round(judge.total_cost, 6),
        "total_tokens": judge.total_tokens,
    }
    (args.output_dir / "summary_aesthetic_reward.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
