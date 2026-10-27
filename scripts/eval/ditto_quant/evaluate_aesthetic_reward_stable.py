#!/usr/bin/env python3
"""Low-variance variant of the hosted aesthetic-reward judge.

A test-retest run of ``evaluate_aesthetic_reward_litellm.py`` scored the same 30
videos twice and disagreed badly: per-video sd 0.995 on a 1-4 scale, 12/30 videos
moving by a full point or more, one by 2.0, and the arm mean drifting 0.19 between
identical inputs. At that noise level an n=30 comparison can only resolve effects
larger than ~0.5, which is bigger than any difference between the arms under test,
so the measurement could not answer the question being asked of it.

Three changes target that, without touching the scoring rubric -- the system and
user prompts are imported from the original module rather than copied, so scores
stay comparable in kind:

* ``--repeats`` independent scorings per video, aggregated per field by median.
  The median of 3 is robust to a single outlying pass, which is the dominant
  failure mode observed.
* ``--temperature 0`` instead of the provider default.
* ``--frames 8`` instead of 4, so a pass sees more of the clip and depends less on
  which moment it happened to land on.

Every repeat is retained in the per-sample record along with the observed spread,
so the noise floor can be re-measured from any run rather than assumed.

The original script is left untouched; this one is additive.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Imported rather than duplicated: the rubric must stay byte-identical to the
# original judge, otherwise "same scale, less noise" would not be true.
from evaluate_aesthetic_reward_litellm import (  # noqa: E402
    DIMENSIONS,
    LiteLLMJudge,
    _build_aesthetic_evaluator_system_prompt,
    _build_aesthetic_evaluator_user_prompt,
    _extract_frames_b64,
    _extract_json_object,
    _generation_path,
    _read_eval_inputs,
)

_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median_field(passes: list[dict[str, Any]], field: str) -> float | None:
    values = [v for v in (_float_or_none(p.get(field)) for p in passes) if v is not None]
    return statistics.median(values) if values else None


def _median_dimension(passes: list[dict[str, Any]], dimension: str) -> float | None:
    values = [
        v
        for v in (_float_or_none((p.get("scores") or {}).get(dimension)) for p in passes)
        if v is not None
    ]
    return statistics.median(values) if values else None


def _spread(passes: list[dict[str, Any]]) -> float | None:
    """Max-minus-min of overall score across repeats: the per-video noise estimate."""
    values = [
        v
        for v in (_float_or_none(p.get("overall_aesthetic_score")) for p in passes)
        if v is not None
    ]
    return max(values) - min(values) if len(values) > 1 else (0.0 if values else None)


def _score_one(
    *,
    judge: LiteLLMJudge,
    row: dict[str, Any],
    method: str,
    generation_root: Path,
    frames: int,
    repeats: int,
    index: int,
    total: int,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or f"eval_{index - 1:03d}")
    instruction = str(row.get("instruction") or row.get("prompt") or "")
    try:
        video_path = _generation_path(generation_root, method, sample_id)
        # Decoded once and reused across repeats: the frames are deterministic, so
        # re-decoding per pass would only add wall-clock, and sharing them keeps the
        # repeats a measurement of judge variance rather than of frame sampling.
        images = _extract_frames_b64(video_path, frames=frames)
        system_prompt = _build_aesthetic_evaluator_system_prompt()
        user_prompt = _build_aesthetic_evaluator_user_prompt(instruction)

        passes: list[dict[str, Any]] = []
        errors: list[str] = []
        for repeat in range(repeats):
            try:
                raw_text = judge.complete(
                    system_prompt=system_prompt,
                    user_text=user_prompt,
                    images_b64=images,
                )
                payload = _extract_json_object(raw_text)
                passes.append(
                    {
                        "repeat": repeat + 1,
                        "scores": payload.get("scores") or {},
                        "overall_aesthetic_score": payload.get("overall_aesthetic_score"),
                        "uncertain": payload.get("uncertain"),
                        "reason": payload.get("reason"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - a dead pass must not void the video
                errors.append(f"repeat {repeat + 1}: {exc!r}")

        if not passes:
            raise RuntimeError("; ".join(errors) or "all repeats failed")

        result = {
            "sample_id": sample_id,
            "method": method,
            "video": str(video_path),
            "frames": len(images),
            "repeats_requested": repeats,
            "repeats_succeeded": len(passes),
            "scores": {d: _median_dimension(passes, d) for d in DIMENSIONS},
            "overall_aesthetic_score": _median_field(passes, "overall_aesthetic_score"),
            "overall_spread": _spread(passes),
            "passes": passes,
            "error": "",
            "partial_error": "; ".join(errors),
        }
    except Exception as exc:  # noqa: BLE001 - record and continue
        result = {
            "sample_id": sample_id,
            "method": method,
            "video": "",
            "frames": 0,
            "repeats_requested": repeats,
            "repeats_succeeded": 0,
            "scores": {},
            "overall_aesthetic_score": None,
            "overall_spread": None,
            "passes": [],
            "error": repr(exc),
            "partial_error": "",
        }
    _log(
        f"[stable_judge] {method} {index}/{total} {sample_id} "
        f"median={result['overall_aesthetic_score']} spread={result['overall_spread']} "
        f"error={bool(result['error'])}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score eval videos with a repeat-and-median hosted judge."
    )
    parser.add_argument("--eval-inputs", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LOW_HIGH_JUDGE_BASE_URL", "http://litellm-sgp.woa.com"),
    )
    parser.add_argument(
        "--model", default=os.environ.get("LOW_HIGH_JUDGE_MODEL", "gemini-3.1-pro-preview")
    )
    parser.add_argument("--api-key-env", default="LOW_HIGH_JUDGE_API_KEY")
    parser.add_argument("--reasoning-effort", default=os.environ.get("LOW_HIGH_JUDGE_REASONING_EFFORT", "high"))
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 pins the sampler; pass a negative value to omit the field entirely",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=3000)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--repeats", type=int, default=3, help="independent scorings per video, combined by median"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-sample-ids", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing judge API key in env var: {args.api_key_env}")

    rows = _read_eval_inputs(args.eval_inputs)
    if args.only_sample_ids:
        keep = {
            line.strip()
            for line in args.only_sample_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
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
        temperature=None if args.temperature < 0 else args.temperature,
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
                    repeats=args.repeats,
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

        scored = [r for r in results if not r["error"]]
        aggregates = {
            d: [v for v in (_float_or_none(r["scores"].get(d)) for r in scored) if v is not None]
            for d in DIMENSIONS
        }
        overall = [
            v for v in (_float_or_none(r["overall_aesthetic_score"]) for r in scored) if v is not None
        ]
        spreads = [v for v in (_float_or_none(r["overall_spread"]) for r in scored) if v is not None]

        summary[method] = {
            "n": total,
            "errors": len(results) - len(scored),
            "scored": len(overall),
            **{d: _mean(values) for d, values in aggregates.items()},
            "overall_aesthetic_score": _mean(overall),
            # The noise floor this run actually observed. Compare against the
            # single-pass sd of 0.995 that motivated this script.
            "mean_within_video_spread": _mean(spreads),
            "max_within_video_spread": max(spreads) if spreads else None,
            "videos_with_zero_spread": sum(1 for v in spreads if v == 0.0),
        }
        _log(f"[stable_judge] {method} done: {json.dumps(summary[method], ensure_ascii=False)}")

    summary["_meta"] = {
        "judge_model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": None if args.temperature < 0 else args.temperature,
        "frames": args.frames,
        "repeats": args.repeats,
        "aggregation": "per-field median across repeats",
        "n_samples": len(rows),
        "sample_ids": [str(row.get("sample_id")) for row in rows],
        "total_cost_usd": round(judge.total_cost, 6),
        "total_tokens": judge.total_tokens,
    }
    (args.output_dir / "summary_aesthetic_reward.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
