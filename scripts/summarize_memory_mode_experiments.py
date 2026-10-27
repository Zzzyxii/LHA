#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize memory mode experiment runs with accepted_rate / attempts_to_accept / failure tag rates "
            "and local-vs-global routing buckets, then emit threshold tuning suggestions."
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Default: <experiment-root>/experiment.runs.jsonl")
    parser.add_argument("--baseline-mode", default="explicit")
    parser.add_argument(
        "--tuning-mode",
        default="hybrid",
        help="Mode used for threshold suggestions. Falls back to baseline when absent.",
    )
    parser.add_argument("--memory-config", default="configs/memory.json")
    parser.add_argument("--output", type=Path, default=None, help="Default: <experiment-root>/analysis.summary.json")
    parser.add_argument(
        "--recommend-output",
        type=Path,
        default=None,
        help="Default: <experiment-root>/threshold.recommendations.json",
    )
    return parser.parse_args()


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: object) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _safe_list(value: object) -> list:
    return list(value) if isinstance(value, list) else []


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = max(0.0, min(1.0, float(q)))
    rank = q * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _load_attempt_payload(attempt_dir: Path) -> tuple[dict, dict]:
    contract_path = attempt_dir / "memory.contract.json"
    if contract_path.exists():
        contract = _read_json(contract_path)
        critic = _safe_dict(contract.get("critic"))
        evaluation = _safe_dict(critic.get("evaluation"))
        reflection = _safe_dict(critic.get("reflection"))
        if evaluation and reflection:
            return evaluation, reflection
    eval_path = attempt_dir / "evaluation.response.json"
    reflection_path = attempt_dir / "reflection.response.json"
    if eval_path.exists() and reflection_path.exists():
        return _read_json(eval_path), _read_json(reflection_path)
    return {}, {}


def _discover_attempt_dirs(run_dir: Path) -> list[Path]:
    attempts = [item for item in run_dir.iterdir() if item.is_dir() and item.name.startswith("attempt_")] if run_dir.exists() else []
    attempts.sort(key=lambda path: path.name)
    return attempts


def _normalize_run_rows(experiment_root: Path, manifest_rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in manifest_rows:
        mode = _clean_text(row.get("mode", "")).lower()
        output_dir_raw = _clean_text(row.get("output_dir", ""))
        if not mode or not output_dir_raw:
            continue
        output_dir = Path(output_dir_raw).expanduser()
        if not output_dir.exists():
            continue
        result_payload = _safe_dict(row.get("result"))
        status = _clean_text(result_payload.get("status", "") or row.get("status", ""))
        normalized.append(
            {
                "mode": mode,
                "sample_id": _clean_text(row.get("sample_id", "")),
                "run_dir": output_dir.resolve(),
                "status": status,
                "attempt_index": _safe_int(result_payload.get("attempt_index", 0), 0),
            }
        )
    if normalized:
        return normalized

    # Fallback discovery: <experiment-root>/runs/<mode>/<sample>/attempt_*
    fallback: list[dict] = []
    runs_root = experiment_root / "runs"
    if not runs_root.exists():
        return fallback
    for mode_dir in sorted(item for item in runs_root.iterdir() if item.is_dir()):
        mode = _clean_text(mode_dir.name).lower()
        for run_dir in sorted(item for item in mode_dir.iterdir() if item.is_dir()):
            attempts = _discover_attempt_dirs(run_dir)
            if not attempts:
                continue
            fallback.append(
                {
                    "mode": mode,
                    "sample_id": _clean_text(run_dir.name),
                    "run_dir": run_dir.resolve(),
                    "status": "",
                    "attempt_index": 0,
                }
            )
    return fallback


def _aggregate_mode(mode: str, run_rows: list[dict]) -> tuple[dict, list[dict]]:
    per_run: list[dict] = []
    attempt_records: list[dict] = []
    for row in run_rows:
        run_dir = Path(row["run_dir"])
        attempt_dirs = _discover_attempt_dirs(run_dir)
        if not attempt_dirs:
            continue

        attempts_payload: list[dict] = []
        for attempt_dir in attempt_dirs:
            evaluation_payload, reflection_payload = _load_attempt_payload(attempt_dir)
            if not evaluation_payload and not reflection_payload:
                continue
            scores = _safe_dict(evaluation_payload.get("scores"))
            overall = _safe_float(scores.get("overall_score", 0.0), 0.0)
            quality = _safe_float(scores.get("quality_score", 0.0), 0.0)
            grounding = _safe_float(scores.get("carrier_grounding", 0.0), 0.0)
            failure_tags = [str(item).strip() for item in _safe_list(evaluation_payload.get("failure_tags")) if str(item).strip()]
            action = _clean_text(reflection_payload.get("action", ""))
            replan_scope = _clean_text(reflection_payload.get("replan_scope", ""))
            attempt_item = {
                "sample_id": row["sample_id"],
                "attempt_dir": str(attempt_dir.resolve()),
                "attempt_index": _safe_int(attempt_dir.name.split("_")[-1], 0),
                "overall_score": overall,
                "quality_score": quality,
                "carrier_grounding": grounding,
                "failure_tags": failure_tags,
                "action": action,
                "replan_scope": replan_scope,
                "accepted": action == "accept",
            }
            attempts_payload.append(attempt_item)
            attempt_records.append(attempt_item)

        if not attempts_payload:
            continue
        final_attempt = attempts_payload[-1]
        accepted = any(item["accepted"] for item in attempts_payload)
        accept_attempt_index = 0
        for item in attempts_payload:
            if item["accepted"]:
                accept_attempt_index = int(item["attempt_index"])
                break
        per_run.append(
            {
                "sample_id": row["sample_id"],
                "run_dir": str(run_dir),
                "attempt_count": len(attempts_payload),
                "accepted": accepted,
                "accept_attempt_index": accept_attempt_index,
                "final_action": final_attempt["action"],
                "final_overall_score": final_attempt["overall_score"],
                "final_failure_tags": final_attempt["failure_tags"],
            }
        )

    run_count = len(per_run)
    accepted_runs = [item for item in per_run if item["accepted"]]
    final_scores = [float(item["final_overall_score"]) for item in per_run]
    attempts_per_run = [float(item["attempt_count"]) for item in per_run]
    attempts_to_accept = [float(item["accept_attempt_index"]) for item in accepted_runs if item["accept_attempt_index"] > 0]

    final_failure_counter: Counter[str] = Counter()
    for item in per_run:
        final_failure_counter.update(item["final_failure_tags"])

    local_attempts = [item for item in attempt_records if item["action"] == "local_refine_prompt"]
    global_attempts = [item for item in attempt_records if item["action"] == "global_replan"]
    total_attempts = len(attempt_records)

    def _bucket_payload(items: list[dict]) -> dict:
        counter: Counter[str] = Counter()
        for item in items:
            counter.update(item.get("failure_tags") or [])
        return {
            "count": len(items),
            "share": (len(items) / total_attempts) if total_attempts else None,
            "mean_overall_score": _mean([_safe_float(item.get("overall_score", 0.0), 0.0) for item in items]),
            "top_failure_tags": [{"tag": key, "count": value} for key, value in counter.most_common(8)],
        }

    mode_summary = {
        "mode": mode,
        "run_count": run_count,
        "accepted_count": len(accepted_runs),
        "accepted_rate": (len(accepted_runs) / run_count) if run_count else None,
        "avg_attempts": _mean(attempts_per_run),
        "avg_attempts_to_accept": _mean(attempts_to_accept),
        "avg_final_overall_score": _mean(final_scores),
        "median_final_overall_score": median(final_scores) if final_scores else None,
        "final_failure_tag_rate": {
            key: (value / run_count) if run_count else None
            for key, value in final_failure_counter.items()
        },
        "local_bucket": _bucket_payload(local_attempts),
        "global_bucket": _bucket_payload(global_attempts),
    }
    return mode_summary, attempt_records


def _comparison_rows(mode_summaries: dict[str, dict], baseline_mode: str) -> list[dict]:
    baseline = mode_summaries.get(baseline_mode)
    if not baseline:
        return []
    baseline_acc = baseline.get("accepted_rate")
    baseline_attempts = baseline.get("avg_attempts_to_accept")
    baseline_score = baseline.get("avg_final_overall_score")
    baseline_global = _safe_dict(baseline.get("global_bucket")).get("share")
    rows: list[dict] = []
    for mode, summary in mode_summaries.items():
        if mode == baseline_mode:
            continue
        rows.append(
            {
                "mode": mode,
                "vs_baseline": baseline_mode,
                "delta_accepted_rate_pp": (
                    (summary.get("accepted_rate") - baseline_acc) * 100.0
                    if summary.get("accepted_rate") is not None and baseline_acc is not None
                    else None
                ),
                "delta_avg_attempts_to_accept": (
                    summary.get("avg_attempts_to_accept") - baseline_attempts
                    if summary.get("avg_attempts_to_accept") is not None and baseline_attempts is not None
                    else None
                ),
                "delta_avg_final_overall_score": (
                    summary.get("avg_final_overall_score") - baseline_score
                    if summary.get("avg_final_overall_score") is not None and baseline_score is not None
                    else None
                ),
                "delta_global_share_pp": (
                    (float(_safe_dict(summary.get("global_bucket")).get("share")) - float(baseline_global)) * 100.0
                    if _safe_dict(summary.get("global_bucket")).get("share") is not None and baseline_global is not None
                    else None
                ),
            }
        )
    return rows


def _load_memory_config(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _suggest_thresholds(
    *,
    attempts_by_mode: dict[str, list[dict]],
    preferred_mode: str,
    fallback_mode: str,
    memory_config: dict,
) -> dict:
    mode = preferred_mode if preferred_mode in attempts_by_mode and attempts_by_mode[preferred_mode] else fallback_mode
    attempts = attempts_by_mode.get(mode, [])
    accepted = [item for item in attempts if item.get("action") == "accept"]
    local = [item for item in attempts if item.get("action") == "local_refine_prompt"]
    global_attempts = [item for item in attempts if item.get("action") == "global_replan"]
    non_accept = [item for item in attempts if item.get("action") != "accept"]

    accept_overall = [_safe_float(item.get("overall_score", 0.0), 0.0) for item in accepted]
    local_overall = [_safe_float(item.get("overall_score", 0.0), 0.0) for item in local]
    global_overall = [_safe_float(item.get("overall_score", 0.0), 0.0) for item in global_attempts]
    non_accept_overall = [_safe_float(item.get("overall_score", 0.0), 0.0) for item in non_accept]
    accept_quality = [_safe_float(item.get("quality_score", 0.0), 0.0) for item in accepted if item.get("quality_score") is not None]
    accept_grounding = [_safe_float(item.get("carrier_grounding", 0.0), 0.0) for item in accepted if item.get("carrier_grounding") is not None]

    current_hidden = _safe_float(
        _safe_dict(_safe_dict(_safe_dict(memory_config.get("implicit")).get("edit")).get("hidden_state")).get(
            "positive_score_threshold", 0.62
        ),
        0.62,
    )
    current_promotion = _safe_dict(_safe_dict(memory_config.get("promotion")).get("promotable"))
    current_min_overall = _safe_float(current_promotion.get("min_overall_score", 0.72), 0.72)
    current_min_quality = _safe_float(current_promotion.get("min_quality_score", 0.62), 0.62)
    current_min_grounding = _safe_float(current_promotion.get("min_carrier_grounding", 0.5), 0.5)

    def _clamp01(value: float | None, default: float) -> float:
        if value is None:
            return default
        return max(0.0, min(1.0, float(value)))

    hidden_suggest = _clamp01(
        (
            (_percentile(accept_overall, 0.5) + _percentile(non_accept_overall, 0.5)) / 2.0
            if accept_overall and non_accept_overall
            else _percentile(accept_overall, 0.25)
        ),
        current_hidden,
    )
    promotion_overall_suggest = _clamp01(_percentile(accept_overall, 0.25), current_min_overall)
    promotion_quality_suggest = _clamp01(_percentile(accept_quality, 0.25), current_min_quality)
    promotion_grounding_suggest = _clamp01(_percentile(accept_grounding, 0.2), current_min_grounding)

    accept_threshold_suggest = _clamp01(
        (
            (_percentile(accept_overall, 0.25) + _percentile(local_overall, 0.75)) / 2.0
            if accept_overall and local_overall
            else _percentile(accept_overall, 0.3)
        ),
        0.78,
    )
    local_floor_suggest = _clamp01(
        (
            (_percentile(local_overall, 0.25) + _percentile(global_overall, 0.75)) / 2.0
            if local_overall and global_overall
            else _percentile(local_overall, 0.25)
        ),
        0.52,
    )

    return {
        "mode_used_for_tuning": mode,
        "sample_sizes": {
            "attempts": len(attempts),
            "accept_attempts": len(accepted),
            "local_attempts": len(local),
            "global_attempts": len(global_attempts),
        },
        "recommendations": {
            "configs_memory_json": {
                "implicit.edit.hidden_state.positive_score_threshold": {
                    "current": current_hidden,
                    "suggested": hidden_suggest,
                },
                "promotion.promotable.min_overall_score": {
                    "current": current_min_overall,
                    "suggested": promotion_overall_suggest,
                },
                "promotion.promotable.min_quality_score": {
                    "current": current_min_quality,
                    "suggested": promotion_quality_suggest,
                },
                "promotion.promotable.min_carrier_grounding": {
                    "current": current_min_grounding,
                    "suggested": promotion_grounding_suggest,
                },
            },
            "critic_runtime_thresholds": {
                "reflector.accept_threshold": {
                    "current": 0.78,
                    "suggested": accept_threshold_suggest,
                },
                "reflector.local_refine_floor": {
                    "current": 0.52,
                    "suggested": local_floor_suggest,
                },
            },
        },
    }


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else (experiment_root / "experiment.runs.jsonl")
    output_path = args.output.expanduser().resolve() if args.output else (experiment_root / "analysis.summary.json")
    recommend_path = (
        args.recommend_output.expanduser().resolve()
        if args.recommend_output
        else (experiment_root / "threshold.recommendations.json")
    )

    manifest_rows = _read_jsonl(manifest_path) if manifest_path.exists() else []
    run_rows = _normalize_run_rows(experiment_root, manifest_rows)
    if not run_rows:
        raise RuntimeError(
            f"No runs found. Checked manifest={manifest_path if manifest_path.exists() else 'missing'} "
            f"and fallback under {experiment_root / 'runs'}"
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in run_rows:
        grouped[_clean_text(row.get("mode", "")).lower()].append(row)

    mode_summaries: dict[str, dict] = {}
    attempts_by_mode: dict[str, list[dict]] = {}
    for mode, rows in sorted(grouped.items()):
        summary, attempt_records = _aggregate_mode(mode, rows)
        mode_summaries[mode] = summary
        attempts_by_mode[mode] = attempt_records

    comparison_rows = _comparison_rows(mode_summaries, _clean_text(args.baseline_mode).lower())
    memory_config_arg = str(args.memory_config or "").strip()
    if memory_config_arg.startswith("configs/"):
        config_path = (PROJECT_ROOT / memory_config_arg).resolve()
    else:
        config_path = Path(memory_config_arg).expanduser().resolve()
    memory_config = _load_memory_config(config_path)
    threshold_recommendations = _suggest_thresholds(
        attempts_by_mode=attempts_by_mode,
        preferred_mode=_clean_text(args.tuning_mode).lower(),
        fallback_mode=_clean_text(args.baseline_mode).lower(),
        memory_config=memory_config,
    )

    summary = {
        "experiment_root": str(experiment_root),
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "run_count": len(run_rows),
        "mode_count": len(mode_summaries),
        "baseline_mode": _clean_text(args.baseline_mode).lower(),
        "mode_summaries": mode_summaries,
        "comparisons_vs_baseline": comparison_rows,
        "threshold_recommendations_path": str(recommend_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    recommend_path.parent.mkdir(parents=True, exist_ok=True)
    recommend_path.write_text(json.dumps(threshold_recommendations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
