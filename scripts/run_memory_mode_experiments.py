#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _dataset import DataSample, get_sample_by_id, get_sample_by_index, load_samples, smoke_sample


VALID_MODES = {"none", "explicit", "implicit", "hybrid"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run closed-loop planner/edit/critic experiments across memory modes "
            "(none/explicit/implicit/hybrid) and save a reproducible run manifest."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modes", default="none,explicit,implicit,hybrid")
    parser.add_argument("--pair-dir", default="", help="Optional pair-dir root. When empty, use smoke samples.")
    parser.add_argument("--sample-id", action="append", default=[], help="Can be repeated.")
    parser.add_argument("--sample-index", action="append", type=int, default=[], help="Can be repeated.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap after resolving sample order.")
    parser.add_argument("--variant", default="wan14b+4steplora")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner-script", default="scripts/run_planner_and_edit.py")
    parser.add_argument(
        "--memory-strategy",
        choices=["per_mode", "shared", "disabled"],
        default="per_mode",
        help=(
            "How to provide memory bank path to runs. per_mode creates one isolated memory file per mode "
            "(copied from --memory-seed-path)."
        ),
    )
    parser.add_argument(
        "--memory-seed-path",
        default="out/memory/global_memory.json",
        help="Seed memory file used by per_mode/shared memory strategy.",
    )
    parser.add_argument(
        "--base-memory-config",
        default="configs/memory.json",
        help="Base memory config JSON used to generate per-mode runtime configs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip dependency preflight checks.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--summary-every", type=int, default=4)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _sanitize_segment(value: str) -> str:
    text = _clean_text(value).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return "".join(ch for ch in text if ch.isalnum() or ch in {"-", "_", "."}) or "unknown"


def _parse_modes(raw: str) -> list[str]:
    modes = [_clean_text(item).lower() for item in str(raw or "").split(",")]
    parsed = [mode for mode in modes if mode]
    if not parsed:
        parsed = ["none", "explicit", "implicit", "hybrid"]
    invalid = [mode for mode in parsed if mode not in VALID_MODES]
    if invalid:
        raise RuntimeError(f"Invalid modes: {invalid}. Valid={sorted(VALID_MODES)}")
    deduped: list[str] = []
    seen: set[str] = set()
    for mode in parsed:
        if mode not in seen:
            deduped.append(mode)
            seen.add(mode)
    return deduped


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _mode_runtime_override(mode: str) -> dict[str, Any]:
    return {
        "runtime": {
            "mode": mode,
            "critic_mode": "",
            "edit_mode": "",
        }
    }


def _build_mode_configs(*, output_root: Path, modes: list[str], base_memory_config: Path) -> dict[str, Path]:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    if base_memory_config.exists():
        base_payload = _load_json(base_memory_config)
    else:
        base_payload = {}

    paths: dict[str, Path] = {}
    for mode in modes:
        merged = _merge_dict(base_payload, _mode_runtime_override(mode))
        path = (config_dir / f"memory.{mode}.json").resolve()
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[mode] = path
    return paths


def _resolve_samples(args: argparse.Namespace) -> list[DataSample]:
    if str(args.pair_dir).strip():
        pair_dir = (PROJECT_ROOT / args.pair_dir).resolve()
        if not pair_dir.exists():
            raise FileNotFoundError(f"pair-dir not found: {pair_dir}")
        selected: list[DataSample] = []
        if args.sample_id:
            for sample_id in args.sample_id:
                selected.append(get_sample_by_id(pair_dir, sample_id))
        elif args.sample_index:
            for idx in args.sample_index:
                selected.append(get_sample_by_index(pair_dir, int(idx)))
        else:
            selected = load_samples(pair_dir)
        if args.max_samples > 0:
            selected = selected[: int(args.max_samples)]
        return selected

    # Smoke fallback
    selected: list[DataSample] = []
    if args.sample_id:
        for sample_id in args.sample_id:
            idx = int(sample_id) - 1 if str(sample_id).isdigit() else 0
            selected.append(smoke_sample(index=max(0, idx)))
    elif args.sample_index:
        for idx in args.sample_index:
            selected.append(smoke_sample(index=max(0, int(idx))))
    else:
        selected = [smoke_sample(index=0), smoke_sample(index=1)]
    if args.max_samples > 0:
        selected = selected[: int(args.max_samples)]

    # de-duplicate by sample_id while preserving order
    deduped: list[DataSample] = []
    seen: set[str] = set()
    for sample in selected:
        if sample.sample_id in seen:
            continue
        deduped.append(sample)
        seen.add(sample.sample_id)
    return deduped


def _prepare_memory_paths(
    *,
    output_root: Path,
    modes: list[str],
    memory_strategy: str,
    memory_seed_path: Path,
) -> dict[str, str]:
    if memory_strategy == "disabled":
        return {mode: "" for mode in modes}

    if memory_strategy == "shared":
        return {mode: str(memory_seed_path.resolve()) for mode in modes}

    # per_mode
    memory_dir = output_root / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    seed_exists = memory_seed_path.exists() and memory_seed_path.is_file()
    for mode in modes:
        mode_path = (memory_dir / f"memory.{mode}.json").resolve()
        if seed_exists:
            shutil.copy2(memory_seed_path, mode_path)
        paths[mode] = str(mode_path)
    return paths


def _parse_json_like(text: str) -> dict[str, Any]:
    payload = (text or "").strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    for line in reversed(lines):
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _preflight_dependencies() -> dict[str, bool]:
    required = ["torch", "qwen_vl_utils"]
    return {name: bool(importlib.util.find_spec(name)) for name in required}


def _has_attempt_outputs(run_dir: Path) -> bool:
    return any(path.is_dir() and path.name.startswith("attempt_") for path in run_dir.iterdir()) if run_dir.exists() else False


def _run_single(
    *,
    args: argparse.Namespace,
    mode: str,
    sample: DataSample,
    config_path: Path,
    memory_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        str(args.python),
        str((PROJECT_ROOT / args.runner_script).resolve()),
        "--variant",
        str(args.variant),
        "--max-attempts",
        str(int(args.max_attempts)),
        "--sample-id",
        str(sample.sample_id),
        "--output-dir",
        str(output_dir),
        "--memory-path",
        str(memory_path),
    ]
    if str(args.pair_dir).strip():
        command.extend(["--pair-dir", str(args.pair_dir)])

    env = dict(os.environ)
    env["LOW_HIGH_MEMORY_CONFIG"] = str(config_path)
    start_ts = _utc_now()
    start = time.perf_counter()
    if args.dry_run:
        end = time.perf_counter()
        return {
            "mode": mode,
            "sample_id": sample.sample_id,
            "pair_dir": str(args.pair_dir or ""),
            "status": "dry_run",
            "started_at": start_ts,
            "ended_at": _utc_now(),
            "duration_sec": end - start,
            "output_dir": str(output_dir.resolve()),
            "memory_config_path": str(config_path),
            "memory_path": str(memory_path),
            "command": command,
            "result": {},
            "return_code": 0,
            "stdout_path": "",
            "stderr_path": "",
        }

    completed = subprocess.run(command, capture_output=True, text=True, env=env)
    end = time.perf_counter()
    logs_dir = output_dir / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    result_payload = _parse_json_like(completed.stdout or "")

    status = str(result_payload.get("status", "") or "")
    if completed.returncode != 0:
        status = status or "runner_error"
    elif not status and _has_attempt_outputs(output_dir):
        status = "attempts_written"
    elif not status:
        status = "unknown"

    return {
        "mode": mode,
        "sample_id": sample.sample_id,
        "pair_dir": str(args.pair_dir or ""),
        "status": status,
        "started_at": start_ts,
        "ended_at": _utc_now(),
        "duration_sec": end - start,
        "output_dir": str(output_dir.resolve()),
        "memory_config_path": str(config_path),
        "memory_path": str(memory_path),
        "command": command,
        "result": result_payload,
        "return_code": int(completed.returncode),
        "stdout_path": str(stdout_path.resolve()),
        "stderr_path": str(stderr_path.resolve()),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(rows: list[dict[str, Any]], *, modes: list[str], dry_run: bool) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        items = [row for row in rows if row.get("mode") == mode]
        accepted = [row for row in items if str((row.get("result") or {}).get("status", "")).strip() == "accepted"]
        errored = [row for row in items if int(row.get("return_code", 0) or 0) != 0]
        by_mode[mode] = {
            "runs": len(items),
            "accepted_runs": len(accepted),
            "accepted_rate": (len(accepted) / len(items)) if items and not dry_run else None,
            "error_runs": len(errored),
            "avg_duration_sec": (sum(float(row.get("duration_sec", 0.0) or 0.0) for row in items) / len(items)) if items else None,
        }
    return {
        "dry_run": bool(dry_run),
        "total_runs": len(rows),
        "by_mode": by_mode,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    modes = _parse_modes(args.modes)
    preflight = _preflight_dependencies()
    if (not args.dry_run) and (not args.skip_preflight):
        missing = [name for name, ok in preflight.items() if not ok]
        if missing:
            raise RuntimeError(
                "Dependency preflight failed. Missing modules: "
                + ", ".join(missing)
                + ". Install them in your training env or run with --skip-preflight."
            )
    samples = _resolve_samples(args)
    if not samples:
        raise RuntimeError("No samples resolved for experiment run.")

    base_memory_config = (PROJECT_ROOT / args.base_memory_config).resolve()
    mode_configs = _build_mode_configs(
        output_root=output_root,
        modes=modes,
        base_memory_config=base_memory_config,
    )
    memory_seed_path = (PROJECT_ROOT / args.memory_seed_path).resolve()
    mode_memory_paths = _prepare_memory_paths(
        output_root=output_root,
        modes=modes,
        memory_strategy=str(args.memory_strategy),
        memory_seed_path=memory_seed_path,
    )

    rows: list[dict[str, Any]] = []
    run_index = 0
    total_runs = len(modes) * len(samples)

    for mode in modes:
        config_path = mode_configs[mode]
        memory_path = mode_memory_paths.get(mode, "")
        for sample in samples:
            run_index += 1
            sample_seg = _sanitize_segment(sample.sample_id)
            run_dir = output_root / "runs" / mode / sample_seg
            if args.skip_existing and _has_attempt_outputs(run_dir):
                row = {
                    "mode": mode,
                    "sample_id": sample.sample_id,
                    "pair_dir": str(args.pair_dir or ""),
                    "status": "skipped_existing",
                    "started_at": _utc_now(),
                    "ended_at": _utc_now(),
                    "duration_sec": 0.0,
                    "output_dir": str(run_dir.resolve()),
                    "memory_config_path": str(config_path),
                    "memory_path": str(memory_path),
                    "command": [],
                    "result": {},
                    "return_code": 0,
                    "stdout_path": "",
                    "stderr_path": "",
                }
                rows.append(row)
                continue

            row = _run_single(
                args=args,
                mode=mode,
                sample=sample,
                config_path=config_path,
                memory_path=memory_path,
                output_dir=run_dir,
            )
            rows.append(row)
            if int(row.get("return_code", 0) or 0) != 0 and not args.continue_on_error:
                summary = _summary(rows, modes=modes, dry_run=bool(args.dry_run))
                summary.update(
                    {
                        "output_root": str(output_root),
                        "modes": modes,
                        "samples": [sample.sample_id for sample in samples],
                        "failed_at_run": run_index,
                    }
                )
                runs_jsonl = output_root / "experiment.runs.jsonl"
                _write_jsonl(runs_jsonl, rows)
                (output_root / "experiment.summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"Run failed at mode={mode} sample_id={sample.sample_id}. "
                    f"See logs: {row.get('stdout_path')} {row.get('stderr_path')}"
                )

            if args.summary_every > 0 and (run_index % int(args.summary_every) == 0 or run_index == total_runs):
                interim = _summary(rows, modes=modes, dry_run=bool(args.dry_run))
                interim.update(
                    {
                        "output_root": str(output_root),
                        "modes": modes,
                        "samples": [sample.sample_id for sample in samples],
                        "completed_runs": run_index,
                        "total_planned_runs": total_runs,
                    }
                )
                (output_root / "experiment.summary.json").write_text(
                    json.dumps(interim, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    final_summary = _summary(rows, modes=modes, dry_run=bool(args.dry_run))
    final_summary.update(
        {
            "output_root": str(output_root),
            "modes": modes,
            "samples": [sample.sample_id for sample in samples],
            "memory_strategy": str(args.memory_strategy),
            "mode_memory_paths": mode_memory_paths,
            "mode_config_paths": {key: str(path) for key, path in mode_configs.items()},
            "runner_script": str((PROJECT_ROOT / args.runner_script).resolve()),
            "preflight": preflight,
            "created_at": _utc_now(),
        }
    )

    runs_jsonl = output_root / "experiment.runs.jsonl"
    _write_jsonl(runs_jsonl, rows)
    summary_path = output_root / "experiment.summary.json"
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
