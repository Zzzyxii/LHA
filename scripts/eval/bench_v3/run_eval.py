#!/usr/bin/env python3
"""Run the bench_v3 evaluation plan (eval_config.json): GT metrics, checklist judge, pairwise judge.

    python3 scripts/eval/bench_v3/run_eval.py --tiers 'track_a.*' [--skip-pairwise]

Per tier, arms whose generation dir is missing or incomplete (does not cover
every task sample) are skipped with a warning, so a tier can be evaluated while
some of its arms are still rendering and topped up later: every stage resumes.
Outputs land in out/bench_v3/eval/<tier>.{gt,checklist,pairwise}/.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "scripts/eval/bench_v3"
EVAL = ROOT / "out/bench_v3/eval"
ANCHORS = {"noop", "oracle"}  # gt_metrics adds these itself


def load_config() -> dict:
    cfg = json.loads((HERE / "eval_config.json").read_text(encoding="utf-8"))
    dirs = cfg["dirs"]
    for tier in cfg["tiers"].values():
        tier["arms"] = {k: v.format(**dirs) for k, v in tier["arms"].items()}
    return cfg


def task_ids(tier: str) -> list[str]:
    path = ROOT / "out/bench_v3/judge_tasks" / f"{tier}.jsonl"
    return [json.loads(line)["sample_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ready_arms(tier: str, arms: dict[str, str]) -> dict[str, Path]:
    ids = task_ids(tier)
    ready = {}
    for name, rel in arms.items():
        folder = (ROOT / rel) if not rel.startswith("/") else Path(rel)
        missing = [s for s in ids if not (folder / f"{s}.mp4").exists()]
        if missing:
            print(f"  [skip] {tier}:{name} missing {len(missing)}/{len(ids)} videos", flush=True)
            continue
        ready[name] = folder
    return ready


def run(cmd: list[str], log: Path, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        code = subprocess.call(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if code:
        raise SystemExit(f"failed ({code}): {' '.join(cmd[:3])} ... see {log}")


def judge_dir(tier: str, stage: str, model: str, primary: str) -> Path:
    """<tier>.<stage> for the primary judge, <tier>.<stage>@<model> for the others."""
    return EVAL / (f"{tier}.{stage}" if model == primary else f"{tier}.{stage}@{model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default="*", help="fnmatch pattern over tier names")
    parser.add_argument("--skip-gt", action="store_true")
    parser.add_argument("--skip-checklist", action="store_true")
    parser.add_argument("--skip-pairwise", action="store_true")
    parser.add_argument("--judges", default="", help="comma-separated override of eval_config judges")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()
    cfg = load_config()
    judges = [j for j in args.judges.split(",") if j] or cfg["judges"]
    primary = cfg["judges"][0]
    py = sys.executable

    for tier, spec in cfg["tiers"].items():
        if not fnmatch.fnmatch(tier, args.tiers):
            continue
        print(f"== {tier}", flush=True)
        arms = ready_arms(tier, spec["arms"])
        tasks = ROOT / "out/bench_v3/judge_tasks" / f"{tier}.jsonl"
        logs = EVAL / "logs"
        if spec.get("gt") and not args.skip_gt:
            cmd = [py, str(ROOT / "scripts/eval/bench_v3/gt_metrics.py"), "--manifest", str(ROOT / "out/bench_v2/manifest.jsonl"),
                   "--hf-cache", str(ROOT / "models/hf_cache/hub"), "--output-dir", str(EVAL / f"{tier}.gt")]
            for k, v in arms.items():
                if k not in ANCHORS:
                    cmd += ["--arm", f"{k}={v}"]
            run(cmd, logs / f"{tier}.gt.log", env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu))
        for model in judges:
            if not args.skip_checklist:
                cmd = [py, str(HERE / "run_checklist.py"), "--tasks", str(tasks), "--model", model,
                       "--output-dir", str(judge_dir(tier, "checklist", model, primary)), "--concurrency", str(args.concurrency)]
                for k, v in arms.items():
                    cmd += ["--arm", f"{k}={v}"]
                run(cmd, logs / f"{tier}.checklist@{model}.log")
            if not args.skip_pairwise:
                for a, b in spec.get("pairwise", []):
                    if a in arms and b in arms:
                        run([py, str(HERE / "run_pairwise.py"), "--tasks", str(tasks), "--model", model,
                             "--arm-a", f"{a}={arms[a]}", "--arm-b", f"{b}={arms[b]}",
                             "--output-dir", str(judge_dir(tier, "pairwise", model, primary)),
                             "--concurrency", str(args.concurrency)], logs / f"{tier}.pairwise@{model}.log")
        print(f"   done {tier}: {sorted(arms)}", flush=True)


if __name__ == "__main__":
    main()
