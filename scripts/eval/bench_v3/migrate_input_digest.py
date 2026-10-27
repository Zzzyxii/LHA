#!/usr/bin/env python3
"""One-off migration: stamp input_digest on judge rows written before it existed.

A row is stamped only if it provably came from the current inputs: same video
path as the current job and, for checklist rows, the same item texts as the
current task (the judge stores the items it was given). Rows that fail the check
are left unstamped, so run_checklist / run_pairwise judge them again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import judges  # noqa: E402
import run_checklist  # noqa: E402
import run_eval  # noqa: E402
import run_pairwise  # noqa: E402


def same_file(a: str | None, b: str | None) -> bool:
    """Rows from older runs store relative paths, current jobs absolute ones."""
    if not a or not b:
        return False
    return Path(a).resolve() == Path(b).resolve()


def migrate(path: Path, jobs: list[dict], key: tuple[str, ...], tasks: dict, checklist: bool) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    by_key = {tuple(j[k] for k in key): j for j in jobs}
    rows = judges.read_jsonl(path)
    stamped = rejected = 0
    for row in rows:
        if "input_digest" in row or row.get("error"):
            continue
        job = by_key.get(tuple(row.get(k) for k in key))
        ok = job is not None and all(same_file(row.get(f), job.get(f)) for f in ("video", "video_a", "video_b") if f in job)
        if ok and checklist:
            task = tasks[row["sample_id"]]
            ok = [c.get("item") for c in row.get("change", [])] == list(task.get("checklist") or []) and \
                 [p.get("item") for p in row.get("preserve", [])] == list(task.get("preserve") or [])
        if ok:
            row["input_digest"] = job["input_digest"]
            stamped += 1
        else:
            rejected += 1
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return stamped, rejected


def main() -> None:
    cfg = run_eval.load_config()
    primary = cfg["judges"][0]
    for tier, spec in cfg["tiers"].items():
        task_path = run_eval.ROOT / "out/bench_v3/judge_tasks" / f"{tier}.jsonl"
        tasks = {t["sample_id"]: t for t in judges.read_jsonl(task_path)}
        arms = run_eval.ready_arms(tier, spec["arms"])
        for model in cfg["judges"]:
            jobs, _ = run_checklist.build_jobs(list(tasks.values()), list(arms.items()))
            path = run_eval.judge_dir(tier, "checklist", model, primary) / "checklist.jsonl"
            s, r = migrate(path, jobs, ("arm", "sample_id"), tasks, True)
            if s or r:
                print(f"{tier} checklist@{model}: stamped {s}, left for re-judging {r}")
            for a, b in spec.get("pairwise", []):
                if a in arms and b in arms:
                    jobs, _ = run_pairwise.build_jobs(list(tasks.values()), arms[a], arms[b])
                    path = run_eval.judge_dir(tier, "pairwise", model, primary) / f"{a}__vs__{b}.jsonl"
                    s, r = migrate(path, jobs, ("sample_id",), tasks, False)
                    if s or r:
                        print(f"{tier} {a}__vs__{b}@{model}: stamped {s}, left for re-judging {r}")


if __name__ == "__main__":
    main()
