#!/usr/bin/env python3
"""Text-only check: does a planner's plan cover the edits that produced the target?

For each low-high-cost sample, the reference is the expert instruction (written
from the LOW and HIGH frames) plus its ``transformation_tags``. An LLM labels each
tag against a candidate text as:

  match    -- the candidate asks for this change with compatible specifics
  partial  -- same object/dimension, different specifics (e.g. pink robe vs green)
  missing  -- the candidate does not ask for this change

and lists the candidate's extra changes, flagging those that contradict the
reference (it says keep it, or asks for something incompatible).

This separates "picked the wrong dimensions" (a planner problem) from "picked the
right dimension but guessed the wrong detail" (the intent never contained it)
without rendering any video.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/eval/ditto_quant"))

from evaluate_aesthetic_reward_litellm import LiteLLMJudge  # noqa: E402

SYSTEM_PROMPT = """You audit a planned video edit against the reference edit that produced the target video.
The reference instruction was written by looking at the source and target frames, so it describes what actually changed.
Judge only the text. Be literal and strict. Return strict JSON only."""

USER_TEMPLATE = """REFERENCE INSTRUCTION (what actually changed from source to target):
{reference}

REFERENCE CHANGE TAGS:
{tags}

CANDIDATE TEXT (what the plan asks the edit model to do):
{candidate}

For every reference tag, label the candidate:
- "match": the candidate asks for this change and its specifics are compatible with the reference (same object, compatible colour/material/style).
- "partial": the candidate changes the same object or dimension, but its specifics differ from the reference (different colour, garment type, material, setting, mood).
- "missing": the candidate does not ask for this change at all.

Then list concrete EXTRA changes the candidate asks for that the reference does not mention. For each, set "contradicts": true if the reference says to keep that thing unchanged, or asks for something incompatible with it.

Return JSON:
{{"tags": [{{"tag": "...", "label": "match|partial|missing", "note": "short"}}],
  "extra": [{{"change": "short", "contradicts": true}}]}}"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _plan_text(prompt_path: Path) -> str:
    """The plan section of a composed agent prompt (intent + plan, no skill blocks)."""
    text = prompt_path.read_text(encoding="utf-8")
    match = re.search(r"Detailed edit plan:\n(.*?)(?:\n\n(?:Craft direction|Preserve unchanged):|\Z)", text, re.S)
    intent = text.split("\n\n", 1)[0]
    return intent + ("\n" + match.group(1).strip() if match else "")


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in judge output")
    return json.loads(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=ROOT / "scripts/eval/bench_v3/data/top50_sources.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out/bench_v2/plan_coverage")
    parser.add_argument("--model", default=os.environ.get("LOW_HIGH_JUDGE_MODEL", "gemini-3.8-flash"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("LOW_HIGH_JUDGE_REASONING_EFFORT", "high"))
    parser.add_argument("--base-url", default=os.environ.get("LOW_HIGH_JUDGE_BASE_URL", "http://litellm-sgp.woa.com"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    api_key = os.environ.get("LOW_HIGH_JUDGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("set LOW_HIGH_JUDGE_API_KEY")
    judge = LiteLLMJudge(base_url=args.base_url, api_key=api_key, model=args.model,
                         reasoning_effort=args.reasoning_effort, temperature=0.0, max_completion_tokens=8000)

    expert = {r["sample_id"]: r["instruction"] for r in json.load(open(ROOT / "reports/top50_video_bundle_20260725_nodup/sample_index.json"))}
    intents = {
        "intent": {r["sample_id"]: r["intent"] for r in _jsonl(ROOT / "scripts/eval/bench_v3/data/intents_specific.jsonl")},
        "generic": {r["sample_id"]: r["intent"] for r in _jsonl(ROOT / "scripts/eval/bench_v3/data/intents_generic.jsonl")},
    }
    planning = ROOT / "out/bench_v2/planning"
    samples = [r for r in _jsonl(args.sources) if r["source_dataset"] == "low_high_cost"]

    # candidate name -> (sample_id -> text)
    candidates: dict[str, dict[str, str]] = {
        "intent_text": intents["intent"],
        "intent_plan": {},
        "generic_plan": {},
    }
    for r in samples:
        sid = r["sample_id"]
        for form in ("intent", "generic"):
            prompt = planning / f"ours_{form}_agent/loop/{sid}/attempt_001/edit.prompt.txt"
            candidates[f"{form}_plan"][sid] = _plan_text(prompt)

    jobs = [(name, r) for name in candidates for r in samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def run(job: tuple[str, dict]) -> dict:
        name, r = job
        sid = r["sample_id"]
        user = USER_TEMPLATE.format(reference=expert[sid], tags=json.dumps(r["transformation_tags"]),
                                    candidate=candidates[name][sid])
        out = {"candidate": name, "sample_id": sid}
        try:
            out.update(_parse(judge.complete(system_prompt=SYSTEM_PROMPT, user_text=user, images_b64=[])))
        except Exception as exc:  # noqa: BLE001 - keep going; the row records the failure
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(run, jobs))

    path = args.output_dir / "coverage.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    print(f"{path}  errors={sum('error' in r for r in results)}  cost=${judge.total_cost:.2f}")
    for name in candidates:
        rows = [r for r in results if r["candidate"] == name and "error" not in r]
        labels = [t.get("label") for r in rows for t in r.get("tags", [])]
        n = max(1, len(labels))
        extra = [e for r in rows for e in r.get("extra", [])]
        print(f"{name:13s} n={len(rows)} tags: match={labels.count('match') / n:.0%} "
              f"partial={labels.count('partial') / n:.0%} missing={labels.count('missing') / n:.0%} | "
              f"extra/sample={len(extra) / max(1, len(rows)):.1f} contradicting/sample="
              f"{sum(bool(e.get('contradicts')) for e in extra) / max(1, len(rows)):.1f}")


if __name__ == "__main__":
    main()
