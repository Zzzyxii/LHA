#!/usr/bin/env python3
"""Time one planner run on one sample; used to size planning-track wall-clock."""
import json
import sys
import time
from pathlib import Path

from planner.core.video_planner import plan_video
from vlm import LiteLLMVisionModel

out = Path(sys.argv[1])
sid = sys.argv[2] if len(sys.argv) > 2 else "heldout1000_0133"
intents = {
    json.loads(l)["sample_id"]: json.loads(l)
    for l in open("scripts/eval/bench_v3/data/intents_specific.jsonl", encoding="utf-8")
    if l.strip()
}
out.mkdir(parents=True, exist_ok=True)
t = time.time()
plan = plan_video(
    sample_id=sid,
    prompt=intents[sid]["intent"],
    low_video=f"reports/top50_video_bundle_20260725_nodup/videos/source_low/{sid}.mp4",
    memory_context={},
    memory_store=None,
    attempt_adaptation_state={},
    state_path=out / "planner.state.json",
    variant="wan14b+ours",
)
print(f"PLANNER_DONE {time.time() - t:.0f}s {LiteLLMVisionModel.usage_summary()}", flush=True)
print(plan.model_dump().get("compiled_edit_prompt", "")[:800], flush=True)
