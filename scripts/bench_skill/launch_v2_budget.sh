#!/usr/bin/env bash
# Re-run execution-track skill v2 with the 512-token prompt budget fix.
# The original v2 run (out/bench_skill/run30_v2) had 16/30 prompts truncated by
# Wan's text encoder; this repeats it with the same planner and critic model
# (gemini-3.1-pro-preview, no reasoning_effort) so the only change is the budget.
#
#   WAIT_PID  optional: wait for this process (e.g. queue_e) to exit first
# Requires LOW_HIGH_JUDGE_API_KEY in the environment.
set -eu
cd "$(dirname "$0")/../.."
source scripts/bench_skill/env.sh
: "${LOW_HIGH_JUDGE_API_KEY:?export LOW_HIGH_JUDGE_API_KEY first}"
export LOW_HIGH_AGENT_LITELLM_MODEL=gemini-3.1-pro-preview
unset LOW_HIGH_AGENT_LITELLM_REASONING_EFFORT
unset LOW_HIGH_AGENT_LITELLM_PLANNER_MODEL LOW_HIGH_AGENT_LITELLM_CRITIC_MODEL
export LOW_HIGH_AGENT_PROMPT_MODE=instruction_first

if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

IDX=reports/top50_video_bundle_20260725_nodup/sample_index.json
SRC=reports/top50_video_bundle_20260725_nodup/videos/source_low
OUT=out/bench_skill/run30_v2_budget
IDS=scripts/eval/bench_v3/data/release30_ids.txt
mkdir -p "$OUT/logs"
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/bench_skill/run_bench.py \
    --arm ours_skill --sample-index "$IDX" --source-dir "$SRC" --output-root "$OUT" \
    --only-sample-ids "$IDS" --shard-index "$i" --shard-count 8 --max-attempts 1 \
    >> "$OUT/logs/ours_skill.shard$i.log" 2>&1 &
done
wait
echo "v2_budget done"
