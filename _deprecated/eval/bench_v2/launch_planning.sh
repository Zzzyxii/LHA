#!/usr/bin/env bash
# bench_v2 planning track: one-line intent in, edited video out.
# Four arms on 50 samples across 8 GPUs (layout sized below):
#   *_direct : intent fed straight to the edit model (no agent)
#   *_agent  : intent expanded by the planner, which sees the low video only
# The upper bound (expert instruction) is the existing ours_raw run.
set -u

cd "$(dirname "$0")/../../.."
source scripts/bench_skill/env.sh

IDX=reports/top50_video_bundle_20260725_nodup/sample_index.json
SRC=reports/top50_video_bundle_20260725_nodup/videos/source_low
OUT=out/bench_v2/planning
mkdir -p "$OUT/logs"

launch () { # arm intents_file prompt_mode shard_index shard_count gpu
  local arm=$1 intents=$2 mode=$3 si=$4 sc=$5 gpu=$6
  CUDA_VISIBLE_DEVICES=$gpu LOW_HIGH_AGENT_PROMPT_MODE=$mode nohup python3 scripts/bench_skill/run_bench.py \
    --arm "$arm" --sample-index "$IDX" --source-dir "$SRC" --output-root "$OUT" \
    --instruction-file "$intents" --max-attempts 1 \
    --shard-index "$si" --shard-count "$sc" \
    >> "$OUT/logs/$arm.shard$si.log" 2>&1 &
  echo "  $arm shard$si/$sc gpu$gpu mode=$mode pid=$!"
}

# Intents are versioned with the code: they were LLM-written once and reviewed,
# and regenerating them would silently change the benchmark.
SPEC=scripts/eval/bench_v3/data/intents_specific.jsonl
GEN=scripts/eval/bench_v3/data/intents_generic.jsonl

# With the gemini-3.8-flash planner an agent sample costs ~710 s (~270 s planner
# + render + critic) and a direct sample ~460 s, so an even 2/2 split keeps the
# longest arm at ~4.9 h. (With gpt-6.1-sol, agent ~1136 s, a 3/1 split was better.)
launch ours_intent_direct  "$SPEC" instruction_first 0 2 0
launch ours_intent_direct  "$SPEC" instruction_first 1 2 1
launch ours_intent_agent   "$SPEC" expand            0 2 2
launch ours_intent_agent   "$SPEC" expand            1 2 3
launch ours_generic_direct "$GEN"  instruction_first 0 2 4
launch ours_generic_direct "$GEN"  instruction_first 1 2 5
launch ours_generic_agent  "$GEN"  expand            0 2 6
launch ours_generic_agent  "$GEN"  expand            1 2 7
