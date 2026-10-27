#!/usr/bin/env bash
# Planning-track ablation: which part of the agent prompt makes edits conservative.
# Prompts come from build_ablation_prompts.py (run it first); all three arms
# render through the raw path (no loop, no memory, no reference image), so they
# differ from each other only in the prompt text.
#   ours_abl_agentprompt : agent prompt verbatim        -> isolates the reference image
#   ours_abl_plan_craft  : minus the preserve list
#   ours_abl_plan_only   : minus the craft lines too
set -u

cd "$(dirname "$0")/../../.."
source scripts/bench_skill/env.sh

IDX=reports/top50_video_bundle_20260725_nodup/sample_index.json
SRC=reports/top50_video_bundle_20260725_nodup/videos/source_low
OUT=out/bench_v2/ablation
P=$OUT/prompts
mkdir -p "$OUT/logs"

launch () { # arm prompts_file shard_index shard_count gpu
  local arm=$1 prompts=$2 si=$3 sc=$4 gpu=$5
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/bench_skill/run_bench.py \
    --arm "$arm" --sample-index "$IDX" --source-dir "$SRC" --output-root "$OUT" \
    --instruction-file "$prompts" \
    --shard-index "$si" --shard-count "$sc" \
    >> "$OUT/logs/$arm.shard$si.log" 2>&1 &
  echo "  $arm shard$si/$sc gpu$gpu pid=$!"
}

launch ours_abl_agentprompt "$P/agentprompt.jsonl" 0 2 ${GPU0:-0}
launch ours_abl_agentprompt "$P/agentprompt.jsonl" 1 2 ${GPU1:-1}
launch ours_abl_plan_craft  "$P/plan_craft.jsonl"  0 2 ${GPU2:-2}
launch ours_abl_plan_craft  "$P/plan_craft.jsonl"  1 2 ${GPU3:-3}
launch ours_abl_plan_only   "$P/plan_only.jsonl"   0 2 ${GPU4:-4}
launch ours_abl_plan_only   "$P/plan_only.jsonl"   1 2 ${GPU5:-5}
