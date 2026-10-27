#!/usr/bin/env bash
# Enhancement-skill redesign (v2): instruction verbatim + deterministic per-dimension
# craft direction, no planner rewrite. Run over the 30 archived manifest samples.
set -u

cd "$(dirname "$0")/../.."
source scripts/bench_skill/env.sh

IDX=reports/top50_video_bundle_20260725_nodup/sample_index.json
SRC=reports/top50_video_bundle_20260725_nodup/videos/source_low
OUT=out/bench_skill/run30_v2
IDS=out/bench_skill/run50_native/manifest30_ids.txt
mkdir -p "$OUT/logs"

launch () { # arm shard_index shard_count gpu extra...
  local arm=$1 si=$2 sc=$3 gpu=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/bench_skill/run_bench.py \
    --arm "$arm" --sample-index "$IDX" --source-dir "$SRC" --output-root "$OUT" \
    --only-sample-ids "$IDS" \
    --shard-index "$si" --shard-count "$sc" "$@" \
    >> "$OUT/logs/$arm.shard$si.log" 2>&1 &
  echo "  $arm shard$si/$sc gpu$gpu $* pid=$!"
}

# max-attempts 1: the closed loop's refinement was measured to regress on 73% of
# samples, so a single pass is the honest test of the prompt redesign in isolation.
for i in 0 1 2 3 4 5 6 7; do
  launch ours_skill "$i" 8 "$i" --max-attempts 1
done
