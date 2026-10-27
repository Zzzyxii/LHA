#!/usr/bin/env bash
# Re-run the skill arm with the prompt-composition fix, alongside the best-of-N
# arm, over the 30 archived manifest samples.
set -u

cd "$(dirname "$0")/../.."
source scripts/bench_skill/env.sh

IDX=reports/top50_video_bundle_20260725_nodup/sample_index.json
SRC=reports/top50_video_bundle_20260725_nodup/videos/source_low
OUT=out/bench_skill/run30_fixed
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

launch ours_skill  0 4 0 --max-attempts 3
launch ours_skill  1 4 1 --max-attempts 3
launch ours_skill  2 4 2 --max-attempts 3
launch ours_skill  3 4 3 --max-attempts 3
launch ours_bestof 0 4 4 --candidates 3
launch ours_bestof 1 4 5 --candidates 3
launch ours_bestof 2 4 6 --candidates 3
launch ours_bestof 3 4 7 --candidates 3
