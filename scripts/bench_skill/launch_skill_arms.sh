#!/usr/bin/env bash
# Relaunch the skill arms (resumable: run_bench.py skips samples already rendered).
set -u

cd "$(dirname "$0")/../.."
source scripts/bench_skill/env.sh

IDX=reports/top50_video_bundle_20260725_nodup/sample_index.json
SRC=reports/top50_video_bundle_20260725_nodup/videos/source_low
OUT=out/bench_skill/run50
mkdir -p "$OUT/logs"

launch () { # arm shard_index shard_count gpu max_attempts
  CUDA_VISIBLE_DEVICES=$4 nohup python3 scripts/bench_skill/run_bench.py \
    --arm "$1" --sample-index "$IDX" --source-dir "$SRC" --output-root "$OUT" \
    --shard-index "$2" --shard-count "$3" --max-attempts "$5" \
    >> "$OUT/logs/$1.shard$2.log" 2>&1 &
  echo "  $1 shard$2/$3 gpu$4 attempts=$5 pid=$!"
}

launch ours_skill    0 3 0 3
launch ours_skill    1 3 1 3
launch ours_skill    2 3 2 3
launch ditto_skill   0 2 3 3
launch ditto_skill   1 2 4 3
launch bernini_skill 0 2 6 3
launch bernini_skill 1 2 7 3
