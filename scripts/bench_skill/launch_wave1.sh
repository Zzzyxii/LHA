#!/usr/bin/env bash
# Wave 1 of the native-duration run: the arms that answer the core question
# (ours+skill vs the Bernini bare baseline, plus the no-skill control).
# Frames and fps follow each source video, so edits preserve clip duration.
set -u

cd "$(dirname "$0")/../.."
source scripts/bench_skill/env.sh

IDX=reports/top50_video_bundle_20260725_nodup/sample_index.json
SRC=reports/top50_video_bundle_20260725_nodup/videos/source_low
OUT=out/bench_skill/run50_native
mkdir -p "$OUT/logs"

launch () { # arm shard_index shard_count gpu max_attempts [extra...]
  local arm=$1 si=$2 sc=$3 gpu=$4 att=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 scripts/bench_skill/run_bench.py \
    --arm "$arm" --sample-index "$IDX" --source-dir "$SRC" --output-root "$OUT" \
    --shard-index "$si" --shard-count "$sc" --max-attempts "$att" "$@" \
    >> "$OUT/logs/$arm.shard$si.log" 2>&1 &
  echo "  $arm shard$si/$sc gpu$gpu attempts=$att $* pid=$!"
}

# ours + skill: the system under test
launch ours_skill  0 3 0 3
launch ours_skill  1 3 1 3
launch ours_skill  2 3 2 3
# ours without skill: isolates the skill's contribution at identical protocol
launch ours_raw    0 2 3 1
launch ours_raw    1 2 4 1
# Bernini bare: the baseline to beat, regenerated at native duration
launch bernini_raw 0 3 5 1
launch bernini_raw 1 3 6 1
launch bernini_raw 2 3 7 1
