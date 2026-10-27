#!/bin/bash
# usage: eval_ive.sh <frames_dir> <name> [gpu] [info_json] [source_frames_dir]
# Score one IVEBench arm with the metrics whose checkpoints exist on this pod
# (see PUBLISHED.md): VideoCLIP-XL compliance/fidelity plus temporal flickering,
# which needs no checkpoint. Quality metrics are skipped and reported as
# unavailable rather than silently substituted.
set -euo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
FRAMES=$1
NAME=$2
GPU=${3:-6}
INFO=${4:-$PROJ/bench_external/ivebench/ive_info.json}
SRCFR=${5:-$PROJ/bench_external/ivebench/src_frames}
# Results go into the project (shared filesystem), not /data: /data only exists on
# the pod, and the report script runs wherever the analysis happens.
OUT=$PROJ/bench_external/ivebench/eval_$NAME
mkdir -p "$OUT"
cd "$PROJ/bench_external/ivebench_code/metrics" || exit 1
CUDA_VISIBLE_DEVICES=$GPU /opt/venv/bin/python -u evaluate.py \
  --output_path "$OUT" \
  --source_videos_path "$SRCFR" \
  --target_videos_path "$FRAMES" \
  --info_json_path "$INFO" \
  --metric overall_semantic_consistency phrase_semantic_consistency semantic_fidelity \
           temporal_flickering \
  --name "$NAME" 2>&1 | tail -5
shopt -s nullglob
CSV_FILES=("$OUT"/*_eval_results.csv)
if [ "${#CSV_FILES[@]}" -ne 1 ] || [ ! -s "${CSV_FILES[0]:-}" ]; then
  echo "IVE evaluator produced no non-empty result CSV for $NAME" >&2
  exit 1
fi
CSV=${CSV_FILES[0]}
/opt/venv/bin/python "$PROJ/bench_external/ivebench/bin/summarize_ive.py" "$CSV" \
  --out "$OUT/summary.json"
test -s "$OUT/summary.json"
