#!/usr/bin/env bash
# Re-score every video we have against its target_high: the 16 published
# methods from the archive plus our own native-duration arms.
#   RELEASE_ROOT  extracted 归档.zip (default: this machine's location)
#   GPU           device for CLIP/DINO (default 0)
set -u
cd "$(dirname "$0")/../../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RELEASE_ROOT=${RELEASE_ROOT:-/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/归档_extract}
ARGS=()
for d in "$RELEASE_ROOT"/methods/*/; do
  m=$(basename "$d")
  ARGS+=(--arm "pub:$m=$d")
done
ARGS+=(--arm "ours_raw=out/bench_skill/run50_native/ours_raw/generations")
ARGS+=(--arm "ours_skill_v1=out/bench_skill/run50_native/ours_skill/generations")
ARGS+=(--arm "ours_skill_v2=out/bench_skill/run30_v2/ours_skill/generations")
ARGS+=(--arm "bernini_raw=out/bench_skill/run50_native/bernini_raw/generations")

CUDA_VISIBLE_DEVICES=${GPU:-0} python3 scripts/eval/bench_v3/gt_metrics.py \
  --manifest out/bench_v2/manifest.jsonl \
  --hf-cache models/hf_cache/hub \
  --output-dir out/bench_v2/gt_all \
  "${ARGS[@]}"
