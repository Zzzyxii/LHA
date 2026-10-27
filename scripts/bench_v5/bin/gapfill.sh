#!/usr/bin/env bash
# Re-render every sample of an arm whose output is missing.
#
# render_all.sh already skips clips that exist, so "filling the gaps" is just
# running it again. It has to be run again because the loop reports rc=0 even for
# samples that died of CUDA OOM (LESSONS.md), so a finished pass is not a complete
# pass. Check the output count, not the rc lines.
#
# usage: gapfill.sh <prompts.jsonl> <out_dir> <gpu>
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
INSTR=$1
OUT=$2
GPU=$3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJ" || exit 1
CUDA_VISIBLE_DEVICES=$GPU nohup bash scripts/bench_v4/bin/render_all.sh \
  wan14b+dittolora "$INSTR" "$OUT" "$GPU" > /data/gapfill_$(basename "$OUT").log 2>&1 &
echo "launched gapfill on gpu $GPU -> /data/gapfill_$(basename "$OUT").log"
