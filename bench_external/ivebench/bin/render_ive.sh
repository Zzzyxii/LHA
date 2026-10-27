#!/bin/bash
# usage: render_ive.sh <instruction_file> <out_dir> <gpu> [source_dir] [variant] [num_frames] [negative_prompt]
# source_dir defaults to the style subset's clips; pass the attribute subset's
# directory (src_mp4_attr) for the point-editing arms.
# Renders a prompt file through the selected local edit backend. Inputs are
# 832x480, 32 frames; render_one normalizes the requested count to the nearest
# supported 4n+1 length (29 by default). Resumable: existing outputs are skipped.
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
INSTR=$1
OUT=$2
GPU=$3
SRC=${4:-$PROJ/bench_external/ivebench/src_mp4}
VARIANT=${5:-wan14b+dittolora}
N_FRAMES=${6:-32}
NEGATIVE_PROMPT=${7:-}
cd "$PROJ" || exit 1
export PYTHONPATH=$PROJ/src:$PROJ/Ditto
export PATH=$PROJ/scripts/bench_skill/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
_PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
if [ ! -f "$INSTR" ] || [ ! -d "$SRC" ] || [ ! -x "$_PY" ]; then
  echo "invalid input, source directory, or Python executable" >&2
  exit 2
fi
SAMPLE_IDS=$("$_PY" -c '
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        print(json.loads(line)["sample_id"])
' "$INSTR") || exit 2
if [ -z "$SAMPLE_IDS" ]; then
  echo "instruction file contains no sample IDs: $INSTR" >&2
  exit 2
fi
mkdir -p "$OUT"
failed=0
RENDER_ARGS=()
if [ -n "$NEGATIVE_PROMPT" ]; then
  RENDER_ARGS+=(--negative-prompt "$NEGATIVE_PROMPT")
fi
while read -r sid; do
  [ -z "$sid" ] && continue
  echo "[$(date +%T)] START $sid"
  "$_PY" "$PROJ/scripts/bench_v4/render_one.py" \
    --variant "$VARIANT" --sample-id "$sid" --instruction-file "$INSTR" \
    --out-dir "$OUT" --source-dir "$SRC" --num-frames "$N_FRAMES" \
    --verify-existing-metadata \
    "${RENDER_ARGS[@]}"
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$OUT/$sid.mp4" ] || [ ! -s "$OUT/$sid.meta.json" ]; then
    echo "[$(date +%T)] FAIL $sid rc=$rc"
    failed=1
  else
    echo "[$(date +%T)] END $sid rc=0"
  fi
done <<< "$SAMPLE_IDS"
echo "[$(date +%T)] ALL DONE"
if [ "$failed" -ne 0 ]; then
  echo "one or more clips failed; see per-sample log above" >&2
  exit 1
fi
