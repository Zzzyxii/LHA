#!/bin/bash
# usage: render_all.sh <variant> <instruction_file> <out_dir> <gpu> [limit]
# Renders every sample in the instruction file sequentially on one GPU.
# Resumable: samples whose output already exist are skipped (pass --overwrite to redo).
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
VARIANT=$1
INSTR=$2
OUT=$3
GPU=$4
LIMIT=${5:-0}
mkdir -p "$OUT"
export PYTHONPATH=$PROJ/src:$PROJ/Ditto
export PATH=$PROJ/scripts/bench_skill/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU
_PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
count=0
while read -r sid; do
  [ -z "$sid" ] && continue
  count=$((count+1))
  if [ "$LIMIT" -gt 0 ] && [ "$count" -gt "$LIMIT" ]; then break; fi
  echo "[$(date +%T)] START $sid ($count)"
  "$_PY" "$PROJ/scripts/bench_v4/render_one.py" \
    --variant "$VARIANT" --sample-id "$sid" --instruction-file "$INSTR" --out-dir "$OUT"
  echo "[$(date +%T)] END $sid rc=$?"
done < <(/opt/venv/bin/python -c "
import json,sys
for line in open('$INSTR'):
    line=line.strip()
    if line: print(json.loads(line)['sample_id'])
")
echo "[$(date +%T)] ALL DONE"
