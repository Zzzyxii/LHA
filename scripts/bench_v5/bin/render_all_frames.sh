#!/bin/bash
# usage: render_all_frames.sh <variant> <instruction_file> <out_dir> <gpu> <num_frames>
# Same as bench_v4/bin/render_all.sh but with an explicit frame count. Needed for
# Bernini: it renders the source's full 81 frames by default, which is 5x the cost
# of the published Bernini protocol (17 frames) and the frame count every released
# Bernini video in 归档_extract uses. Matching it keeps the comparison honest and
# turns a 20 min/clip job into a 3 min/clip one.
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
VARIANT=$1
INSTR=$2
OUT=$3
GPU=$4
NF=${5:-0}
mkdir -p "$OUT"
export PYTHONPATH=$PROJ/src:$PROJ/Ditto
export PATH=$PROJ/scripts/bench_skill/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
_PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
while read -r sid; do
  [ -z "$sid" ] && continue
  echo "[$(date +%T)] START $sid"
  "$_PY" "$PROJ/scripts/bench_v4/render_one.py" \
    --variant "$VARIANT" --sample-id "$sid" --instruction-file "$INSTR" \
    --out-dir "$OUT" --num-frames "$NF"
  echo "[$(date +%T)] END $sid rc=$?"
done < <(/opt/venv/bin/python -c "
import json
for line in open('$INSTR'):
    line=line.strip()
    if line: print(json.loads(line)['sample_id'])
")
echo "[$(date +%T)] ALL DONE"
