#!/bin/bash
# usage: render_one.sh <variant> <sample_id> <instruction_file> <out_dir> <gpu>
set -euo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
export PYTHONPATH=$PROJ/src:$PROJ/Ditto
export PATH=$PROJ/scripts/bench_skill/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${5:-6}
_PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
exec "$_PY" "$PROJ/scripts/bench_v4/render_one.py" \
  --variant "$1" --sample-id "$2" --instruction-file "$3" --out-dir "$4"
