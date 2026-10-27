#!/bin/bash
#SBATCH -J stage2-top50-chunk
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH -t 03:00:00

set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
BASE_MODEL=${BASE_MODEL:-/home/l/liyj/shiying/models/Wan2.1-VACE-14B}
LINE=$((SLURM_ARRAY_TASK_ID + 1))

IFS=$'\t' read -r TASK_ID METHOD LATENT_MODE CKPT CHUNK_INPUT OUT < <(sed -n "${LINE}p" "$RUN/infer_tasks.tsv")
if test -z "${METHOD:-}" || test -z "${CKPT:-}" || test -z "${CHUNK_INPUT:-}"; then
  echo "No chunk task for array index ${SLURM_ARRAY_TASK_ID}"
  exit 2
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate metagen
cd "$PROJECT"

export PYTHONPATH=$PROJECT/src:$PROJECT/Ditto
export HF_HOME=/home/l/liyj/shiying/hf_cache
export TORCH_HOME=/home/l/liyj/.cache/torch
export XDG_CACHE_HOME=/home/l/liyj/shiying/low-high-new/models/cache
export TMPDIR=/mnt/scratch/l/liyj/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/logs" "$TMPDIR"

ARGS=(
  scripts/eval/ditto_quant/run_inference.py
  --eval-inputs "$CHUNK_INPUT"
  --output-dir "$OUT"
  --checkpoint "$METHOD=$CKPT"
  --checkpoint-path "$BASE_MODEL"
  --python-executable "$HOME/miniconda3/envs/metagen/bin/python"
  --device-id 0
  --height 480
  --width 832
  --num-frames 17
  --num-inference-steps 20
  --cfg-scale 1.0
  --seed 42
  --fps 20
  --quality 5
  --default-lora-alpha 1.0
  --skip-existing
  --fail-fast
)
if test "$LATENT_MODE" != "none"; then
  ARGS+=(--latent-memory-mode "$LATENT_MODE" --latent-memory-tokens 4 --latent-memory-hidden-dim 0 --latent-memory-scale 1.0 --latent-memory-init-std 0.02)
fi

echo "TASK_ID=$TASK_ID METHOD=$METHOD CHUNK_INPUT=$CHUNK_INPUT"
"$HOME/miniconda3/envs/metagen/bin/python" "${ARGS[@]}"
echo "DONE infer chunk task=$TASK_ID method=$METHOD $(date)"
