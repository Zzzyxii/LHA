#!/bin/bash
#SBATCH -J stage2-top50-infer
#SBATCH -p gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH -t 24:00:00

set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
BASE_MODEL=${BASE_MODEL:-/home/l/liyj/shiying/models/Wan2.1-VACE-14B}
LINE=$((SLURM_ARRAY_TASK_ID + 1))

IFS=$'\t' read -r METHOD FAMILY LATENT_MODE CKPT < <(sed -n "${LINE}p" "$RUN/methods.tsv")
if test -z "${METHOD:-}" || test -z "${CKPT:-}"; then
  echo "No method for task ${SLURM_ARRAY_TASK_ID}"
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

OUT=$RUN/infer/$METHOD
ARGS=(
  scripts/eval/ditto_quant/run_inference.py
  --eval-inputs "$RUN/eval_inputs.top50.jsonl"
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

echo "METHOD=$METHOD"
echo "CKPT=$CKPT"
echo "LATENT_MODE=$LATENT_MODE"
"$HOME/miniconda3/envs/metagen/bin/python" "${ARGS[@]}"
echo "DONE infer $METHOD $(date)"
