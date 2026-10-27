#!/bin/bash
#SBATCH -J stage2-top50-pick
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 04:00:00

set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
LINE=$((SLURM_ARRAY_TASK_ID + 1))
IFS=$'\t' read -r METHOD FAMILY LATENT_MODE CKPT < <(sed -n "${LINE}p" "$RUN/methods.tsv")

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate metagen
cd "$PROJECT"
export HF_HOME=/mnt/scratch/l/liyj/pickscore_hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

python scripts/eval/pickscore/evaluate_manifest_group.py \
  --manifest "$RUN/full_metrics/manifest_all.jsonl" \
  --output-dir "$RUN/pickscore" \
  --method "$METHOD" \
  --expected-count 50 \
  --frames 16 \
  --device cuda
echo "DONE pickscore $METHOD $(date)"
