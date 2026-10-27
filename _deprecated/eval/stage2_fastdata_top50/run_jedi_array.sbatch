#!/bin/bash
#SBATCH -J stage2-top50-jedi
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 04:00:00

set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
JEDI_MODEL_DIR=${JEDI_MODEL_DIR:-/mnt/scratch/l/liyj/jedi_val30_7methods/model}
JEDI_CONFIG=${JEDI_CONFIG:-$JEDI_MODEL_DIR/vith16_ssv2_16x2x3.yaml}
LINE=$((SLURM_ARRAY_TASK_ID + 1))
IFS=$'\t' read -r METHOD FAMILY LATENT_MODE CKPT < <(sed -n "${LINE}p" "$RUN/methods.tsv")

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate metagen
cd "$PROJECT"
export PYTHONPATH=$PROJECT/src:$PROJECT/Ditto
export HF_HOME=/home/l/liyj/shiying/hf_cache
export TMPDIR=/mnt/scratch/l/liyj/tmp
mkdir -p "$RUN/jedi/features" "$TMPDIR"

python scripts/eval/jedi/extract_val30_features.py \
  --manifest "$RUN/full_metrics/manifest_all.jsonl" \
  --model-dir "$JEDI_MODEL_DIR" \
  --config "$JEDI_CONFIG" \
  --output-dir "$RUN/jedi/features" \
  --method-id "$METHOD" \
  --display-name "$METHOD" \
  --expected-count 50 \
  --frames 16 \
  --batch-size 2
echo "DONE jedi $METHOD $(date)"
