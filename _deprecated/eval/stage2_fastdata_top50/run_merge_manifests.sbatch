#!/bin/bash
#SBATCH -J stage2-top50-merge
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00

set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate metagen
cd "$PROJECT"

python scripts/eval/stage2_fastdata_top50/merge_inference_manifests.py \
  --run-root "$RUN" \
  --project-root "$PROJECT" \
  --num-shards 6 \
  --expected-samples 50
echo "DONE merge $(date)"
