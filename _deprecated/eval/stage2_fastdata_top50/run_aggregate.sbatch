#!/bin/bash
#SBATCH -J stage2-top50-agg
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

python scripts/eval/pickscore/aggregate_manifest_groups.py --input-dir "$RUN/pickscore"
python scripts/eval/stage2_fastdata_top50/aggregate_jedi_stage2_fastdata_top50.py --run-root "$RUN"
python scripts/eval/val30_full_metrics/aggregate_full_metrics.py \
  --root "$RUN/full_metrics" \
  --output-dir "$RUN/full_metrics_aggregate"
python scripts/eval/stage2_fastdata_top50/aggregate_stage2_fastdata_top50.py --run-root "$RUN"
echo "DONE aggregate $(date)"
