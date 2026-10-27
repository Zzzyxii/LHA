#!/bin/bash
#SBATCH -J stage2-top50-submit1
#SBATCH -p normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 00:15:00

set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
cd "$PROJECT"

CORE_JOB=$(sbatch --parsable \
  --export=ALL,PROJECT="$PROJECT",FULL_METRICS_ROOT="$RUN/full_metrics",FULL_METRICS_CODE="$PROJECT/scripts/eval/val30_full_metrics" \
  scripts/eval/val30_full_metrics/run_core_array.sbatch)
echo "CORE_JOB=$CORE_JOB"

REWARD_JOB=$(sbatch --parsable \
  --export=ALL,PROJECT="$PROJECT",FULL_METRICS_ROOT="$RUN/full_metrics" \
  scripts/eval/val30_full_metrics/run_reward_array.sbatch)
echo "REWARD_JOB=$REWARD_JOB"

PICK_SUBMIT_JOB=$(sbatch --parsable \
  --dependency=afterok:"$CORE_JOB":"$REWARD_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/submit_pick_stage.sbatch)
echo "PICK_SUBMIT_JOB=$PICK_SUBMIT_JOB"

{
  echo "CORE_JOB=$CORE_JOB"
  echo "REWARD_JOB=$REWARD_JOB"
  echo "PICK_SUBMIT_JOB=$PICK_SUBMIT_JOB"
} >> "$RUN/submitted_jobs.env"
