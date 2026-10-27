#!/bin/bash
#SBATCH -J stage2-top50-submit3
#SBATCH -p normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 00:15:00

set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
cd "$PROJECT"

METHODS=$(wc -l < "$RUN/methods.tsv" | tr -d ' ')
LAST=$((METHODS - 1))

JEDI_HIGH_JOB=$(sbatch --parsable \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_jedi_high.sbatch)
echo "JEDI_HIGH_JOB=$JEDI_HIGH_JOB"

JEDI_JOB=$(sbatch --parsable --array=0-"$LAST"%6 \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_jedi_array.sbatch)
echo "JEDI_JOB=$JEDI_JOB"

AGG_JOB=$(sbatch --parsable \
  --dependency=afterok:"$JEDI_HIGH_JOB":"$JEDI_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_aggregate.sbatch)
echo "AGG_JOB=$AGG_JOB"

{
  echo "JEDI_HIGH_JOB=$JEDI_HIGH_JOB"
  echo "JEDI_JOB=$JEDI_JOB"
  echo "AGG_JOB=$AGG_JOB"
} >> "$RUN/submitted_jobs.env"
