#!/bin/bash
#SBATCH -J stage2-top50-submit2
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

PICK_JOB=$(sbatch --parsable --array=0-"$LAST"%6 \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_pickscore_array.sbatch)
echo "PICK_JOB=$PICK_JOB"

JEDI_SUBMIT_JOB=$(sbatch --parsable \
  --dependency=afterok:"$PICK_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/submit_jedi_stage.sbatch)
echo "JEDI_SUBMIT_JOB=$JEDI_SUBMIT_JOB"

{
  echo "PICK_JOB=$PICK_JOB"
  echo "JEDI_SUBMIT_JOB=$JEDI_SUBMIT_JOB"
} >> "$RUN/submitted_jobs.env"
