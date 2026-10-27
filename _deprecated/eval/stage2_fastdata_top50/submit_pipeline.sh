#!/bin/bash
set -euo pipefail

PROJECT=${PROJECT:-/home/l/liyj/shiying/low-high-new}
RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
BASE_MODEL=${BASE_MODEL:-/home/l/liyj/shiying/models/Wan2.1-VACE-14B}
CKPT_ROOT=${CKPT_ROOT:-$RUN/hf_checkpoints}
HF_REPO=${HF_REPO:-Ouzhang/low-high-stage2-fastdata-checkpoints-public}
HF_BASE_URL=${HF_BASE_URL:-https://hf-mirror.com/$HF_REPO/resolve/main/checkpoints/stage2_exp1_step7000_fastdata}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate metagen
cd "$PROJECT"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/l/liyj/shiying/hf_cache}
export TMPDIR=${TMPDIR:-/mnt/scratch/l/liyj/tmp}
mkdir -p "$RUN/logs" "$CKPT_ROOT" "$TMPDIR"

echo "Download checkpoints from $HF_BASE_URL"
for NAME in \
  stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata
do
  DEST_DIR="$CKPT_ROOT/checkpoints/stage2_exp1_step7000_fastdata/$NAME"
  mkdir -p "$DEST_DIR"
  for STEP in 1000 2000 2500 3000 4000 5000
  do
    FILE="step-${STEP}.safetensors"
    DEST="$DEST_DIR/$FILE"
    URL="$HF_BASE_URL/$NAME/$FILE"
    if test -s "$DEST"; then
      echo "skip existing $DEST"
      continue
    fi
    echo "curl $URL"
    if ! curl -fL --retry 6 --retry-delay 10 --connect-timeout 30 --continue-at - \
      -o "$DEST" "$URL"; then
      echo "missing or failed: $URL"
      rm -f "$DEST"
    fi
  done
done

python scripts/eval/stage2_fastdata_top50/prepare_stage2_fastdata_top50.py \
  --sample-index reports/top50_video_bundle_20260725_nodup/sample_index.json \
  --checkpoint-root "$CKPT_ROOT" \
  --run-root "$RUN" \
  --expected-samples 50

METHODS=$(wc -l < "$RUN/methods.tsv" | tr -d ' ')
LAST=$((METHODS - 1))
echo "methods=$METHODS"

INFER_JOB=$(sbatch --parsable --array=0-"$LAST"%6 \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN",BASE_MODEL="$BASE_MODEL" \
  scripts/eval/stage2_fastdata_top50/run_infer_array.sbatch)
echo "INFER_JOB=$INFER_JOB"

PREP_JOB=$(sbatch --parsable \
  --dependency=afterok:"$INFER_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_merge_manifests.sbatch)
echo "PREP_METRICS_JOB=$PREP_JOB"

CORE_JOB=$(sbatch --parsable --dependency=afterok:"$PREP_JOB" \
  --export=ALL,PROJECT="$PROJECT",FULL_METRICS_ROOT="$RUN/full_metrics",FULL_METRICS_CODE="$PROJECT/scripts/eval/val30_full_metrics" \
  scripts/eval/val30_full_metrics/run_core_array.sbatch)
echo "CORE_JOB=$CORE_JOB"

REWARD_JOB=$(sbatch --parsable --dependency=afterok:"$PREP_JOB" \
  --export=ALL,PROJECT="$PROJECT",FULL_METRICS_ROOT="$RUN/full_metrics" \
  scripts/eval/val30_full_metrics/run_reward_array.sbatch)
echo "REWARD_JOB=$REWARD_JOB"

PICK_JOB=$(sbatch --parsable --array=0-"$LAST"%6 --dependency=afterok:"$PREP_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_pickscore_array.sbatch)
echo "PICK_JOB=$PICK_JOB"

JEDI_HIGH_JOB=$(sbatch --parsable --dependency=afterok:"$PREP_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_jedi_high.sbatch)
echo "JEDI_HIGH_JOB=$JEDI_HIGH_JOB"

JEDI_JOB=$(sbatch --parsable --array=0-"$LAST"%6 --dependency=afterok:"$PREP_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_jedi_array.sbatch)
echo "JEDI_JOB=$JEDI_JOB"

AGG_JOB=$(sbatch --parsable \
  --dependency=afterok:"$CORE_JOB":"$REWARD_JOB":"$PICK_JOB":"$JEDI_HIGH_JOB":"$JEDI_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_aggregate.sbatch)
echo "AGG_JOB=$AGG_JOB"

cat > "$RUN/submitted_jobs.env" <<EOF
INFER_JOB=$INFER_JOB
PREP_METRICS_JOB=$PREP_JOB
CORE_JOB=$CORE_JOB
REWARD_JOB=$REWARD_JOB
PICK_JOB=$PICK_JOB
JEDI_HIGH_JOB=$JEDI_HIGH_JOB
JEDI_JOB=$JEDI_JOB
AGG_JOB=$AGG_JOB
RUN=$RUN
EOF
cat "$RUN/submitted_jobs.env"
