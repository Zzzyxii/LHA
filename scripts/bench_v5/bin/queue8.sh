#!/bin/bash
# Eighth stage: bern_g2 -- the feedback round on Bernini.
#
# bern_g0 showed that the grounded prompt makes Bernini overshoot (delivery 0.50 ->
# 1.10 but magnitude 0.85 -> 1.49, waste 0.59 -> 0.90), which is the opposite of
# what the same prompt does on Ditto. That is exactly the residual the feedback
# round is for: measure what the render actually moved, subtract the overshoot,
# re-render. Mirror of the Ditto ladder (g2 = feedback from g1).
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
NF=17

(
  while ! grep -q "BRANCH H DONE" /data/queue_h.log 2>/dev/null; do sleep 120; done
  echo "[$(date +%T)] Bernini g0/g1 done, scoring to compute the feedback residual"
  ARMS=$($PY scripts/bench_v5/list_arms.py --cache-dir scripts/bench_v5/data/cache_pod)
  CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/score_profile.py \
    --cache-dir scripts/bench_v5/data/cache_pod \
    --axis-validation scripts/bench_v5/data/axis_validation.json \
    --arms "$ARMS" --out scripts/bench_v5/data/scores_bern2.jsonl > /data/score_bern2.log 2>&1
  $PY scripts/bench_v5/compile_axis_v5.py \
    --base-prompts scripts/bench_v4/data/instructions_grounded_clean48.jsonl \
    --profile scripts/bench_v5/data/axis_profile_calib.json --mode feedback \
    --feedback-scores scripts/bench_v5/data/scores_bern2.jsonl --feedback-arm bern_g1 \
    --out scripts/bench_v5/data/instructions_bern_g2.jsonl
  echo "[$(date +%T)] bern_g2 prompts compiled, rendering"
  bash scripts/bench_v5/bin/render_all_frames.sh bernini1.3b \
    scripts/bench_v5/data/instructions_bern_g2.jsonl out/bench_v5/runs/bern_g2 7 $NF \
    > /data/render_bern_g2.log 2>&1
  echo "[$(date +%T)] bern_g2: $(ls out/bench_v5/runs/bern_g2/*.mp4 2>/dev/null | wc -l)"
  $PY scripts/bench_v5/build_run_jobs.py --run-dir out/bench_v5/runs/bern_g2 --arm bern_g2 \
    --out scripts/bench_v5/data/jobs_bern_g2.jsonl
  CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/extract_cache.py \
    --jobs scripts/bench_v5/data/jobs_bern_g2.jsonl \
    --cache-dir scripts/bench_v5/data/cache_pod --workers 2 > /data/extract_bern_g2.log 2>&1
  echo "[$(date +%T)] BRANCH J DONE"
) > /data/queue_j.log 2>&1 &

echo "queue8 launched: branch J (bern_g2 feedback round)"
