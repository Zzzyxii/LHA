#!/bin/bash
# Chain the next GPU stage onto each render that is running now, so no card sits
# idle between stages.
#
#   branch A (gpu 7): g1 renders -> extract -> score -> compile g2 (feedback) -> render g2
#   branch B (gpu 6): g0 renders -> gapfill -> IVEBench direct -> IVEBench skill
#
# Every stage checks the *outputs*, not the exit code (LESSONS.md: render_all.sh
# prints rc=0 for clips that died of CUDA OOM).
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Wait for a render pass to finish. The driver loop can die without ever printing
# ALL DONE (OOM, someone else taking the card), so a pass also counts as finished
# when its log has not moved for 30 minutes: the stages after it re-render whatever
# is missing instead of waiting forever on a log line that will never come.
wait_all_done() {
  local log=$1
  while true; do
    grep -q "ALL DONE" "$log" 2>/dev/null && return 0
    local age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
    [ "$age" -gt 1800 ] && { echo "[$(date +%T)] $log stalled ${age}s, continuing"; return 0; }
    sleep 120
  done
}

# ---------------- branch A: g1 -> g2 ----------------
(
  wait_all_done /data/render_g1.log
  echo "[$(date +%T)] g1 pass finished, gap filling"
  CUDA_VISIBLE_DEVICES=7 bash scripts/bench_v4/bin/render_all.sh wan14b+dittolora \
    scripts/bench_v5/data/instructions_g1_axis.jsonl out/bench_v5/runs/g1_axis_prior 7 \
    > /data/gapfill_g1.log 2>&1
  echo "[$(date +%T)] g1 gapfill done: $(ls out/bench_v5/runs/g1_axis_prior/*.mp4 | wc -l) clips"
  $PY scripts/bench_v5/build_run_jobs.py --run-dir out/bench_v5/runs/g1_axis_prior \
    --arm g1_axis_prior --out scripts/bench_v5/data/jobs_g1.jsonl
  CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/extract_cache.py \
    --jobs scripts/bench_v5/data/jobs_g1.jsonl --cache-dir scripts/bench_v5/data/cache_pod \
    --workers 2 > /data/extract_g1.log 2>&1
  CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/score_profile.py \
    --cache-dir scripts/bench_v5/data/cache_pod \
    --axis-validation scripts/bench_v5/data/axis_validation.json \
    --arms "direct_intent=direct_intent,g0=g0_grounded_clean,g1=g1_axis_prior" \
    --out scripts/bench_v5/data/scores_g1.jsonl > /data/score_g1.log 2>&1
  echo "[$(date +%T)] g1 scored"
  $PY scripts/bench_v5/compile_axis_v5.py \
    --base-prompts scripts/bench_v4/data/instructions_grounded_clean48.jsonl \
    --profile scripts/bench_v5/data/axis_profile_calib.json --mode feedback \
    --feedback-scores scripts/bench_v5/data/scores_g1.jsonl --feedback-arm g1 \
    --out scripts/bench_v5/data/instructions_g2_feedback.jsonl
  echo "[$(date +%T)] g2 prompts compiled, rendering"
  CUDA_VISIBLE_DEVICES=7 bash scripts/bench_v4/bin/render_all.sh wan14b+dittolora \
    scripts/bench_v5/data/instructions_g2_feedback.jsonl out/bench_v5/runs/g2_axis_feedback 7 \
    > /data/render_g2.log 2>&1
  echo "[$(date +%T)] BRANCH A DONE"
) > /data/queue_a.log 2>&1 &

# ---------------- branch B: g0 -> IVEBench ----------------
(
  wait_all_done /data/render_g0.log
  echo "[$(date +%T)] g0 pass finished, gap filling"
  CUDA_VISIBLE_DEVICES=6 bash scripts/bench_v4/bin/render_all.sh wan14b+dittolora \
    scripts/bench_v4/data/instructions_grounded_clean48.jsonl out/bench_v5/runs/g0_grounded_clean 6 \
    > /data/gapfill_g0.log 2>&1
  echo "[$(date +%T)] g0 gapfill done: $(ls out/bench_v5/runs/g0_grounded_clean/*.mp4 | wc -l) clips"
  $PY scripts/bench_v5/build_run_jobs.py --run-dir out/bench_v5/runs/g0_grounded_clean \
    --arm g0_grounded_clean --out scripts/bench_v5/data/jobs_g0.jsonl
  CUDA_VISIBLE_DEVICES=6 LH_DEVICE=cuda $PY -u scripts/bench_v5/extract_cache.py \
    --jobs scripts/bench_v5/data/jobs_g0.jsonl --cache-dir scripts/bench_v5/data/cache_pod \
    --workers 2 > /data/extract_g0.log 2>&1
  echo "[$(date +%T)] g0 features cached; starting IVEBench arms"
  bash bench_external/ivebench/bin/render_ive.sh bench_external/ivebench/ive_direct.jsonl \
    out/bench_v5/runs/ive_direct 6 > /data/render_ive_direct.log 2>&1
  echo "[$(date +%T)] ive_direct done: $(ls out/bench_v5/runs/ive_direct/*.mp4 | wc -l)"
  bash bench_external/ivebench/bin/render_ive.sh bench_external/ivebench/ive_skill.jsonl \
    out/bench_v5/runs/ive_skill 6 > /data/render_ive_skill.log 2>&1
  echo "[$(date +%T)] ive_skill done: $(ls out/bench_v5/runs/ive_skill/*.mp4 | wc -l)"
  CUDA_VISIBLE_DEVICES=6 bash scripts/bench_v4/bin/render_all.sh wan14b+dittolora \
    scripts/bench_v4/data/instructions_expert.jsonl out/bench_v5/runs/direct_expert_v5 6 \
    > /data/render_expert.log 2>&1
  echo "[$(date +%T)] BRANCH B DONE"
) > /data/queue_b.log 2>&1 &

echo "queue launched: branch A (gpu7 g1->g2), branch B (gpu6 g0->ive->expert)"
