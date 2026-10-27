#!/bin/bash
# Second-stage automation: turn finished renders into numbers without waiting for
# a human to poll.
#
#   branch C: g2 renders -> extract g2 (+expert if it exists) -> score every arm
#             in the cache -> final_report.py
#   branch D: IVEBench direct/skill renders -> frame folders -> IVEBench metrics
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}

# The log does not exist until its render branch starts, and a missing file must
# not be mistaken for a stalled one: `stat` would return 0 and the "stalled"
# branch would fire immediately, running the evaluation on an empty output
# directory and then never running again.
wait_all_done() {
  local log=$1
  while [ ! -f "$log" ]; do sleep 120; done
  while true; do
    grep -q "ALL DONE" "$log" 2>/dev/null && return 0
    local age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
    [ "$age" -gt 1800 ] && { echo "[$(date +%T)] $log stalled, continuing"; return 0; }
    sleep 120
  done
}

# ---------------- branch C: final low->high scoring ----------------
(
  # Wait for the gap-filled arm (branch I), not for the first render pass: the first
  # pass can leave clips missing after an OOM and still print ALL DONE.
  while ! grep -q "BRANCH I DONE" /data/queue_i.log 2>/dev/null; do sleep 120; done
  wait_all_done /data/render_g2.log
  echo "[$(date +%T)] g2 pass ended: $(ls out/bench_v5/runs/g2_axis_feedback/*.mp4 2>/dev/null | wc -l) clips"
  for ARM in g2_axis_feedback direct_expert_v5; do
    [ -d "out/bench_v5/runs/$ARM" ] || continue
    $PY scripts/bench_v5/build_run_jobs.py --run-dir "out/bench_v5/runs/$ARM" --arm "$ARM" \
      --out "scripts/bench_v5/data/jobs_${ARM}.jsonl"
    CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/extract_cache.py \
      --jobs "scripts/bench_v5/data/jobs_${ARM}.jsonl" \
      --cache-dir scripts/bench_v5/data/cache_pod --workers 2 > "/data/extract_${ARM}.log" 2>&1
    echo "[$(date +%T)] extracted $ARM"
  done
  ARMS=$($PY scripts/bench_v5/list_arms.py --cache-dir scripts/bench_v5/data/cache_pod)
  CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/score_profile.py \
    --cache-dir scripts/bench_v5/data/cache_pod \
    --axis-validation scripts/bench_v5/data/axis_validation.json \
    --arms "$ARMS" --out scripts/bench_v5/data/scores_final.jsonl > /data/score_final.log 2>&1
  echo "[$(date +%T)] scored final: $ARMS"
  $PY scripts/bench_v5/final_report.py --scores scripts/bench_v5/data/scores_final.jsonl \
    --prov low_high_cost \
    --ive "$PROJ/bench_external/ivebench/eval_direct/summary.json" \
          "$PROJ/bench_external/ivebench/eval_skill/summary.json" \
          "$PROJ/bench_external/ivebench/eval_attr_direct/summary.json" \
          "$PROJ/bench_external/ivebench/eval_attr_skill/summary.json" \
    --out out/bench_v5/RESULTS_FINAL.md >> /data/score_final.log 2>&1
  $PY scripts/bench_v5/leaderboard.py --scores scripts/bench_v5/data/scores_final.jsonl \
    --metric profile_error --prov low_high_cost > out/bench_v5/LEADERBOARD.txt 2>&1
  echo "[$(date +%T)] BRANCH C DONE"
) > /data/queue_c.log 2>&1 &

# ---------------- branch D: IVEBench evaluation ----------------
(
  for ARM in direct skill; do
    wait_all_done /data/render_ive_${ARM}.log
    echo "[$(date +%T)] ive_${ARM} pass ended: $(ls out/bench_v5/runs/ive_${ARM}/*.mp4 2>/dev/null | wc -l)"
    $PY bench_external/ivebench/out2frames.py --mp4-dir out/bench_v5/runs/ive_${ARM} \
      --out-dir bench_external/ivebench/frames_${ARM} > /data/ive_frames_${ARM}.log 2>&1
    bash bench_external/ivebench/bin/eval_ive.sh \
      "$PROJ/bench_external/ivebench/frames_${ARM}" ${ARM} 6 > /data/ive_eval_${ARM}.log 2>&1
    echo "[$(date +%T)] ive_${ARM} evaluated"
  done
  echo "[$(date +%T)] BRANCH D DONE"
) > /data/queue_d.log 2>&1 &

echo "queue2 launched: branch C (final scoring), branch D (ivebench eval)"
