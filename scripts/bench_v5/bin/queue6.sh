#!/bin/bash
# Sixth stage: run the same agent ladder on Bernini-R-1.3B, the second base model.
# The point of the project is that the skill + loop helps whichever base model it
# sits on, so the ladder has to be repeated on a second one, not just Ditto.
# 17 frames: the published Bernini protocol and the frame count of every released
# Bernini video in 归档_extract (see render_all_frames.sh).
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
NF=17

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

(
  # Wait for branch I (g2 gap-fill) as well: g2's own re-run is still on gpu 7 when
  # the first pass prints ALL DONE, and two jobs there would OOM at 97/98 GB.
  while ! grep -q "BRANCH I DONE" /data/queue_i.log 2>/dev/null; do sleep 120; done
  wait_all_done /data/render_g2.log
  echo "[$(date +%T)] starting Bernini ladder on gpu 7"
  # baselines and agent arms, in the same order as the Ditto ladder
  bash scripts/bench_v5/bin/render_all_frames.sh bernini1.3b \
    scripts/bench_v5/data/instructions_intent.jsonl out/bench_v5/runs/bern_direct_intent 7 $NF \
    > /data/render_bern_direct.log 2>&1
  echo "[$(date +%T)] bern_direct: $(ls out/bench_v5/runs/bern_direct_intent/*.mp4 2>/dev/null | wc -l)"
  bash scripts/bench_v5/bin/render_all_frames.sh bernini1.3b \
    scripts/bench_v4/data/instructions_grounded_clean48.jsonl out/bench_v5/runs/bern_g0 7 $NF \
    > /data/render_bern_g0.log 2>&1
  echo "[$(date +%T)] bern_g0: $(ls out/bench_v5/runs/bern_g0/*.mp4 2>/dev/null | wc -l)"
  bash scripts/bench_v5/bin/render_all_frames.sh bernini1.3b \
    scripts/bench_v5/data/instructions_g1_axis.jsonl out/bench_v5/runs/bern_g1 7 $NF \
    > /data/render_bern_g1.log 2>&1
  echo "[$(date +%T)] bern_g1: $(ls out/bench_v5/runs/bern_g1/*.mp4 2>/dev/null | wc -l)"
  for ARM in bern_direct_intent bern_g0 bern_g1; do
    [ -d "out/bench_v5/runs/$ARM" ] || continue
    $PY scripts/bench_v5/build_run_jobs.py --run-dir "out/bench_v5/runs/$ARM" --arm "$ARM" \
      --out "scripts/bench_v5/data/jobs_${ARM}.jsonl"
    CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/extract_cache.py \
      --jobs "scripts/bench_v5/data/jobs_${ARM}.jsonl" \
      --cache-dir scripts/bench_v5/data/cache_pod --workers 2 > "/data/extract_${ARM}.log" 2>&1
    echo "[$(date +%T)] extracted $ARM"
  done
  echo "[$(date +%T)] BRANCH H DONE"
) > /data/queue_h.log 2>&1 &

echo "queue6 launched: branch H (bernini ladder on gpu 7)"
