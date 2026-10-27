#!/bin/bash
# Seventh stage: gap-fill g2 before anything scores it.
# render_all.sh prints rc=0 for clips that die of CUDA OOM (LESSONS.md), and g2 runs
# once with no gap-fill of its own -- at 97/98 GB on its card a death is plausible.
# Branch A renders g2; this re-runs the same loop (it skips clips that exist) and
# only then declares the arm complete, so branch C cannot score a partial arm.
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}

wait_all_done() {
  local log=$1
  while [ ! -f "$log" ]; do sleep 120; done
  while true; do
    grep -q "ALL DONE" "$log" 2>/dev/null && return 0
    local age=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || echo 0) ))
    [ "$age" -gt 3600 ] && { echo "[$(date +%T)] $log stalled, continuing"; return 0; }
    sleep 120
  done
}

(
  wait_all_done /data/render_g2.log
  echo "[$(date +%T)] g2 pass ended: $(ls out/bench_v5/runs/g2_axis_feedback/*.mp4 2>/dev/null | wc -l) clips"
  CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/bench_v4/bin/render_all.sh wan14b+dittolora \
    scripts/bench_v5/data/instructions_g2_feedback.jsonl out/bench_v5/runs/g2_axis_feedback 7 \
    > /data/render_g2_gapfill.log 2>&1
  echo "[$(date +%T)] g2 after gap-fill: $(ls out/bench_v5/runs/g2_axis_feedback/*.mp4 2>/dev/null | wc -l) clips"
  echo "[$(date +%T)] BRANCH I DONE"
) > /data/queue_i.log 2>&1 &

echo "queue7 launched: branch I (g2 gap-fill)"
