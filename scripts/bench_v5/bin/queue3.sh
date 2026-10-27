#!/bin/bash
# Third stage: the point-editing half of the external benchmark
# (IVEBench attribute_editing), on gpu 7 once the g2 loop arm is done.
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
    [ "$age" -gt 1800 ] && { echo "[$(date +%T)] $log stalled, continuing"; return 0; }
    sleep 120
  done
}

(
  # Both this branch and the Bernini ladder (queue6.sh, branch H) start when g2
  # finishes, and both were on gpu 7: two 14B/1.3B jobs would share a card and
  # either OOM or halve each other's speed. Attribute arms wait for branch H.
  while ! grep -q "BRANCH H DONE" /data/queue_h.log 2>/dev/null; do sleep 120; done
  wait_all_done /data/render_g2.log
  echo "[$(date +%T)] starting IVEBench attribute arms"
  for ARM in direct skill; do
    bash bench_external/ivebench/bin/render_ive.sh \
      bench_external/ivebench/ive_attr_${ARM}.jsonl out/bench_v5/runs/ive_attr_${ARM} 7 \
      "$PROJ/bench_external/ivebench/src_mp4_attr" > /data/render_ive_attr_${ARM}.log 2>&1
    echo "[$(date +%T)] ive_attr_${ARM}: $(ls out/bench_v5/runs/ive_attr_${ARM}/*.mp4 2>/dev/null | wc -l) clips"
    $PY bench_external/ivebench/out2frames.py --mp4-dir out/bench_v5/runs/ive_attr_${ARM} \
      --out-dir bench_external/ivebench/frames_attr_${ARM} > /data/ive_attr_frames_${ARM}.log 2>&1
    bash bench_external/ivebench/bin/eval_ive.sh \
      "$PROJ/bench_external/ivebench/frames_attr_${ARM}" attr_${ARM} 7 \
      "$PROJ/bench_external/ivebench/ive_info_attr.json" \
      "$PROJ/bench_external/ivebench/src_frames_attr" > /data/ive_attr_eval_${ARM}.log 2>&1
    echo "[$(date +%T)] ive_attr_${ARM} evaluated"
  done
  echo "[$(date +%T)] BRANCH E DONE"
) > /data/queue_e.log 2>&1 &

echo "queue3 launched: branch E (ivebench attribute editing on gpu 7)"
