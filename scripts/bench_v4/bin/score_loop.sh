#!/bin/bash
# Score whatever has been rendered so far, every 10 minutes, until both arms are complete.
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
export PATH=$PROJ/scripts/bench_skill/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
while true; do
  /opt/venv/bin/python $PROJ/scripts/bench_v4/score_automated.py \
    --arms "direct_intent=$PROJ/out/bench_v4/runs/direct_intent,grounded_v4=$PROJ/out/bench_v4/runs/grounded_v4" \
    --sources $PROJ/reports/top50_video_bundle_20260725_nodup/videos/source_low \
    --targets $PROJ/reports/top50_video_bundle_20260725_nodup/videos/target_high \
    --instructions $PROJ/scripts/bench_v4/data/instructions_intent.jsonl \
    --out $PROJ/out/bench_v4/scores_v2.jsonl >> $PROJ/out/bench_v4/score_loop.log 2>&1
  echo "[$(date +%T)] pass done direct=$(ls $PROJ/out/bench_v4/runs/direct_intent/*.mp4 2>/dev/null | wc -l) skill=$(ls $PROJ/out/bench_v4/runs/grounded_v4/*.mp4 2>/dev/null | wc -l)" >> $PROJ/out/bench_v4/score_loop.log
  if [ "$(ls $PROJ/out/bench_v4/runs/direct_intent/*.mp4 2>/dev/null | wc -l)" -ge 50 ] && [ "$(ls $PROJ/out/bench_v4/runs/grounded_v4/*.mp4 2>/dev/null | wc -l)" -ge 50 ]; then
    echo "[$(date +%T)] both arms complete" >> $PROJ/out/bench_v4/score_loop.log
    break
  fi
  sleep 600
done
