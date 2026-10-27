#!/usr/bin/env bash
# Chain the remaining skill arms after queue F, one after another on the same GPUs:
#   decisive -> execution-only -> plan+card compile
# WAIT_PID=<queue_f pid> bash scripts/eval/bench_v3/launch_queue_chain.sh
set -u
cd "$(dirname "$0")/../../.."
source scripts/bench_skill/env.sh
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
S=out/bench_v3/skills_v3
R=out/bench_v3/runs
nohup python3 scripts/eval/bench_v3/launch_queue.py --gpus "${GPUS:-2,3,4,5,6}" --shards 5 \
  --job "ours_t1_direct:$S/with_card_decisive:$R/skillv3_decisive" \
  --job "ours_t1_direct:$S/with_card_exec:$R/skillv3_exec" \
  --job "ours_t1_direct:$S/plan_card:$R/skillv3_plancard" \
  > out/bench_v3/logs/queue_chain.log 2>&1 &
echo "queue_chain pid=$!"
