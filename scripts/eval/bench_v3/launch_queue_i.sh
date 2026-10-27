#!/usr/bin/env bash
# Queue I (corrected): render the grounded skill arms. The earlier v3 renders were
# made from prompts written without frames (the extractor silently returned none),
# so all four arms are re-rendered from the v3b prompt set.
#   grounded : writer must list visible objects first, then edit only those
#   card     : full skill card
#   plancard : planner plan compressed with the card
#   nocard   : same writer, no card (ablation)
set -u
cd "$(dirname "$0")/../../.."
source scripts/bench_skill/env.sh
S=out/bench_v3/skills_v3
R=out/bench_v3/runs
nohup python3 scripts/eval/bench_v3/launch_queue.py --gpus "${GPUS:-2,3,4,5,6}" --shards 5 \
  --job "ours_t1_direct:$S/with_card_grounded:$R/skillv3b_grounded" \
  --job "ours_t1_direct:$S/with_card:$R/skillv3b_card" \
  --job "ours_t1_direct:$S/plan_card:$R/skillv3b_plancard" \
  --job "ours_t1_direct:$S/no_card:$R/skillv3b_nocard" \
  > out/bench_v3/logs/queue_i.log 2>&1 &
echo "queue_i pid=$!"
