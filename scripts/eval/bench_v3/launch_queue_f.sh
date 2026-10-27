#!/usr/bin/env bash
# Queue F (2026-10-20): skill-v3 arms on all 50 bench samples.
#   ours_skillv3 = short spec written with the category skill card
#   ours_nocard  = same writer, same budget, no card (the ablation)
# Both use the prompts already written by plan_v3.py, so this is render-only.
set -u
cd "$(dirname "$0")/../../.."
source scripts/bench_skill/env.sh

S=out/bench_v3/skills_v3
R=out/bench_v3/runs
nohup python3 scripts/eval/bench_v3/launch_queue.py --gpus "${GPUS:-2,3,4,5,6}" --shards 5 \
  --job "ours_t1_direct:$S/with_card:$R/skillv3_card" \
  --job "ours_t1_direct:$S/no_card:$R/skillv3_nocard" \
  > out/bench_v3/logs/queue_f.log 2>&1 &
echo "queue_f pid=$!"
