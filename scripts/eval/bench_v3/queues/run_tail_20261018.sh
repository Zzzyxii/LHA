#!/usr/bin/env bash
# Tail of the 2026-10-18 render plan, chained after queue D:
#  1. GPUs 1-7 help finish track_a t1_agent_fresh (GPU 0 is already walking it
#     forward; --reverse meets it in the middle and run_bench skips finished
#     samples, so no sample renders twice).
#  2. Bernini cross-backbone t1 arms (queues/launch_queue_d.sh with BERNINI=1).
set -u
cd "$(dirname "$0")/../../../.."
source scripts/bench_skill/env.sh

until grep -q "ALL DONE" out/bench_v3/logs/queue_d.log 2>/dev/null; do sleep 60; done
for g in 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python3 scripts/bench_skill/run_bench.py --arm ours_t1_agent \
    --sample-index out/bench_v3/inputs/fresh/track_a.t1_agent.index.jsonl \
    --instruction-file out/bench_v3/inputs/fresh/track_a.t1_agent.instructions.jsonl \
    --output-root out/bench_v3/runs/track_a.t1_fresh --max-attempts 1 \
    --shard-index $((g - 1)) --shard-count 7 --reverse \
    >> "out/bench_v3/logs/track_a.t1_fresh.helper$g.log" 2>&1 &
done
wait
GPUS=1,2,3,4,5,6,7 BERNINI=1 bash scripts/eval/bench_v3/queues/launch_queue_d.sh
