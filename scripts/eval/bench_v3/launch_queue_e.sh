#!/usr/bin/env bash
# Queue E (2026-10-19): render-only follow-ups, no API calls.
#   1. Memory-free agent (track_a.t1_fresh) at seeds 1055 / 2068, so the T1 main
#      result has the same seed check as the old memory-carrying arm.
#   2. style_track reference tier (t2) at seed 1055 for both arms. The run root
#      carries a copy of the seed-42 translation cache, so both seeds render the
#      identical translated instruction and only the seed differs; a cache miss
#      would raise rather than re-translate.
# Resumable: rerunning skips finished samples.
set -u
cd "$(dirname "$0")/../../.."
source scripts/bench_skill/env.sh

I=out/bench_v3/inputs
R=out/bench_v3/runs
nohup python3 scripts/eval/bench_v3/launch_queue.py --gpus "${GPUS:-0,1,2,3,4,5,6,7}" --shards 4 \
  --job "ours_t1_agent:$I/fresh/track_a.t1_agent:$R/track_a.t1_fresh_seed1055:1055" \
  --job "ours_t1_agent:$I/fresh/track_a.t1_agent:$R/track_a.t1_fresh_seed2068:2068" \
  --job "ours_t2_agent:$I/style_track.t2:$R/style_track.t2_seed1055:1055" \
  --job "ours_t2_direct_ref:$I/style_track.t2:$R/style_track.t2_seed1055:1055" \
  > out/bench_v3/logs/queue_e.log 2>&1 &
echo "queue_e pid=$!"
