#!/usr/bin/env bash
# Queue D (2026-10-18): generation-only follow-ups, no API calls (agent prompts are
# precomputed in the *_agent instruction files).
#   P1  track_a t1_direct at seeds 1055/2068: makes the headline agent-vs-direct
#       comparison seed-balanced (agent and t3 already have three seeds).
#   P3  style_track / ditto_track agent and direct at seed 1055.
# Bernini cross-backbone runs (bernini_t1_*) need diffusers + veomni in this env;
# they are a separate queue (set BERNINI=1) so a missing dependency cannot fail
# the Ours jobs.
# Outputs go to new run dirs only. Resumable: rerunning skips finished samples.
set -u
cd "$(dirname "$0")/../../../.."
source scripts/bench_skill/env.sh

I=out/bench_v3/inputs
R=out/bench_v3/runs
JOBS=(
  --job "ours_t1_direct:$I/track_a.t1:$R/track_a.t1_seed1055:1055"
  --job "ours_t1_direct:$I/track_a.t1:$R/track_a.t1_seed2068:2068"
  --job "ours_t1_direct:$I/style_track.t1:$R/style_track.t1_seed1055:1055"
  --job "ours_t1_agent:$I/style_track.t1_agent:$R/style_track.t1_seed1055:1055"
  --job "ours_t1_direct:$I/ditto_track.t1:$R/ditto_track.t1_seed1055:1055"
  --job "ours_t1_agent:$I/ditto_track.t1_agent:$R/ditto_track.t1_seed1055:1055"
)
if [ "${BERNINI:-0}" = 1 ]; then
  JOBS=(
    --job "bernini_t1_direct:$I/track_a.t1:$R/track_a.t1_bernini_v3"
    --job "bernini_t1_agent:$I/fresh/track_a.t1_agent:$R/track_a.t1_bernini_v3"
  )
fi
LOG=out/bench_v3/logs/queue_d.log
[ "${BERNINI:-0}" = 1 ] && LOG=out/bench_v3/logs/queue_d_bernini.log
nohup python3 scripts/eval/bench_v3/launch_queue.py --gpus "${GPUS:-0,1,2,3,4,5,6,7}" --shards 4 "${JOBS[@]}" \
  > "$LOG" 2>&1 &
echo "queue_d pid=$!"
