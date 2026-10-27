# Shared environment for bench_skill / bench_v2 runs. Source it, don't execute it.
# PROJ is derived from this file's location so the repo can live anywhere.
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH=$PROJ/src:$PROJ/Ditto
export PATH=$PROJ/scripts/bench_skill/bin:$PATH
# bin/ffmpeg is a symlink to imageio-ffmpeg's bundled binary, whose absolute path
# differs per machine. Re-point it whenever it does not resolve (e.g. after the
# repo is copied elsewhere). bin/ffprobe is a PyAV-backed shim and needs nothing.
if [ ! -x "$PROJ/scripts/bench_skill/bin/ffmpeg" ]; then
  _ffmpeg_exe="$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null)"
  if [ -n "$_ffmpeg_exe" ]; then
    ln -sfn "$_ffmpeg_exe" "$PROJ/scripts/bench_skill/bin/ffmpeg"
  else
    echo "env.sh: imageio-ffmpeg not installed; pip install imageio-ffmpeg" >&2
  fi
  unset _ffmpeg_exe
fi
export LOW_HIGH_AGENT_KEEP_EDIT_ON_GPU=1
export LOW_HIGH_AGENT_KEEP_VLM_ON_GPU=0
export LOW_HIGH_AGENT_USE_LITELLM_VLM=1
export LOW_HIGH_JUDGE_BASE_URL=${LOW_HIGH_JUDGE_BASE_URL:-http://litellm-sgp.woa.com}

# Planner (the agent that writes the edit plan) and critic (closed-loop evaluate +
# reflect) share this model. A role can be overridden independently with
# LOW_HIGH_AGENT_LITELLM_<ROLE>_MODEL / _REASONING_EFFORT (ROLE = PLANNER or
# CRITIC); see vlm.resolve_vision_model. gpt-6.1-sol accepts reasoning_effort
# none/low/medium/high/xhigh (not max). Switched to gemini-3.8-flash on
# 2026-10-17 for the planning track; judge numbers from gpt-6.1-sol or
# gemini-3.1-pro-preview are not comparable with these.
export LOW_HIGH_AGENT_LITELLM_MODEL=gemini-3.8-flash
export LOW_HIGH_AGENT_LITELLM_REASONING_EFFORT=high
# Offline judges (evaluate_aesthetic_reward_*.py, compare_pairwise.py).
export LOW_HIGH_JUDGE_MODEL=gemini-3.8-flash
export LOW_HIGH_JUDGE_REASONING_EFFORT=high

# Frame count and fps follow each source video so edits preserve clip duration.
unset LOW_HIGH_BENCH_NUM_FRAMES
unset LOW_HIGH_BENCH_FPS
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# LOW_HIGH_JUDGE_API_KEY is deliberately not set here: export it in your shell.
