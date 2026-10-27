#!/usr/bin/env bash
# Re-render a controlled 30-clip comparison:
# direct Ditto, skill/compiler Ditto, Wan2.1-VACE-14B base, and Bernini-R-1.3B.
set -euo pipefail

PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
GPU=${1:?usage: run_frozen30_baselines.sh <gpu>}
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}
NEGATIVE_PROMPT="low quality, blurry, flicker, distorted anatomy, extra limbs, duplicate subjects, text, subtitles, watermark"
cd "$PROJ"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to start the benchmark" >&2
  exit 2
fi
FREE_MIB=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
  awk -F, -v gpu="$GPU" '$1 + 0 == gpu {gsub(/[[:space:]]/, "", $2); print $2}')
if [ -z "$FREE_MIB" ] || [ "$FREE_MIB" -lt 55000 ]; then
  echo "GPU $GPU has ${FREE_MIB:-unknown} MiB free; need at least 55000 MiB" >&2
  exit 2
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%S)_$$"
RUN_ROOT="out/bench_v5/runs/frozen30_controlled_${RUN_ID}"
IVE="$PROJ/bench_external/ivebench"
if [ -e "$RUN_ROOT" ]; then
  echo "run output already exists: $RUN_ROOT" >&2
  exit 3
fi
for label in direct skill vace_base bernini; do
  for track in style point; do
    eval_name="frozen30_${RUN_ID}_${label}_${track}"
    if [ -e "$IVE/eval_${eval_name}" ] || [ -e "$IVE/frames_${eval_name}" ]; then
      echo "evaluation output already exists for $eval_name" >&2
      exit 3
    fi
  done
done
mkdir -p "$RUN_ROOT"
python3 "$IVE/score_frozen30.py"

render_and_eval() {
  local label=$1
  local variant=$2
  local style_instructions=$3
  local point_instructions=$4
  local style_name="frozen30_${RUN_ID}_${label}_style"
  local point_name="frozen30_${RUN_ID}_${label}_point"
  local style_out="$RUN_ROOT/${label}/style"
  local point_out="$RUN_ROOT/${label}/point"
  local style_frames="$IVE/frames_${style_name}"
  local point_frames="$IVE/frames_${point_name}"

  bash "$IVE/bin/render_ive.sh" \
    "$IVE/$style_instructions" "$style_out" "$GPU" "$IVE/src_mp4" \
    "$variant" 32 "$NEGATIVE_PROMPT"
  bash "$IVE/bin/render_ive.sh" \
    "$IVE/$point_instructions" "$point_out" "$GPU" \
    "$IVE/src_mp4_attr" "$variant" 32 "$NEGATIVE_PROMPT"

  "$PY" "$IVE/out2frames.py" --mp4-dir "$style_out" --out-dir "$style_frames"
  "$PY" "$IVE/out2frames.py" --mp4-dir "$point_out" --out-dir "$point_frames"
  bash "$IVE/bin/eval_ive.sh" "$style_frames" "$style_name" "$GPU" \
    "$IVE/ive_info.json" "$IVE/src_frames"
  bash "$IVE/bin/eval_ive.sh" "$point_frames" "$point_name" "$GPU" \
    "$IVE/ive_info_attr.json" "$IVE/src_frames_attr"
  test -s "$IVE/eval_${style_name}/summary.json"
  test -s "$IVE/eval_${point_name}/summary.json"
}

render_and_eval "direct" "wan14b+dittolora" \
  "FROZEN30_STYLE.jsonl" "FROZEN30_POINT_EDIT.jsonl"
render_and_eval "skill" "wan14b+dittolora" \
  "FROZEN30_STYLE_SKILL.jsonl" "FROZEN30_POINT_EDIT_SKILL.jsonl"
render_and_eval "vace_base" "wan14b-vace-base" \
  "FROZEN30_STYLE.jsonl" "FROZEN30_POINT_EDIT.jsonl"
render_and_eval "bernini" "bernini1.3b" \
  "FROZEN30_STYLE.jsonl" "FROZEN30_POINT_EDIT.jsonl"

"$PY" "$IVE/score_frozen30.py" \
  --style-arm "direct=eval_frozen30_${RUN_ID}_direct_style" \
  --point-arm "direct=eval_frozen30_${RUN_ID}_direct_point" \
  --style-arm "skill=eval_frozen30_${RUN_ID}_skill_style" \
  --point-arm "skill=eval_frozen30_${RUN_ID}_skill_point" \
  --style-arm "VACE-base-Ditto-runtime=eval_frozen30_${RUN_ID}_vace_base_style" \
  --point-arm "VACE-base-Ditto-runtime=eval_frozen30_${RUN_ID}_vace_base_point" \
  --style-arm "Bernini-R-1.3B=eval_frozen30_${RUN_ID}_bernini_style" \
  --point-arm "Bernini-R-1.3B=eval_frozen30_${RUN_ID}_bernini_point" \
  --report-out "$IVE/FROZEN30_SCORECARD_${RUN_ID}.md"

echo "RUN_ID=$RUN_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "REPORT=$IVE/FROZEN30_SCORECARD_${RUN_ID}.md"
echo "NEGATIVE_PROMPT=$NEGATIVE_PROMPT"
