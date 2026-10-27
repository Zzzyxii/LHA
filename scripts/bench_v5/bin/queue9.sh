#!/bin/bash
# Ninth stage: IVEBench attribute/point-editing arms on gpu 6, then the expert
# baseline on the same card.
#
# Order was changed after the style subset came back: the point-editing arms are the
# half of the task the style subset cannot speak to (定点编辑), and the role
# direct_expert was going to play -- "is our win only against a weak baseline?" --
# is already covered by pub_Ditto, the released Ditto output. So attribute arms run
# first and expert runs after whatever is left of the card.
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}

(
  for ARM in direct skill; do
    bash bench_external/ivebench/bin/render_ive.sh \
      bench_external/ivebench/ive_attr_${ARM}.jsonl out/bench_v5/runs/ive_attr_${ARM} 6 \
      "$PROJ/bench_external/ivebench/src_mp4_attr" > /data/render_ive_attr_${ARM}.log 2>&1
    echo "[$(date +%T)] ive_attr_${ARM}: $(ls out/bench_v5/runs/ive_attr_${ARM}/*.mp4 2>/dev/null | wc -l)"
    $PY bench_external/ivebench/out2frames.py --mp4-dir out/bench_v5/runs/ive_attr_${ARM} \
      --out-dir bench_external/ivebench/frames_attr_${ARM} > /data/ive_attr_frames_${ARM}.log 2>&1
    bash bench_external/ivebench/bin/eval_ive.sh \
      "$PROJ/bench_external/ivebench/frames_attr_${ARM}" attr_${ARM} 6 \
      "$PROJ/bench_external/ivebench/ive_info_attr.json" \
      "$PROJ/bench_external/ivebench/src_frames_attr" > /data/ive_attr_eval_${ARM}.log 2>&1
    echo "[$(date +%T)] ive_attr_${ARM} evaluated"
  done
  echo "[$(date +%T)] BRANCH K DONE"
  CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash scripts/bench_v4/bin/render_all.sh wan14b+dittolora \
    scripts/bench_v4/data/instructions_expert.jsonl out/bench_v5/runs/direct_expert_v5 6 \
    > /data/render_expert.log 2>&1
  echo "[$(date +%T)] expert: $(ls out/bench_v5/runs/direct_expert_v5/*.mp4 2>/dev/null | wc -l)"
  echo "[$(date +%T)] BRANCH K2 DONE"
) > /data/queue_k.log 2>&1 &

echo "queue9 launched: branch K (attribute arms + eval), then expert"
