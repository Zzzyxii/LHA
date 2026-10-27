#!/bin/bash
# Score every arm on the VBench consistency dimensions (subject / background),
# which are the ones this method is supposed to win: both reward keeping the
# shot's content, and --load_ckpt_from_local is required or VBench tries wget.
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
for arm in noop50 oracle50 direct_intent grounded_v4; do
  bash "$PROJ/scripts/bench_v4/bin/run_vbench.sh" \
    "$PROJ/out/bench_v4/runs/$arm" \
    "$PROJ/scripts/bench_v4/data/instructions_intent.jsonl" \
    "$PROJ/out/bench_v4/vbench_cons_${arm}.json" 0 \
    subject_consistency background_consistency \
    > "$PROJ/out/bench_v4/vbench_cons_${arm}.log" 2>&1
  echo "[$(date +%T)] $arm done" >> "$PROJ/out/bench_v4/vbench_cons_progress.log"
done
echo ALLDONE >> "$PROJ/out/bench_v4/vbench_cons_progress.log"
