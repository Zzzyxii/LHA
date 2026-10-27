#!/bin/bash
# usage: render_retry.sh <variant> <instruction_file> <out_dir> <gpu> [max_passes]
# Render every sample, retrying the ones that failed, until all exist or the pass
# cap is hit. Needed because the GPU box is shared: a render that starts while
# another tenant is holding the card dies with OOM, and a one-pass loop would
# silently leave holes in the arm.
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
VARIANT=$1
INSTR=$2
OUT=$3
GPU=$4
MAX=${5:-12}
mkdir -p "$OUT"
wait_free() {
  # The box is shared. Starting a 14B render on a card another tenant is filling
  # dies with OOM and leaves a hole in the arm, so wait for real headroom first.
  need=${NEED_MB:-64000}
  for _ in $(seq 1 90); do
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits -i "$GPU" | tr ',' ' ')
    free=$((total - used))
    if [ "$free" -ge "$need" ]; then return 0; fi
    echo "[$(date +%T)] gpu$GPU only ${free}MB free (need $need), waiting"
    sleep 120
  done
  return 1
}

for pass in $(seq 1 "$MAX"); do
  wait_free || { echo "[$(date +%T)] gpu$GPU never freed up"; break; }
  missing=$("$PROJ/scripts/bench_v4/bin/missing.py" "$INSTR" "$OUT")
  if [ -z "$missing" ]; then
    echo "[$(date +%T)] pass $pass: nothing missing, done"
    break
  fi
  n=$(printf "%s\n" "$missing" | wc -l)
  echo "[$(date +%T)] pass $pass: $n missing"
  printf "%s\n" "$missing" > "$OUT/.todo.pass$pass"
  bash "$PROJ/scripts/bench_v4/bin/render_all.sh" "$VARIANT" "$OUT/.todo.pass$pass" "$OUT" "$GPU"
  sleep 60
done
echo "[$(date +%T)] render_retry finished after $MAX passes max"
