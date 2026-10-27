#!/bin/bash
# Fifth stage: assemble the IVEBench table once both external subsets are scored.
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}

(
  while ! grep -q "BRANCH D DONE" /data/queue_d.log 2>/dev/null; do sleep 120; done
  while ! grep -q "BRANCH E DONE" /data/queue_e.log 2>/dev/null; do sleep 120; done
  echo "[$(date +%T)] both IVEBench subsets evaluated"
  $PY bench_external/ivebench/bin/report_ive.py \
    --out bench_external/ivebench/RESULTS_IVEBENCH_TABLE.md
  echo "[$(date +%T)] BRANCH G DONE"
) > /data/queue_g.log 2>&1 &

echo "queue5 launched: branch G (ivebench table)"
