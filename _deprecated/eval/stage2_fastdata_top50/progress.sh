#!/bin/bash
set -euo pipefail

RUN=${RUN:-/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727}
USER_NAME=${USER_NAME:-liyj}

echo "=== jobs ==="
squeue -u "$USER_NAME" -o "%.18i %.9P %.32j %.8T %.10M %.6D %R" | grep -E 'JOBID|stage2-top50|val30-full|wrap' || true

echo "=== methods ==="
test -f "$RUN/methods.tsv" && wc -l "$RUN/methods.tsv" || true

echo "=== generated videos ==="
if test -d "$RUN/infer"; then
  find "$RUN/infer" -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' '
  find "$RUN/infer" -maxdepth 3 -path '*/generations/*' -type d 2>/dev/null | sort | sed -n '1,30p'
else
  echo 0
fi

echo "=== manifests ==="
if test -d "$RUN/infer"; then
  find "$RUN/infer" -path '*/manifests/quant_manifest.jsonl' -type f 2>/dev/null | sort | while read -r p; do printf "%5s %s\n" "$(wc -l < "$p" | tr -d ' ')" "$p"; done
fi
test -f "$RUN/full_metrics/manifest_all.jsonl" && wc -l "$RUN/full_metrics/manifest_all.jsonl" || true

echo "=== metric outputs ==="
for DIR in "$RUN/full_metrics/results" "$RUN/pickscore" "$RUN/jedi"; do
  test -d "$DIR" && find "$DIR" -type f 2>/dev/null
done | sort | sed -n '1,120p'

echo "=== final score files ==="
ls -lh "$RUN"/stage2_fastdata_top50_scores.* 2>/dev/null || true
