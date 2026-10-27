#!/usr/bin/env bash
# One-shot progress summary for the bench_skill run.
cd "$(dirname "$0")/../.."
OUT=out/bench_skill/run50

echo "=== generations ==="
for a in ours_skill ours_raw ditto_skill bernini_skill; do
  n=$(ls "$OUT/$a/generations"/*.mp4 2>/dev/null | wc -l)
  printf "  %-15s %2d/50\n" "$a" "$n"
done

echo
echo "=== live shards ==="
ps -eo args | grep "[r]un_bench.py" | grep -oP "arm \K\w+" | sort | uniq -c | sed 's/^/  /'

echo
echo "=== status rows by outcome ==="
cat "$OUT"/*/status.shard*.jsonl 2>/dev/null | python3 -c "
import sys, json
from collections import Counter
per_arm = {}
for line in sys.stdin:
    row = json.loads(line)
    arm = row.get('arm', '?')
    key = (row.get('error') or '').split(':')[0] or row.get('status', 'ok')
    per_arm.setdefault(arm, Counter())[key] += 1
for arm in sorted(per_arm):
    parts = ', '.join(f'{k}={v}' for k, v in per_arm[arm].most_common())
    print(f'  {arm:<15} {parts}')
"

echo
echo "=== closed-loop outcomes (skill arms) ==="
cat "$OUT"/*_skill/status.shard*.jsonl 2>/dev/null | python3 -c "
import sys, json
from collections import Counter
per_arm = {}
for line in sys.stdin:
    row = json.loads(line)
    if row.get('error'):
        continue
    per_arm.setdefault(row.get('arm','?'), Counter())[row.get('status','?')] += 1
for arm in sorted(per_arm):
    parts = ', '.join(f'{k}={v}' for k, v in per_arm[arm].most_common())
    print(f'  {arm:<15} {parts}')
"
