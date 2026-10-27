#!/usr/bin/env bash
# Score the bench_v2 planning track once launch_planning.sh has finished.
#
# 1. GT-referenced metrics (deterministic, no judge): all four planning arms plus
#    the expert-instruction upper bound (ours_raw); noop/oracle are added
#    automatically by gt_metrics.py.
# 2. Pairwise judge on the comparisons that answer "does the agent help":
#      agent vs direct, for each intent form   -> the agent's contribution
#      agent vs expert upper bound             -> how much of the gap it closes
#
#   GPU   device for CLIP/DINO (default 0)
# Requires LOW_HIGH_JUDGE_API_KEY in the environment.
set -u
cd "$(dirname "$0")/../../.."
source scripts/bench_skill/env.sh

P=out/bench_v2/planning
UPPER=out/bench_skill/run50_native/ours_raw/generations

echo "== GT-referenced metrics =="
CUDA_VISIBLE_DEVICES=${GPU:-0} python3 scripts/eval/bench_v3/gt_metrics.py \
  --manifest out/bench_v2/manifest.jsonl --hf-cache models/hf_cache/hub \
  --output-dir out/bench_v2/gt_planning \
  --arm "intent_direct=$P/ours_intent_direct/generations" \
  --arm "intent_agent=$P/ours_intent_agent/generations" \
  --arm "generic_direct=$P/ours_generic_direct/generations" \
  --arm "generic_agent=$P/ours_generic_agent/generations" \
  --arm "expert_upper=$UPPER"

# Flat directory of symlinks so compare_pairwise.py can resolve <root>/<arm>/<id>.mp4.
G=out/bench_v2/pairwise_gen
mkdir -p "$G"
for a in intent_direct intent_agent generic_direct generic_agent; do
  ln -sfn "$PWD/$P/ours_$a/generations" "$G/$a"
done
ln -sfn "$PWD/$UPPER" "$G/expert_upper"

pair () { # arm_a arm_b intents_file
  # The judge sees the instruction as context; give it the same intent the arms got.
  python3 - "$3" > "$G/_eval_inputs.$1.$2.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        print(json.dumps({"sample_id": row["sample_id"], "instruction": row["intent"]}, ensure_ascii=False))
PY
  python3 scripts/eval/ditto_quant/compare_pairwise.py \
    --eval-inputs "$G/_eval_inputs.$1.$2.jsonl" --generation-root "$G" \
    --arm-a "$1" --arm-b "$2" --frames 8 --concurrency 4 \
    --output-dir out/bench_v2/pairwise_planning
}

SPEC=scripts/eval/bench_v3/data/intents_specific.jsonl
GEN=scripts/eval/bench_v3/data/intents_generic.jsonl
echo "== pairwise: agent vs direct =="
pair intent_agent  intent_direct  "$SPEC"
pair generic_agent generic_direct "$GEN"
echo "== pairwise: agent vs expert upper bound =="
pair intent_agent  expert_upper   "$SPEC"
pair generic_agent expert_upper   "$GEN"
