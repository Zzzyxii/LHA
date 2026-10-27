#!/usr/bin/env bash
# Re-score every arm with the original paper metrics, using the original scripts
# unchanged (frame counts and settings copied from val30_full_metrics/run_core_array.sbatch
# and the stage2 pickscore/jedi jobs). Methods are spread over GPUs 0..N-1.
# Run build_orig_metrics_manifest.py first. JEDi and the reward judge run separately.
set -u
cd "$(dirname "$0")/../../.."
ROOT=$PWD
PATCH=$ROOT/out/hf_eval_bundle/eval_bundles/patches
OUT=$ROOT/out/bench_v2/orig_metrics
MANIFEST=$OUT/manifest_all.jsonl
NGPU=${NGPU:-8}
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export HF_HOME=$ROOT/models/hf_cache TORCH_HOME=$ROOT/models/torch_cache XDG_CACHE_HOME=$ROOT/models/cache
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs"

mapfile -t METHODS < <(python3 -c "
import json,sys
seen=[]
for l in open('$MANIFEST'):
    m=json.loads(l)['method']
    if m not in seen: seen.append(m)
print('\n'.join(seen))")

score_method () { # method gpu
  local m=$1 gpu=$2 dir=$OUT/$1
  mkdir -p "$dir"
  python3 -c "
import json
rows=[l for l in open('$MANIFEST') if json.loads(l)['method']=='$m']
open('$dir/manifest.jsonl','w').write(''.join(rows))"
  local n; n=$(wc -l < "$dir/manifest.jsonl")
  export CUDA_VISIBLE_DEVICES=$gpu
  test -s "$dir/traditional.csv" || python3 "$PATCH/reference/benchmarks/edit/run_traditional_metrics.py" \
    --manifest "$dir/manifest.jsonl" --output "$dir/traditional.csv" \
    --frames-per-video 8 --resize 224 --pyiqa-metrics musiq,niqe --lpips --device cuda
  test -s "$dir/clip/summary.json" || python3 scripts/eval/ditto_quant/evaluate.py \
    --manifest "$dir/manifest.jsonl" --output-dir "$dir/clip" \
    --clip-model openai/clip-vit-large-patch14 --device cuda --frames-per-video 16 --clip-batch-size 8
  test -s "$dir/supp.summary.json" || python3 "$PATCH/scripts/eval/val30_full_metrics/evaluate_supp_clip_dino.py" \
    --manifest "$dir/manifest.jsonl" --output "$dir/supp.csv" --summary "$dir/supp.summary.json"
  test -s "$dir/laion.summary.json" || python3 "$PATCH/scripts/eval/val30_full_metrics/evaluate_laion_aesthetic.py" \
    --manifest "$dir/manifest.jsonl" --output "$dir/laion.csv" --summary "$dir/laion.summary.json"
  test -s "$dir/pickscore/$m/summary.json" || python3 "$PATCH/scripts/eval/pickscore/evaluate_manifest_group.py" \
    --manifest "$dir/manifest.jsonl" --output-dir "$dir/pickscore" --method "$m" \
    --expected-count "$n" --frames 16 --device cuda
  echo "DONE $m $(date +%T)"
}

for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in "${!METHODS[@]}"; do
      if [ $((i % NGPU)) -eq "$g" ]; then score_method "${METHODS[$i]}" "$g"; fi
    done
  ) > "$OUT/logs/gpu$g.log" 2>&1 &
done
wait
echo "ALL DONE $(date +%T)"
