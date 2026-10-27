#!/bin/bash
# Fourth stage: the supporting tables that go with the final scores.
# Waits for branch C to finish scoring, then regenerates the paired tests, the
# per-axis breakdown and the metric audit from the same scores file, so the three
# documents cannot disagree with each other.
set -uo pipefail
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
cd "$PROJ" || exit 1
PY=${LOW_HIGH_PY:-/opt/venv/bin/python}

(
  while ! grep -q "BRANCH C DONE" /data/queue_c.log 2>/dev/null; do sleep 120; done
  echo "[$(date +%T)] branch C done, building supporting tables"
  $PY scripts/bench_v5/report_profile.py --scores scripts/bench_v5/data/scores_final.jsonl \
    > out/bench_v5/PAIRED_TESTS.txt 2>&1
  $PY scripts/bench_v5/axis_breakdown.py --scores scripts/bench_v5/data/scores_final.jsonl \
    --prov low_high_cost > out/bench_v5/AXIS_BREAKDOWN.txt 2>&1
  $PY scripts/bench_v5/axis_breakdown.py --scores scripts/bench_v5/data/scores_final.jsonl \
    --prov low_high_cost \
    --arms direct_intent,direct_expert_v5,g0_grounded_clean,g1_axis_prior,g2_axis_feedback,pub_Ditto,pub_ours_full \
    > out/bench_v5/AXIS_BREAKDOWN_KEY.txt 2>&1
  $PY scripts/bench_v5/metric_audit.py --scores scripts/bench_v5/data/scores_final.jsonl \
    > out/bench_v5/METRIC_AUDIT.txt 2>&1
  # Robustness: the 13-axis set is validated on all 50 pairs, which is the same set
  # the profile (and therefore the compiler) uses. Re-score under the 12-axis set
  # validated on the 37 reported pairs only and the 9-axis set validated on the
  # calibration split only, so no claim depends on one axis selection.
  ARMS=$($PY scripts/bench_v5/list_arms.py --cache-dir scripts/bench_v5/data/cache_pod)
  for V in lowhigh calib13; do
    CUDA_VISIBLE_DEVICES=7 LH_DEVICE=cuda $PY -u scripts/bench_v5/score_profile.py \
      --cache-dir scripts/bench_v5/data/cache_pod \
      --axis-validation "scripts/bench_v5/data/axis_validation_${V}.json" \
      --arms "$ARMS" --out "scripts/bench_v5/data/scores_final_${V}.jsonl" \
      > "/data/score_final_${V}.log" 2>&1
  done
  {
    echo "# Axis-set robustness (profile_error, low_high_cost)"
    echo
    for V in "" _lowhigh _calib13; do
      echo "## axis set: ${V#_} (scores_final${V}.jsonl)"
      echo
      $PY scripts/bench_v5/leaderboard.py --scores "scripts/bench_v5/data/scores_final${V}.jsonl" \
        --metric profile_error --prov low_high_cost
      echo
    done
  } > out/bench_v5/ROBUSTNESS.txt 2>&1
  # The curated 30-sample list (归档_extract/manifest.jsonl) is reported alongside
  # the full set: it predates every arm in this benchmark, so it is a pre-registered
  # subset rather than a subset chosen after seeing results.
  $PY scripts/bench_v5/final_report.py --scores scripts/bench_v5/data/scores_final.jsonl \
    --prov ALL --ids scripts/bench_v5/data/manifest30_ids.txt \
    --out out/bench_v5/RESULTS_MANIFEST30.md >> /data/queue_f.log 2>&1
  echo "[$(date +%T)] BRANCH F DONE"
) > /data/queue_f.log 2>&1 &

echo "queue4 launched: branch F (paired tests, axis breakdown, metric audit)"
