# bench_v5

The low->high benchmark, rebuilt on axis profiles. Supersedes bench_v4's headline.

## Why

bench_v4's headline was `gain_dino` (DINO cosine to `target_high` minus the
source's). Every arm scored below `noop` on it, so the only way to gain was to edit
less -- and the per-sample change in `gain_dino` correlated -0.733 with the change
in how much the output moved. The next recipe in line ("do not swap or add any
garment, object or background element") was that pressure turning into a design.
Meanwhile the axes that were reachable (instruction adherence, zero-shot
production value) already had the baseline sitting *at* the oracle, so they could
not separate two arms either. Both ends were unusable.

bench_v5 replaces one scalar with a vector: the edit is measured along validated
axes and compared to the target's own movement along the same axes.

## Pipeline

```
axes.py             16 candidate axes, each a CLIP antonym margin
extract_cache.py    CLIP axis vector + image embedding per clip (cache dir)
validate_axes.py    keep axes whose low->high shift is consistent across pairs
make_profile.py     freeze a calibration split, estimate the global profile
score_profile.py    per-sample delivery / alignment / waste / magnitude / change / adherence
report_profile.py   per-arm means, anchors, paired tests split by provenance
axis_breakdown.py   per-axis cv vs target: which knobs each arm under-delivers
compile_axis_v5.py prior | feedback prompt rewrite from the measured residual
```

## Metrics

| metric | definition | noop | oracle |
|---|---|---|---|
| `delivery` | (cv . tp_hat) / \|tp\| -- fraction of the target's change delivered | 0 | 1 |
| `alignment` | cos(cv, tp) -- direction only | 0 | 1 |
| `waste` | \|cv - proj_tp cv\| / \|tp\| -- off-target edit | 0 | 0 |
| `magnitude` | \|cv\| / \|tp\| | 0 | 1 |
| `change` | 1 - cos(clip_img(out), clip_img(src)) | 0 | -- |
| `adherence` | cos(out, intent) - cos(src, intent) | 0 | -- |

`cv` = output axis vector - source axis vector; `tp` = target - source.
Reported separately, never merged (LESSONS.md). `delivery` is the headline because
it is the only one that cannot be improved by doing nothing.

## Leakage rules

* No arm prompt is built from `target_high`.
* The prior used by `compile_axis_v5.py` is a *task-level* profile estimated on a
  frozen 13-clip calibration split (`data/split.json`), stratified inside
  `low_high_cost` only. Split-half cosine against its complement: 0.966.
* `feedback` mode uses `profile - what this clip's round-0 render actually moved`,
  which needs only (source, output).
* The target is used for evaluation only.

## Running

```bash
# features (on the GPU pod, ~40 s for 300 clips)
bin/pod 'cd <outt> && CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python -u \
  scripts/bench_v5/extract_cache.py --jobs scripts/bench_v5/data/jobs_v4arms.jsonl \
  --cache-dir scripts/bench_v5/data/cache_pod --workers 4'

python3 scripts/bench_v5/validate_axes.py --cache scripts/bench_v5/data/cache_pod \
  --out scripts/bench_v5/data/axis_validation.json
python3 scripts/bench_v5/make_profile.py --cache-dir scripts/bench_v5/data/cache_pod \
  --prov low_high_cost --out scripts/bench_v5/data/axis_profile_calib.json \
  --split-out scripts/bench_v5/data/split.json

# score + report (score needs the CLIP backend, so run it on the pod)
bin/pod 'cd <outt> && CUDA_VISIBLE_DEVICES=6 /opt/venv/bin/python -u \
  scripts/bench_v5/score_profile.py --cache-dir scripts/bench_v5/data/cache_pod \
  --axis-validation scripts/bench_v5/data/axis_validation.json \
  --arms "noop=noop50,oracle=oracle50,direct_intent=direct_intent,grounded_v4=grounded_v4" \
  --out scripts/bench_v5/data/scores_v5.jsonl'
python3 scripts/bench_v5/report_profile.py --scores scripts/bench_v5/data/scores_v5.jsonl
python3 scripts/bench_v5/axis_breakdown.py --scores scripts/bench_v5/data/scores_v5.jsonl
```

Render with `scripts/bench_v4/bin/render_all.sh wan14b+dittolora <prompts.jsonl>
<out_dir> <gpu>`; it is resumable and skips clips that already exist.

## Results

`out/bench_v5/RESULTS_v5.md`.
