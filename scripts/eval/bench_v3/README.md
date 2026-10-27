# bench_v3

Measures whether the agent (planner / reference translator in front of the Wan2.1-VACE-14B
"ours" edit model) beats feeding the user's request straight to the edit model, at three
levels of user input, judged by two LLM judges plus GT metrics where a target exists.
Run everything from the repo root.

## Tracks and tiers

| track | rows | source | tiers |
|---|---|---|---|
| `track_a` | 37 low-high-cost samples | `build_track_a.py` | `t1` one-line intent; `t2_same` intent + mid frame of this sample's own `target_high`; `t2_cross` intent + mid frame of the tag-closest *other* sample; `t3` expert instruction |
| `style_track` (B_opuscar) | 48 = 6 OPUSCAR looks x 8 low-high sources | `build_style_track.py` | `t1` "make it look like <film> (<year>)"; `t2` OPUSCAR still; `t3` LLM-written look-only spec; no GT |
| `ditto_track` (B_ditto) | 13 Ditto-1M samples of top50 | `build_style_track.py` | `t1` bench_v2 intent; `t2` mid frame of `target_high`; `t3` original instruction |

Track files live in `data/{track_a,style_track,ditto_track}.jsonl`.

## Arms (`scripts/bench_skill/run_bench.py` ARMS)

All bench_v3 arms are `wan14b+ours`; only the prompt path and the VACE reference differ.

| arm | mode | varies |
|---|---|---|
| `ours_t1_direct` / `ours_t3_direct` | raw | instruction file rendered as-is (t1 intent / t3 spec) |
| `ours_t1_agent` | raw | instruction = intent + precomputed planner plan (`*_agent` input) |
| `ours_t2_direct_ref` | raw + `vace_reference` | intent as-is, first reference image given to VACE |
| `ours_t2_agent` | translate | agent turns intent + reference into a text instruction; no VACE ref |
| `ours_t2_agent_ref` | translate + `vace_reference` | translated text **and** VACE reference |
| `bernini_t1_direct` / `bernini_t1_agent` | raw (bernini1.3b) | cross-backbone check of the t1 prompts |

`eval_config.json` names arms per tier (`t1_agent`, `t2_direct_ref`, ...) and maps them to
generation dirs; some are reused bench_v2 / bench_skill renders (e.g. `track_a.t1`
`t1_agent` = `bench_v2/ablation/ours_abl_plan_only`, `t3_expert` = `bench_skill/run50_native/ours_raw`).
`noop` / `oracle` are the source / target anchors; `adv_recolor` / `adv_swap` are adversarial controls;
`*_s1055` / `*_s2068` are seed repeats. `pairwise` lists one-variable comparisons.

## Pipeline

API steps need `source /root/.low_high_judge.env`; GPU steps need `source scripts/bench_skill/env.sh`.

1. **Track files**
   ```
   python3 scripts/eval/bench_v3/build_track_a.py [--output scripts/eval/bench_v3/data/track_a.jsonl] [--ref-dir out/bench_v3/refs]
   python3 scripts/eval/bench_v3/build_style_track.py [--offline] [--model M] [--reasoning-effort high]
   ```
   `build_style_track` caches LLM calls in `out/bench_v3/style_track/cache/`; `--offline` fails on a cache miss.

2. **Instruction files**, one per (track, tier) -> `<dir>/<track>.<tier>.{instructions,index}.jsonl`
   ```
   python3 scripts/eval/bench_v3/make_instruction_file.py --track scripts/eval/bench_v3/data/track_a.jsonl \
       --tier t2_same --output-dir out/bench_v3/inputs
   ```
   `--tier` is one of `t1 t2_same t2_cross` (track A), `t2` (track B), `t3`. `--intent-file F.jsonl`
   (`{sample_id, intent}`) overrides the intent and writes stem `<track>.<tier>_<--name>` (default `_override`).

3. **Precompute agent work** (API only, so renders never call the planner and every arm sees the same text)
   ```
   python3 scripts/eval/bench_v3/precompute_plans.py --input-stem out/bench_v3/inputs/track_a.t1 \
       --output-root out/bench_v3/runs/track_a.t1 [--concurrency 6]
   python3 scripts/eval/bench_v3/precompute_translations.py \
       --instruction-file out/bench_v3/inputs/track_a.t2_same.instructions.jsonl \
       --output-root out/bench_v3/runs/track_a.t2_same [--concurrency 8]
   ```
   - `precompute_plans` (t1 agent) writes `<root>/plans/<id>.json` and `<input-stem>_agent.{instructions,index}.jsonl`.
     Plans here are **memoryless** (`memory_context={}`, no memory store) and use the plan_only composition,
     fitted to `utils.prompt_budget.token_limit()` tokens.
   - `precompute_translations` (t2) fills `<root>/translations/`, the cache the translate arms read.
     `--output-root` **must equal** the `OUTPUT_ROOT` of the matching launch_queue job. Exits 1 on any error.

4. **Render** across GPUs (`run_bench.py --max-attempts 1`, sharded; resumable)
   ```
   python3 scripts/eval/bench_v3/launch_queue.py --gpus 0,1,2,3 --shards 4 \
       --job ours_t2_agent:out/bench_v3/inputs/track_a.t2_same:out/bench_v3/runs/track_a.t2_same \
       --job ours_t1_agent:out/bench_v3/inputs/track_a.t1_agent:out/bench_v3/runs/track_a.t1_seed1055:1055 \
       [--log-dir out/bench_v3/logs]
   ```
   `--job` is `ARM:INPUT_STEM:OUTPUT_ROOT[:SEED]` (seed defaults to run_bench's 42). Output videos land in
   `OUTPUT_ROOT/<arm>/generations/<id>.mp4`. Exits non-zero if any job failed; rerun to resume.

5. **Judge tasks** -> `out/bench_v3/judge_tasks/<track>.<tier>.jsonl`
   ```
   python3 scripts/eval/bench_v3/build_judge_tasks.py [--offline] [--model gpt-5.6-luna] [--concurrency 8] [--output-dir DIR]
   ```
   Checklists are fixed before any output exists and are tier-specific: track A t3 = transformation_tags;
   track A t1 / t2_* and track B t1 / t2 are generated by `--model` (default gpt-5.6-luna, a different family
   from the gemini agent); track B t1 / t2 use look-only prompts; track B t3 uses the track's style checklist.
   Cached under `<output-dir>/cache/`; `--offline` fails on a cache miss.

6. **Adversarial controls** -> `out/bench_v3/eval/_adv_{recolor,swap}/` (no flags)
   ```
   python3 scripts/eval/bench_v3/make_adversarial.py
   ```

7. **Evaluate** the plan in `eval_config.json`
   ```
   python3 scripts/eval/bench_v3/run_eval.py [--tiers 'track_a.*'] [--skip-gt] [--skip-checklist] [--skip-pairwise] \
       [--judges gemini-3.8-flash,gpt-5.6-luna] [--concurrency 10] [--gpu 0]
   ```
   - Arms whose dir does not cover every task sample are skipped, so a tier can be evaluated mid-render.
   - Each stage runs once per judge in `eval_config.json` (gemini-3.8-flash primary, gpt-5.6-luna second).
     Outputs: `out/bench_v3/eval/<tier>.gt/`, `<tier>.<stage>/` (primary) and `<tier>.<stage>@<model>/`
     (others); logs in `out/bench_v3/eval/logs/`.
   - Resume is by `input_digest` (task as the judge sees it + video path/size/mtime): a stored row is reused only
     if its digest matches, so a rebuilt task file or re-rendered video is judged again.

8. **Aggregate** -> `out/bench_v3/RESULTS.md` + `results.json` (no flags)
   ```
   python3 scripts/eval/bench_v3/aggregate.py
   ```
   Per arm: fidelity / preservation / amount changed; per pair: paired deltas with sign tests and each judge's
   pairwise verdicts plus a **consensus** column (samples where every judge picks the same side).
   Draw conclusions from the consensus columns only.

## Side tools

- `make_label_kit.py build --tasks T --arm-a NAME=DIR --arm-b NAME=DIR [-n 30] [--seed 0] --output-dir D [--absolute-paths]`
  writes a blind HTML labelling page; `make_label_kit.py score --key D/answer_key.json --answers answers.json --judge PAIRWISE.jsonl`
  scores a pairwise judge against human labels.
- `migrate_input_digest.py` (no flags): one-off; stamps `input_digest` on judge rows written before
  it existed (only rows that provably match current inputs). Not part of a normal run.
- `run_checklist.py` / `run_pairwise.py`: the per-stage judges `run_eval.py` calls (`--tasks`, `--arm`/`--arm-a`/`--arm-b`
  `name=dir`, `--output-dir`, `--frames 4`, `--concurrency 6`, `--limit`); `judges.py` holds the shared logic.
- `queues/launch_queue_d.sh`: past-run record of queue D (2026-10-18: track_a t1_direct seeds 1055/2068,
  style/ditto t1 at seed 1055; `BERNINI=1` for the bernini cross-backbone jobs). Kept for provenance; rerunning resumes.

## Conventions

- Input stems: `out/bench_v3/inputs/<track>.<tier>[_agent]`; run roots: `out/bench_v3/runs/<track>.<tier>[_seedN]`.
- Judge outputs: `out/bench_v3/eval/<tier>.<stage>[@model]/`, append-only jsonl; the latest row per job wins.
- All LLM-built artefacts (style specs, checklists, plans, translations) are cached on disk; with a warm cache
  reruns make no API calls.
- Adding an arm or pair = edit `eval_config.json`, then rerun `run_eval.py` + `aggregate.py`.

## Tests

```
python3 -m pytest -q scripts/eval/bench_v3/tests
```
