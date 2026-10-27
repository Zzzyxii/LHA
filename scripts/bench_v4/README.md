# bench_v4

Benchmark for the low-cost -> high-cost video edit task, rebuilt from bench_v3
after two of its properties made its headline numbers unusable:

1. the headline metric was an LLM judge, and two judges from different families
   disagreed on which arm won (LESSONS.md: gemini 20/9 for the agent, luna 20/7
   for the direct arm; kappa 0.11). The number was measuring the judge;
2. the agent's prompt came from an LLM writer, so the arm could not be
   re-derived from its inputs.

bench_v4 fixes both: **the primary metrics are LLM-free** (pure functions of
source frames, output frames and the instruction) and **the agent arm is a
deterministic compiler** (skill_compile.py), so every number can be recomputed.

## Task

Input: a low-cost source clip plus a one-line user intent
(`data/instructions_intent.jsonl`, 50 samples from
`reports/top50_video_bundle_20260725_nodup`).
Output: an edited clip that looks like a higher-cost production of the same shot.

The target clips (`videos/target_high`) are **not** an input to any arm. They are
used only as the `oracle` anchor and, optionally, for `target_gain`.

## Arms

| arm | prompt | purpose |
|---|---|---|
| `noop` | source copied | floor; must score exactly 0 on every gain |
| `oracle` | target_high copied | ceiling; a metric on which oracle <= noop is broken |
| `direct_intent` | the one-line intent, verbatim | **the baseline to beat** (single-pass edit) |
| `skill_v4` | `skill_compile.py` output | method, deterministic (no frames; ablated at n=11) |
| `grounded_v4` | `skill_compile_grounded.py` output | method, with frame grounding (local Qwen3-VL-8B) |

All rendered arms use the same variant (`wan14b+dittolora`), the same seed, the
same frame count (follows the source) and the same negative prompt. Only the
prompt text differs, which is the whole point.

## Metrics (score_automated.py)

| metric | compared against | meaning |
|---|---|---|
| `change` | source | how far the output moved from the low-cost input |
| `preservation` | source | 1 - change, reported separately |
| `clip_t_gain` | instruction | did the edit move toward what was asked |
| `aes_gain` | source | LAION aesthetic head (VBench's `aesthetic_quality`) |
| `zsq_gain` | source | CLIP zero-shot "high-budget vs amateur" margin |
| `musiq_gain` | source | MUSIQ-SPAQ. **FAILED the anchor: oracle -1.40 (5/10 wins). Diagnostic only, never headline.** |
| `target_gain` | target_high | cosine to the real high-cost clip, minus the source's |
| `temporal_out` | itself | frame-to-frame coherence |

Never merged into one score: how much changed, whether it went the right way,
and whether it looks better are three different questions (LESSONS.md).

## Metric validation came first

Run the anchors before trusting a metric. On 10 samples:

```
metric           noop mean  oracle mean  oracle>noop   wins
change             -0.0002       0.1201      +0.1202  10/10
aes_gain            0.0000       0.0173      +0.0173   8/10
zsq_gain            0.0000       0.0123      +0.0123   9/10
clip_t_gain         0.0000       0.0097      +0.0097   7/10
musiq_gain          0.0000      -1.4043      -1.4043   5/10  <- rejected
```

## Running

GPU work happens on the taiji pod (see LESSONS.md for the access path).

```bash
# 1. compile the arm's prompts (deterministic, no API)
python3 scripts/bench_v4/skill_compile.py \
  --intents scripts/bench_v4/data/instructions_intent.jsonl \
  --out scripts/bench_v4/data/instructions_skill_v4.jsonl

# 2. render one arm on one GPU (resumable, skips existing outputs)
bash scripts/bench_v4/bin/render_all.sh wan14b+dittolora \
  scripts/bench_v4/data/instructions_skill_v4.jsonl \
  out/bench_v4/runs/skill_v4 7

# 3. score (append-only, resumes by (arm, sample_id))
CUDA_VISIBLE_DEVICES=0 python3 scripts/bench_v4/score_automated.py \
  --arms "noop=out/bench_v4/runs/noop,oracle=out/bench_v4/runs/oracle,direct_intent=out/bench_v4/runs/direct_intent,skill_v4=out/bench_v4/runs/skill_v4" \
  --sources reports/top50_video_bundle_20260725_nodup/videos/source_low \
  --targets reports/top50_video_bundle_20260725_nodup/videos/target_high \
  --instructions scripts/bench_v4/data/instructions_intent.jsonl \
  --out out/bench_v4/scores.jsonl
```

## Why there is a grounded arm

At n=11 the deterministic arm was behind the baseline on `gain_dino` (-0.076 vs
-0.060) while winning the zero-shot quality axis 9/2. The gap matches bench_v3's
own finding: the arm that could see the frames (their VLM planner) scored
`gain_dino` +0.008 where the direct arm scored -0.072, and the biggest measured
failure mode was specifying carriers that are not in frame. The deterministic
compiler cannot see frames, so `skill_compile_grounded.py` runs Qwen3-VL-8B
locally over four frames per clip and writes the prompt from what is actually
there. It also tells the writer to *upgrade* rather than *restyle*: `source_low`
is already a cheap version of the same shot in the same genre, so the target does
not change genre, it raises production value.

## Result

See `out/bench_v4/RESULTS_v4.md`. Short version: on the 37 low-high-cost clips the
grounded arm beats the single-pass baseline on `gain_dino` (+0.0210, CI
[+0.0026, +0.0403], 24/13, p = 0.099), preserves better and changes less; it loses
on the LAION/VBench aesthetic axis, where the larger edit wins. Both arms are still
negative against an oracle of +0.2136, so the gap to the target is mostly open.

## Known gaps

- No LLM judge. The litellm gateway key is out of quota (every model returns
  `budget_error`), so a second-opinion judge is not available; the automated
  metrics carry the result and the judge is deferred.
- `direct_expert` (expert instruction as prompt) was started and then dropped to
  free a GPU for the method arm; re-run it when a GPU is spare.
- `musiq` needs `musiq_spaq_ckpt-358bb6af.pth` in `~/.cache/torch/hub/pyiqa`
  (copied from `junjhzhu/vbench_cache`); the default `musiq` variant downloads
  koniq weights and hangs offline.
