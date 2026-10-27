# LESSONS

Reusable lessons from the low-high benchmark work. Each entry: what went wrong or
what worked, and the rule to follow next time.

## Evaluation

- **Check what a metric compares against before trusting it.** The original top50
  table had 11 metrics and none compared an output to its own target: CLIP-T and
  PickScore compare to the instruction, CLIP-F/DINO-F to adjacent frames,
  Src-Edit/LPIPS to the source, LAION/MUSIQ/NIQE/Reward to nothing, JEDi only as a
  set distribution. Rule: for every metric, write down "compared against what".
- **Always score `noop` (source as output) and `oracle` (target as output).** They
  expose metrics that reward doing nothing (noop beat most published methods on
  JEDi) and catch pipeline bugs (noop gain must be exactly 0).
- **Report change amount next to fidelity.** "Did not break it" and "got it right"
  look the same on a target-similarity metric unless you also report how much the
  output moved away from the source (1 − src similarity).
- **Fidelity and aesthetics disagree; never merge them.** Judges that only see the
  output prefer bigger edits; target-referenced metrics penalise edits in the wrong
  direction. Report both, separately.
- **Judges must see the source.** A judge that sees only output frames cannot tell
  "upgraded the costume" from "replaced the person".
- **Pairwise judges: ask both orders, disagreement = tie.** One-direction position
  bias reached 0.80.
- **A self-trained reward model is not an independent referee.** The project's own
  reward ranked its own model first; two independent judges did not reproduce it.
  Treat such a gap as a reward-hacking signal before using that reward to optimise.
- **Split mixed benchmarks by source before concluding.** top50 is 37 low-high-cost
  + 13 Ditto style-transfer samples; the agent helped on one and not the other.
- **Freeze compact mixed-task slices from metadata, never scores.** The IVEBench
  30-clip reporting slice round-robins style subcategories and samples color,
  scale, and position edits evenly. Keep task-wise deltas separate; a pooled
  average can hide a trade-off and does not create a valid universal score.
- **A shared generation setting can still invalidate a task.** The inherited
  negative prompt explicitly banned anime/cartoon while several style-edit prompts
  requested anime. Matching that setting across arms was not enough: the new
  controlled comparison uses one neutral artifact-only negative prompt for every
  method and records it in each render's metadata.
- **Every arm in a paired report must come from the same run and frozen manifest.**
  A new VACE/Bernini result combined with an old direct/skill CSV is not a paired
  four-arm experiment even if filenames line up. Pass all evaluation roots
  explicitly, reject manifest drift, and fail on stale/ambiguous evaluator outputs.
- **Calibrate reproduced metrics against the published table first.** Recomputing
  the published videos and matching CLIP-F/PickScore/JEDi to the table to the third
  decimal is what makes every other number in the re-score trustworthy.
- **Re-run every headline comparison with a second judge from another model
  family.** On bench_v3 T1 (one-line intent), gemini-3.8-flash said the agent won
  fidelity 20/9 and quality 25/5; gpt-5.6-luna said the direct arm won both (20/7,
  21/6). Both agreed on preservation (27/3, 24/0), and both agreed on every T3
  question. Vague requirements make fidelity/quality verdicts judge-dependent;
  only claims that survive both judges (or human labels) go into conclusions.
- **Generate checklists with a different model family from the agent under test**,
  and freeze them before any arm is judged, so the agent is not graded against its
  own phrasing.
- **Do not let the agent's model family judge the agent.** The bench_v3 planner and
  translator are gemini-3.8-flash; the gemini judge favoured the agent arms on
  fidelity/quality while gpt-5.6-luna favoured the direct arms (t1 quality kappa
  0.11). Where no agent was involved (Bernini vs expert) the two judges agreed. Use
  a judge panel without the agent's family, and report consensus only.
- **Check that the oracle tops every arm before trusting a checklist.** On bench_v3
  the target clip itself scored *below* some arms on fidelity and below almost every
  arm on preservation, because preservation rewards editing less. A checklist on
  which the oracle is not the ceiling measures something other than "reached the
  target".
- **A reference image cut from the target leaks the answer.** t2_same (reference =
  a target frame) reached 41% of the oracle's gain and its translation beat the
  expert instruction; with a reference from another clip (t2_cross) the same arm
  was below noop. Use same-shot references only as a leakage ceiling.
- **Choosing the arm configuration on the test set inflates it.** plan_only was
  picked after an ablation on the same 37 samples; report it as exploratory or
  confirm on held-out samples.

## Experiments

- **Diff the full render configuration between arms, not just the prompt.** The
  agent arm silently passed the source's first frame as a VACE reference image in
  35/50 samples (memory-driven `reference_from_low_video`); the direct arm never
  did. Check `implicit_controls_applied` in `edit.response.json`.
- **Ablate by deleting sections from the exact prompt that was sent**, then verify
  the full-prompt arm reproduces the original outputs (it did, bit-for-bit, on the
  samples without the hidden control). That proves the render path is deterministic
  and the ablation isolates what it claims.
- **Know which pipeline stage a stored field represents.** `planner.response.json`
  `compiled_edit_prompt` is the *composed* prompt (plan + skill blocks), not the raw
  plan; the raw plan is in `planner.call_004_final.response.txt`. Mixing them
  produced a false "plan truncated in 50/50" claim.
- **The Wan text encoder reads 512 tokens and silently drops the rest.** Budget
  prompts with the model's own tokenizer; never let truncation decide what is lost.
- **Do not parse fields that were truncated for storage.** `raw_vlm_response` is
  `json.dumps(...)[:12000]` and is invalid JSON for long planner outputs; the plan
  text now has its own field (`VideoPlan.planner_plan`).
- **Arms that should differ in one variable must share every other input.** The two
  T2 translate arms (with / without VACE reference) originally translated
  separately and could get different instructions; translations are now
  precomputed once per output root and shared.
- **Reasoning models count thinking in `max_completion_tokens`.** Reserve headroom
  and double the budget on an empty, length-truncated reply.

## Data and paths

- **Archived manifests point at the original host.** `sample_index.json`
  `source_video` is a `/home/l/liyj/...` path and bundle videos are dangling
  symlinks; use the unpacked release archive and pass `--source-dir`.
- **Count statuses by field, not by substring.** `grep -c '"error"'` also matches
  the empty `"error": ""` field present in every record.

## Process

- **Resolve remote-tool paths from the actual mount before declaring a compute
  blocker.** The Taiji skill was mounted at
  `/apdcephfs_szcf/share_304169095/juliayxzhu/Tiny/skills`, not under
  `.../hunyuan/juliayxzhu/Tiny`; checking the literal user path alone would
  incorrectly suggest that remote GPU work was unavailable.
- **Never write a new run into an output directory you did not create in this
  session.** A Bernini run appeared in `runs/track_a.t1_bernini/` from a process not
  started here, using older prompts; launching mine into the same root would have
  skipped its finished samples and mixed two treatments under one arm name. New
  runs get a fresh root (`track_a.t1_bernini_v3`).
- **Key stored results on their inputs, not only on their ids.** Judge rows are now
  stamped with `input_digest` (task text + video identity); resume re-judges a row
  whose inputs changed instead of keeping a stale verdict. Migrations of old rows
  must compare paths after `resolve()` (old rows stored repo-relative paths).

- **A transient "classifier error" on a long compound command usually clears when
  the same work is split into smaller commands.** Do not route around it in other
  ways.

## Infrastructure (2026-10-25)

- **The litellm gateway key is out of quota; every model returns `budget_error` (-30012).**
  `/root/.low_high_judge.env` still resolves and `/v1/models` lists 221 models, but
  any chat call fails, so planner / critic / judge stages have no LLM. Check the
  gateway with one cheap call *before* designing a stage that needs one, and keep
  the primary metrics LLM-free so the result does not depend on it.
- **The GPU pod can only reach `mirrors.tencent.com`.** huggingface.co, github.com,
  pypi.org, hf-mirror.com and modelscope.cn all time out (curl exit 000). Anything
  that downloads HF/GitHub weights hangs instead of failing: `pyiqa.create_metric("musiq")`
  sat on a koniq download forever. Either pre-place the weights in the cache or
  pick the offline-capable variant (`musiq-spaq`, weights in
  `junjhzhu/vbench_cache/pyiqa_model`).
- **Reusable local assets on this filesystem** (all reachable from the pod, all
  offline): `junjhzhu/VBench` (VBench + VBench-2.0 source) with every checkpoint in
  `junjhzhu/vbench_cache`; a working env at `junjhzhu/venvs/vbench-eval`
  (py3.11 / torch 2.4.1 / openai-clip); CLIP ViT-L/14 at `~/.cache/clip`; DINO
  ViT-B/16 at `junjhzhu/vbench_cache/dino_model` (repo + `dino_vitbase16_pretrain.pth`);
  LAION aesthetic head at `junjhzhu/vbench_cache/aesthetic_model/emb_reader`.
- **GPU access path**: the 8xH20 box is taiji instance
  `8b0fba19a0be2b9d01a0c881e92706fd` (task `basic_train_juliayxzhu_20260922174533_e407eb10`,
  single pod `ts-8b0fba19a0be2b9d01a0c881e92706fd-launcher`, host network, node
  28.45.1.187). Reach it with
  `python3 scripts/connect_mcp.py call exec_hunyuan_train_instance_command` from
  `Tiny/skills/ref_skills/taiji-basic-platform-skills/sub-skills/instance-skill`.
  Only GPUs 6 and 7 are free (0-5 hold other people's vLLM-Omni services at
  42-66 GB); do not schedule onto 0-5. `nohup ... &` returns immediately, so long
  renders can be launched and polled. Only `/data` survives a restart.
- **Render cost to plan around**: Wan2.1-VACE-14B + Ditto LoRA at 832x480, 20 steps,
  frames following the source, is ~7.7 min and ~50 GB VRAM per clip on one H20
  (already bf16). 50 clips is ~6.4 GPU-hours, so two arms on two cards is a
  ~6.5 h wall-clock job. Budget arms accordingly before launching.
- **`wan14b` and any `backend="diffsynth"` variant cannot run here**: they need
  `DiffSynth-Studio/` at the repo root, which is an empty submodule. `backend="ditto"`
  variants (`wan14b+dittolora`, `wan14b+ours`, `bernini1.3b`) use `Ditto/diffsynth`
  and work.

## Metrics (2026-10-25)

- **Two quality metrics ranked the *source* above the *target* and had to be dropped.**
  MUSIQ-SPAQ: oracle `musiq_gain` -1.40 (5/10 wins). VBench `imaging_quality`:
  noop 0.696 vs oracle 0.664. Both reward technical cleanliness (sharpness, low
  noise); a cinematic high-cost grade lowers both. "High-cost" is an aesthetic
  property, so validate every candidate metric on the anchors before using it.
- **Metrics that survived the anchor test** (10 samples, oracle vs noop):
  `change` 10/10, `zsq_gain` (CLIP zero-shot high-budget vs amateur margin) 9/10,
  VBench `aesthetic_quality` (noop 0.5411 -> oracle 0.5705) 8/10, `clip_t_gain` 7/10.
  Effect sizes are small, so these need the whole 50 samples and paired tests.
- **CLIP image similarity cannot express "closer to the target" on this data.**
  Source and target sit at 0.93 cosine, so a CLIP `cos(out,tgt) - cos(src,tgt)` is
  bounded at +0.07 and reads *negative* for every real edit (-0.13, -0.04, -0.02 on
  the first three samples) - it degenerates into "editing moves you away". DINO puts
  the same pair at ~0.73, which is what gives `gain_dino` real headroom; that is why
  bench_v2/v3 used DINO for this number.
- **VBench on your own videos needs `--mode custom_input --prompt_file`**, and the
  prompt file is JSON keyed by *absolute* video paths (a plain-text file gives
  `JSONDecodeError`; relative names give an "prompts not found" assertion). Default
  mode filters against VBench's own prompt suite and silently scores nothing, which
  surfaces as `ZeroDivisionError: float division by zero`.
- **ffmpeg image2 numbering starts at 1.** Writing 0-based padded frames into the
  same directory collided with an existing name (`SameFileError`). Keep ffmpeg output
  in a `raw/` subdirectory and write the canonical names afterwards.

## Reproducibility (2026-10-25)

- **An LLM-written prompt makes the arm unreproducible.** Two runs of the same
  writer give different prompts, and when the judges disagreed there was no way to
  tell whether the prompt or the judge moved the number. bench_v4's agent arm is a
  deterministic compiler (`scripts/bench_v4/skill_compile.py`), so the arm can be
  re-derived from (intent, card) and diffed.
- **Card actions are advice to an LLM writer, not prompt text.** They are fragments
  ("upgrade to period headdress: ...", "period hat, styled hair with ..."); emitting
  them bare produced subjectless prompts. Attach the carrier explicitly
  ("On the headwear, upgrade to ...").

## Full bench_v4 run (2026-10-25, 50 clips x 2 arms)

- **Split by data provenance before reading the headline.** top50 is 37
  `low_high_cost` + 13 `ditto` clips. The grounded arm beat the single-pass
  baseline by +0.0210 `gain_dino` on the 37 (24/13) and *lost* by -0.0215 on the 13
  (5/8); pooled over 50 the effect halves to +0.0099, p 0.32. The split is recorded
  in `top50_sources.jsonl` before any result exists, so reporting the low-high
  subset is legitimate -- reporting the pooled number as the headline is not.
- **Compute the anchors on the same sample set as the arms.** On 10 samples VBench
  `aesthetic_quality` gave oracle 0.5705, which was *below* the 50-clip baseline
  score of 0.5804 and made the metric look broken. On all 50 the oracle is 0.5826
  and the ordering is clean (oracle 0.5826 > direct 0.5804 > grounded 0.5569 >
  noop 0.5383). A 10-sample anchor is not comparable to a 50-sample arm.
- **Killing a render job means killing the driver loop, not just the python
  child.** `render_all.sh` respawns the next sample immediately, so killing only
  `render_one.py` left the loop holding the GPU and the next job OOMed three times
  before I noticed. Match on the output directory, kill the parent, verify the
  card is at 0 MiB, then launch.
- **Frame grounding is what moves `gain_dino`; more prompt text does not.** The
  frame-blind deterministic compiler was neutral (-0.0025 at n=15) and the grounded
  writer (Qwen3-VL-8B over 4 frames) gained +0.0210 on the low-high subset. But
  both arms are still negative against an oracle of +0.2136, so grounding closes
  roughly 10% of the gap: it is the right lever, not the whole answer.
- **Qwen3-VL-8B-Thinking will not reliably emit JSON.** Its reasoning arrives as
  prose full of braces, so "take the last JSON object" still fails; ask for plain
  text and take whatever follows `</think>`. ~25 s per clip with 4 frames.
- **Put guardrails before detail in a fixed word budget.** Emitting carriers first
  left 47/50 compiled prompts with no preservation clause and 0/50 with the
  off-carrier rule. Order: global clause, guardrails, then carriers, then the
  decisive clause; carriers are the expendable part.

## The contamination bug (worst mistake of the run)

- **A Thinking model that runs out of `max_new_tokens` before closing `<think>`
  returns reasoning, not an answer -- and it looks like a prompt.** 26/50 of the
  first grounded arm were 90 words of prose ("Got it, let's break down the user's
  request...") sent to the edit model as the edit instruction. It was only caught
  when a later run printed one; nothing in the pipeline complained.
- **Validate the generated artefact, not just that generation returned.** Any
  writer output needs an assertion that it is the thing you asked for. Here it is
  a marker list (`looks_like_reasoning`) plus a retry at double the token budget.
  A prompt that is 90 words, contains "Step 1" and no imperative clause is not a
  prompt.
- **The contamination produced a plausible-looking win.** The contaminated arm
  beat the baseline by +0.0210, and the mechanism is visible in the data: per-sample
  `Δgain_dino` correlates -0.733 with `Δchange`, and rambling prompts make this
  model under-edit. Any metric that rewards "did not move much" will reward broken
  input. Before believing a win, check that the winning arm's inputs are what you
  think they are.

## Benchmark design (2026-10-25, bench_v5)

- **A headline metric on which every arm scores below `noop` is broken, and it will
  quietly turn the method into "edit less".** `gain_dino` (DINO cosine to
  `target_high` minus the source's) put every arm negative (direct_intent -0.036,
  grounded_v4 -0.026) against an oracle of +0.2136, and the only way to gain was to
  change less: the per-sample change in `gain_dino` correlated -0.733 with the change
  in DINO change. The next recipe being built ("do not swap or add any garment,
  object or background element") was that pressure expressing itself as a design.
  If the only winning move on your metric is to do nothing, the metric is the bug.
- **Check that the ceiling is reachable *and* that the floor is not already at it.**
  On the axes that were reachable (instruction adherence, zero-shot production
  value) the single-pass baseline already sat at the oracle (adherence 0.0221 vs
  0.0193; zsq 0.0122 vs 0.0123), so no arm could ever be separated there either.
  A metric needs headroom at both ends.
- **Replace "closer to the target" with "did it push the right knobs".** bench_v5
  measures the edit as a vector of validated axis scores (`scripts/bench_v5/axes.py`,
  CLIP antonym margins differenced against the source) and compares it to the
  target's own vector: `delivery` = (edit . target_dir)/|target| (noop 0, oracle 1),
  `alignment` = cos(edit, target), `waste` = orthogonal part. `delivery` cannot be
  won by editing less (a smaller edit in the right direction scores lower) and cannot
  be won by editing more (the orthogonal part shows up in `waste`).
- **Validate each axis before admitting it.** 16 candidate axes, kept the 13 whose
  low->high shift had |paired t| >= 2 across the 50 pairs. Three died:
  `makeup_grooming` (t=-0.14), `depth_of_field` (t=+0.77), `craft` (t=-0.86).
  Record the rejects in code (`axes.REJECTED_AXES`), not in a commit message.
- **Check how much of the result rests on one axis.** With all 16 axes the agent arm
  beat the baseline on `delivery` +0.145 (p=0.034); dropping `costume_ornament` cut
  it to +0.070 (p=0.35). A headline that survives only because of one axis is not a
  headline yet. `waste` was robust across every axis set (p=3e-04 to 8e-07).
- **The biggest low->high effects in this data are content, not grade.** On the 37
  `low_high_cost` pairs the target's own axis profile is led by `costume_ornament`
  (+0.047) and `headwear` (+0.024), both larger than `color_grade` (+0.019) or
  `material_detail` (+0.011). A recipe that forbids swapping garments is therefore
  wrong for this dataset by construction, whatever it does to the score.
- **Estimate a task-level prior on a frozen calibration split, and stratify it.**
  Taking every third clip across *all* provenances pulled the prior's `skin_finish`
  and `headwear` terms past their own magnitude because the `ditto` clips have a
  different transformation. Restricted to `low_high_cost`, split-half cosine between
  the calibration prior and its complement is 0.966.

## Engineering (2026-10-25)

- **`torch.set_num_threads()` inside a library's lazy `_load()` overrides whatever
  the pool worker set.** `featurize._load()` called
  `torch.set_num_threads(min(32, os.cpu_count()))`, so 5 workers each ran 32
  intra-op threads: 160 threads on a 32-core box and ~6x slowdown. Thread counts
  come from the environment only (`LH_THREADS`), never from `os.cpu_count()`.
- **Subsetting an axis vector: index into the *full* name list, not the kept one.**
  `idx = [names.index(n) for n in names]` (kept list) silently returned 0..k-1 and
  shifted every axis by one; the profile printed `jewelry` where `headwear` belonged.
  Build the index from `axes.axis_names()`, and sanity-check by comparing a
  subsample's profile against its complement.
- **`pkill -f <pattern>` kills the shell that is running it** when the pattern
  appears in that shell's own command line (the command contained `extract_cache`
  and exited 143). Split the literal (`pkill -f "extract""_cache"`) or kill by PID.
- **A background `nohup ... &` started inside an exec_command call does not survive
  that call.** It was gone by the next command with no error in the log. Long jobs
  run in the foreground of a session that is then polled, or on the pod with
  `nohup` (where it does survive).
- **The GPU pod is ~100x faster than the contended CPU box for this workload.**
  300 clips x 8 frames of CLIP: >60 min on the workstation (load average ~200,
  shared with other users) vs 40 s on one H20. Extract features on the pod; it
  mounts the same filesystem, so no transfer is needed.
- **open_clip and openai's `clip` spell the same checkpoint differently**
  (`ViT-L-14` vs `ViT-L/14`). The pod has `clip` with the weights cached and cannot
  reach pypi, so `featurize.py` tries open_clip and falls back to `clip`.

## Compute access (2026-10-25)

- **The taiji instance-skill lives at `/apdcephfs_szcf/share_304169095/juliayxzhu/Tiny/...`,
  without the `hunyuan/` path segment.** `scripts/bench_v5/bin/pod '<cmd>'` wraps
  `connect_mcp.py call exec_hunyuan_train_instance_command`. Credentials are in
  `~/.config/taiji/credentials.json`. The gateway returns 504 if the command takes
  more than about a minute, so launch with `nohup` and poll the log.
- **GPUs 6 and 7 only.** 0-5 hold 42-66 GB of other people's vLLM services.
- **Render cost scales with frames.** ~7.7 min/clip at the source's ~81 frames;
  capping IVEBench clips at 32 frames should roughly halve it.

- **`render_all.sh` prints `rc=0` for samples that actually died of CUDA OOM.**
  A run can report 30/50 finished while only 6 mp4s exist; the traceback is in the
  log a few hundred lines above the `END` line and `$?` is 0 anyway. Count the
  *outputs*, never the `rc` lines, and re-run the loop to fill the gaps (it skips
  clips whose output already exists). Set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before relaunching.
- **Another agent is running renders on the same pod and took GPU 6 while I was
  using it.** Two 45-50 GB jobs on one 95 GB card OOMed 24 of my samples. Check
  `nvidia-smi --query-gpu=index,memory.used` before trusting a long run, and re-run
  the loop afterwards rather than assuming a clean pass.

## Contamination, second round (2026-10-25) -- the bug that invalidated a whole result table

- **A generated artefact must be validated at the point it is created, not only
  before it is rendered.** The `grounded_v4` arm was rendered with 26/50 prompts
  that were VLM reasoning text ("Got it, let's break down the user's request...");
  every bench_v5 number that compares `grounded_v4` (RESULTS_v5.md §4, the axis
  breakdown, the robustness table) is therefore void. Worse, the *next* arms were
  compiled from that same file: `axis_v5` and `loop_r2` inherited 27/50
  contaminated prompts and were already rendering when it was caught. Compiling on
  top of a dirty base propagates the bug silently, so `compile_axis_v5.py` now
  calls `looks_contaminated()` on the base prompt *and* on its own output and
  aborts instead of writing.
- **The two clean grounded files differ in one important way.**
  `instructions_grounded_clean.jsonl` still has 2 contaminated rows (0477, 0218);
  `instructions_grounded_clean48.jsonl` is the 48 that pass the audit. Use the
  audited file, and always run `audit_prompts.py` on the exact file a render reads.
- **`audit_prompts.py` globs produce false positives on non-prompt files.**
  Pointing it at `scripts/bench_v5/data/*.jsonl` flagged `scores_*.jsonl` and
  `top50_sources.jsonl` as "empty". Name the instruction files explicitly.
- **Check the render inputs of an arm before trusting any result computed from it.**
  `out/bench_v4/runs/<arm>/*.meta.json` stores the exact instruction that was
  sent; counting contamination there (26/50) took one command and settled whether
  the arm could be used at all.

## Metric design (2026-10-25)

- **`delivery` can be inflated by overshooting along the target's direction**
  (cv = 3 * tp scores 3.0), so it must be reported with a term that penalises
  doing too much. `profile_error = sqrt((delivery - 1)^2 + waste^2)` is derived
  from the same vectors (noop 1, oracle 0) and is the only scalar of the set that
  neither under- nor over-editing can win.

## Edit-backend configuration (2026-10-25)

- **`denoising_strength` is a diffsynth-`vt2v` parameter; the ditto backend ignores
  it.** `EditVariant.denoising_strength = 0.7` looks like it should damp every arm,
  but `_render_with_ditto()` never passes it, so `wan14b+dittolora` renders at the
  pipeline default. Do not reason about arm strength from that field; read the
  backend function. (Ditto's own `inference/infer_ditto.py` defaults are
  20 steps, cfg 1.0, 832x480 -- the same as ours, so the arms are Ditto at
  published settings.)

## Reusable assets found in-tree (2026-10-25)

- **`归档_extract/` is the released comparison bundle**: 50 clips with
  `source_low/`, `target_high/` and `methods/<16 published methods>/`. It is
  byte-identical to `reports/top50_video_bundle_.../videos/source_low`, so the
  published outputs can be scored with bench_v5 metrics at zero render cost.
- **The released method outputs do not share the source's frame count** (most are
  17 @20 fps, Ditto/VACE 73, InsV2V 49) but the clips are near-static (source
  frame-to-frame CLIP similarity 0.997, first-to-last 0.957), so frame-averaged
  features are not sensitive to which span was exported.
- **Scoring 785 clips on the pod: CPU ~1.6 clips/min (8 workers, contended box),
  GPU ~4 clips/s.** Feature extraction belongs on the GPU; it costs ~3 GB and can
  share a card with a render.

## Access (2026-10-25)

- **Direct SSH to the pod is blocked by a network ACL, not by sshd.** The pod runs
  `sshd` on port **36000** with `PermitRootLogin yes`, and a second instance can be
  started on any port with its own `AuthorizedKeysFile`. From the DevCloud
  workstation the TCP connect succeeds but the handshake gets
  `HTTP/1.1 403 ... x-tai-errcode: acl.14001` ("访问 IDC 环境需要通过 igate 开通访问权限"):
  DevCloud -> IDC needs an igate grant, which cannot be self-served. Use the MCP
  `exec_hunyuan_train_instance_command` path (`scripts/bench_v5/bin/pod`) with
  `nohup` + poll.

## Negative result: the intent text cannot predict which axes to move (2026-10-25)

- **Reweighting the axis prior by CLIP text margins of the intent sentence does not
  predict the target's direction better than the unweighted prior.**
  `scripts/bench_v5/intent_goal.py` scores each axis as
  `cos(intent, high_caption) - cos(intent, low_caption)` and gates/adds it onto the
  task prior; on the 13-clip calibration split the plain prior already won
  (mean cos(goal, tp) 0.716 vs 0.714 for the best variant) and on the 37-clip
  complement the best variant gained +0.006 (0.610 vs 0.604). A one-line intent
  ("Make it look like ...") carries almost no axis-level information.
- **The consequence is a hard ceiling on any prior-aiming arm.** Mean
  cos(prior, per-sample tp) is 0.60 on the held-out clips, so an arm that only aims
  at the dataset average cannot exceed roughly that fraction of the target's change
  however well it executes. Clip-specific information has to come from somewhere
  else: the source frames (grounding) or the render itself (feedback). This is the
  argument for the loop, and it is worth stating in the paper.
- **Negative results belong in the repo, not in a chat.** `intent_goal.py --eval`
  prints both splits so the variant-selection / reporting split stays visible.

## Audit false positives (2026-10-25)

- **`audit_prompts.py`'s 110-word limit is a rule for *generated* prompts, not for
  the expert baseline.** `instructions_expert.jsonl` trips it on 21/50 rows because
  those instructions are deliberately detailed. The word limit exists to catch a
  writer that ran past its budget; do not run the audit on the expert file, and do
  not "fix" it by shortening the baseline.

## Shell gotchas that cost real GPU hours (2026-10-25)

- **`cd X && A & B &` only background `A`.** `&&` binds tighter than `&`, so
  `cd X && nohup A & nohup B &` runs `A` in `X` and `B` in the *original* directory.
  One arm of a two-GPU launch died with "No such file or directory" and the card sat
  idle for half an hour before it was noticed. Put `cd X;` on its own line, or use a
  subshell `( cd X && ... ) &`.
- **A missing log file is not a stalled log file.** A wait loop of the form
  `[ $(date +%s) - $(stat -c %Y "$log") -gt 1800 ]` treats a log that does not exist
  yet as infinitely old (`stat` fails, `|| echo 0`), so the stage fires immediately
  on an empty output directory and then never runs again. Wait for the file to exist
  first (see `wait_all_done` in `scripts/bench_v5/bin/queue2.sh`).
- **`pkill -f queue.sh` kills the pod shell running it**, because the pattern is in
  that shell's own command line (exit 143). Kill by PID from `ps -eo pid,args`.

## Environment (2026-10-25)

- **This workstation can reach huggingface.co and github.com; the GPU pod cannot.**
  Download benchmark assets here and render there -- they mount the same
  `/apdcephfs_szcf` filesystem, so no transfer step is needed.

## Numbers worth keeping (2026-10-25)

- **The axis metrics are not noise-limited.** Split-half (even vs odd frames)
  correlation of the paired arm difference is 0.986-0.999 for every arm, i.e.
  full-frame reliability >= 0.99. Adding frames per clip would not tighten a single
  p-value; only more clips would.
- **`profile_error` is not a "do less" metric.** For a synthetic `cv = a * tp` it is
  `|a - 1|` (unique optimum at the exact target change), and within an arm the
  correlation with how much the clip changed is *negative* for the good arms
  (pub_Ditto -0.448, pub_ours_full -0.418): the clips they changed more are the
  clips they scored better on.

## Tests (2026-10-25)

- **`scripts/bench_v5/test_metrics.py` pins the metric curve on synthetic vectors**
  (noop, oracle, six scalings `cv = a*tp`, a purely orthogonal edit, the
  error-as-hypotenuse identity, and a degenerate target). The noop/oracle anchors
  in a results table are not enough: they cannot catch a sign error or a missing
  sqrt, and a metric that rewarded under-editing would still put noop at 1 and
  oracle at 0.
- **Writing the test corrected a wrong expectation of mine, not a bug in the
  code.** I expected a purely orthogonal edit of size |tp| to score 1.0 (the noop
  value); it scores sqrt(2). That is correct and desirable -- an edit entirely in
  the wrong direction is worse than no edit -- and it is the property that stops an
  arm from winning by editing hard in the wrong direction.

## Second base model: Bernini (2026-10-26)

- **"The agent beats the base model" has to be shown on more than one base model**,
  or it is a claim about one checkpoint. The ladder is repeated on
  **Bernini-R-1.3B** (`bernini1.3b`), the other released base model available here.
- **Bernini's default frame count makes it 6x more expensive than its published
  protocol.** `render_one.py` follows the source (81 frames) unless told otherwise;
  the released Bernini videos in `归档_extract` are 17 frames. At 81 frames a clip
  took >20 min, at 17 frames 3.3 min. Use `scripts/bench_v5/bin/render_all_frames.sh
  ... 17` and compare within-model.
- **The Bernini backend needs the ffprobe shim on PATH.** It runs
  `third_party/Bernini/infer_single_gpu.py` as a subprocess and the pod has no
  ffprobe; `render_all.sh` puts `scripts/bench_skill/bin` (a PyAV-backed ffprobe
  shim) on PATH, so a hand-run command that omits it fails with
  `FileNotFoundError: 'ffprobe'`.
- **Two queue branches waiting on the same milestone will collide on the same GPU.**
  Branch E (attribute arms) and branch H (Bernini ladder) both started when g2
  finished and both used gpu 7. Branch E now waits for "BRANCH H DONE" first.

## The shared filesystem can fill up with no warning (2026-10-26)

- /apdcephfs_szcf/share_304169095 hit 102T/102T, 0 bytes available. Every write --
  renders, reports, even `echo x > file` -- fails with "Disk quota exceeded". 14 of 40
  IVEBench clips died this way and looked like ordinary render failures. Check df
  before concluding a render bug is a code bug.
- Freeing our own regenerable intermediates (~6.5 GB: IVEBench frame dumps, the
  downloaded source mp4s, the pre-cut source frame folders) did not restore
  writability: the volume is saturated by other tenants.
- The pod has its own disk: `df -h /data` showed 19T free while the shared mount was
  full. When the shared filesystem is unwritable, render to /data (it survives pod
  restarts) and copy back when space appears. Scripts must not hard-code the project
  path for output directories.
- Deleting prior benchmarks' renders is not worth it: 1.7 GB will not change a
  saturated 102T volume, and old renders are evidence, not temp files.

## Scorecard provenance (2026-10-26)

- **A score file's metadata can list an arm that has no score on a given sample.**
  Count actual per-row arm fields after filtering to the frozen manifest; do not
  infer coverage from the metadata arm list or render-directory totals.
- **Use the score file whose axis-validation metadata matches the reported metric.**
  `scores_final.jsonl` uses 13 axes, while `scores_final_lowhigh.jsonl` uses 12.
  A report generated from the latter must not be presented as the 13-axis
  `RESULTS_MANIFEST30.md` result.
- **An audited prompt exclusion is not automatically a render gap.**
  `heldout1000_0477` is
  absent from the audited `instructions_grounded_clean48.jsonl`; adding it back
  with a prompt from another arm would change the treatment. Keep paired n=29
  unless the original grounding process can produce a validated prompt and the
  arm protocol explicitly allows that regeneration.

## Information parity is part of the benchmark (2026-10-26)

- **The archived `pub_Ditto` result is answer-aware, not a same-information
  baseline.** `out/REPORT_ALL.md` §0.1 states that the detailed expert
  instruction was written by a VLM after looking at both the low and high videos.
  It therefore contains target-side details such as garments, colors, objects and
  backgrounds that are unavailable to the agent's low+one-line-intent input.
- **The same checkpoint does not make two arms comparable if prompt information
  differs.** `direct_intent` and `g0_grounded_clean` both use `wan14b+dittolora`,
  seed 42, 832x480 and 20 steps. Their comparison is causal. `pub_Ditto` is an
  upper reference for a full instruction, not a fair control.
- **Reports must separate information conditions.** The generated
  `INFORMATION_AUDIT.md` records prompt provenance, protocol metadata and
  coverage. Future leaderboards must label one-line-intent, source-grounded and
  answer-aware expert arms instead of sorting them as one neutral ranking.
