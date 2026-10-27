# IVEBench run (external benchmark)

48 clips of the **style editing** category, 32 frames at 832x480, rendered with the
same `wan14b+dittolora` variant used on the low->high benchmark. Scored with
IVEBench's own metrics; published reference numbers are in `PUBLISHED.md`.

## What can be computed here

Only the metrics whose checkpoints exist on this pod: `overall_semantic_consistency`
(OSC), `phrase_semantic_consistency` (PSC), `semantic_fidelity` (SF),
`temporal_flickering` (TF). The rest need Qwen2.5-VL-72B, GroundingDINO, CoTracker
or AMT weights that cannot be downloaded from here, so **no IVEBench total is
quoted** -- only per-metric comparisons on identical clips.

## Anchor: the source clips scored against themselves

| metric | source | published Ditto (short, all categories) | published VACE |
|---|--|--|--|
| OSC (compliance) | 0.2363 | 0.25 | 0.23 |
| PSC (compliance) | 0.1210 | 0.24 | 0.22 |
| SF (fidelity) | 1.0000 | 0.89 | 0.97 |
| TF (quality) | 0.9668 | 0.97 | 0.98 |

Two things to read into this before the arms are compared:

* **Compliance starts at 0.236 and the best published method reaches 0.25.** On
  style editing the whole compliance headroom is small and VideoCLIP-XL moves
  little; an arm-to-arm difference of +0.01 is a real effect here, unlike on
  fidelity where the ceiling is 1.0 and the interesting question is how little an
  arm gives up.
* **Fidelity is the axis where a bad edit shows up**: the source is at 1.0000 by
  construction and published methods sit at 0.89-0.97, so any arm below ~0.89 is
  changing more than the published baselines do.

## Arms

| arm | prompt |
|---|--|
| `src` | (source clips, anchor above) |
| `direct` | IVEBench's own edit prompt verbatim |
| `skill` | the same instruction through `compile_ive.py --mode skill`: one decisive restyle clause, then the preservation guardrails, nothing else |

Rendered by `scripts/bench_v5/bin/queue.sh` (branch B) and evaluated by branch D.

## Result: style editing (48 clips, paired per clip)

| metric | source | direct | skill | skill - direct | wins | p |
|---|--|--|--|--|--|--|
| OSC (compliance, global) | 0.2363 | 0.2317 | 0.2300 | -0.0016 | 21/48 | 0.43 |
| PSC (compliance, phrase) | 0.1210 | 0.1456 | 0.1511 | **+0.0055** | 31/48 | 0.0084 |
| SF (fidelity) | 1.0000 | 0.8562 | 0.8236 | **-0.0327** | 13/48 | 0.0054 |
| TF (quality) | 0.9668 | 0.9652 | 0.9667 | +0.0015 | 28/48 | 0.13 |

Read honestly: the compiler moves along the compliance-fidelity frontier, it does
not beat it. It buys +0.0055 phrase-level compliance (significant, 31/48 clips) for
-0.033 semantic fidelity (significant, 13/48 clips). On this benchmark a stronger
restyle *is* a less faithful edit, and the two metrics move together; there is no
free lunch to report here, so this is reported as a trade-off rather than a win.

What can also be said: both arms land in the published fidelity band (0.82-0.86
against Ditto 0.89 / VACE 0.97), and global semantic consistency is not the metric
that measures a restyle on these clips (it barely moves from the source anchor at
all, -0.002 to -0.006), which is why PSC is the number to read.

## The point-editing subset (`attribute_editing`, 40 clips)

Style editing only tests restyling. The project's task also covers 定点编辑, so the
external run includes IVEBench's `attribute_editing` category -- change one
subject's colour, size or position and leave everything else alone:

| subcategory | n | example instruction |
|---|--|--|
| color adjustment | 18 | "Change the turquoise waves to black." |
| subject scaling | 12 | "Reduce the size of the golden statue of Lord Murugan." |
| position change | 10 | (relocate a subject within the shot) |

This is the harder test of a skill system, because "be decisive" applied globally is
exactly wrong here: it restyles the whole frame. `compile_ive.py --mode skill_attr`
keeps the instruction verbatim as the only change and spends its budget on
everything else staying identical. Arms: `attr_direct` (instruction verbatim) and
`attr_skill` (`skill_attr`).

Source anchor for this subset (source clips scored against themselves, 40/40 valid):

| metric | attribute anchor | style anchor |
|---|--|--|
| OSC | 0.2446 | 0.2363 |
| PSC | 0.1846 | 0.1210 |
| SF | 1.0000 | 1.0000 |
| TF | 0.9685 | 0.9668 |

PSC (the phrase-level check against the edited attribute) starts much higher here
than on style clips, so it is the metric with the most headroom for a point edit;
OSC barely moves for a localised change, and for these clips the discriminative
question is how much fidelity an arm gives up, not how much compliance it gains.

Selected by `build_attr_subset.py` (evenly spaced within subcategory, no seed).
Rendered on gpu 7 after the `g2` arm (`scripts/bench_v5/bin/queue3.sh`).

Scoring:

```bash
bash bench_external/ivebench/bin/eval_ive.sh <frames_dir> <arm_name> <gpu>
```

Frames come from `out2frames.py --mp4-dir <run_dir> --out-dir <frames_dir>`.

## Attribute / point editing: the compiler wins (40 clips, paired)

| metric | direct | skill_attr | skill - direct | wins | p |
|---|--|--|--|--|--|
| PSC (compliance, phrase) | 0.1891 | **0.1924** | **+0.0033** | 30/40 | **0.011** |
| OSC (compliance, global) | 0.2389 | 0.2397 | +0.0008 | 24/40 | 0.58 |
| SF (fidelity) | 0.9156 | 0.9172 | +0.0016 | 24/40 | 0.77 |
| TF (quality) | 0.9692 | **0.9720** | +0.0027 | 24/40 | 0.013 |

This is the result the style subset could not give: on point edits the compiler
raises phrase-level compliance **without paying fidelity** (+0.0033 PSC, p=0.011,
with SF unchanged at +0.0016, p=0.77) and slightly improves temporal quality. Compare
with style editing, where the same kind of gain cost -0.033 SF: a point edit is
supposed to change one thing and leave the rest alone, and `skill_attr` (instruction
verbatim + "only that subject, everything else identical") is exactly that, so the
two metrics stop pulling against each other.

Against the source anchor (unchanged clips), `direct` costs -0.084 SF and gains only
+0.0045 PSC (p=0.13) -- so the edit itself buys almost nothing; `skill_attr` is what
makes it buy something.

Evaluated with a copy of `ivebench_code` on the pod's local disk: IVEBench writes its
log file into its own code directory, and with the shared filesystem full that made
OSC/PSC/SF return the -1 "failed" sentinel silently (only the CSV column shows it).

## Frozen 30-clip comparison slice

`score_frozen30.py` freezes a metadata-only balanced slice (15 style + 15 point
edits), writes `FROZEN30_MANIFEST.jsonl` and track-specific prompt files, and
generates `FROZEN30_SCORECARD.md` from the saved per-video CSVs. The current
scorecard is the skill/compiler arm vs direct prompting with the same Ditto
generator; it is not a full vision-agent run.
It is a compact report, not a replacement for the full 48+40 results.

That first frozen scorecard reuses the earlier 48/40 renders. Those renders
inherited a default negative prompt containing `cartoon, anime`, which conflicts
with some style-edit instructions. They remain useful as a historical paired
result, but should not be treated as the final controlled comparison.

The corrected run is `scripts/bench_v5/run_frozen30_baselines.sh <gpu>`. It
re-renders all four arms on the same 30 clips and uses the same neutral
artifact-only negative prompt for each arm: direct Ditto, skill/compiler Ditto,
VACE base weights via the local Ditto runtime, and Bernini-R-1.3B. Per-video
metadata records the exact negative prompt and render configuration. The script
refuses to start if the requested GPU has less than 55 GB free. Its new
run-stamped scorecard supersedes the historical frozen report for claims about
the corrected protocol; the earlier outputs and scores are retained.

The renderer accepts a backend variant and an explicit negative prompt so methods
can use identical source clips, instructions, seed and generation constraints. For
example, on the configured GPU pod:

```bash
python3 bench_external/ivebench/score_frozen30.py
```

The aggregate driver renders, evaluates, and reports all four arms. It consumes
multiple GPU-hours.

The renderer normalizes 32 requested frames to the backend's supported 4n+1
length, uses the benchmark's verbatim instruction and seed 42, and exits nonzero
if any requested clip fails. Other methods require their own validated
backend/runtime; do not use published all-category numbers as substitutes for
these per-video outputs.
