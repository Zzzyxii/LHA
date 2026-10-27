# IVEBench published numbers (for comparison)

Source: IVEBench paper (ICLR 2026, arXiv:2510.11647) Table 2 / the project's
leaderboard, retrieved 2026-10-25. **Short subset** (32-128 frames), which is the
subset our 32-frame clips belong to.

| method | Total | Quality | Compliance | Fidelity |
|---|--|--|--|--|
| Ditto | 0.67 | 0.78 | 0.49 | 0.73 |
| InsV2V | 0.67 | 0.80 | 0.39 | 0.82 |
| Lucy-Edit-Dev | 0.64 | 0.82 | 0.34 | 0.75 |
| VACE | 0.63 | 0.80 | 0.25 | 0.83 |
| ICVE | 0.60 | 0.71 | 0.45 | 0.64 |
| Omni-Video | 0.59 | 0.78 | 0.44 | 0.54 |
| AnyV2V | 0.58 | 0.73 | 0.42 | 0.59 |
| StableV2V | 0.51 | 0.69 | 0.43 | 0.41 |

Per-metric (Short), the three we can compute offline are marked `[*]`:

| method | SC | BC | TF | MS | VTSS | OSC `[*]` | PSC `[*]` | IS | QA | SF `[*]` | MF | CF |
|---|--|--|--|--|--|--|--|--|--|--|--|--|
| Ditto | 0.96 | 0.98 | 0.97 | 0.99 | 0.038 | 0.25 | 0.24 | 3.87 | 0.30 | 0.89 | 0.79 | 3.64 |
| InsV2V | 0.94 | 0.96 | 0.97 | 0.97 | 0.045 | 0.24 | 0.23 | 3.06 | 0.30 | 0.95 | 0.86 | 4.05 |
| Lucy-Edit-Dev | 0.95 | 0.96 | 0.98 | 0.99 | 0.051 | 0.24 | 0.22 | 2.84 | 0.20 | 0.93 | 0.68 | 3.83 |
| VACE | 0.95 | 0.97 | 0.98 | 0.98 | 0.045 | 0.23 | 0.22 | 2.16 | 0.20 | 0.97 | 0.89 | 4.03 |
| ICVE | 0.95 | 0.97 | 0.99 | 1.00 | 0.017 | 0.23 | 0.23 | 3.62 | 0.30 | 0.85 | 0.46 | 3.55 |

`OSC` = `overall_semantic_consistency`, `PSC` = `phrase_semantic_consistency`,
`SF` = `semantic_fidelity` (all VideoCLIP-XL). The rest need checkpoints this pod
cannot download (Qwen2.5-VL-72B for `IS`/`CF`, GroundingDINO for `QA`, CoTracker for
`MF`, AMT for `MS`) or weights that are not in the local cache (`BC` wants CLIP
ViT-B/32, `SC` wants DINO via torch.hub).

## How we compare against them

We run **the same base model and LoRA as `Ditto`** (`wan14b+dittolora`), on 48
clips of the style-editing category, 32 frames at 832x480 -- the resolution and
frame-length regime Ditto was published at. Three arms get scored:

* `src` -- the source clips themselves (fidelity ceiling, compliance floor);
* `direct` -- IVEBench's own edit prompt verbatim;
* `skill` -- the same instruction through our compiler.

Caveats that must travel with any number quoted from here:

1. Published totals cover all 8 editing categories; ours is style editing only, so
   only per-metric comparisons on the same subset are meaningful, not totals.
2. We cannot re-run the other 7 methods, so `direct` is the only same-model
   baseline we can render. The published Ditto row is a *context* number, not a
   same-subset control.
3. Our sources are trimmed to 32 frames (IVEBench is 32-1024); every arm gets the
   identical input, so arm-vs-arm comparisons are unaffected.
