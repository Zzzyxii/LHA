# Frozen 30-clip IVEBench scorecard

Frozen selection: 15 style-editing + 15 point-editing clips. Style selection round-robins sorted subcategories; point-edit selection takes five evenly spaced sample IDs in each of color adjustment, subject scaling, and position change. Selection uses prompt metadata only, never scores. The selection is retrospective (the 48/40 runs already existed), so this is a compact exploratory reporting slice, not an independent confirmatory test set.

All arms use the local IVEBench metrics on the same selected clips. For each method, the table shows its mean, paired difference against direct, and the skill/compiler-minus-method difference. The 95% intervals are paired bootstrap percentile intervals (20,000 resamples); p values are two-sided exact sign tests of direction, unadjusted across method/metric/track tests. Intervals estimate mean differences; sign tests are not tests of those means.

Frozen manifest: `FROZEN30_MANIFEST.jsonl`.

## Style transfer (n=15)

| Metric | Method | Mean | Δ vs direct (95% CI) | p | Skill−method (95% CI) | p |
|---|---|---:|---:|---:|---:|---:|
| OSC | direct | 0.2298 | — | — | +0.0004 [-0.0016, +0.0024] | 1.0000 |
| OSC | skill | 0.2302 | +0.0004 [-0.0016, +0.0024] | 1.0000 | 0 (self) | — |
| PSC | direct | 0.1464 | — | — | +0.0036 [-0.0016, +0.0105] | 0.6072 |
| PSC | skill | 0.1500 | +0.0036 [-0.0016, +0.0105] | 0.6072 | 0 (self) | — |
| SF | direct | 0.8569 | — | — | -0.0131 [-0.0273, -0.0007] | 0.6072 |
| SF | skill | 0.8438 | -0.0131 [-0.0273, -0.0007] | 0.6072 | 0 (self) | — |
| TF | direct | 0.9688 | — | — | +0.0011 [-0.0009, +0.0032] | 0.6072 |
| TF | skill | 0.9699 | +0.0011 [-0.0009, +0.0032] | 0.6072 | 0 (self) | — |

Selected IDs: short_0001.mp4, short_0002.mp4, short_0003.mp4, short_0004.mp4, short_0005.mp4, short_0006.mp4, short_0007.mp4, short_0010.mp4, short_0012.mp4, short_0019.mp4, short_0034.mp4, short_0049.mp4, short_0054.mp4, short_0101.mp4, short_0149.mp4.

## Point editing (n=15)

| Metric | Method | Mean | Δ vs direct (95% CI) | p | Skill−method (95% CI) | p |
|---|---|---:|---:|---:|---:|---:|
| OSC | direct | 0.2389 | — | — | -0.0005 [-0.0039, +0.0030] | 1.0000 |
| OSC | skill | 0.2384 | -0.0005 [-0.0039, +0.0030] | 1.0000 | 0 (self) | — |
| PSC | direct | 0.1880 | — | — | +0.0039 [+0.0006, +0.0072] | 0.0352 |
| PSC | skill | 0.1919 | +0.0039 [+0.0006, +0.0072] | 0.0352 | 0 (self) | — |
| SF | direct | 0.9038 | — | — | +0.0042 [-0.0104, +0.0188] | 0.3018 |
| SF | skill | 0.9080 | +0.0042 [-0.0104, +0.0188] | 0.3018 | 0 (self) | — |
| TF | direct | 0.9668 | — | — | +0.0035 [+0.0015, +0.0060] | 0.0074 |
| TF | skill | 0.9703 | +0.0035 [+0.0015, +0.0060] | 0.0074 | 0 (self) | — |

Selected IDs: short_0018.mp4, short_0046.mp4, short_0073.mp4, short_0104.mp4, short_0108.mp4, short_0115.mp4, short_0184.mp4, short_0189.mp4, short_0222.mp4, short_0267.mp4, short_0269.mp4, short_0284.mp4, short_0346.mp4, short_0353.mp4, short_0354.mp4.

## Scope and next comparison

This scorecard compares our skill/compiler arm with direct Ditto prompting and any additional evaluated methods supplied on the same frozen clips.
Independent open-source methods (for example Bernini-R or VACE) do not yet have outputs on these same clips, so they are absent. Published all-category IVEBench rows are not substituted for paired results.

Full 48-style / 40-point paired results remain in `RESULTS_IVEBENCH.md`; this frozen 30 is a compact, balanced reporting slice, not a replacement for the full results.
