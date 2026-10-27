# IVEBench results (generated)

Metrics available on this pod: OSC/PSC (compliance, VideoCLIP-XL), SF (fidelity), TF (quality). No IVEBench total is quoted -- 8 of 12 metrics need checkpoints that cannot be fetched here.


## Style editing (48 clips)

| arm | n | OSC | PSC | SF | TF | dOSC vs source |
|---|---|---|---|---|---|---|
| source (anchor) |  | 0.2363 | 0.1210 | 1.0000 | 0.9668 | +0.0000 |
| direct (benchmark prompt) |  | 0.2317 | 0.1456 | 0.8562 | 0.9652 | -0.0047 |
| skill |  | 0.2300 | 0.1511 | 0.8236 | 0.9667 | -0.0063 |

Published Ditto / VACE (short subset, **all categories** -- context, not a same-subset control): OSC: Ditto 0.25 / VACE 0.23, PSC: Ditto 0.24 / VACE 0.22, SF: Ditto 0.89 / VACE 0.97, TF: Ditto 0.97 / VACE 0.98

## Attribute / point editing (40 clips)

| arm | n | OSC | PSC | SF | TF | dOSC vs source |
|---|---|---|---|---|---|---|
| source (anchor) |  | 0.2446 | 0.1846 | 1.0000 | 0.9685 | +0.0000 |
| direct (benchmark prompt) |  | 0.2389 | 0.1891 | 0.9156 | 0.9692 | -0.0057 |
| skill_attr |  | 0.2397 | 0.1924 | 0.9172 | 0.9720 | -0.0049 |

Published Ditto / VACE (short subset, **all categories** -- context, not a same-subset control): OSC: Ditto 0.25 / VACE 0.23, PSC: Ditto 0.24 / VACE 0.22, SF: Ditto 0.89 / VACE 0.97, TF: Ditto 0.97 / VACE 0.98
