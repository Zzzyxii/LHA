# Top50 Video Bundle

- Samples: `50`
- Grid playback speed: `2.0x`
- Methods configured: `16`
- Main demo: `demo/top50_all_samples_grid.mp4`
- Per-sample grids: `demo/per_sample/`
- Organized videos: `videos/`

## Method Scores

| Method | CLIP-F ↑ | DINO-F ↑ | PickScore ↑ | JEDi ↓ | Reward overall ↑ | Narr./emotion ↑ | Style/world ↑ | Comp./lighting ↑ | Color/texture ↑ | Hierarchy/readability ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VACE | 0.9884 | 0.9896 | 18.09 | 0.21 | 2.31 | 2.34 | 2.34 | 2.34 | 2.26 | 2.53 |
| Bernini-R-1.3B | 0.9953 | 0.9954 | 19.32 | 0.50 | 3.77 | 3.80 | 3.80 | 3.80 | 3.74 | 3.80 |
| Ditto | 0.9911 | 0.9913 | 18.83 | 0.16 | 3.22 | 3.24 | 3.24 | 3.30 | 3.18 | 3.42 |
| InsV2V | 0.9857 | 0.9865 | 18.02 | 0.33 | 1.89 | 1.96 | 1.94 | 1.96 | 1.84 | 2.16 |
| InsViE | 0.9918 | 0.9923 | 18.25 | 0.30 | 1.89 | 2.09 | 2.09 | 2.09 | 1.71 | 2.03 |
| TokenFlow | 0.9935 | 0.9956 | 18.85 | 0.50 | 2.98 | 3.00 | 2.98 | 3.00 | 2.96 | 3.18 |
| ICVE | 0.9933 | 0.9976 | 17.94 | 0.65 | 2.53 | 2.54 | 2.54 | 2.56 | 2.52 | 2.84 |
| Kiwi-Edit | 0.9931 | 0.9958 | 17.79 | 0.50 | 2.28 | 2.28 | 2.28 | 2.28 | 2.28 | 2.31 |
| SANA-Streaming | 0.9937 | 0.9964 | 18.06 | 0.52 | 2.33 | 2.42 | 2.42 | 2.36 | 2.28 | 2.50 |
| Senorita | 0.9953 | 0.9980 | 18.18 | 0.48 | 2.54 | 2.56 | 2.52 | 2.56 | 2.52 | 2.68 |
| ReCo | 0.9962 | 0.9977 | 18.88 | 0.62 | 3.44 | 3.48 | 3.46 | 3.48 | 3.40 | 3.43 |
| Lucy-Edit | 0.9946 | 0.9977 | 18.65 | 0.61 | 3.59 | 3.60 | 3.60 | 3.58 | 3.58 | 3.63 |
| Ditto + Ditto data | 0.9967 | 0.9986 | 18.97 | 0.54 | 3.55 | 3.56 | 3.56 | 3.60 | 3.52 | 3.68 |
| Ours + Ditto data | 0.9917 | 0.9920 | 19.04 | 0.54 | 3.66 | 3.72 | 3.70 | 3.72 | 3.62 | 3.75 |
| Ditto + Ditto + ours data | 0.9968 | 0.9985 | 18.87 | 0.52 | 3.78 | 3.80 | 3.80 | 3.80 | 3.76 | 3.82 |
| Ours | 0.9967 | 0.9985 | 18.83 | 0.48 | 3.87 | 3.88 | 3.88 | 3.88 | 3.86 | 3.91 |

## Inference Parameters

### VACE

- Generation dir: `/home/l/liyj/shiying/hf_data/eval1000_all_modelscope_top50/generation_root_reward_missing_three/generations/Wan`
- model: `Wan2.1-VACE-14B baseline`
- protocol: `top50, 832x480, 17 frames, 20 fps export, fixed per-example seed`

### Bernini-R-1.3B

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/Bernini-R-1.3B/generations/Bernini-R-1.3B`
- model: `ByteDance/Bernini-R-1.3B-Diffusers`
- protocol: `top50 wrapper, 17-frame export, fixed per-example seed`

### Ditto

- Generation dir: `/home/l/liyj/shiying/hf_data/eval1000_all_modelscope_top50/generation_root_reward_missing_three/generations/Wan+Ditto`
- model: `Wan2.1-VACE-14B + Ditto global checkpoint`
- protocol: `top50, 832x480, 17 frames, 20 fps export, fixed per-example seed`

### InsV2V

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_missing_methods_20260720/InsV2V/generations/InsV2V`
- model: `InsV2V available native-output run`
- protocol: `approximate comparison, native 384x384/49 frames, fixed top50 inputs`

### InsViE

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_missing_methods_20260720/InsViE/generations/InsViE`
- model: `InsViE available native-output run`
- protocol: `approximate comparison, native 720x480/49 frames; currently 35/50 available`

### TokenFlow

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/TokenFlow/generations/TokenFlow`
- model: `TokenFlow wrapper`
- protocol: `top50 strict wrapper, 832x480/17-frame export where applicable`

### ICVE

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/ICVE/generations/ICVE`
- model: `ICVE wrapper`
- protocol: `top50 strict wrapper, 832x480/17-frame export where applicable`

### Kiwi-Edit

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/KiwiEdit_top50/generations/Kiwi-Edit`
- model: `Kiwi-Edit`
- protocol: `top50 strict wrapper, prompt-only, no reference images`

### SANA-Streaming

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/SANA-Streaming/generations/SANA-Streaming`
- model: `SANA-Streaming`
- protocol: `top50 strict wrapper`

### Senorita

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/Senorita_top50_driver/generations/Señorita`
- model: `Señorita`
- protocol: `driver wrapper, 832x480, 17 frames, fixed per-example seed`

### ReCo

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/ReCo/generations/ReCo`
- model: `ReCo`
- protocol: `top50 strict wrapper, 832x480/17-frame export where applicable`

### Lucy-Edit

- Generation dir: `/mnt/scratch/l/liyj/paper_top50_strict_480p_17f20s_20260723/Lucy-Edit/generations/Lucy-Edit`
- model: `Lucy-Edit`
- protocol: `top50 strict wrapper, 832x480/17-frame export where applicable`

### Ditto + Ditto data

- Generation dir: `/home/l/liyj/shiying/hf_data/eval1000_all_modelscope_top50/generation_root/generations/exp1_ditto_data_step7000`
- model: `Ditto architecture fine-tuned on Ditto data`
- checkpoint: `exp1_ditto_data_step7000`
- protocol: `top50, 832x480, 17 frames, 20 denoising steps, fixed per-example seed`

### Ours + Ditto data

- Generation dir: `/home/l/liyj/shiying/hf_data/eval1000_all_modelscope_top50/generation_root/generations/exp3_auxall_ditto_data_step1000`
- model: `memory + aux, Ditto data`
- checkpoint: `exp3_auxall_ditto_data_step1000`
- protocol: `top50, 832x480, 17 frames, 20 denoising steps, fixed per-example seed`

### Ditto + Ditto + ours data

- Generation dir: `/home/l/liyj/shiying/hf_data/eval1000_all_modelscope_top50/generation_root/generations/exp6_ditto_all_data_step2000`
- model: `Ditto architecture fine-tuned on Ditto + ours data`
- checkpoint: `exp6_ditto_all_data_step2000`
- protocol: `top50, 832x480, 17 frames, 20 denoising steps, fixed per-example seed`

### Ours

- Generation dir: `/home/l/liyj/shiying/hf_data/eval1000_all_modelscope_top50/generation_root/generations/exp5_auxall_all_data_step6000`
- model: `ours memory + aux, Ditto + ours data`
- checkpoint: `exp5_auxall_all_data_step6000`
- protocol: `top50, 832x480, 17 frames, 20 denoising steps, fixed per-example seed`

## Missing Videos

- `InsViE`: 15 missing: heldout1000_0644, heldout1000_0374, heldout1000_0477, heldout1000_0679, heldout1000_0027, heldout1000_0903, heldout1000_0569, heldout1000_0656, heldout1000_0264, heldout1000_0357, heldout1000_0272, heldout1000_0887 ...
