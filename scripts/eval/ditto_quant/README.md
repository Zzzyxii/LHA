# Ditto Paper-Style Quantitative Evaluator

This directory provides a reproducible, paper-style evaluator for Ditto/Editto experiments. It is not the official Ditto paper evaluator.

The implementation follows the metric descriptions in the paper, but the paper does not release the exact test set, CLIP backbone, VLM judge prompt, frame sampling policy, or human-study protocol. Treat these numbers as internal comparison metrics, not official Table 1 reproduction numbers.

## Models And Data Layout

The evaluator does not require models or videos to be placed in one fixed directory. All important paths can be passed by command-line arguments. For cluster runs, prefer explicit local paths so the job does not try to download models at runtime.

Recommended layout:

```text
low-high-new/
  models/
    openai_clip-vit-large-patch14/
    Qwen3-VL-8B-Thinking/
  out/
    ditto_eval_inputs/
      manifest.jsonl
      videos/
        source/
        ours/
        baseline/
    ditto_quant_eval/
```

### Required Data

You must prepare the evaluation videos yourself. This repository does not include the paper's 50 OOD videos x 5 instructions test set.

Each manifest row should point to:

- `edited_video`: generated video to evaluate.
- `instruction`: edit instruction used to generate the video.
- `source_video`: optional but strongly recommended, required for meaningful VLM source-preservation scoring.
- `method`: method name for grouping, for example `ours`, `tokenflow`, `insv2v`, or `baseline`.

Paths may be absolute, or relative to the manifest file location.

### CLIP Model

`CLIP-T` and `CLIP-F` require a Transformers-compatible CLIP checkpoint. By default the script uses:

```text
openai/clip-vit-large-patch14
```

If the machine has internet access, Transformers will download it into the Hugging Face cache automatically. For offline cluster runs, download it first and pass the local path:

```bash
hf download openai/clip-vit-large-patch14 \
  --local-dir models/openai_clip-vit-large-patch14
```

Then run with:

```bash
--clip-model models/openai_clip-vit-large-patch14
```

You can also use a shared absolute model path:

```bash
--clip-model /mnt/dhwfile/raise/user/panjiabao/huxiaobin/shiying/low-high-new/models/openai_clip-vit-large-patch14
```

If you only want to test the manifest/report path without loading CLIP, use `--skip-clip`.

### VLM Judge Model

The VLM score is optional. If `--vlm-backend none` is used, no Qwen model is needed.

For `--vlm-backend local-qwen`, provide a local Qwen-VL model directory:

```bash
--vlm-backend local-qwen \
--vlm-model models/Qwen3-VL-8B-Thinking \
--vlm-device cuda:0
```

The evaluator uses the existing project loader in `src/vlm/local_qwen.py`. That loader also respects:

```bash
LOW_HIGH_AGENT_LOCAL_QWEN_MODEL=/path/to/Qwen3-VL-8B-Thinking
LOW_HIGH_AGENT_LOCAL_QWEN_DEVICE=cuda:0
LOW_HIGH_AGENT_USE_LOCAL_QWEN=1
```

If your model is not named `Qwen3-VL-8B-Thinking`, that is fine as long as it is compatible with `AutoModelForImageTextToText` and `AutoProcessor`.

### Runtime Dependencies

The evaluator expects the project runtime dependencies:

```bash
pip install -r requirements.txt
```

Important packages are `torch`, `transformers`, `imageio[ffmpeg]`, `pillow`, and, for the local Qwen judge, `qwen-vl-utils`.

## Metrics

- `CLIP-T`: text-video similarity. The evaluator samples frames from the edited video, embeds each frame with CLIP, embeds the edit instruction with CLIP text, averages cosine similarity, and multiplies by 100.
- `CLIP-F`: temporal consistency. The evaluator embeds sampled edited frames with CLIP, averages adjacent-frame cosine similarity, and multiplies by 100.
- `VLM`: optional local Qwen-VL judge. It samples source and edited frames, asks the VLM to score instruction fidelity, source preservation, visual quality, safety, and overall quality on a 0-10 scale.

## Manifest

Use JSONL. Each line is one generated result for one method.

Required fields:

- `edited_video` or `output_video` or `video`
- `instruction` or `prompt` or `edit_instruction`

Recommended fields:

- `sample_id`
- `method`
- `source_video`

Example:

```jsonl
{"sample_id":"case_000","method":"ours","source_video":"/data/source_000.mp4","edited_video":"/data/ours_000.mp4","instruction":"Render it in the style of pixel art."}
{"sample_id":"case_000","method":"baseline","source_video":"/data/source_000.mp4","edited_video":"/data/baseline_000.mp4","instruction":"Render it in the style of pixel art."}
```

See `manifest.example.jsonl`.

## Run CLIP Metrics

```bash
PYTHONPATH=src python scripts/eval/ditto_quant/evaluate.py \
  --manifest path/to/eval_manifest.jsonl \
  --output-dir out/ditto_quant_eval \
  --device cuda \
  --frames-per-video 16
```

If the CLIP model is already downloaded locally, pass its path:

```bash
PYTHONPATH=src python scripts/eval/ditto_quant/evaluate.py \
  --manifest path/to/eval_manifest.jsonl \
  --output-dir out/ditto_quant_eval \
  --clip-model /path/to/openai_clip-vit-large-patch14 \
  --device cuda
```

## Run With Local Qwen-VL Judge

```bash
PYTHONPATH=src python scripts/eval/ditto_quant/evaluate.py \
  --manifest path/to/eval_manifest.jsonl \
  --output-dir out/ditto_quant_eval_vlm \
  --device cuda \
  --vlm-backend local-qwen \
  --vlm-model /path/to/Qwen3-VL-8B-Thinking \
  --vlm-device cuda:0 \
  --vlm-frames-per-video 4
```

The local judge uses the existing project loader in `src/vlm/local_qwen.py`.

## Outputs

The output directory contains:

- `per_sample_metrics.jsonl`: full records with raw manifest data and optional raw VLM response.
- `per_sample_metrics.csv`: compact table for spreadsheet inspection.
- `summary.json`: grouped means, standard deviations, standard errors, and counts by method.
- `summary.md`: compact markdown table with `CLIP-T`, `CLIP-F`, `VLM Overall`, `VLM Instr`, `VLM Source`, `VLM Quality`, and `VLM Safety`.

## Generate Videos From Checkpoints

`evaluate.py` only scores videos that already exist. To compare checkpoints end-to-end, first run Ditto inference for a fixed eval input set.

Eval input JSONL rows should contain:

- `sample_id`
- `source_video`
- `instruction`
- optional `control_video`
- optional `reference_image`

Example:

```jsonl
{"sample_id":"case_000","source_video":"/data/eval/source_000.mp4","instruction":"Render it in the style of pixel art."}
{"sample_id":"case_001","source_video":"/data/eval/source_001.mp4","instruction":"Replace the man's clothes with a black suit."}
```

See `eval_inputs.example.jsonl`.

Run inference across multiple LoRA checkpoints:

```bash
PYTHONPATH=src python scripts/eval/ditto_quant/run_inference.py \
  --eval-inputs path/to/eval_inputs.jsonl \
  --output-dir out/edit_model_eval \
  --checkpoint-path /path/to/models/Wan2.1-VACE-14B \
  --python-executable /path/to/python \
  --checkpoint ditto_global=/path/to/ditto_global.safetensors \
  --checkpoint step-5000=/path/to/step-5000.safetensors \
  --checkpoint step-10000=/path/to/step-10000.safetensors \
  --device-id 0 \
  --height 480 \
  --width 832 \
  --num-frames 73 \
  --num-inference-steps 20
```

The script writes generated videos under:

```text
out/edit_model_eval/generations/<checkpoint-name>/
```

It also writes the quant manifest consumed by `evaluate.py`:

```text
out/edit_model_eval/manifests/quant_manifest.jsonl
```

Then evaluate the generated videos:

```bash
PYTHONPATH=src python scripts/eval/ditto_quant/evaluate.py \
  --manifest out/edit_model_eval/manifests/quant_manifest.jsonl \
  --output-dir out/edit_model_eval/metrics \
  --clip-model /path/to/openai_clip-vit-large-patch14 \
  --device cuda
```

With VLM judging:

```bash
PYTHONPATH=src python scripts/eval/ditto_quant/evaluate.py \
  --manifest out/edit_model_eval/manifests/quant_manifest.jsonl \
  --output-dir out/edit_model_eval/metrics_vlm \
  --clip-model /path/to/openai_clip-vit-large-patch14 \
  --device cuda \
  --vlm-backend local-qwen \
  --vlm-model /path/to/Qwen3-VL-8B-Thinking \
  --vlm-device cuda:0
```

## Re-Aggregate Existing Results

If you only need to regenerate summaries from existing per-sample metric files:

```bash
PYTHONPATH=src python scripts/eval/ditto_quant/report.py \
  --metrics-jsonl out/run_a/per_sample_metrics.jsonl out/run_b/per_sample_metrics.jsonl \
  --output-dir out/ditto_quant_merged_summary
```

## Notes

- `CLIP-T` and `CLIP-F` depend on frame sampling, video decoding, and the CLIP checkpoint. Keep these fixed for fair comparisons.
- The VLM score is judge-prompt dependent. Keep `vlm_judge.py` unchanged during a comparison run.
- Human evaluation metrics from the paper (`Edit-Acc`, `Temp-Con`, `Overall`) are not automated here. Use VLM scores as a scalable proxy, not a replacement for a blinded user study.
