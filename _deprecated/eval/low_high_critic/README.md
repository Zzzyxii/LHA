# Low-High Critic Evaluation

This directory is for low-high agent critic evaluation. It is separate from
`scripts/eval/ditto_quant`, which is only for Ditto/Editto checkpoint evaluation.

## Four Critic Dimensions

`evaluate.py` computes:

- `high_vs_low_distribution_alignment`: CLIP embedding distribution score comparing generated videos to high-cost and low-cost reference sets.
- `temporal_consistency`: CLIP-F style adjacent-frame consistency.
- `high_cost_aesthetic_quality`: VLM score when enabled, otherwise optional distribution proxy.
- `edit_semantic_preservation`: VLM score when enabled, otherwise source-edited CLIP similarity proxy.

## Strong VLM Teacher JSONL

`build_strong_vlm_teacher_jsonl.py` prepares small batches for a stronger VLM
teacher to label the last two dimensions:

- `high_cost_aesthetic_quality`
- `edit_semantic_preservation`

The recommended teacher model is `gpt-5.4` with:

- `reasoning_effort=medium`
- `max_output_tokens=900`
- deterministic sampling if the caller supports it, for example `temperature=0`
- 4 uniformly sampled source frames and 4 uniformly sampled edited frames for small-batch direct vision calls

For a teacher-owned caller that can load videos from paths, use the generic format:

```bash
PYTHONPATH=src python scripts/eval/low_high_critic/build_strong_vlm_teacher_jsonl.py \
  --metadata /home/featurize/data/Ditto-1M/training_metadata/global.json \
  --ditto-root /home/featurize/data/Ditto-1M \
  --output-jsonl out/low_high_critic_teacher/ditto_small_teacher_inputs.jsonl \
  --output-format teacher-generic \
  --media-mode paths \
  --sample-size 20 \
  --shuffle \
  --seed 1 \
  --model gpt-5.4 \
  --reasoning-effort medium \
  --max-output-tokens 900
```

For a directly callable OpenAI Batch `/v1/responses` JSONL, use sampled frames:

```bash
PYTHONPATH=src python scripts/eval/low_high_critic/build_strong_vlm_teacher_jsonl.py \
  --metadata /home/featurize/data/Ditto-1M/training_metadata/global.json \
  --ditto-root /home/featurize/data/Ditto-1M \
  --output-jsonl out/low_high_critic_teacher/openai_batch_small.jsonl \
  --output-format openai-batch-responses \
  --media-mode sampled-frames \
  --frames-per-video 4 \
  --sample-size 20 \
  --shuffle \
  --seed 1 \
  --model gpt-5.4 \
  --reasoning-effort medium \
  --max-output-tokens 900
```

`sampled-frames` embeds JPEG data URLs into the JSONL, so use it only for small
batches unless you have checked file-size limits. For large datasets, generate
generic path-based JSONL and let the teacher-side caller handle video loading.

Expected model output schema:

```json
{
  "high_cost_aesthetic_quality": 0,
  "edit_semantic_preservation": 0,
  "decision": "accept",
  "failure_reasons": [],
  "refinement_suggestions": [],
  "rationale": ""
}
```

