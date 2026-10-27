# Latent Memory / VACE Hint Analysis

这些脚本用于回答：`vace_hint` latent memory 是否学到了可解释的 high-cost transformation basis。

## 1. 静态检查 checkpoint

不跑 Wan/Ditto，只检查 checkpoint 里的 latent memory 参数。

```bash
python3 scripts/codebook/inspect_latent_memory_checkpoint.py \
  --checkpoint out/edit_model_face_stage1/base_sft_rebuild_merged_8gpu_latent_vace_hint/real_train_output/step-15214.safetensors \
  --mode vace_hint \
  --output-dir out/codebook/vace_hint_static
```

输出：

- `summary.json`: latent memory key、shape、selector keys
- `memory_norms.csv`: 每个 VACE hint layer、每个 token 的 memory norm
- `token_cosines.csv`: 同一层不同 token 之间的 cosine

## 2. 运行一条 inference 并记录 routing

建议用训练/eval 时 Ditto 的 prompt，也就是 metadata 里的 `prompt`，不要随便写 prompt。这样解释的是模型真实训练分布上的行为。

```bash
python3 scripts/codebook/trace_vace_hint_routing.py \
  --sample-id val_0000 \
  --category sci_fi \
  --input-video path/to/low.mp4 \
  --control-video path/to/low.mp4 \
  --output-video out/codebook/traces/val_0000.mp4 \
  --checkpoint-path models/Wan2.1-VACE-14B \
  --lora-path out/edit_model_face_stage1/base_sft_rebuild_merged_8gpu_latent_vace_hint/real_train_output/step-15214.safetensors \
  --prompt "your training/eval prompt" \
  --trace-output out/codebook/traces/val_0000.trace.json \
  --device-id 0 \
  --num-frames 17 \
  --num-inference-steps 1 \
  --latent-memory-tokens 4 \
  --latent-memory-hidden-dim 0 \
  --latent-memory-scale 1.0
```

输出 trace 里包含：

- 每个 `hint_index` 的 `weights`
- `top_token`
- `weight_entropy`
- `delta_norm`
- `hint_norm`
- `delta_to_hint_norm`

`--num-inference-steps 1` 已经足够观察 routing；不需要为可解释性跑完整高质量视频。

也可以从 `metadata.val.csv` 自动生成一批 trace 命令：

```bash
python3 scripts/codebook/build_trace_commands.py \
  --metadata-csv out/edit_model_face_stage1/base_sft_rebuild_merged_8gpu_latent_vace_hint/real_train/metadata.val.csv \
  --output-sh out/codebook/run_trace_vace_hint_8.sh \
  --repo-root . \
  --lora-path out/edit_model_face_stage1/base_sft_rebuild_merged_8gpu_latent_vace_hint/real_train_output/step-15214.safetensors \
  --trace-dir out/codebook/traces \
  --limit 8 \
  --num-inference-steps 1

bash out/codebook/run_trace_vace_hint_8.sh
```

如果 `metadata.val.csv` 里是旧的视频路径，但当前机器上 `datas/ditto_face/low/` 和 `datas/ditto_face2/low/`
里是 hash 文件名，可以用 `ditto_filtered_1.json` / `ditto_filtered_2.json` 按 prompt/instruction 映射：

```bash
python3 scripts/codebook/build_trace_commands.py \
  --metadata-csv out/edit_model_face_stage1/base_sft_rebuild_merged_8gpu_latent_vace_hint/real_train/metadata.val.csv \
  --output-sh out/codebook/run_trace_vace_hint_8.sh \
  --repo-root . \
  --lora-path out/edit_model_face_stage1/base_sft_rebuild_merged_8gpu_latent_vace_hint/real_train_output/step-15214.safetensors \
  --trace-dir out/codebook/traces \
  --limit 8 \
  --num-inference-steps 1 \
  --ditto-filtered-json ditto_filtered_1.json \
  --ditto-filtered-json ditto_filtered_2.json \
  --ditto-low-roots datas/ditto_face/low,datas/ditto_face2/low

bash out/codebook/run_trace_vace_hint_8.sh
```

## 3. 聚合成 heatmap CSV

多跑一些样本后：

```bash
python3 scripts/codebook/aggregate_vace_hint_traces.py \
  --trace-glob "out/codebook/traces/*.trace.json" \
  --output-dir out/codebook/vace_hint_aggregate
```

输出：

- `routing_heatmap.csv`: category x layer x token 平均权重
- `delta_norms.csv`: category x layer 的 delta norm / entropy

可以用这些表画：

- sci-fi / 古风 / 服装 / 场景 是否激活不同 token
- 浅层和深层 VACE hint 是否 delta norm 不同
- routing entropy 是否随类别变化

## 解释边界

如果不同类别的 routing 权重没有明显分开，就不要强行叫 `codebook`。更稳妥的说法是：

`prompt-conditioned latent memory basis` 或 `VACE hint latent memory adapter`。
