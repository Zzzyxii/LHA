# SI: 从 Ditto + Ditto Data Step7000 继续训练的四组实验命令

这份命令用于从下面这个模型继续训练：

```text
方法名：Ditto + Ditto data
实验名：exp1_ditto_data_step7000
SI checkpoint：
/mnt/si002961ale4/default/lgy/shiying/low-high-new/out/edit_model_face_stage1/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
HF checkpoint：
Ouzhang/low-high-models/checkpoints/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
```

对应你给的 top50 分数是：

```text
CLIP-T 22.6013
CLIP-F 99.6678
CLIP Src-Edit 0.8283
TCLIPFrame 0.9980
DINO-F 0.9986
LPIPS 0.4249
LAION Aesthetic 5.7991
MUSIQ 56.9112
NIQE 5.5754
PickScore 18.61
Reward overall 3.55
```

## 0. 进入 SI 仓库和检查环境

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto

$PY - <<'PY'
import sys
print("Python:", sys.executable)
import imageio, imageio_ffmpeg, modelscope, accelerate, diffsynth
import diffsynth.models.latent_memory_runtime as lm
from diffsynth.models.easyvfx_frequency_runtime import EasyVFXFrequencyLoss
print("Dependencies OK")
print("latent_memory_runtime:", lm.__file__)
print("EasyVFXFrequencyLoss:", EasyVFXFrequencyLoss)
PY
```

如果 checkpoint 不在本地，先从 HF 拉下来：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export HF_ENDPOINT=https://hf-mirror.com
hf download Ouzhang/low-high-models \
  checkpoints/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors \
  --repo-type model \
  --local-dir .
```

## 1. 准备 ours-only 训练数据

默认使用之前约定的 ours 子集：

```bash
DATA=out/edit_model_face_stage1/ours_drop_last10/train.jsonl
test -f "$DATA" && wc -l "$DATA"
```

如果这个文件还没有，需要先找 SI 上现有的 ours-only JSONL：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

find out/edit_model_face_stage1 -name '*.jsonl' \
  | grep -Ei 'ours|low_high|low-high|cost' \
  | sort
```

然后把找到的 ours-only JSONL 填到 `OURS_RAW_JSONL`，构建每个子文件夹丢掉最后 10 个样本的数据：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export OURS_RAW_JSONL=path/to/ours.train_no_heldout1000.jsonl

$PY - <<'PY'
import json
from collections import defaultdict
from pathlib import Path
import os

src = Path(os.environ["OURS_RAW_JSONL"])
dst = Path("out/edit_model_face_stage1/ours_drop_last10/train.jsonl")

rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
groups = defaultdict(list)
for row in rows:
    p = str(row.get("low_video_path") or row.get("source_video") or row.get("vace_video") or "")
    groups[str(Path(p).parent)].append(row)

kept = []
for key, items in groups.items():
    items = sorted(items, key=lambda r: str(r.get("sample_id") or r.get("low_video_path") or r.get("source_video") or ""))
    kept.extend(items[:-10] if len(items) > 10 else [])

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + "\n", encoding="utf-8")
print("input", len(rows))
print("groups", len(groups))
print("output", len(kept))
print("output_path", dst)
PY
```

如果你想用 ours 全量而不是 drop-last10 子集，后面命令里只要把 `DATA=out/edit_model_face_stage1/ours_drop_last10/train.jsonl` 换成你的 ours-only 全量 JSONL。

## 2. 通用变量

这四组实验都保持相同 base、数据、分辨率、帧数和训练超参：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto

export BASE_CKPT=out/edit_model_face_stage1/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10/train.jsonl
export MAX_TRAIN_STEPS=5000

test -f "$BASE_CKPT"
test -f "$DATA"
mkdir -p out/edit_model_face_stage1/logs
```

`MAX_TRAIN_STEPS=5000` 是建议的 stage2 起点；如果要跑完整 epoch，可以改成 `0`。

注意：下面每个实验默认占用 8 张可见 GPU。`gradient-accumulation-steps=1`，这样全局 batch 约等于原来的 4GPU x grad-accum 2，但速度更快。如果同一台机器同时跑多组，不要让它们共享同一组卡。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7          # 跑实验 1
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15    # 跑实验 2
```

如果只有 8 张空卡，就一次只跑一个实验。

## 3. 实验 1：只用最新 FX 方法

含义：从 `Ditto + Ditto data step7000` 继续，在 ours data 上只加 EasyVFX/Frequency full loss；不加 text/hint memory，不加 aux loss，不加 SA-RoPE。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

nohup "$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir out/edit_model_face_stage1/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k \
  --stage base_sft \
  --recipe official_vace14b_continue \
  --launcher accelerate \
  --python-executable "$PY" \
  --accelerate-num-processes 8 \
  --accelerate-config-file scripts/training/edit-model/accelerate_config_8gpu_bf16.yaml \
  --real-train-backend ditto \
  --models-root models \
  --dataset-repeat 1 \
  --dataset-num-workers 0 \
  --gradient-accumulation-steps 1 \
  --save-steps 1000 \
  --save-step-list 2500,5000 \
  --weight-decay 0.01 \
  --learning-rate 5e-5 \
  --num-epochs 1 \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --seed 42 \
  --model-init-device cuda \
  --lora-checkpoint "$BASE_CKPT" \
  --num-frames 17 \
  --height 480 \
  --width 832 \
  --latent-memory-mode none \
  --aux-loss-preset none \
  --easyvfx-frequency-mode full \
  --easyvfx-low-ratio 0.25 \
  --easyvfx-temporal-low-ratio 0.35 \
  --easyvfx-low-weight 1e-3 \
  --easyvfx-high-weight 1e-3 \
  --easyvfx-temporal-weight 5e-4 \
  --easyvfx-descriptor-weight 1e-3 \
  --easyvfx-adaptive-weight 1e-3 \
  --easyvfx-adaptive-temperature 0.7 \
  --run-real-train \
  > out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_easyvfx_full_only.launch.log 2>&1 &

echo "PID=$!"
```

## 4. 实验 2：只用 text mem + hint mem

含义：从同一个 step7000 继续，在 ours data 上只打开 `text+vace_hint` memory；不加 aux loss，不加 FX。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

nohup "$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k \
  --stage base_sft \
  --recipe official_vace14b_continue \
  --launcher accelerate \
  --python-executable "$PY" \
  --accelerate-num-processes 8 \
  --accelerate-config-file scripts/training/edit-model/accelerate_config_8gpu_bf16.yaml \
  --real-train-backend ditto \
  --models-root models \
  --dataset-repeat 1 \
  --dataset-num-workers 0 \
  --gradient-accumulation-steps 1 \
  --save-steps 1000 \
  --save-step-list 2500,5000 \
  --weight-decay 0.01 \
  --learning-rate 5e-5 \
  --num-epochs 1 \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --seed 42 \
  --model-init-device cuda \
  --lora-checkpoint "$BASE_CKPT" \
  --num-frames 17 \
  --height 480 \
  --width 832 \
  --latent-memory-mode text+vace_hint \
  --latent-memory-tokens 4 \
  --latent-memory-hidden-dim 0 \
  --latent-memory-scale 1.0 \
  --latent-memory-init-std 0.02 \
  --aux-loss-preset none \
  --easyvfx-frequency-mode none \
  --run-real-train \
  > out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_text_hint_mem_only.launch.log 2>&1 &

echo "PID=$!"
```

## 5. 实验 3：text/hint mem + aux loss

含义：从同一个 step7000 继续，在 ours data 上打开 `text+vace_hint` memory，并加入当前全部 aux loss；不加 FX。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

nohup "$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k \
  --stage base_sft \
  --recipe official_vace14b_continue \
  --launcher accelerate \
  --python-executable "$PY" \
  --accelerate-num-processes 8 \
  --accelerate-config-file scripts/training/edit-model/accelerate_config_8gpu_bf16.yaml \
  --real-train-backend ditto \
  --models-root models \
  --dataset-repeat 1 \
  --dataset-num-workers 0 \
  --gradient-accumulation-steps 1 \
  --save-steps 1000 \
  --save-step-list 2500,5000 \
  --weight-decay 0.01 \
  --learning-rate 5e-5 \
  --num-epochs 1 \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --seed 42 \
  --model-init-device cuda \
  --lora-checkpoint "$BASE_CKPT" \
  --num-frames 17 \
  --height 480 \
  --width 832 \
  --latent-memory-mode text+vace_hint \
  --latent-memory-tokens 4 \
  --latent-memory-hidden-dim 0 \
  --latent-memory-scale 1.0 \
  --latent-memory-init-std 0.02 \
  --aux-loss-preset all \
  --latent-memory-code-diversity-weight 1e-3 \
  --latent-memory-gate-entropy-weight 1e-4 \
  --latent-memory-gate-usage-weight 1e-3 \
  --latent-feature-alignment-weight 5e-3 \
  --latent-feature-alignment-margin 0.0 \
  --latent-feature-relation-weight 5e-4 \
  --latent-feature-relation-margin 0.0 \
  --latent-feature-relation-max-tokens 256 \
  --easyvfx-frequency-mode none \
  --run-real-train \
  > out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_text_hint_mem_auxall.launch.log 2>&1 &

echo "PID=$!"
```

## 6. 实验 4：所有新方法

含义：从同一个 step7000 继续，在 ours data 上同时使用：

- text memory + VACE hint memory
- aux loss all
- EasyVFX/Frequency full
- SA-RoPE `source+memory`
- identity distillation + first-frame MMD

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

nohup "$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir out/edit_model_face_stage1/stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k \
  --stage base_sft \
  --recipe official_vace14b_continue \
  --launcher accelerate \
  --python-executable "$PY" \
  --accelerate-num-processes 8 \
  --accelerate-config-file scripts/training/edit-model/accelerate_config_8gpu_bf16.yaml \
  --real-train-backend ditto \
  --models-root models \
  --dataset-repeat 1 \
  --dataset-num-workers 0 \
  --gradient-accumulation-steps 1 \
  --save-steps 1000 \
  --save-step-list 2500,5000 \
  --weight-decay 0.01 \
  --learning-rate 5e-5 \
  --num-epochs 1 \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --seed 42 \
  --model-init-device cuda \
  --lora-checkpoint "$BASE_CKPT" \
  --num-frames 17 \
  --height 480 \
  --width 832 \
  --latent-memory-mode text+vace_hint \
  --latent-memory-tokens 4 \
  --latent-memory-hidden-dim 0 \
  --latent-memory-scale 1.0 \
  --latent-memory-init-std 0.02 \
  --aux-loss-preset all \
  --latent-memory-code-diversity-weight 1e-3 \
  --latent-memory-gate-entropy-weight 1e-4 \
  --latent-memory-gate-usage-weight 1e-3 \
  --latent-feature-alignment-weight 5e-3 \
  --latent-feature-alignment-margin 0.0 \
  --latent-feature-relation-weight 5e-4 \
  --latent-feature-relation-margin 0.0 \
  --latent-feature-relation-max-tokens 256 \
  --easyvfx-frequency-mode full \
  --easyvfx-low-ratio 0.25 \
  --easyvfx-temporal-low-ratio 0.35 \
  --easyvfx-low-weight 1e-3 \
  --easyvfx-high-weight 1e-3 \
  --easyvfx-temporal-weight 5e-4 \
  --easyvfx-descriptor-weight 1e-3 \
  --easyvfx-adaptive-weight 1e-3 \
  --easyvfx-adaptive-temperature 0.7 \
  --sa-rope-mode source+memory \
  --sa-rope-hidden-dim 128 \
  --sa-rope-init-scale 0.1 \
  --sa-rope-min-scale 0.5 \
  --sa-rope-max-scale 1.5 \
  --sa-rope-reg-weight 1e-4 \
  --sa-rope-default-head-role alternate \
  --identity-distill-weight 5e-4 \
  --first-frame-mmd-weight 5e-4 \
  --identity-distill-max-tokens 256 \
  --run-real-train \
  > out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_all_new_methods.launch.log 2>&1 &

echo "PID=$!"
```

如果只想把“所有新方法”理解为 `memory + aux + EasyVFX full`，不想同时打开 SA-RoPE/identity/MMD，就删掉实验 4 里从 `--sa-rope-mode source+memory` 到 `--identity-distill-max-tokens 256` 这一段。

## 7. 进度和错误检查

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

ps -ef | grep -E 'stage2_exp1_step7000|model_training/train.py|accelerate' | grep -v grep

for OUT in \
  stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k \
  stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k \
  stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k \
  stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k
do
  echo
  echo "=== $OUT ==="
  ls -lh "out/edit_model_face_stage1/$OUT/real_train.log" "out/edit_model_face_stage1/$OUT/real_train_status.json" 2>/dev/null
  find "out/edit_model_face_stage1/$OUT/real_train_output" -maxdepth 1 -name 'step-*.safetensors' -print 2>/dev/null | sort -V | tail
  grep -nEi 'Traceback|Error|Exception|Failed|Killed|OutOfMemory|CUDA|No such file|not found' \
    "out/edit_model_face_stage1/$OUT/real_train.log" 2>/dev/null | tail -20
done
```
