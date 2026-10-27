# SI 四台 8GPU 机器分别启动四组 Stage2 实验

四台机器共享同一个仓库和数据路径：

```text
/mnt/si002961ale4/default/lgy/shiying/low-high-new
```

四组实验都从同一个 base checkpoint 继续：

```text
out/edit_model_face_stage1/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
```

四组实验都使用同一份 SI 可读的 ours-only drop-last10 数据：

```text
out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
```

每台机器复制对应一整段命令即可。不要在同一台 8 卡机器同时复制两段。

重要：这些机器如果检测到前台命令结束，可能会自动回收容器。因此下面命令不使用 `nohup ... &`，而是前台阻塞运行，并用 `tee` 同时把日志写到共享目录。看到训练命令开始后不要关闭终端；如果必须断开，需要用平台自带的长期任务机制，而不是 shell 后台进程。

## 先构建 SI 可读的 ours-only drop-last10 数据

之前失败的原因是旧输出目录里的 `metadata.train.csv` 还保留了 `/mnt/a100b/...` 旧绝对路径。下面这段使用已经验证过的快速构建方式：只做字符串路径映射，不对 10 万行数据逐条全量 `exists()` 扫共享盘；构建完成后只抽样检查路径是否存在。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3

$PY - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

src = Path("out/edit_model_face_stage1/new_eval_split_50/stage2_low_high_cost.train_no_eval50.si.jsonl")
if not src.exists():
    src = Path("out/edit_model_face_stage1/heldout1000_balanced_v1/low_high_cost.train_no_heldout1000.jsonl")
if not src.exists():
    raise FileNotFoundError("No ours-only source jsonl found")

dst = Path("out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl")
dst.parent.mkdir(parents=True, exist_ok=True)

path_keys = [
    "video", "vace_video",
    "low_video_path", "high_video_path",
    "source_video", "target_video", "edited_video",
    "source_path", "target_path",
]

def fix_path(v):
    if not isinstance(v, str):
        return v
    v = v.replace("/mnt/a100b/default/lgy/shiying", "/mnt/si002961ale4/default/lgy/shiying")
    v = v.replace("/home/l/liyj/shiying", "/mnt/si002961ale4/default/lgy/shiying")
    return v

groups = defaultdict(list)
n = 0
with src.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        for key in path_keys:
            if key in row:
                row[key] = fix_path(row[key])
        key_path = row.get("low_video_path") or row.get("source_video") or row.get("vace_video") or ""
        groups[str(Path(key_path).parent)].append(row)

kept = []
dropped = 0
for key, items in groups.items():
    items = sorted(items, key=lambda r: str(r.get("sample_id") or r.get("low_video_path") or r.get("source_video") or r.get("vace_video") or ""))
    if len(items) > 10:
        kept.extend(items[:-10])
        dropped += 10
    else:
        dropped += len(items)

with dst.open("w", encoding="utf-8") as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("source", src)
print("input", n)
print("groups", len(groups))
print("dropped", dropped)
print("output", len(kept))
print("output_path", dst)
PY

ls -lh out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
wc -l out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl

export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
test -s "$DATA" || { echo "empty or missing DATA=$DATA"; exit 1; }
grep -n '/mnt/a100b' "$DATA" | head

$PY - <<'PY'
import json
from pathlib import Path
p = Path("out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl")
for i, line in enumerate(p.open()):
    if i >= 20:
        break
    row = json.loads(line)
    for key in ["video", "vace_video", "high_video_path", "low_video_path"]:
        value = row.get(key)
        if isinstance(value, str):
            print(i, key, Path(value).exists(), value)
PY
```

## 机器 1：只用最新 FX / EasyVFX 方法

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata
export MAX_TRAIN_STEPS=5000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -m py_compile scripts/training/edit-model/train.py
$PY -c "from diffsynth.models.easyvfx_frequency_runtime import EasyVFXFrequencyLoss; print('EasyVFX OK', EasyVFXFrequencyLoss)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu.launch.log
```

## 机器 2：只用 text mem + hint mem

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata
export MAX_TRAIN_STEPS=5000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -m py_compile scripts/training/edit-model/train.py
$PY -c "import diffsynth.models.latent_memory_runtime as lm; print('latent memory OK', lm.__file__)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu.launch.log
```

## 机器 3：text/hint mem + aux loss

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata
export MAX_TRAIN_STEPS=5000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -m py_compile scripts/training/edit-model/train.py
$PY -c "import diffsynth.models.latent_memory_runtime as lm; print('latent memory OK', lm.__file__)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu.launch.log
```

## 机器 4：所有新方法

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/exp1_ditto_model_ditto_data_4gpu_save1k/real_train_output/step-7000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata
export MAX_TRAIN_STEPS=5000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -m py_compile scripts/training/edit-model/train.py
$PY -c "from diffsynth.models.easyvfx_frequency_runtime import EasyVFXFrequencyLoss; import diffsynth.models.latent_memory_runtime as lm; print('EasyVFX OK', EasyVFXFrequencyLoss); print('latent memory OK', lm.__file__)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_all_new_methods_8gpu.launch.log
```

## 续训：四组都从各自 step-5000 继续到总 20000 step

这里的“训到 20000”指每组以自己已经完成的 `step-5000.safetensors` 为起点，再训练 `15000` step。新输出目录里的 `step-15000.safetensors` 对应累计约 `5000 + 15000 = 20000` step。

不要把输出目录设回原来的 `*_fastdata` 目录，避免覆盖已经完成的 step-5000 产物。下面四段仍然是一台 8 卡机器跑一段，不要在同一台机器同时跑两段。

### 续训机器 1：EasyVFX 从 5000 到 20000

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata/real_train_output/step-5000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata_from5k_to20k
export MAX_TRAIN_STEPS=15000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -m py_compile scripts/training/edit-model/train.py
$PY -c "from diffsynth.models.easyvfx_frequency_runtime import EasyVFXFrequencyLoss; print('EasyVFX OK', EasyVFXFrequencyLoss)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  --save-step-list 5000,10000,15000 \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_from5k_to20k.launch.log
```

### 续训机器 2：text/hint mem 从 5000 到 20000

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata/real_train_output/step-5000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata_from5k_to20k
export MAX_TRAIN_STEPS=15000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -m py_compile scripts/training/edit-model/train.py
$PY -c "import diffsynth.models.latent_memory_runtime as lm; print('latent memory OK', lm.__file__)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  --save-step-list 5000,10000,15000 \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_from5k_to20k.launch.log
```

### 续训机器 3：text/hint mem + aux loss 从 5000 到 20000

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata/real_train_output/step-5000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata_from5k_to20k
export MAX_TRAIN_STEPS=15000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -m py_compile scripts/training/edit-model/train.py
$PY -c "import diffsynth.models.latent_memory_runtime as lm; print('latent memory OK', lm.__file__)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  --save-step-list 5000,10000,15000 \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_from5k_to20k.launch.log
```

### 续训机器 4：所有新方法从 5000 到 20000

机器 4 使用 `sa_rope`，启动前会先热修 `adaptive_rope_runtime.py` 里的 CPU/GPU device mismatch。如果已经修过，这行补丁不会重复改。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export BASE_CKPT=out/edit_model_face_stage1/stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata/real_train_output/step-5000.safetensors
export DATA=out/edit_model_face_stage1/ours_drop_last10_si_existing_fast/train.jsonl
export OUTDIR=out/edit_model_face_stage1/stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata_from5k_to20k
export MAX_TRAIN_STEPS=15000

$PY -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
$PY -m pip install sentencepiece -i https://pypi.org/simple

test -f "$BASE_CKPT" || { echo "missing BASE_CKPT=$BASE_CKPT"; exit 1; }
test -f "$DATA" || { echo "missing DATA=$DATA"; exit 1; }
mkdir -p out/edit_model_face_stage1/logs

$PY -c 'from pathlib import Path; p=Path("Ditto/diffsynth/models/adaptive_rope_runtime.py"); s=p.read_text(); old="    role_mask = roles.view(1, num_heads, 1) == TEMPORAL_HEAD\n"; new="    role_mask = (roles.view(1, num_heads, 1) == TEMPORAL_HEAD).to(device=temporal_freqs.device)\n"; p.write_text(s.replace(old, new)); print("sa_rope role_mask patched" if old in s else "sa_rope role_mask already patched or target not found")'
$PY -m py_compile scripts/training/edit-model/train.py Ditto/diffsynth/models/adaptive_rope_runtime.py
$PY -c "from diffsynth.models.easyvfx_frequency_runtime import EasyVFXFrequencyLoss; import diffsynth.models.latent_memory_runtime as lm; print('EasyVFX OK', EasyVFXFrequencyLoss); print('latent memory OK', lm.__file__)"

"$PY" scripts/training/edit-model/train.py \
  --dataset "$DATA" \
  --output-dir "$OUTDIR" \
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
  --save-step-list 5000,10000,15000 \
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
  2>&1 | tee out/edit_model_face_stage1/logs/stage2_exp1_step7000_ours_all_new_methods_8gpu_from5k_to20k.launch.log
```

查看四组续训进度：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

for OUT in \
  stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata_from5k_to20k \
  stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata_from5k_to20k \
  stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata_from5k_to20k \
  stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata_from5k_to20k
do
  echo
  echo "=== $OUT ==="
  cat "out/edit_model_face_stage1/$OUT/real_train_status.json" 2>/dev/null | grep -E '"status"|"returncode"' || true
  find "out/edit_model_face_stage1/$OUT/real_train_output" -maxdepth 1 -name 'step-*.safetensors' -print 2>/dev/null | sort -V | tail
  tail -20 "out/edit_model_face_stage1/$OUT/real_train.log" 2>/dev/null || true
done
```

## HF 上传和评测：Stage2 fastdata checkpoints

说明：

- 每组从 `step-7000` 继续训练后产生的所有 `step-*.safetensors` 都可以上传，不只上传 `step-5000`。
- 对已经完成到 5000 step 的组，通常会上传 `step-1000/2000/2500/3000/4000/5000.safetensors`。
- 其他三组如果目录里还没有 checkpoint，下面上传命令会自动跳过。
- 续训到总 20000 step 后，新目录里的 `step-15000.safetensors` 才等价于累计 `5000 + 15000 = 20000`。
- `evaluate.py` 不能直接评 checkpoint；必须先用 checkpoint 跑 inference 生成视频，再用生成视频 manifest 评测。

### 在 SI 机器上传已完成的 1000-5000 checkpoints 到 HF

下面用 public model repo 存模型权重。不要把 token 写进命令或粘到聊天里，用 `read -s` 在远程终端本地输入。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export HF_ENDPOINT=https://hf-mirror.com
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public

read -s -p "HF token: " HF_TOKEN
echo
export HF_TOKEN

for NAME in \
  stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata
do
  CKPT_DIR="out/edit_model_face_stage1/$NAME/real_train_output"
  if test -d "$CKPT_DIR"; then
    find "$CKPT_DIR" -maxdepth 1 -name 'step-*.safetensors' -print | sort -V | while read -r CKPT; do
      STEP_FILE="$(basename "$CKPT")"
      REMOTE="checkpoints/stage2_exp1_step7000_fastdata/$NAME/$STEP_FILE"
      echo "upload $CKPT -> $HF_REPO/$REMOTE"
      hf upload "$HF_REPO" "$CKPT" "$REMOTE" --repo-type model --commit-message "upload $NAME $STEP_FILE"
    done
  else
    echo "skip missing $CKPT_DIR"
  fi
done
```

如果镜像端点连不上，再换官方端点重跑：

```bash
export HF_ENDPOINT=https://huggingface.co
```

### 续训完成后上传 5000-15000 续训 checkpoints

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export HF_ENDPOINT=https://hf-mirror.com
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public

read -s -p "HF token: " HF_TOKEN
echo
export HF_TOKEN

for NAME in \
  stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata_from5k_to20k \
  stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata_from5k_to20k \
  stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata_from5k_to20k \
  stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata_from5k_to20k
do
  CKPT_DIR="out/edit_model_face_stage1/$NAME/real_train_output"
  if test -d "$CKPT_DIR"; then
    find "$CKPT_DIR" -maxdepth 1 -name 'step-*.safetensors' -print | sort -V | while read -r CKPT; do
      STEP_FILE="$(basename "$CKPT")"
      REMOTE="checkpoints/stage2_exp1_step7000_fastdata_from5k_to20k/$NAME/$STEP_FILE"
      echo "upload $CKPT -> $HF_REPO/$REMOTE"
      hf upload "$HF_REPO" "$CKPT" "$REMOTE" --repo-type model --commit-message "upload $NAME $STEP_FILE"
    done
  else
    echo "skip missing $CKPT_DIR"
  fi
done
```

### 下载校验

在任意机器上先看 HF repo 里有哪些文件：

```bash
export HF_ENDPOINT=https://huggingface.co
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public

hf download "$HF_REPO" --repo-type model --include '*.safetensors' --dry-run
```

下载到本地目录：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export HF_ENDPOINT=https://huggingface.co
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public

mkdir -p out/edit_model_face_stage1/hf_downloaded_stage2_fastdata

hf download "$HF_REPO" \
  --repo-type model \
  --local-dir out/edit_model_face_stage1/hf_downloaded_stage2_fastdata

find out/edit_model_face_stage1/hf_downloaded_stage2_fastdata -name '*.safetensors' -print | sort -V | xargs -r ls -lh
```

当前本地 Mac 检查结果：

- 本地 `models/` 目录没有 `Wan2.1-VACE-14B` 基础模型权重。
- 本地 Python 环境没有 `torch`，也没有 CUDA。
- 因此本地只能做 HF 下载和文件大小校验；完整评测需要在 SI/GPU 机器上跑下面的 inference + evaluate。

### 用下载后的 checkpoints 跑 inference

下面以 50 条 eval split 为例。真实评测要在 GPU 机器上跑；Mac 本机通常只能下载校验，不能实际生成 Wan2.1-VACE-14B 视频。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export CKPT_DIR=out/edit_model_face_stage1/hf_downloaded_stage2_fastdata/checkpoints/stage2_exp1_step7000_fastdata
export EVAL_INPUTS=out/eval_newsplit50/metric_details/eval_inputs.jsonl
export EVAL_OUT=out/eval_stage2_fastdata_hf_step5000

"$PY" scripts/eval/ditto_quant/run_inference.py \
  --eval-inputs "$EVAL_INPUTS" \
  --output-dir "$EVAL_OUT" \
  --checkpoint-path models/Wan2.1-VACE-14B \
  --python-executable "$PY" \
  --checkpoint easyvfx_step5000="$CKPT_DIR/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata/step-5000.safetensors" \
  --checkpoint text_hint_mem_step5000="$CKPT_DIR/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata/step-5000.safetensors" \
  --checkpoint text_hint_mem_auxall_step5000="$CKPT_DIR/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata/step-5000.safetensors" \
  --checkpoint all_new_methods_step5000="$CKPT_DIR/stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata/step-5000.safetensors" \
  --device-id 0 \
  --height 480 \
  --width 832 \
  --num-frames 17 \
  --num-inference-steps 20
```

如果只上传了一部分 checkpoint，就把不存在的 `--checkpoint ...=...` 行删掉。

### 评测生成视频

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export PYTHONPATH=$PWD/src:$PWD/Ditto
export EVAL_OUT=out/eval_stage2_fastdata_hf_step5000

"$PY" scripts/eval/ditto_quant/evaluate.py \
  --manifest "$EVAL_OUT/manifests/quant_manifest.jsonl" \
  --output-dir "$EVAL_OUT/metrics" \
  --clip-model models/openai_clip-vit-large-patch14 \
  --device cuda
```

带 VLM judge：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export PYTHONPATH=$PWD/src:$PWD/Ditto
export EVAL_OUT=out/eval_stage2_fastdata_hf_step5000

"$PY" scripts/eval/ditto_quant/evaluate.py \
  --manifest "$EVAL_OUT/manifests/quant_manifest.jsonl" \
  --output-dir "$EVAL_OUT/metrics_vlm" \
  --clip-model models/openai_clip-vit-large-patch14 \
  --device cuda \
  --vlm-backend local-qwen \
  --vlm-model models/Qwen3-VL-8B-Thinking \
  --vlm-device cuda:0
```

### 评测结果记录

2026-07-27 本地确认：

- Public repo: `Ouzhang/low-high-stage2-fastdata-checkpoints-public`
- Repo API confirmed `private=false`, commit sha `2fa6f28659ef1fe03cc37cbf599433bb5db0b799`
- HF 上共有 18 个 `safetensors`：EasyVFX only、text/hint mem only、text/hint mem + aux all 各 6 个 checkpoint。
- 每组已上传 `step-1000/2000/2500/3000/4000/5000.safetensors`。
- all new methods 机器四尚未成功产出 `real_train_output`，所以目前没有上传。
- 本地已用直链下载校验 `text_hint_mem_auxall step-5000`：
  `out/edit_model_face_stage1/hf_download_check/auxall_step-5000.safetensors`
- 下载文件大小：`252M`
- sha256: `93e38cedbfa4df1361b960b2c9a52f119dc1306af046286508ac7345cb084436`
- 本地无 `Wan2.1-VACE-14B` 基础模型、无 `torch/CUDA`，因此本地不能直接生成视频评测；真实指标仍需在 SI/GPU 机器跑 inference + evaluate。

| method | checkpoint | HF path | download check | inference | metric output | result summary | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EasyVFX only | `step-1000/2000/2500/3000/4000/5000` | `checkpoints/stage2_exp1_step7000_fastdata/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata/` | API confirmed | pending | pending | pending | 已上传 6 个 checkpoint |
| text/hint mem only | `step-1000/2000/2500/3000/4000/5000` | `checkpoints/stage2_exp1_step7000_fastdata/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata/` | API confirmed | pending | pending | pending | 已上传 6 个 checkpoint |
| text/hint mem + aux all | `step-1000/2000/2500/3000/4000/5000` | `checkpoints/stage2_exp1_step7000_fastdata/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata/` | API confirmed; step-5000 downloaded | pending | pending | pending | 已上传 6 个 checkpoint |
| all new methods | all existing `step-*.safetensors` | `checkpoints/stage2_exp1_step7000_fastdata/stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata/` | not uploaded | pending | pending | pending | 机器四未产出 checkpoint |
| EasyVFX only | continuation `step-*.safetensors` | `checkpoints/stage2_exp1_step7000_fastdata_from5k_to20k/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata_from5k_to20k/` | pending | pending | pending | pending | 续训完成后上传 |
| text/hint mem only | continuation `step-*.safetensors` | `checkpoints/stage2_exp1_step7000_fastdata_from5k_to20k/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata_from5k_to20k/` | pending | pending | pending | pending | 续训完成后上传 |
| text/hint mem + aux all | continuation `step-*.safetensors` | `checkpoints/stage2_exp1_step7000_fastdata_from5k_to20k/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata_from5k_to20k/` | pending | pending | pending | pending | 续训完成后上传 |
| all new methods | continuation `step-*.safetensors` | `checkpoints/stage2_exp1_step7000_fastdata_from5k_to20k/stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata_from5k_to20k/` | pending | pending | pending | pending | 续训完成后上传 |

## Val50 / Top50 同配置评测上传 checkpoints

目标：把 HF 上已上传的 18 个 checkpoint 在论文表同一 Top50/val50 协议下评一遍：

- 样本：`reports/top50_video_bundle_20260725_nodup/sample_index.json` 的 50 条。
- 生成：`832x480`、`17 frames`、`20 denoising steps`、`fps=20`、固定 seed。
- 指标列：`CLIP-F`、`DINO-F`、`PickScore`、`JEDi`、`Reward overall`、`Narr./emotion`、`Style/world`、`Comp./lighting`、`Color/texture`、`Hierarchy/readability`。
- 当前可评：EasyVFX only、text/hint mem only、text/hint mem + aux all，各 `step-1000/2000/2500/3000/4000/5000`，共 18 个方法。
- 机器四 all new methods 还没产出 checkpoint，等它有 `real_train_output/step-*.safetensors` 后再加入。

注意：`out/eval_newsplit50/metric_details/eval_inputs.jsonl` 实际是 59 行，dedup 版是 46 行；不要用它复现这张 Top50 表。下面命令从 `top50_video_bundle_20260725_nodup/sample_index.json` 生成严格 50 行输入。

### 1. 下载 HF checkpoints 并构建 Top50 eval inputs

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export ROOT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export PYTHONPATH=$ROOT/src:$ROOT/Ditto
export HF_ENDPOINT=https://hf-mirror.com
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public
export RUN=out/eval_stage2_fastdata_top50_20260727
export CKPT_ROOT=$RUN/hf_checkpoints
export TOP50_INDEX=reports/top50_video_bundle_20260725_nodup/sample_index.json
export TOP50_INPUTS=$RUN/eval_inputs.top50.si.jsonl

mkdir -p "$RUN" "$CKPT_ROOT"

hf download "$HF_REPO" \
  --repo-type model \
  --include 'checkpoints/stage2_exp1_step7000_fastdata/**/*.safetensors' \
  --local-dir "$CKPT_ROOT"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("out/eval_stage2_fastdata_top50_20260727")
src = Path("reports/top50_video_bundle_20260725_nodup/sample_index.json")
dst = root / "eval_inputs.top50.si.jsonl"
high_manifest = root / "high_manifest.top50.jsonl"

prefix_pairs = [
    ("/home/l/liyj/shiying", "/mnt/si002961ale4/default/lgy/shiying"),
    ("/mnt/a100b/default/lgy/shiying", "/mnt/si002961ale4/default/lgy/shiying"),
]

def fix_path(value):
    if not isinstance(value, str):
        return value
    for old, new in prefix_pairs:
        if value.startswith(old + "/"):
            return new + value[len(old):]
    return value

items = json.loads(src.read_text(encoding="utf-8"))
if len(items) != 50:
    raise SystemExit(f"expected 50 top50 items, got {len(items)}")

eval_rows = []
high_rows = []
missing = []
for item in items:
    source = fix_path(item["source_video"])
    target = fix_path(item["target_video"])
    if not Path(source).is_file():
        missing.append(source)
    if not Path(target).is_file():
        missing.append(target)
    row = {
        "sample_id": item["sample_id"],
        "source_video": source,
        "control_video": source,
        "target_video": target,
        "instruction": item["instruction"],
    }
    eval_rows.append(row)
    high_rows.append({**row, "method": "high", "edited_video": target})

if missing:
    print("missing path sample:")
    print("\n".join(missing[:20]))
    raise SystemExit(f"missing {len(missing)} top50 source/target paths")

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    for row in eval_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
with high_manifest.open("w", encoding="utf-8") as f:
    for row in high_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("wrote", dst, len(eval_rows))
print("wrote", high_manifest, len(high_rows))
PY
```

### 2. 生成 18 组视频

EasyVFX-only 不启用 latent memory：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export PYTHONPATH=$PWD/src:$PWD/Ditto
export RUN=out/eval_stage2_fastdata_top50_20260727
export CKPT_DIR=$RUN/hf_checkpoints/checkpoints/stage2_exp1_step7000_fastdata/stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata

ARGS=()
for STEP in 1000 2000 2500 3000 4000 5000; do
  ARGS+=(--checkpoint "easyvfx_s${STEP}=$CKPT_DIR/step-${STEP}.safetensors")
done

"$PY" scripts/eval/ditto_quant/run_inference.py \
  --eval-inputs "$RUN/eval_inputs.top50.si.jsonl" \
  --output-dir "$RUN/infer_easyvfx" \
  --checkpoint-path models/Wan2.1-VACE-14B \
  --python-executable "$PY" \
  "${ARGS[@]}" \
  --device-id 0 \
  --height 480 \
  --width 832 \
  --num-frames 17 \
  --num-inference-steps 20 \
  --fps 20 \
  --seed 1 \
  --skip-existing \
  --fail-fast
```

text/hint memory only：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export PYTHONPATH=$PWD/src:$PWD/Ditto
export RUN=out/eval_stage2_fastdata_top50_20260727
export CKPT_DIR=$RUN/hf_checkpoints/checkpoints/stage2_exp1_step7000_fastdata/stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata

ARGS=()
for STEP in 1000 2000 2500 3000 4000 5000; do
  ARGS+=(--checkpoint "textmem_s${STEP}=$CKPT_DIR/step-${STEP}.safetensors")
done

"$PY" scripts/eval/ditto_quant/run_inference.py \
  --eval-inputs "$RUN/eval_inputs.top50.si.jsonl" \
  --output-dir "$RUN/infer_textmem" \
  --checkpoint-path models/Wan2.1-VACE-14B \
  --python-executable "$PY" \
  "${ARGS[@]}" \
  --device-id 0 \
  --height 480 \
  --width 832 \
  --num-frames 17 \
  --num-inference-steps 20 \
  --fps 20 \
  --seed 1 \
  --latent-memory-mode text+vace_hint \
  --latent-memory-tokens 4 \
  --latent-memory-hidden-dim 0 \
  --latent-memory-scale 1.0 \
  --latent-memory-init-std 0.02 \
  --skip-existing \
  --fail-fast
```

text/hint memory + aux all：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export PYTHONPATH=$PWD/src:$PWD/Ditto
export RUN=out/eval_stage2_fastdata_top50_20260727
export CKPT_DIR=$RUN/hf_checkpoints/checkpoints/stage2_exp1_step7000_fastdata/stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata

ARGS=()
for STEP in 1000 2000 2500 3000 4000 5000; do
  ARGS+=(--checkpoint "auxall_s${STEP}=$CKPT_DIR/step-${STEP}.safetensors")
done

"$PY" scripts/eval/ditto_quant/run_inference.py \
  --eval-inputs "$RUN/eval_inputs.top50.si.jsonl" \
  --output-dir "$RUN/infer_auxall" \
  --checkpoint-path models/Wan2.1-VACE-14B \
  --python-executable "$PY" \
  "${ARGS[@]}" \
  --device-id 0 \
  --height 480 \
  --width 832 \
  --num-frames 17 \
  --num-inference-steps 20 \
  --fps 20 \
  --seed 1 \
  --latent-memory-mode text+vace_hint \
  --latent-memory-tokens 4 \
  --latent-memory-hidden-dim 0 \
  --latent-memory-scale 1.0 \
  --latent-memory-init-std 0.02 \
  --skip-existing \
  --fail-fast
```

### 3. 合并 manifest 并准备全指标目录

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export RUN=out/eval_stage2_fastdata_top50_20260727
export FULL=$RUN/full_metrics

"$PY" - <<'PY'
import json
from pathlib import Path

run = Path("out/eval_stage2_fastdata_top50_20260727")
inputs = {
    json.loads(line)["sample_id"]: json.loads(line)
    for line in (run / "eval_inputs.top50.si.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
}
manifest_paths = [
    run / "infer_easyvfx/manifests/quant_manifest.jsonl",
    run / "infer_textmem/manifests/quant_manifest.jsonl",
    run / "infer_auxall/manifests/quant_manifest.jsonl",
]
rows = []
for path in manifest_paths:
    if not path.is_file():
        raise SystemExit(f"missing manifest: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref = inputs[row["sample_id"]]
        row["target_video"] = ref["target_video"]
        rows.append(row)

methods = sorted({row["method"] for row in rows})
counts = {method: sum(row["method"] == method for row in rows) for method in methods}
bad = {method: count for method, count in counts.items() if count != 50}
if len(methods) != 18 or bad:
    raise SystemExit(f"expected 18 methods x 50 rows, got methods={len(methods)} bad={bad}")

out = run / "full_metrics/manifest_all.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("wrote", out, "rows", len(rows), "methods", len(methods))
PY

"$PY" scripts/eval/val30_full_metrics/prepare_full_metrics.py \
  --manifest "$FULL/manifest_all.jsonl" \
  --output-root "$FULL" \
  --num-shards 6 \
  --expected-methods 18 \
  --expected-samples 50
```

### 4. 跑 CLIP-F、DINO-F、传统指标、Reward

这些命令可按 shard 并行跑；如果只用一台机器，就循环跑完 6 个 shard。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export PYTHONPATH=$PWD/src:$PWD/Ditto
export RUN=out/eval_stage2_fastdata_top50_20260727
export FULL=$RUN/full_metrics
export TMPDIR=/mnt/si002961ale4/default/lgy/shiying/tmp
mkdir -p "$TMPDIR"

for SHARD in 00 01 02 03 04 05; do
  MANIFEST="$FULL/manifests/shard_${SHARD}.jsonl"
  OUT="$FULL/results/shard_${SHARD}"
  mkdir -p "$OUT"

  "$PY" reference/benchmarks/edit/run_traditional_metrics.py \
    --manifest "$MANIFEST" \
    --output "$OUT/traditional.csv" \
    --frames-per-video 8 \
    --resize 224 \
    --pyiqa-metrics musiq,niqe \
    --lpips \
    --device cuda

  "$PY" scripts/eval/ditto_quant/evaluate.py \
    --manifest "$MANIFEST" \
    --output-dir "$OUT/clip" \
    --clip-model openai/clip-vit-large-patch14 \
    --device cuda \
    --frames-per-video 16 \
    --clip-batch-size 8

  "$PY" scripts/eval/val30_full_metrics/evaluate_supp_clip_dino.py \
    --manifest "$MANIFEST" \
    --output "$OUT/supp_clip_dino.csv" \
    --summary "$OUT/supp_clip_dino.summary.json"

  "$PY" scripts/eval/val30_full_metrics/evaluate_laion_aesthetic.py \
    --manifest "$MANIFEST" \
    --output "$OUT/laion_aesthetic.csv" \
    --summary "$OUT/laion_aesthetic.summary.json"
done
```

Reward aesthetic judge：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export PYTHONPATH=$PWD/src:$PWD/Ditto
export RUN=out/eval_stage2_fastdata_top50_20260727
export FULL=$RUN/full_metrics

mkdir -p "$FULL/results/reward"

while read -r METHOD; do
  test -n "$METHOD" || continue
  OUT="$FULL/results/reward/$METHOD"
  SUMMARY="$OUT/summary_aesthetic_reward.json"
  if test -s "$SUMMARY"; then
    echo "SKIP completed method=$METHOD"
    continue
  fi
  "$PY" scripts/eval/ditto_quant/evaluate_aesthetic_reward.py \
    --eval-inputs "$FULL/reward_inputs/$METHOD.jsonl" \
    --generation-root "$FULL/generation_root" \
    --methods "$METHOD" \
    --base-model models/Qwen3-VL-8B-Thinking \
    --adapter /home/l/liyj/shiying/hf_data/critic_models/evaluator_sft_qwen8b_aesthetic5/real_sft_checkpoint \
    --output-dir "$OUT" \
    --frames 2 \
    --device cuda </dev/null
done < "$FULL/methods.txt"
```

如果 SI 上没有 reward adapter，把 `--adapter` 改成实际路径；否则 reward 六列会缺失。

### 5. 跑 PickScore

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export RUN=out/eval_stage2_fastdata_top50_20260727
export FULL=$RUN/full_metrics
export PICK=$RUN/pickscore
mkdir -p "$PICK"

while read -r METHOD; do
  test -n "$METHOD" || continue
  "$PY" scripts/eval/pickscore/evaluate_manifest_group.py \
    --manifest "$FULL/manifest_all.jsonl" \
    --output-dir "$PICK" \
    --method "$METHOD" \
    --expected-count 50 \
    --frames 16 \
    --device cuda
done < "$FULL/methods.txt"

"$PY" scripts/eval/pickscore/aggregate_manifest_groups.py --input-dir "$PICK"
```

### 6. 跑 JEDi

需要 VideoJEDi/V-JEPA 模型目录。先确认下面两个路径存在；不存在就把它们改成你机器上的实际路径。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export RUN=out/eval_stage2_fastdata_top50_20260727
export FULL=$RUN/full_metrics
export JEDI=$RUN/jedi
export JEDI_MODEL_DIR=/mnt/scratch/l/liyj/jedi_val30_7methods/model
export JEDI_CONFIG=$JEDI_MODEL_DIR/vith16_ssv2_16x2x3.yaml
mkdir -p "$JEDI/features" "$JEDI/results"

test -d "$JEDI_MODEL_DIR" || { echo "missing JEDI_MODEL_DIR=$JEDI_MODEL_DIR"; exit 1; }
test -f "$JEDI_CONFIG" || { echo "missing JEDI_CONFIG=$JEDI_CONFIG"; exit 1; }

"$PY" scripts/eval/jedi/extract_val30_features.py \
  --manifest "$RUN/high_manifest.top50.jsonl" \
  --model-dir "$JEDI_MODEL_DIR" \
  --config "$JEDI_CONFIG" \
  --output-dir "$JEDI/features" \
  --method-id high \
  --display-name high \
  --expected-count 50 \
  --frames 16 \
  --resolution 224 \
  --batch-size 2

while read -r METHOD; do
  test -n "$METHOD" || continue
  "$PY" scripts/eval/jedi/extract_val30_features.py \
    --manifest "$FULL/manifest_all.jsonl" \
    --model-dir "$JEDI_MODEL_DIR" \
    --config "$JEDI_CONFIG" \
    --output-dir "$JEDI/features" \
    --method-id "$METHOD" \
    --display-name "$METHOD" \
    --expected-count 50 \
    --frames 16 \
    --resolution 224 \
    --batch-size 2
done < "$FULL/methods.txt"

"$PY" - <<'PY'
import csv
import json
import re
from pathlib import Path
import numpy as np

run = Path("out/eval_stage2_fastdata_top50_20260727")
feature_dir = run / "jedi/features"
output_dir = run / "jedi/results"
output_dir.mkdir(parents=True, exist_ok=True)

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")

def official_jedi(reference, test):
    gamma = 1.0 / reference.shape[1]
    xx = np.square(gamma * (reference @ reference.T))
    yy = np.square(gamma * (test @ test.T))
    xy = np.square(gamma * (reference @ test.T))
    return float((xx.mean() + yy.mean() - 2.0 * xy.mean()) * 100.0)

reference = np.load(feature_dir / "high.npy")
if reference.shape != (50, 1280):
    raise SystemExit(f"bad high feature shape: {reference.shape}")

methods = [line.strip() for line in (run / "full_metrics/methods.txt").read_text().splitlines() if line.strip()]
rows = []
for method in methods:
    test = np.load(feature_dir / f"{safe_name(method)}.npy")
    if test.shape != reference.shape:
        raise SystemExit(f"bad feature shape for {method}: {test.shape}")
    rows.append({"method": method, "n": len(test), "jedi": official_jedi(reference, test)})

with (output_dir / "jedi_val50.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, ["method", "n", "jedi"])
    writer.writeheader()
    writer.writerows(rows)
(output_dir / "jedi_val50.json").write_text(json.dumps({"results": rows}, indent=2) + "\n")
print("wrote", output_dir / "jedi_val50.csv")
PY
```

### 7. 汇总成和论文表同列的结果

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PY=/opt/conda/bin/python3
export RUN=out/eval_stage2_fastdata_top50_20260727
export FULL=$RUN/full_metrics

"$PY" scripts/eval/val30_full_metrics/aggregate_full_metrics.py \
  --root "$FULL" \
  --output-dir "$RUN/full_metrics_aggregate"

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

run = Path("out/eval_stage2_fastdata_top50_20260727")
full = json.loads((run / "full_metrics_aggregate/summary.json").read_text())
records = {row["method"]: row for row in full["records"]}

pick = {}
for path in (run / "pickscore").glob("*/summary.json"):
    payload = json.loads(path.read_text())
    pick[payload["method"]] = payload.get("pickscore_mean")

jedi = {}
jedi_path = run / "jedi/results/jedi_val50.csv"
if jedi_path.is_file():
    with jedi_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            jedi[row["method"]] = float(row["jedi"])

fields = [
    "Method",
    "CLIP-F ↑",
    "DINO-F ↑",
    "PickScore ↑",
    "JEDi ↓",
    "Reward overall ↑",
    "Narr./emotion ↑",
    "Style/world ↑",
    "Comp./lighting ↑",
    "Color/texture ↑",
    "Hierarchy/readability ↑",
]

def fmt(value, digits=4):
    return "" if value is None else f"{float(value):.{digits}f}"

rows = []
for method in sorted(records):
    r = records[method]
    rows.append({
        "Method": method,
        "CLIP-F ↑": fmt((r.get("clip_frame_t") or 0) / 100.0, 4) if r.get("clip_frame_t") is not None else "",
        "DINO-F ↑": fmt(r.get("dino_frame"), 4),
        "PickScore ↑": fmt(pick.get(method), 2),
        "JEDi ↓": fmt(jedi.get(method), 2),
        "Reward overall ↑": fmt(r.get("reward_overall"), 2),
        "Narr./emotion ↑": fmt(r.get("reward_narrative"), 2),
        "Style/world ↑": fmt(r.get("reward_style"), 2),
        "Comp./lighting ↑": fmt(r.get("reward_composition"), 2),
        "Color/texture ↑": fmt(r.get("reward_color"), 2),
        "Hierarchy/readability ↑": fmt(r.get("reward_hierarchy"), 2),
    })

csv_path = run / "stage2_fastdata_top50_scores.csv"
md_path = run / "stage2_fastdata_top50_scores.md"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Stage2 Fastdata Top50 Scores",
    "",
    "| " + " | ".join(fields) + " |",
    "| " + " | ".join(["---"] + ["---:"] * (len(fields) - 1)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(row[field] for field in fields) + " |")
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path)
print(csv_path)
PY
```

结果出来后，把 `$RUN/stage2_fastdata_top50_scores.md` 的表贴回这里，或者直接把那段表追加到本 md 的下面。

### Val50 / Top50 结果表

| Method | CLIP-F ↑ | DINO-F ↑ | PickScore ↑ | JEDi ↓ | Reward overall ↑ | Narr./emotion ↑ | Style/world ↑ | Comp./lighting ↑ | Color/texture ↑ | Hierarchy/readability ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easyvfx_s1000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| easyvfx_s2000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| easyvfx_s2500 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| easyvfx_s3000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| easyvfx_s4000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| easyvfx_s5000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| textmem_s1000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| textmem_s2000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| textmem_s2500 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| textmem_s3000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| textmem_s4000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| textmem_s5000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| auxall_s1000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| auxall_s2000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| auxall_s2500 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| auxall_s3000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| auxall_s4000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| auxall_s5000 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

### liyj 机器实际提交状态

2026-07-27 已在 `liyj@xlogin.comp.nus.edu.sg` 上启动评测，路径：

```bash
PROJECT=/home/l/liyj/shiying/low-high-new
RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727
```

已从 public HF mirror 下载并识别到 18 个 checkpoint：

- `easyvfx_s1000` 到 `easyvfx_s5000`
- `textmem_s1000` 到 `textmem_s5000`
- `auxall_s1000` 到 `auxall_s5000`

`stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata` 当前 HF 上仍未找到 checkpoint，下载返回 404，本轮评测暂不包含机器四。

Slurm job（第一次提交）：

```text
INFER_JOB=700201
PREP_METRICS_JOB=700202
STAGE1_SUBMIT_JOB=700216
```

当前状态：

```text
700201_[0-17%6]  gpu-long  stage2-top50-infer    PENDING  (Priority)
700202           normal    stage2-top50-merge    PENDING  (Dependency)
700216           normal    stage2-top50-submit1  PENDING  (Dependency)
```

因为集群有 `AssocMaxSubmitJobLimit`，后续指标任务改成分阶段提交：推理完成后自动合并 manifest，然后依次提交 core/reward、PickScore、JEDi、aggregate，避免一次性提交过多数组任务。

2026-07-27 为了尽快拿结果，已把推理数组并发从 `%6` 提高到最大 `%18`。也尝试过更激进的 5-sample chunk 方案，但 180 个数组元素会触发 `AssocMaxSubmitJobLimit`，不能提交。当前恢复为稳定的 18-task 最大并发方案：

```text
INFER_JOB=700277
PREP_METRICS_JOB=700278
STAGE1_SUBMIT_JOB=700279
TASKS=18
ARRAY_THROTTLE=18
```

当前队列状态：

```text
700277_[0-17%18]  gpu-long  stage2-top50-infer    PENDING  (Priority)
700278            normal    stage2-top50-merge    PENDING  (Dependency)
700279            normal    stage2-top50-submit1  PENDING  (Dependency)
```

说明：`%18` 已经允许 18 个 checkpoint 同时跑，能用多少张 A100-80 取决于 Slurm 实际释放多少卡。短分区 `gpu` 最大时间只有 3 小时，不适合单 checkpoint 50 条视频的完整任务；强行切成 180 个小任务又被账号提交额度拒绝，所以当前这个是可提交且不容易 OOM 的最快稳定方案。

2026-07-27 账号权限诊断：

```text
结论：`liyj` 账号不是没有 GPU 权限，当前主要是 Slurm 排队优先级问题。

证据：
- 当前正式任务 `700277_[0-17%18]` 已被 Slurm 接受，状态是 `PENDING (Priority)`，不是 `InvalidAccount`、`QOSMaxGRES`、`ReqNodeNotAvail` 或权限拒绝。
- 临时提交的最小 GPU 测试任务也能进入队列：`700367 gpu wrap PENDING (Priority)`，说明账号可以向 GPU 分区提交任务。该测试任务已取消。
- `scontrol show job 700277_0` 显示：
  `Account=allusers QOS=normal`
  `JobState=PENDING Reason=Priority`
  `ReqTRES=cpu=10,mem=160G,node=1,billing=21,gres/gpu=1,gres/gpu:a100-80=1`
- `sacctmgr show assoc where user=liyj` 显示该账号有关联到 `allusers/normal`，并有 `MaxJobs=16`、`MaxSubmit=32` 一类限制。
- `sshare` 显示 `liyj` fairshare 很低：`FairShare=0.001361`，同时 RawUsage 很高，所以排队优先级会偏低。
```

因此现在不需要换命令重跑，也不是脚本没启动；要么等调度，要么请管理员/组里调高账号 fairshare/QOS/priority，或者换一个 fairshare 更高、能跑 `gpu-long` A100-80 的账号。

### SI 四台机器并行跑完整 Top50 评测（不下载 checkpoint）

这段是在 SI 机器上跑的版本，直接使用本地训练输出：

```text
out/edit_model_face_stage1/<实验名>/real_train_output/step-*.safetensors
```

四台机器分别评一个 family，每台机器跑该 family 的 6 个 checkpoint：

```text
机器一：easyvfx_s1000/s2000/s2500/s3000/s4000/s5000
机器二：textmem_s1000/s2000/s2500/s3000/s4000/s5000
机器三：auxall_s1000/s2000/s2500/s3000/s4000/s5000
机器四：allnew_s1000/s2000/s2500/s3000/s4000/s5000
```

说明：

- 每台 SI 机器先粘自己对应的变量块，然后继续粘“每台 SI 机器都继续执行这一段”。
- 这版不从 HF 下载 checkpoint，也不使用 `sbatch`；它在当前容器里前台阻塞运行。
- 一个 family 只有 6 个 checkpoint；如果只按 checkpoint 并行，8 卡机器最多只用 6 张卡。下面推荐的 8 卡版会把 Top50 再切成 chunk，让 8 张卡都能持续拿任务。
- 看到命令开始后不要关闭终端；命令会 `wait` 到 6 个推理进程全部结束。
- 如果某台机器只有 4 张可用 GPU，把 `GPU_IDS=(0 1 2 3 4 5 6 7)` 改成 `GPU_IDS=(0 1 2 3)`。
- 机器四必须先确认 `allnew` 目录里已经有 `step-*.safetensors`。

#### SI 机器一：easyvfx

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PROJECT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export BASE_MODEL=$PROJECT/models/Wan2.1-VACE-14B
export FAMILY_SHORT=easyvfx
export LATENT_MODE=none
export FAMILY_DIR=stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata
export LOCAL_CKPT_DIR=$PROJECT/out/edit_model_face_stage1/$FAMILY_DIR/real_train_output
export RUN=$PROJECT/out/eval_stage2_fastdata_top50_easyvfx_$(date +%Y%m%d_%H%M%S)
```

#### SI 机器二：text memory only

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PROJECT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export BASE_MODEL=$PROJECT/models/Wan2.1-VACE-14B
export FAMILY_SHORT=textmem
export LATENT_MODE=text+vace_hint
export FAMILY_DIR=stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata
export LOCAL_CKPT_DIR=$PROJECT/out/edit_model_face_stage1/$FAMILY_DIR/real_train_output
export RUN=$PROJECT/out/eval_stage2_fastdata_top50_textmem_$(date +%Y%m%d_%H%M%S)
```

#### SI 机器三：text memory + aux all

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PROJECT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export BASE_MODEL=$PROJECT/models/Wan2.1-VACE-14B
export FAMILY_SHORT=auxall
export LATENT_MODE=text+vace_hint
export FAMILY_DIR=stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata
export LOCAL_CKPT_DIR=$PROJECT/out/edit_model_face_stage1/$FAMILY_DIR/real_train_output
export RUN=$PROJECT/out/eval_stage2_fastdata_top50_auxall_$(date +%Y%m%d_%H%M%S)
```

#### SI 机器四：all new methods

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PROJECT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export PY=/opt/conda/bin/python3
export BASE_MODEL=$PROJECT/models/Wan2.1-VACE-14B
export FAMILY_SHORT=allnew
export LATENT_MODE=text+vace_hint
export FAMILY_DIR=stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata
export LOCAL_CKPT_DIR=$PROJECT/out/edit_model_face_stage1/$FAMILY_DIR/real_train_output
export RUN=$PROJECT/out/eval_stage2_fastdata_top50_allnew_$(date +%Y%m%d_%H%M%S)
```

#### SI 环境准备和路径检查

每台 SI 机器选完上面的变量块后，先跑这一段。它会安装/补齐推理和评测脚本需要的 Python 包，并检查基础模型、Top50 数据、本地 LoRA checkpoint 是否存在。

```bash
set -euo pipefail

cd "$PROJECT"

export PYTHONPATH=$PROJECT/src:$PROJECT/Ditto
export HF_HOME=$PROJECT/models/cache/huggingface
export TORCH_HOME=$PROJECT/models/cache/torch
export XDG_CACHE_HOME=$PROJECT/models/cache
export TMPDIR=$PROJECT/out/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$RUN/logs"

echo "=== python ==="
"$PY" --version

echo "=== install python deps ==="
"$PY" -m pip install \
  imageio imageio-ffmpeg ftfy modelscope pynvml peft pandas datasets \
  safetensors einops qwen-vl-utils pillow protobuf accelerate pydantic transformers \
  -i https://pypi.org/simple
"$PY" -m pip install sentencepiece -i https://pypi.org/simple

echo "=== import checks ==="
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
import safetensors, pandas, datasets, transformers, accelerate
print("basic deps OK")
try:
    from diffsynth.models.latent_memory_runtime import LatentMemoryConfig
    print("latent memory OK")
except Exception as e:
    print("latent memory import failed:", repr(e))
try:
    from diffsynth.models.easyvfx_frequency_runtime import EasyVFXFrequencyLoss
    print("EasyVFX OK")
except Exception as e:
    print("EasyVFX import skipped/failed:", repr(e))
PY

echo "=== model path check ==="
test -d "$BASE_MODEL"
ls -lh "$BASE_MODEL" | head
find "$BASE_MODEL" -maxdepth 1 \( -name 'diffusion_pytorch_model-*.safetensors' -o -name 'models_t5_umt5-xxl-enc-bf16.pth' -o -name 'Wan2.1_VAE.pth' \) -print | sort

echo "=== Top50 data check ==="
if ! test -f reports/top50_video_bundle_20260725_nodup/sample_index.json; then
  echo "missing reports/top50_video_bundle_20260725_nodup/sample_index.json"
  echo "先把本地 low-high-new/reports/top50_video_bundle_20260725_nodup/ 同步到 SI 的 reports/ 下面，或者 git pull 到包含这个目录的版本。"
  exit 2
fi
"$PY" - <<'PY'
import json
from pathlib import Path
p = Path("reports/top50_video_bundle_20260725_nodup/sample_index.json")
rows = json.loads(p.read_text())
print("top50 samples", len(rows))
print(rows[0]["sample_id"])
for key in ("source_video", "target_video"):
    path = Path(rows[0][key])
    print(key, path, path.exists())
    if not path.exists():
        candidates = [
            str(path).replace("/home/l/liyj/shiying", "/mnt/si002961ale4/default/lgy/shiying"),
            str(path).replace("/mnt/scratch/l/liyj", "/mnt/si002961ale4/default/lgy/shiying"),
        ]
        for candidate in candidates:
            c = Path(candidate)
            print(" candidate", c, c.exists())
PY

echo "=== local ckpt check ==="
echo "LOCAL_CKPT_DIR=$LOCAL_CKPT_DIR"
test -d "$LOCAL_CKPT_DIR"
find "$LOCAL_CKPT_DIR" -maxdepth 1 -name 'step-*.safetensors' -print | sort -V | xargs -r ls -lh
test "$(find "$LOCAL_CKPT_DIR" -maxdepth 1 -name 'step-*.safetensors' -print | wc -l | tr -d ' ')" -ge 1

echo "=== git/code check ==="
test -f scripts/eval/stage2_fastdata_top50/prepare_stage2_fastdata_top50.py
test -f scripts/eval/stage2_fastdata_top50/prepare_infer_chunks.py
test -f scripts/eval/ditto_quant/run_inference.py
test -f Ditto/inference/infer_ditto.py
"$PY" -m py_compile \
  scripts/eval/stage2_fastdata_top50/prepare_stage2_fastdata_top50.py \
  scripts/eval/stage2_fastdata_top50/prepare_infer_chunks.py \
  scripts/eval/stage2_fastdata_top50/merge_inference_manifests.py \
  scripts/eval/ditto_quant/run_inference.py

echo "SI env ready"
```

如果失败停在 `=== Top50 data check ===` 后面，先跑这个确认：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

ls -lh reports/top50_video_bundle_20260725_nodup/ 2>/dev/null || true
test -f reports/top50_video_bundle_20260725_nodup/sample_index.json && echo "sample_index exists" || echo "sample_index missing"

/opt/conda/bin/python3 - <<'PY'
import json
from pathlib import Path
p = Path("reports/top50_video_bundle_20260725_nodup/sample_index.json")
if not p.exists():
    raise SystemExit("missing sample_index.json: 需要把 reports/top50_video_bundle_20260725_nodup/ 同步到 SI")
rows = json.loads(p.read_text())
print("samples", len(rows), "first", rows[0]["sample_id"])
for key in ("source_video", "target_video"):
    raw = Path(rows[0][key])
    candidates = [
        raw,
        Path(str(raw).replace("/home/l/liyj/shiying", "/mnt/si002961ale4/default/lgy/shiying")),
        Path(str(raw).replace("/mnt/scratch/l/liyj", "/mnt/si002961ale4/default/lgy/shiying")),
    ]
    print(key)
    for c in candidates:
        print(" ", c, c.exists())
PY
```

如果 `sample_index missing`，要先同步这个目录；如果 `sample_index exists` 但视频候选路径全是 `False`，说明 Top50 的 source/target 视频也不在这台 SI 机器上，需要先把 Top50 对应的 low/high 视频同步到 SI，或者改 `sample_index.json` 指到 SI 上真实存在的视频路径。

#### 每台 SI 机器都继续执行这一段

推荐用这个 8 卡 chunk 版：

```bash
set -euo pipefail

cd "$PROJECT"
export PYTHONPATH=$PROJECT/src:$PROJECT/Ditto
export HF_HOME=$PROJECT/models/cache/huggingface
export TORCH_HOME=$PROJECT/models/cache/torch
export XDG_CACHE_HOME=$PROJECT/models/cache
export TMPDIR=$PROJECT/out/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/logs" "$RUN/local_checkpoints/checkpoints/stage2_exp1_step7000_fastdata" "$TMPDIR"

echo "PROJECT=$PROJECT"
echo "RUN=$RUN"
echo "LOCAL_CKPT_DIR=$LOCAL_CKPT_DIR"
echo "BASE_MODEL=$BASE_MODEL"

test -d "$LOCAL_CKPT_DIR"
find "$LOCAL_CKPT_DIR" -maxdepth 1 -name 'step-*.safetensors' -print | sort -V
test "$(find "$LOCAL_CKPT_DIR" -maxdepth 1 -name 'step-*.safetensors' -print | wc -l | tr -d ' ')" -ge 1

LINK_ROOT="$RUN/local_checkpoints/checkpoints/stage2_exp1_step7000_fastdata/$FAMILY_DIR"
ln -sfn "$LOCAL_CKPT_DIR" "$LINK_ROOT"

"$PY" scripts/eval/stage2_fastdata_top50/prepare_stage2_fastdata_top50.py \
  --sample-index reports/top50_video_bundle_20260725_nodup/sample_index.json \
  --checkpoint-root "$RUN/local_checkpoints" \
  --run-root "$RUN" \
  --expected-samples 50

echo "=== methods ==="
cat "$RUN/methods.tsv"

"$PY" scripts/eval/stage2_fastdata_top50/prepare_infer_chunks.py \
  --run-root "$RUN" \
  --chunk-size 5

echo "=== chunk tasks ==="
wc -l "$RUN/infer_tasks.tsv"
head "$RUN/infer_tasks.tsv"

GPU_IDS=(0 1 2 3 4 5 6 7)
MAX_PARALLEL=${#GPU_IDS[@]}
TASKS_TOTAL=$(wc -l < "$RUN/infer_tasks.tsv" | tr -d ' ')
NEXT_TASK=1
pids=()
pid_names=()

launch_task() {
  local line_no="$1"
  local gpu="$2"
  local row task_id method latent_mode ckpt chunk_input out log
  row=$(sed -n "${line_no}p" "$RUN/infer_tasks.tsv")
  IFS=$'\t' read -r task_id method latent_mode ckpt chunk_input out <<< "$row"
  log="$RUN/logs/infer_task${task_id}_${method}_gpu${gpu}.log"
  echo "launch task=$task_id/$TASKS_TOTAL method=$method gpu=$gpu chunk=$chunk_input log=$log"

  ARGS=(
    scripts/eval/ditto_quant/run_inference.py
    --eval-inputs "$chunk_input"
    --output-dir "$out"
    --checkpoint "$method=$ckpt"
    --checkpoint-path "$BASE_MODEL"
    --python-executable "$PY"
    --device-id "$gpu"
    --height 480
    --width 832
    --num-frames 17
    --num-inference-steps 20
    --cfg-scale 1.0
    --seed 42
    --fps 20
    --quality 5
    --default-lora-alpha 1.0
    --skip-existing
    --fail-fast
  )
  if test "$latent_mode" != "none"; then
    ARGS+=(--latent-memory-mode "$latent_mode" --latent-memory-tokens 4 --latent-memory-hidden-dim 0 --latent-memory-scale 1.0 --latent-memory-init-std 0.02)
  fi

  "$PY" "${ARGS[@]}" >"$log" 2>&1 &
  pids[$gpu]=$!
  pid_names[$gpu]="task=$task_id method=$method log=$log"
}

for gpu in "${GPU_IDS[@]}"
do
  if test "$NEXT_TASK" -le "$TASKS_TOTAL"; then
    launch_task "$NEXT_TASK" "$gpu"
    NEXT_TASK=$((NEXT_TASK + 1))
  fi
done

status=0
while true
do
  active=0
  for gpu in "${GPU_IDS[@]}"
  do
    pid="${pids[$gpu]:-}"
    if test -z "$pid"; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      active=$((active + 1))
      continue
    fi
    if wait "$pid"; then
      echo "done gpu=$gpu ${pid_names[$gpu]}"
    else
      echo "failed gpu=$gpu ${pid_names[$gpu]}"
      status=1
    fi
    pids[$gpu]=""
    if test "$NEXT_TASK" -le "$TASKS_TOTAL"; then
      launch_task "$NEXT_TASK" "$gpu"
      NEXT_TASK=$((NEXT_TASK + 1))
      active=$((active + 1))
    fi
  done
  if test "$active" -eq 0; then
    break
  fi
  echo "progress mp4=$(find "$RUN/infer_chunks" -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ') / expected=$((TASKS_TOTAL * 5)) next_task=$NEXT_TASK/$TASKS_TOTAL"
  sleep 60
done

if test "$status" -ne 0; then
  echo "At least one inference chunk failed. Inspect logs:"
  ls -lh "$RUN/logs"
  tail -80 "$RUN"/logs/infer_task*.log
  exit "$status"
fi

"$PY" scripts/eval/stage2_fastdata_top50/merge_inference_manifests.py \
  --run-root "$RUN" \
  --project-root "$PROJECT" \
  --num-shards 6 \
  --expected-samples 50

echo "DONE inference + manifest merge"
echo "RUN=$RUN"
find "$RUN/infer_chunks" -type f -name '*.mp4' | wc -l
find "$RUN/infer_chunks" -type f -name '*.mp4' | sort | tail -20
```

#### 只想最快拿到所有生成视频

上面这段 8 卡 chunk 命令就是“只生成视频最快版”：它不会先跑指标，先把该 family 的所有 mp4 生成出来，然后合并 manifest。

每台机器预期视频数：

```text
单台机器一个 family：6 checkpoints x 50 samples = 300 mp4
三台机器前三组：easyvfx/textmem/auxall = 900 mp4
四台机器全跑：easyvfx/textmem/auxall/allnew = 1200 mp4
```

判断该机器是否生成完：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

echo "RUN=$RUN"
echo "mp4 count:"
find "$RUN/infer_chunks" -type f -name '*.mp4' 2>/dev/null | wc -l

echo "manifest count:"
find "$RUN/infer_chunks" -type f -path '*/manifests/quant_manifest.jsonl' 2>/dev/null | wc -l

echo "merged manifest:"
ls -lh "$RUN/full_metrics/manifest_all.jsonl" 2>/dev/null || true

echo "recent logs:"
ls -lt "$RUN/logs"/infer_task*.log 2>/dev/null | head -20
tail -50 "$RUN"/logs/infer_task*.log 2>/dev/null
```

完成标准：

```text
mp4 count = 300
manifest_all.jsonl 存在
wc -l "$RUN/full_metrics/manifest_all.jsonl" = 300
```

如果某个 checkpoint/chunk 失败，重新运行同一段 8 卡 chunk 命令即可；它带了 `--skip-existing`，已经存在的 mp4 会跳过，只补没生成完的部分。

#### SI 上查看生成进度

同一个终端正在前台跑时，不要中断。另开一个终端查：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

echo "RUN=/path/to/your/run"
find "$RUN/infer" -type f -name '*.mp4' 2>/dev/null | wc -l
find "$RUN/infer" -type f -name '*.mp4' 2>/dev/null | sort | tail -20
ls -lh "$RUN/logs" 2>/dev/null
tail -50 "$RUN"/logs/infer_*.log 2>/dev/null
```

#### SI 上继续跑指标

生成和 `merge_inference_manifests.py` 完成后，`$RUN/full_metrics/manifest_all.jsonl` 会存在。指标可以按你之前同配置的 metric 脚本跑；如果是在有 Slurm 的环境，用下面这些脚本。如果 SI 容器没有 `sbatch`，就先把 `$RUN` 目录同步到能跑指标的机器，或把各 metric 脚本按单进程方式拆开跑。

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

export PROJECT=/mnt/si002961ale4/default/lgy/shiying/low-high-new
export FULL_METRICS_ROOT=$RUN/full_metrics
export FULL_METRICS_CODE=$PROJECT/scripts/eval/val30_full_metrics

sbatch --export=ALL,PROJECT="$PROJECT",FULL_METRICS_ROOT="$FULL_METRICS_ROOT",FULL_METRICS_CODE="$FULL_METRICS_CODE" \
  scripts/eval/val30_full_metrics/run_core_array.sbatch

sbatch --export=ALL,PROJECT="$PROJECT",FULL_METRICS_ROOT="$FULL_METRICS_ROOT" \
  scripts/eval/val30_full_metrics/run_reward_array.sbatch

sbatch --array=0-5%6 --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_pickscore_array.sbatch

sbatch --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_jedi_high.sbatch

sbatch --array=0-5%6 --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_jedi_array.sbatch
```

等指标全部完成后聚合：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

/opt/conda/bin/python3 scripts/eval/pickscore/aggregate_manifest_groups.py --input-dir "$RUN/pickscore"
/opt/conda/bin/python3 scripts/eval/stage2_fastdata_top50/aggregate_jedi_stage2_fastdata_top50.py --run-root "$RUN"
/opt/conda/bin/python3 scripts/eval/val30_full_metrics/aggregate_full_metrics.py \
  --root "$RUN/full_metrics" \
  --output-dir "$RUN/full_metrics_aggregate"
/opt/conda/bin/python3 scripts/eval/stage2_fastdata_top50/aggregate_stage2_fastdata_top50.py --run-root "$RUN"

cat "$RUN/stage2_fastdata_top50_scores.md"
```

### NUS/liyj Slurm + HF 备用版

下面这一段是给 `liyj@xlogin` 机器用的，会从 HF mirror 下载 checkpoint；如果你在 SI 本地跑，用上面的 SI 版本。

#### 四台机器并行跑完整 Top50 评测

目标：四台机器分别评一个 family，每台机器跑该 family 的 6 个 checkpoint：

```text
机器一：easyvfx_s1000/s2000/s2500/s3000/s4000/s5000
机器二：textmem_s1000/s2000/s2500/s3000/s4000/s5000
机器三：auxall_s1000/s2000/s2500/s3000/s4000/s5000
机器四：allnew_s1000/s2000/s2500/s3000/s4000/s5000
```

说明：

- 每台机器只下载并评测自己的一个目录，避免四台重复跑 18/24 个 checkpoint。
- 当前 public HF 上已经确认有前三组：`easyvfx`、`textmem`、`auxall`。
- `allnew` 目录之前返回 404；机器四要先确认 HF 上已经上传了 `stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata/step-*.safetensors`，否则 `prepare_stage2_fastdata_top50.py` 会找不到 checkpoint。
- 每台机器默认提交 6 个推理任务，`%6` 并发；如果账号限制更低，把 `%6` 改成 `%2` 或 `%1`。
- 这套命令会自动：下载 checkpoint -> 生成 Top50 输入 -> 提交 inference array -> inference 完成后合并 manifest -> 分阶段提交 core/reward/PickScore/JEDi/aggregate。

#### 机器一：easyvfx

```bash
cd /home/l/liyj/shiying/low-high-new

export PROJECT=/home/l/liyj/shiying/low-high-new
export BASE_MODEL=/home/l/liyj/shiying/models/Wan2.1-VACE-14B
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public
export FAMILY_DIR=stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata
export RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_easyvfx_$(date +%Y%m%d_%H%M%S)
```

#### 机器二：text memory only

```bash
cd /home/l/liyj/shiying/low-high-new

export PROJECT=/home/l/liyj/shiying/low-high-new
export BASE_MODEL=/home/l/liyj/shiying/models/Wan2.1-VACE-14B
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public
export FAMILY_DIR=stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata
export RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_textmem_$(date +%Y%m%d_%H%M%S)
```

#### 机器三：text memory + aux all

```bash
cd /home/l/liyj/shiying/low-high-new

export PROJECT=/home/l/liyj/shiying/low-high-new
export BASE_MODEL=/home/l/liyj/shiying/models/Wan2.1-VACE-14B
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public
export FAMILY_DIR=stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata
export RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_auxall_$(date +%Y%m%d_%H%M%S)
```

#### 机器四：all new methods

```bash
cd /home/l/liyj/shiying/low-high-new

export PROJECT=/home/l/liyj/shiying/low-high-new
export BASE_MODEL=/home/l/liyj/shiying/models/Wan2.1-VACE-14B
export HF_REPO=Ouzhang/low-high-stage2-fastdata-checkpoints-public
export FAMILY_DIR=stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata
export RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_allnew_$(date +%Y%m%d_%H%M%S)
```

#### 每台机器都继续执行这一段

在上面四组变量中选好对应机器后，同一台机器继续粘贴下面整段：

```bash
set -euo pipefail

export CKPT_ROOT="$RUN/hf_checkpoints"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/l/liyj/shiying/hf_cache
export TMPDIR=/mnt/scratch/l/liyj/tmp
export HF_BASE_URL="https://hf-mirror.com/$HF_REPO/resolve/main/checkpoints/stage2_exp1_step7000_fastdata"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate metagen
cd "$PROJECT"

mkdir -p "$RUN/logs" "$CKPT_ROOT" "$TMPDIR"

DEST_DIR="$CKPT_ROOT/checkpoints/stage2_exp1_step7000_fastdata/$FAMILY_DIR"
mkdir -p "$DEST_DIR"

for STEP in 1000 2000 2500 3000 4000 5000
do
  FILE="step-${STEP}.safetensors"
  URL="$HF_BASE_URL/$FAMILY_DIR/$FILE"
  DEST="$DEST_DIR/$FILE"
  if test -s "$DEST"; then
    echo "skip existing $DEST"
  else
    echo "download $URL"
    curl -fL --retry 8 --retry-delay 10 --connect-timeout 30 --continue-at - \
      -o "$DEST" "$URL"
  fi
done

python scripts/eval/stage2_fastdata_top50/prepare_stage2_fastdata_top50.py \
  --sample-index reports/top50_video_bundle_20260725_nodup/sample_index.json \
  --checkpoint-root "$CKPT_ROOT" \
  --run-root "$RUN" \
  --expected-samples 50

METHODS=$(wc -l < "$RUN/methods.tsv" | tr -d ' ')
LAST=$((METHODS - 1))
echo "RUN=$RUN"
echo "METHODS=$METHODS"
cat "$RUN/methods.tsv"

INFER_JOB=$(sbatch --parsable --array=0-"$LAST"%6 \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN",BASE_MODEL="$BASE_MODEL" \
  scripts/eval/stage2_fastdata_top50/run_infer_array.sbatch)
echo "INFER_JOB=$INFER_JOB"

MERGE_JOB=$(sbatch --parsable \
  --dependency=afterok:"$INFER_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/run_merge_manifests.sbatch)
echo "MERGE_JOB=$MERGE_JOB"

STAGE1_SUBMIT_JOB=$(sbatch --parsable \
  --dependency=afterok:"$MERGE_JOB" \
  --export=ALL,PROJECT="$PROJECT",RUN="$RUN" \
  scripts/eval/stage2_fastdata_top50/submit_metrics_stage1.sbatch)
echo "STAGE1_SUBMIT_JOB=$STAGE1_SUBMIT_JOB"

cat > "$RUN/submitted_jobs.env" <<EOF
RUN=$RUN
FAMILY_DIR=$FAMILY_DIR
METHODS=$METHODS
INFER_JOB=$INFER_JOB
MERGE_JOB=$MERGE_JOB
STAGE1_SUBMIT_JOB=$STAGE1_SUBMIT_JOB
EOF

cat "$RUN/submitted_jobs.env"
squeue -u "$USER" -o '%.18i %.9P %.28j %.8T %.14M %.20R' | grep -E 'JOBID|stage2-top50'
```

#### 四台机器分别查进度

每台机器用自己的 `$RUN` 查：

```bash
cd /home/l/liyj/shiying/low-high-new

cat "$RUN/submitted_jobs.env"
RUN="$RUN" bash scripts/eval/stage2_fastdata_top50/progress.sh

echo "=== job detail ==="
squeue -u "$USER" -o '%.18i %.9P %.28j %.8T %.14M %.20R' | grep -E 'JOBID|stage2-top50'

echo "=== generated ==="
find "$RUN/infer" -type f -name '*.mp4' | wc -l
find "$RUN/infer" -type f -name '*.mp4' | sort | tail -20

echo "=== final scores ==="
ls -lh "$RUN"/stage2_fastdata_top50_scores.* 2>/dev/null || true
test -f "$RUN/stage2_fastdata_top50_scores.md" && cat "$RUN/stage2_fastdata_top50_scores.md"
```

#### 如果只想先抢一个视频

这段只跑 `auxall_s5000` 的 Top50 第一条样本，适合验证模型/环境是否通：

```bash
cd /home/l/liyj/shiying/low-high-new

export PROJECT=/home/l/liyj/shiying/low-high-new
export RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727
export QUICK=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_quickone_$(date +%Y%m%d_%H%M%S)
export BASE_MODEL=/home/l/liyj/shiying/models/Wan2.1-VACE-14B

mkdir -p "$QUICK"
head -1 "$RUN/eval_inputs.top50.jsonl" > "$QUICK/eval_one.jsonl"

cat > "$QUICK/run_quick_h200.sbatch" <<'EOF'
#!/bin/bash
#SBATCH -J stage2-quick-h200
#SBATCH -p gpu-long
#SBATCH --gres=gpu:h200-141:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH -t 04:00:00

set -euo pipefail
PROJECT=/home/l/liyj/shiying/low-high-new
RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727
QUICK=${QUICK:?missing QUICK}
BASE_MODEL=/home/l/liyj/shiying/models/Wan2.1-VACE-14B
CKPT=$(awk -F '\t' '$1=="auxall_s5000" {print $4}' "$RUN/methods.tsv")
OUTDIR=$QUICK/out_h200/auxall_s5000

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate metagen
cd "$PROJECT"

export PYTHONPATH=$PROJECT/src:$PROJECT/Ditto
export HF_HOME=/home/l/liyj/shiying/hf_cache
export TORCH_HOME=/home/l/liyj/.cache/torch
export XDG_CACHE_HOME=/home/l/liyj/shiying/low-high-new/models/cache
export TMPDIR=/mnt/scratch/l/liyj/tmp
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$OUTDIR"

"$HOME/miniconda3/envs/metagen/bin/python" scripts/eval/ditto_quant/run_inference.py \
  --eval-inputs "$QUICK/eval_one.jsonl" \
  --output-dir "$OUTDIR" \
  --checkpoint "auxall_s5000=$CKPT" \
  --checkpoint-path "$BASE_MODEL" \
  --python-executable "$HOME/miniconda3/envs/metagen/bin/python" \
  --device-id 0 \
  --height 480 \
  --width 832 \
  --num-frames 17 \
  --num-inference-steps 20 \
  --cfg-scale 1.0 \
  --seed 42 \
  --fps 20 \
  --quality 5 \
  --default-lora-alpha 1.0 \
  --latent-memory-mode text+vace_hint \
  --latent-memory-tokens 4 \
  --latent-memory-hidden-dim 0 \
  --latent-memory-scale 1.0 \
  --latent-memory-init-std 0.02 \
  --skip-existing \
  --fail-fast
EOF

JOB=$(sbatch --parsable --export=ALL,QUICK="$QUICK" "$QUICK/run_quick_h200.sbatch")
echo "QUICK_JOB=$JOB"
squeue -j "$JOB" -o '%.18i %.9P %.24j %.8T %.14M %.20R'
```

查看 liyj 机器上的评测进度：

```bash
ssh -J liyj@stujump.comp.nus.edu.sg liyj@xlogin.comp.nus.edu.sg
cd /home/l/liyj/shiying/low-high-new
RUN=/mnt/scratch/l/liyj/eval_stage2_fastdata_top50_20260727 \
  bash scripts/eval/stage2_fastdata_top50/progress.sh
```

## 统一查看进度

任意一台机器上运行：

```bash
cd /mnt/si002961ale4/default/lgy/shiying/low-high-new

ps -ef | grep -E 'stage2_exp1_step7000|model_training/train.py|accelerate' | grep -v grep

for OUT in \
  stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata \
  stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata
do
  echo
  echo "=== $OUT ==="
  ls -lh "out/edit_model_face_stage1/$OUT/real_train.log" "out/edit_model_face_stage1/$OUT/real_train_status.json" 2>/dev/null
  find "out/edit_model_face_stage1/$OUT/real_train_output" -maxdepth 1 -name 'step-*.safetensors' -print 2>/dev/null | sort -V | tail
  grep -nEi 'Traceback|Error|Exception|Failed|Killed|OutOfMemory|CUDA|No such file|not found' \
    "out/edit_model_face_stage1/$OUT/real_train.log" "out/edit_model_face_stage1/$OUT/real_train_error.json" 2>/dev/null | tail -20
done
```
