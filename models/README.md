# Models Directory

这个目录用于存放 `low-high-new` 项目运行时依赖的本地模型与权重文件。

当前代码已经改为优先从仓库内的 `models/` 目录读取资源，因此本目录是默认的模型根目录。


## 目录结构

当前主要包含以下几类模型：

- `Wan2.1-VACE-14B/`
  - Wan 视频编辑主模型
  - 用于 `wan14b`、`wan14b-vace-base`、`wan14b+4steplora`、`wan14b+dittolora`、`wan14b+dittolora+4steplora` 等 variant

- `Wan2.1-Distill-Loras/`
  - Wan 的蒸馏 LoRA
  - 当前主要使用：
    - `wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors`

- `Ditto_models/`
  - Ditto 相关 LoRA 权重
  - 当前主要使用：
    - `models/ditto_global.safetensors`

- `Qwen3-VL-8B-Thinking/`
  - planner / critic 默认使用的本地视觉语言模型
  - 当前 `src/vlm/local_qwen.py` 会优先从这里加载模型


## 各目录作用

### 1. Wan2.1-VACE-14B

这是视频编辑主干模型目录。

项目运行时主要依赖以下内容：

- `diffusion_pytorch_model-*.safetensors`
- `models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.1_VAE.pth`
- `google/umt5-xxl/`

如果这个目录不完整，`src/edit/video_editor.py` 中的 Wan 相关 variant 会直接报错，无法运行 edit model。

`wan14b-vace-base` 使用这套 VACE 权重、不挂任何 LoRA，通过仓库内
`Ditto/diffsynth` 的 WanVideoPipeline 调用；推理步数和 CFG 固定为 50 / 5.0，
与该 pipeline 的 VACE 默认值一致。用于 benchmark 时应在结果中标明这是
“VACE base weights via Ditto runtime”，避免误写成独立的官方推理栈复现。


### 2. Wan2.1-Distill-Loras

这个目录存放 Wan 的 4-step LoRA 权重。

当前项目里最常用的是：

- `wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors`

对应的 edit variant 是：

- `wan14b+4steplora`
- `wan14b+dittolora+4steplora`

如果缺失，涉及 `4steplora` 的 variant 无法运行。


### 3. Ditto_models

这个目录存放 Ditto 的 LoRA 权重。

当前项目主要使用：

- `models/ditto_global.safetensors`

对应的 edit variant 是：

- `wan14b+dittolora`
- `wan14b+dittolora+4steplora`

如果缺失，所有带 `dittolora` 的 variant 都无法运行。


### 4. Qwen3-VL-8B-Thinking

这是本地 planner / critic 使用的视觉语言模型目录。

当前由以下代码路径使用：

- `src/vlm/local_qwen.py`
- `src/planner/`
- `src/critic/`

如果缺失，以下链路会失败：

- `scripts/run_planner_and_edit.py`
- `scripts/test_video_planner.py`
- `scripts/test_critic.py`

当前代码已优先从以下路径加载该模型：

- `/home/featurize/low-high-new/models/Qwen3-VL-8B-Thinking`


## 当前代码中的默认读取逻辑

### Edit Model

`src/edit/video_editor.py` 当前会优先从仓库内 `models/` 读取：

- `Wan2.1-VACE-14B`
- `Wan2.1-Distill-Loras`
- `Ditto_models`

如果仓库内不存在，才会回退到旧路径。


### Local Qwen

`src/vlm/local_qwen.py` 当前会优先从仓库内 `models/Qwen3-VL-8B-Thinking` 读取本地 Qwen 模型。

因此通常不需要再手动设置：

- `LOW_HIGH_AGENT_LOCAL_QWEN_MODEL`

除非你想显式切换到别的模型目录。


## 常用检查方法

### 1. 检查 edit variant 是否可运行

```bash
python scripts/test_edit_model.py --list-variants
```

如果某个 variant 缺失权重，脚本会直接显示缺少的本地路径。


### 2. 检查 planner / critic 的本地 Qwen 是否可读

如果本地 Qwen 路径不正确，运行以下脚本时会报模型路径相关错误：

```bash
python scripts/test_video_planner.py
python scripts/test_critic.py
python scripts/run_planner_and_edit.py
```


## 建议

- 不要随意改动本目录结构，当前代码默认依赖这些固定路径
- 如果新增模型，建议沿用当前命名方式，避免后续再改代码
- 如果需要迁移目录位置，优先修改代码中的路径解析逻辑，而不是在多个脚本里硬编码路径


## 总结

`models/` 是本项目的统一本地模型根目录。

当前最关键的四类资源分别是：

- Wan 主模型
- Wan 4-step LoRA
- Ditto LoRA
- Qwen3-VL-8B-Thinking

只要这几部分完整，planner、critic 和 edit model 三条主链路就都具备本地运行条件。
