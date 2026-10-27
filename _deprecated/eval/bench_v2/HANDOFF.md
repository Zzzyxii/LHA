# 交接文档:low→high bench 修正与 skill / agent 评测

最后更新:2026-10-16。本文档覆盖 `bench_skill`(skill 评测)和 `bench_v2`(修正后的 bench)两部分工作。

---

## 0. 现状一句话

修正后的 bench(**执行 track**)已经做完并出了结果;用来测 agent 的**规划 track** 代码、数据、启动脚本都已就绪并验证过,**只差生成**。原机器的 8 张卡被别的任务占满(每张约 80 GB),所以换机器跑。

本次交接前,planner、critic、judge 已统一换成 **`gpt-6.1-sol`(reasoning_effort=xhigh)**。切换时发现并修掉了两个会让规划 track 结果失效的问题,详见 2.6。

---

## 1. 关键结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | **原 bench 的指令泄露答案** | 指令是 VLM 同时看着 low 和 high 帧写的(`datas/low_high_cost/invert_high2low_prompts_api.py` 的 `VIDEO_USER_TEMPLATE`),带着 target 的具体颜色和物件 |
| 2 | **原 bench 的指标不逐条和 target_high 比** | CLIP-T / PickScore 比指令,CLIP-F / DINO-F 比自身相邻帧,Src-Edit 比 low,LAION / MUSIQ / NIQE / Reward 只看输出;target_high 只在 JEDi 里以集合分布出现 |
| 3 | **原 bench 任务混杂** | 50 条中 34 条内容重写、11 条刻意低保真的滤镜(VHS、胶片颗粒、黑白)、4 条其他风格滤镜、1 条其他 |
| 4 | **按 GT 衡量,ours 明显赢 Bernini** | 增益差 +0.068,37/13,p=0.0009;已发表视频 +0.095,45/5,p=4×10⁻⁹。旧 Reward 里排第 2 的 Bernini,增益比"什么都不做"还低 |
| 5 | **在执行 track 上 skill 没有帮助,只是让编辑更保守** | skill v2 对 GT 的增益与不加 skill 持平(15/15);对原片的相似度 +0.084,30/0,即每一条都更接近原片 |
| 6 | **单次 pointwise 美学 judge 不可靠** | 同一批视频打两次:逐条差值 sd 0.995,12/30 变化 ≥1 分;只输出 7 个档位(粒度 0.5);成对比较单向问时位置偏差 0.80 |

第 5 条是在意料之中的:执行 track 给的就是照着 GT 写的专家指令,agent 没什么可补的。**agent 的价值只能在规划 track 上测出来**,这就是待做的部分。

---

## 2. 已完成的工作

### 2.1 推理管线修复(`src/` 下的改动)

| 文件 | 改动 | 原因 |
|---|---|---|
| `src/edit/video_editor.py` | 新增 `wan14b+ours` variant、`_attach_trained_latent_memory()` | 原来 `load_lora` 只加载 LoRA,训练出的 latent memory(`latent_memory_hint.*` / `pipe.latent_memory_text.*`)**被静默丢弃**,跑的其实是普通 LoRA |
| 同上 | 新增 `bernini1.3b` backend(`_render_with_bernini`) | 让 Bernini 能接入 agent 闭环 |
| 同上 | `_apply_bench_protocol()` | 闭环不透传帧数 / fps,用环境变量统一协议 |
| `src/vlm/litellm_vlm.py`(新) | OpenAI 兼容网关的 VLM 后端 | planner / critic 走托管模型;429 退避、把网关伪装成 400 的瞬时故障当作可重试、帧图降采样防 413 |
| `src/vlm/__init__.py` | `resolve_vision_model(role=...)` | 按环境变量选后端,planner / critic 可分别配置 |
| `src/critic/model_loader.py` | critic 使用 `role="critic"`,延迟加载 | |
| `src/planner/core/video_planner.py` | 改用 `resolve_vision_model()` | |
| `src/planner/core/pipeline.py` | `_compose_edit_prompt()`;候选为空时的兜底 | 见 2.4;原来场景落在 3 个 style family 之外时 planner 直接崩溃 |
| `src/utils/rule_checker.py` | `validate_selected_strategy` 空候选兜底 | 同上 |
| `src/skills/enhancement*.py`(新) | 按编辑维度划分的 skill(见 2.4) | |
| `src/utils/prompt_budget.py`(新) | 按 Wan 的 umT5 tokenizer 把 prompt 控制在 512 token 内 | 见 2.6 |
| `src/planner/prompts/planner_prompts.py` | 最终方案要求英文、≤180 词 | 见 2.6 |

### 2.2 bench 诊断

逐条看了 30 条样本的画面(对比图在原机器 `out/bench_skill/judge/bench_sheet_*.png`、`bench_lowA.png`):

- low 是从 high **退化**出来的(原始数据是 `high2low_prompt`)。内容重写类的 low,一部分是已经专业拍好的剧照,另一部分是平涂、有伪影的廉价渲染,**不是日常 vlog**。
- 低保真滤镜类的"high"刻意降低制作感,和 low→high 的方向相反。
- ours 倾向于少改,Bernini 倾向于改过头。再加上不少 low 本来就是好看的剧照,"基本不改"在旧指标上也能拿到不错的分。

### 2.3 bench_v2:执行 track(已完成)

代码在 `scripts/eval/bench_v2/`,说明见同目录 `README.md`,完整结果见原机器 `out/bench_v2/RESULTS.md`。

- **主指标 `gain_dino`** = 输出与 target_high 按时间对齐后的 DINO 相似度 − low 原样不动时的相似度。原样返回为 0,往错误方向改为负。
- 自带 `noop`(low 本身)和 `oracle`(target 本身)两个参照臂。实测 noop 增益 0.0000、oracle 相似度 1.0000。
- 按时间戳对齐取帧,兼容 17 / 49 / 73 / 原生帧数的不同导出协议。

前几名(n=50):`pub:ours_full` +0.071、`pub:ditto_all_data` +0.060、`ours_raw` +0.055 …… `noop` 0.000 …… `pub:Bernini-R-1.3B` −0.024、`pub:Lucy-Edit` −0.060、`pub:ReCo` −0.093。

**局限**:它衡量的是"离这一个 target 有多近",DINO/CLIP 在 224 分辨率下比较粗,不衡量好不好看。建议把它作为"有没有改对"的主指标,美学 judge 作为"好不好看"的补充,两者都报。

### 2.4 skill 的两次修改

- **v1(原设计)**:planner 用改写**替换**掉原始指令,丢掉了"保持构图 / 表情 / 身份"等保护子句,还会凭空加内容(174 字的"柔焦粉彩"被扩写到 987 字,出现"羽绒被当反光板")。
- **原 skill 库的问题**:模板是"把整个场景改造成某种世界观"(和局部编辑的任务相反);只有 3 个 style family,bench 里 22 条挤在同一个、0 条匹配 anime、7 条三个都不匹配;排序依赖的字段 11/13 个模板是空的;模板是中文,而底座文本编码器是以英文为主的 umT5。
- **v2(新设计,`src/skills/enhancement*.py`)**:按编辑维度划分(调色、布光、质感、服装、头饰、首饰、妆容、背景、道具、氛围),维度和触发词从 bench 指令中统计得出;关键词触发而非由模型改写;英文;每个 skill 声明必须保护的内容。registry 构造时就做校验。
- **`_compose_edit_prompt` 有两种模式**(环境变量 `LOW_HIGH_AGENT_PROMPT_MODE`):
  - `instruction_first`(默认,执行 track 用):原指令逐字放最前 + 工艺约束 + 保护项,**不使用** planner 的改写
  - `expand`(规划 track 的 agent 臂用):一句话意图 + planner 根据 low 画面写出的详细方案 + 工艺约束 + 保护项。输入只有一句话时,planner 的方案正是 agent 的价值所在

### 2.5 judge

- `evaluate_aesthetic_reward_litellm.py`:托管模型版美学 judge,rubric 与原训练版完全一致
- `evaluate_aesthetic_reward_stable.py`:重复 N 次取中位数;repeats=5 时单次等效噪声从 0.70 降到 0.36,但仍受 0.5 档位粒度限制
- `compare_pairwise.py`:成对比较,**正反顺序各问一次**,两次不一致记为平局;字节完全相同的视频直接判平。输出里带 `first_slot_win_rate` 用来监控位置偏差

> **注意**:到目前为止所有 judge 数字都是 `gemini-3.1-pro-preview` 打的。换成 gpt-6.1-sol 之后要重新打分,**不要和旧数字混在一起比较**。GT 参照指标(`gt_metrics.py`)不用 judge,不受影响。
> 同理,之前 skill v1 / v2 的 agent 臂用的是 gemini 的 planner,新跑的规划 track 用的是 gpt-6.1-sol 的 planner。

### 2.6 切换到 gpt-6.1-sol 时发现的两个问题(都已修复)

**问题 1:推理把输出预算吃光,planner 卡死。** planner 最后一步 `call_004_final` 只给了 8192 个输出 token。gpt-6.1-sol 在 xhigh 下思考 234 秒用光配额,返回空正文;客户端用同样的预算重试,注定再次失败,最坏会循环 8 次约 30 分钟。修复(`src/vlm/litellm_vlm.py`):
- 开启推理时在调用方要求的长度之上**额外预留推理预算**(xhigh +32768,high +16384,上限 128000)
- 如果仍因长度截断返回空正文,**重试时预算翻倍**
- 重试时打印日志(之前是静默重试,卡住和慢根本分不出来)

离线 judge(`evaluate_aesthetic_reward_litellm.py` 的 `LiteLLMJudge`)也加了同样的翻倍逻辑。

**问题 2:prompt 超过 Wan 文本编码器的 512 token 上限,尾部被静默截掉。** Wan2.1 的 prompter 固定 `text_len=512`,超出部分模型完全看不到,而保护块恰好放在最后。实测:

| prompt | token 数 | 后果 |
|---|---|---|
| 专家指令本身 | 中位 153,最大 259 | 无 |
| 执行 track skill v2 | 中位 522,最大 647 | **16/30 超限**,保护块尾部被截 |
| gpt-6.1-sol planner 的方案(修复前) | **2176** | 模型只看到意图和方案的前 1/4,工艺约束和保护块整段丢失 |

修复:
- 源头:planner 最终方案要求**英文、≤180 词**(gpt-6.1-sol 照字面执行原规则"高密度",写出了 2000 多字的中文;gemini 碰巧没有这个问题)
- 兜底:`src/utils/prompt_budget.py` 用 Wan 自己的 tokenizer 计算 token 数,超限时先逐条删通用工艺约束,再按句截短方案;**意图和保护块永不裁剪**。expand 模式下无论 skill 是否触发都会带上通用保护项
- expand 模式下**不再拼接 planner 自己产出的 `hard_preserve_constraints`**:gpt-6.1-sol 给出的是约 30 条按逗号切碎的中文片段("构图"、"时长"……),而保护块不参与裁剪,结果把方案挤出了 512 窗口。通用保护项已覆盖身份、表情、姿态、机位。默认模式下 `hard_preserve` 来自用户的专家指令,保持不变
- 回归:原本没超限的 14 条 v2 prompt 逐字节不变;30 条全部 ≤512 且保护块完整
- 实测(修复后 1 条 smoke):planner 方案 168 个英文词、288 token;最终 prompt 504 token,方案、工艺约束、保护块都完整

> 已有的执行 track v2 结果是在**部分被截断**的 prompt 上测出来的(16/30)。如果要严格引用 v2 的数字,需要用修复后的代码重跑一次 `ours_skill`。

---

## 3. 待做(按优先级)

### P0:规划 track 生成 + 打分(测 agent 价值的核心实验)

4 条臂 × 50 条,输入都是一句话意图,不给专家指令:

| 意图 | 直接喂给模型 | agent 扩写(planner 只看 low 画面) |
|---|---|---|
| 逐条一句话(保留题材,去掉细节) | `ours_intent_direct` | `ours_intent_agent` |
| 统一一句话"Make this video look like a high-budget cinematic film." | `ours_generic_direct` | `ours_generic_agent` |

上界是专家指令直接喂(即已有的 `ours_raw`)。**agent 的价值 = agent 臂比 direct 臂高出多少,以及补上了 direct 到上界之间差距的多少。**

```bash
source scripts/bench_skill/env.sh
export LOW_HIGH_JUDGE_API_KEY='<你的 key>'
bash scripts/eval/bench_v2/launch_planning.sh   # 8 卡,可断点续跑,已完成的视频会跳过
# 全部跑完后:
bash scripts/eval/bench_v2/eval_planning.sh     # GT 指标 + 4 组成对比较
```

意图文件已固定在仓库里:`scripts/eval/bench_v2/data/intents_specific.jsonl`(50 条,只根据指令文字生成,没看过任何帧,自动检查 0 条泄露)。其中 0319、0977 两条措辞不太像用户的话,跑之前可以人工改。

**耗时与成本(实测)**:

| | 单条耗时 | 单条成本 |
|---|---|---|
| direct 臂(只渲染) | 约 460 秒 | 0 |
| agent 臂 planner(gpt-6.1-sol xhigh,6 次调用) | **约 676 秒** | **约 1.9 美元** |
| agent 臂合计(planner + 渲染 + critic) | 约 1150 秒 | 约 2 美元以上 |

脚本按 agent 臂各 3 卡、direct 臂各 1 卡分配,8 卡总时长约 **6.5 小时**。planner 部分约 200 美元,另加 critic 和 judge 的费用。planner 阶段 GPU 是空闲的,如果嫌慢,可以把 `LOW_HIGH_AGENT_LITELLM_REASONING_EFFORT` 降到 `high`。

### P1:用 gpt-6.1-sol 重新打执行 track 的 judge 分

视频都已经有了,不用重新生成:
- `ours_raw` vs `bernini_raw`(成对比较)
- `ours_skill_v2` vs `ours_raw`(成对比较)

### P2:可选

- 把规划 track 的 agent 臂改成完整闭环(`--max-attempts 3`)。注意之前在执行 track 上,73% 的样本里闭环的修正轮比更早的轮次差
- `ditto_skill` / `bernini_skill` 两条臂:回答"skill 的效果是 ours 特有的,还是对所有底座都成立"
- 在规划 track 上加 `ours_bestof` 臂(同样预算,3 个独立 seed,critic 挑最好的)

---

## 4. 在新机器上准备环境

### 4.1 代码

所有改动**都还没有 commit**。建议在原机器上先 commit 再拉到新机器;或者整个目录拷过去。改动清单:

```
 M src/critic/model_loader.py
 M src/edit/video_editor.py
 M src/planner/core/pipeline.py
 M src/planner/core/video_planner.py
 M src/planner/prompts/planner_prompts.py
 M src/utils/rule_checker.py
 M src/vlm/__init__.py
?? scripts/bench_skill/                      # 生成 runner、环境配置、ffprobe shim
?? scripts/eval/bench_v2/                    # 修正后的 bench + 意图数据
?? scripts/eval/ditto_quant/compare_pairwise.py
?? scripts/eval/ditto_quant/evaluate_aesthetic_reward_litellm.py
?? scripts/eval/ditto_quant/evaluate_aesthetic_reward_stable.py
?? src/skills/enhancement.py
?? src/skills/enhancement_templates.py
?? src/utils/prompt_budget.py
?? src/vlm/litellm_vlm.py
```

`Ditto/` 是 submodule,但本机的 `Ditto/diffsynth/models/` 下带有 `latent_memory_runtime.py`、`easyvfx_frequency_runtime.py` 等项目自己的补丁,**要用这份 Ditto,不要用上游的**。

### 4.2 需要拷贝的数据和模型(都被 gitignore,不在 git 里)

| 内容 | 放到仓库内的路径 | 来源 |
|---|---|---|
| Wan2.1-VACE-14B | `models/Wan2.1-VACE-14B/` | 原机器同路径 |
| 训练好的 ours checkpoint | `out/hf_checkpoints/checkpoints/exp5_ours_auxall_all_data_4gpu_save1k/real_train_output/step-6000.safetensors` | 原机器同路径,或 HF `Ouzhang/low-high-models` 同名路径。代码里写死了这个位置(`OURS_EXP5_CHECKPOINT`) |
| CLIP-L / DINO 缓存(GT 指标用) | `models/hf_cache/hub/models--openai--clip-vit-large-patch14`、`models--facebook--dino-vits16` | 原机器同路径 |
| 上界臂视频(50 个) | `out/bench_skill/run50_native/ours_raw/generations/` | 原机器同路径 |
| bench 归档 | 任意位置,用 `RELEASE_ROOT` 指定 | 原机器 `/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/归档.zip`,解压即可 |
| Ditto 公开 LoRA(只有 ditto 臂需要) | `models/Ditto_models/models/ditto_global.safetensors` | 原机器同路径 |
| Bernini(只有 bernini 臂需要) | `models/Bernini-R-1.3B-Diffusers/`,代码 `third_party/Bernini/` | 原机器同路径 |

low 视频和 `sample_index.json` 在 `reports/top50_video_bundle_20260725_nodup/` 下,已被 git 跟踪,不用拷。

需要 `--max-attempts > 1` 才会用到执行 track 的完整结果目录(`out/bench_skill/run50_native/`、`run30_v2/`);只跑 P0 的话只需要上表。

### 4.3 依赖

PyPI 不通时用腾讯镜像 `-i https://mirrors.tencent.com/pypi/simple/`。原机器装过的:

```bash
pip install -i https://mirrors.tencent.com/pypi/simple/ imageio imageio-ffmpeg peft ftfy av
# 下面这些只有 Bernini 臂需要;--no-deps 是为了不动现有的 torch
pip install --no-deps -i https://mirrors.tencent.com/pypi/simple/ diffusers==0.35.2 veomni==0.1.11 decord
# flash-attn 没有对应 torch 2.9 / cu129 / py312 的预编译包,需要源码编译(约 30 分钟)
MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="9.0" \
  pip install --no-deps --no-build-isolation -i https://mirrors.tencent.com/pypi/simple/ flash-attn==2.8.3
```

`TORCH_CUDA_ARCH_LIST="9.0"` 是 H20 / H100 的;别的卡要改。原机器环境:torch 2.9.1+cu129、Python 3.12。

没有系统 ffmpeg / ffprobe 也没关系:`scripts/bench_skill/bin/ffprobe` 是基于 PyAV 的 shim,`bin/ffmpeg` 软链到 imageio-ffmpeg 自带的二进制。软链在新机器上失效时,`source env.sh` 会自动重建。

### 4.4 API key

key **没有**写进任何文件。每次在 shell 里设置:

```bash
export LOW_HIGH_JUDGE_API_KEY='<key>'
```

这个 key 在对话里以明文出现过,建议轮换。

### 4.5 GPU

- Wan14B 常驻 GPU(`LOW_HIGH_AGENT_KEEP_EDIT_ON_GPU=1`)时**每卡约 70 到 90 GB**,81 帧单条约 460 秒(17 帧时约 60 秒)
- 开跑前先 `nvidia-smi` 确认卡是空的。原机器就是因为卡被别人的任务占满,导致 8 个 shard 全部 OOM

### 4.6 开跑前的自检

```bash
cd <repo>
source scripts/bench_skill/env.sh
export LOW_HIGH_JUDGE_API_KEY='<key>'
python3 -c "
from vlm import resolve_vision_model
from critic.model_loader import CriticTextModel
from edit.video_editor import list_variant_status
p=resolve_vision_model(); c=CriticTextModel(kind='evaluator')._fallback
print('planner', p.model, p.reasoning_effort); print('critic', c.model, c.reasoning_effort)
print([s['variant'] for s in list_variant_status() if s['available']])
"
# 期望:planner / critic 都是 gpt-6.1-sol xhigh,可用 variant 里有 wan14b+ours
```

重建 manifest(路径是绝对路径,换机器必须重建):

```bash
python3 scripts/eval/bench_v2/build_manifest.py \
  --release-root <解压后的归档目录> --output out/bench_v2/manifest.jsonl
```

---

## 5. 模型配置(都在 `scripts/bench_skill/env.sh`)

| 角色 | 环境变量 | 当前值 |
|---|---|---|
| planner + critic | `LOW_HIGH_AGENT_LITELLM_MODEL` / `_REASONING_EFFORT` | `gpt-6.1-sol` / `xhigh` |
| 单独覆盖某个角色 | `LOW_HIGH_AGENT_LITELLM_PLANNER_MODEL`、`..._CRITIC_MODEL`(及对应 `_REASONING_EFFORT`) | 未设置 |
| 离线 judge | `LOW_HIGH_JUDGE_MODEL` / `LOW_HIGH_JUDGE_REASONING_EFFORT` | `gpt-6.1-sol` / `xhigh` |
| 网关 | `LOW_HIGH_JUDGE_BASE_URL` | `http://litellm-sgp.woa.com` |

gpt-6.1-sol 实测:能看图,接受 `temperature=0`,`reasoning_effort` 支持 none / low / medium / high / xhigh,**不支持 max**。

---

## 6. 坑

- **不要把 judge 分数和 GT 指标混用来下结论。** 前者衡量好不好看,后者衡量有没有改对,两者在 Bernini、Lucy-Edit、ReCo 上方向相反
- **单向成对比较不可信**(位置偏差最高到 0.80),必须正反各问一次
- **n=30 的 pointwise judge 只能分辨约 0.5 以上的差异**
- 闭环跑满轮数仍未 accept 时,要取 critic 分数最高的那一轮,不是最后一轮(`run_bench.py` 的 `_accepted_video` 已经这么做;历史数据可以用 `scripts/bench_skill/reselect_attempts.py` 修正)
- 原 bench 的 17 帧协议只有 0.85 秒;现在所有臂都跟随源视频的帧数和 fps(81 / 101 帧,4 到 6 秒)
- `heldout1000_0747` 的 skill 臂视频是后来补跑的
- **prompt 有 512 token 的硬上限**,超出部分被静默丢弃。改 prompt 组装或 planner 提示词之后,用 `utils.prompt_budget.count_tokens` 检查一遍
- **推理模型的 `max_completion_tokens` 包含思考 token**,只按答案长度设置会返回空正文。日志里出现 `[litellm_vlm] ... empty completion` 就是这个原因
- 换 planner / critic 模型时注意它们的输出语言和长度:同一份中文提示词,gemini 写英文,gpt-6.1-sol 写中文

---

## 7. 文件索引

| 用途 | 路径 |
|---|---|
| 本文档 | `scripts/eval/bench_v2/HANDOFF.md` |
| bench_v2 说明 | `scripts/eval/bench_v2/README.md` |
| 生成 runner(所有臂) | `scripts/bench_skill/run_bench.py` |
| 共享环境配置 | `scripts/bench_skill/env.sh` |
| 规划 track 启动 / 打分 | `scripts/eval/bench_v2/launch_planning.sh`、`eval_planning.sh` |
| GT 参照指标 | `scripts/eval/bench_v2/gt_metrics.py`、`run_gt_all.sh` |
| 意图数据 | `scripts/eval/bench_v2/data/intents_{specific,generic}.jsonl` |
| prompt token 预算 | `src/utils/prompt_budget.py` |
| planner 单条计时 | `scripts/eval/bench_v2/time_planner.py <输出目录> [sample_id]` |
| 执行 track 结果(原机器) | `out/bench_v2/RESULTS.md`、`out/bench_v2/gt_all/` |
