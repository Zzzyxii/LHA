# Low-High New

## 1. 项目概述

本项目旨在探索一个面向 `low-cost -> high-cost` 视频生成/编辑的 agent 化框架。

核心思想不是把视频编辑视为一次性的 prompt-to-video 过程，而是将其改造成一个自适应闭环：

1. 理解低成本输入视频与用户 prompt。
2. 规划哪些内容需要增强、保留或重写。
3. 调用视频编辑模型执行生成。
4. 使用 critic 对结果进行评估。
5. 根据评估结果进行局部修正或全局重规划。
6. 将成功和失败经验写入 memory，供后续样本复用。

从长期目标来看，这个 story 与 SPIRAL 一类的闭环生成框架一致，但本项目面向的不是长时序动作世界模型，而是 `low-cost / high-cost` 视频增强与编辑。


## 2. 问题定义

传统视频编辑流程通常依赖：

- 一段用户 prompt
- 一个 low-cost 输入视频
- 一次单轮编辑生成

但对于 `low-cost -> high-cost` 任务，仅靠 prompt 往往不够。系统还需要自动判断：

- 视频里哪些区域是真正的 low-cost，应该重点增强
- 哪些主体、结构和布局必须保持稳定
- 不同区域应该增强到什么强度
- 当前结果是否满足身份一致性、时序一致性、审美质量和物理真实感

因此，这个任务天然适合引入 agent、critic 和 memory，形成可迭代优化的闭环系统。


## 3. 目标 Story

本项目的目标系统应具备以下能力：

- 能按编辑维度自适应规划，例如主体、脸部、服装、道具、背景、光照、氛围等
- 能结合 Wan / Ditto 等视频编辑底座执行编辑
- 能通过 critic 对 prompt 对齐、结构保持、身份一致、时序连贯、审美质量、artifact 等维度进行评估
- 当计划基本正确但实现较弱时，进行局部 prompt refinement
- 当当前计划本身不合理时，触发全局 replan
- 能跨样本积累 scene archetype、失败模式、prompt recipe、风格偏好等记忆
- 最终支持 RL / GRPO 类训练，将 planner/critic 的反馈逐步内化进生成模型本身


## 4. 当前仓库状态

当前仓库并不是一个空框架，而是已经具备一个可运行的闭环原型。

目前已经实现的模块包括：

- `Planner`
  - 输入 prompt 和 low-cost 视频
  - 分析视频内容、选择风格族、生成候选策略并汇总成结构化编辑计划
  - 关键文件：
    - `src/planner/core/video_planner.py`
    - `src/planner/core/pipeline.py`

- `Edit Model Wrapper`
  - 负责调用 DiffSynth/Wan 或 Ditto LoRA 变体执行视频编辑
  - 关键文件：
    - `src/edit/video_editor.py`

- `Critic`
  - 对编辑结果进行评估
  - 输出结构化评分、失败标签和修正建议
  - 关键文件：
    - `src/critic/evaluator.py`
    - `src/critic/reflector.py`

- `Memory`
  - 存储 run 记录、evaluation 结果、failure pattern 和 prompt recipe
  - 为后续规划与修正提供历史经验检索
  - 关键文件：
    - `src/memory/store.py`
    - `src/memory/retriever.py`
    - `src/memory/evolver.py`

- `Closed-Loop Runner`
  - 串联执行 `planner -> edit -> critic -> reflection -> memory`
  - 当前支持三类结果分支：
    - `accept`
    - `local_refine_prompt`
    - `global_replan`
  - 关键文件：
    - `src/loop/run_loop.py`
    - `scripts/run_planner_and_edit.py`


## 5. 当前实现与 Agent Story 的对应关系

当前实现已经能映射到一个基础版的 agent 闭环：

- `PlanAgent`
  - 由 planner 模块承担
  - 负责决定 low-cost 视频应该如何向 high-cost 转化

- `World Model / Edit Model`
  - 当前由现有视频编辑底座实现
  - 接收编辑计划或 prompt，生成编辑后视频

- `CriticAgent`
  - 负责对结果打分，并输出失败标签和文本反馈
  - 决定当前结果应该被接受、局部修正还是全局重规划

- `Memory`
  - 负责存储显式经验
  - 当前更接近“可检索经验库”，还不是参数化或隐式记忆


## 6. 当前已经具备的能力

目前仓库已经支持：

- 单样本级别的视频规划
- 多种编辑 backend 变体调用
- critic 结构化评分
- 基于 reflection 的 prompt patch
- 从局部修正退化到全局重规划
- memory 随运行过程持续增长
- planner / critic / memory / edit 的 smoke test 脚本

因此，当前阶段更准确的定位是：

`Phase 0：agent 化 low/high 视频编辑原型`


## 7. 与最终研究目标相比的主要缺口

尽管闭环已经搭起来了，但距离最终研究目标还有明显差距。

当前主要缺口包括：

- planner 仍然是原型，还不是专门为 low/high edit 训练出来的规划器
- 规划维度还不够明确，尤其缺少更系统的并行 edit dimension 设计
- critic 还没有针对 low/high 偏好和视频编辑目标进行专门训练
- memory 目前是显式、符号化的，还没有偏好条件化或参数化能力
- 还没有接入真正的 RL / GRPO 训练流程
- 数据构建、批量化处理、标注与训练管线还没有完整并入当前仓库
- 缺少标准化、研究级的 benchmark 和评估协议


## 8. 推荐技术路线

### 8.1 Planner

planner 后续应该从“结构化 prompt 编译器”演化成“专门面向 low/high 任务的编辑规划器”。

推荐的规划维度包括：

- 主体身份
- 脸部与发型
- 服装与材质丰富度
- 前景道具
- 背景与 world-building
- 光照与氛围特效
- 与运动相关的敏感区域

planner 需要回答的问题包括：

- 哪些内容必须 preserve
- 哪些内容适合轻量增强
- 哪些内容应该强重写
- transformation 应该挂载在哪些 visible carrier 上
- 每个区域的增强强度应该是多少

关键设计判断：

- 更推荐走 `并行编辑维度规划`
- 不应过度依赖动作序列式串行规划

本项目的核心不是动作分解，而是面向区域和语义维度的自适应增强。


### 8.2 Critic

critic 不应只是一个通用打分器，而应该紧贴 low/high 编辑目标。

推荐重点评估维度：

- prompt alignment
- structure preservation
- subject ID consistency
- pose / geometry consistency
- temporal coherence
- world/background realization
- transformation strength
- aesthetics
- artifact level / realism

critic 的输出应包括：

- 总分
- 分维度评分
- failure tags
- 文本化修正建议
- 对 `local refine` 和 `global replan` 的决策


### 8.3 Memory

memory 建议分两步发展。

第一阶段：显式记忆

- scene archetypes
- 常见 failure patterns
- prompt compilation recipes
- style conversion priors
- model behavior priors

第二阶段：偏好记忆

- 不同用户/数据风格下的 high-cost 偏好
- 对“什么算 high-cost”的可复用先验
- 如果实验结果足够支持，再考虑做 latent / parameterized memory

短期建议是先把显式 memory 做扎实，不要一开始就直接上隐式记忆。


### 8.4 RL / GRPO

RL 不应该等所有模块都完善后才考虑。

更合理的路径是尽早验证以下问题：

- critic 分数是否真的和更好的视觉结果相关
- reflection 相比 one-shot baseline 是否确实有收益
- grouped candidates 是否可以稳定排序
- 相对偏好优化是否能带来可测提升

推荐最短路径：

1. 先得到一个可靠的 critic
2. 在小规模数据上验证 reward 是否有效
3. 引入 grouped candidate comparison


## 9. 外部依赖与非 Git 目录

当前仓库有几类目录虽然在本地存在，但**默认不会被 Git 跟踪**。  
这是有意为之，不是目录缺失。

对应规则在：

- [.gitignore](/home/featurize/low-high-new/.gitignore)

当前被整体忽略的目录包括：

- `DiffSynth-Studio/`
- `Ditto/`
- `models/`
- `datas/`
- `out/`
- `reference/`

这意味着：

- 你在本地能看到这些目录
- 但 `git status` / GitHub / 代码审查页面里通常看不到它们的内容
- 如果没有额外说明，别人很容易误以为项目缺文件


### 9.1 DiffSynth-Studio

- 路径：
  - [DiffSynth-Studio](/home/featurize/low-high-new/DiffSynth-Studio)
- 上游仓库：
  - `https://github.com/modelscope/DiffSynth-Studio`
- 作用：
  - 本地 `Wan / VACE` 推理与训练依赖
  - 当前 `edit-model` 的真实训练入口优先走这里
- 当前用途：
  - `wan14b`
  - `wan14b+4steplora`
  - `base_sft`
  - `planner_conditioned`

它属于外部代码依赖，不是本仓库原生源码，因此默认不纳入 Git。


### 9.2 Ditto

- 路径：
  - [Ditto](/home/featurize/low-high-new/Ditto)
- 上游仓库：
  - `https://github.com/ezioby/Ditto`
- 作用：
  - 参考 Ditto 原始训练/推理实现
  - 某些编辑 variant 和训练兼容逻辑仍会复用这里的代码
- 当前用途：
  - `wan14b+dittolora`
  - `wan14b+dittolora+4steplora`
  - Ditto 论文式 low/high 编辑参考实现

它也是外部代码依赖，因此默认不纳入 Git。


### 9.3 models

- 路径：
  - [models](/home/featurize/low-high-new/models)
- 作用：
  - 存放本地模型权重、LoRA、tokenizer 和训练产物挂载

典型内容包括：

- `Wan2.1-VACE-14B`
- `Wan2.1-Distill-Loras`
- `Ditto_models`
- `edit-model/base_sft`
- `edit-model/planner_conditioned`
- `critic/evaluator`
- `critic/reflector`
- `critic/pairwise`

这个目录通常非常大，而且常包含：

- 基座模型
- LoRA checkpoint
- 本地训练输出软链接

因此默认被 `.gitignore` 排除。


### 9.4 reference

- 路径：
  - [reference](/home/featurize/low-high-new/reference)
- 作用：
  - 存放论文、笔记、对话记录、参考实现、临时研究材料

例如：

- SPIRAL 论文
- 项目讨论笔记
- 风格/规划/critic 设计参考材料

这些内容通常不是项目必须的可执行源码，而且可能持续变动、体积较大或带版权限制，所以默认不纳入 Git。


### 9.5 datas

- 路径：
  - [datas](/home/featurize/low-high-new/datas)
- 作用：
  - 存放 Ditto-1M 原始数据、pair 数据、训练集导出结果

例如：

- `datas/Ditto-1M`
- `datas/ditto`
- `datas/ditto_pairs.jsonl`

这是数据目录，不应作为代码仓库内容提交。


### 9.6 out

- 路径：
  - [out](/home/featurize/low-high-new/out)
- 作用：
  - 存放所有实验输出

包括：

- planner 单测输出
- edit-model 出片结果
- critic 测试结果
- 训练日志
- checkpoint
- 对比实验视频

这是典型实验工作目录，也不应进入 Git。


### 9.7 如果别人只拿到 Git 仓库，会缺什么

如果别人只 `git clone` 本仓库，而没有你本地这些忽略目录，那么默认会缺：

- `DiffSynth-Studio/`
- `Ditto/`
- `models/`
- `datas/`
- `reference/`

也就是说，代码框架本身能看到，但：

- 不能直接本地训练
- 不能直接本地推理
- 不能直接复现实验结果

必须额外补：

- 外部代码依赖
- 本地模型权重
- Ditto 数据
- 参考材料


### 9.8 推荐理解方式

可以把当前仓库分成两层：

- **Git 管理层**
  - `src/`
  - `scripts/`
  - `configs/`
  - `README.md`

- **本地工作区层**
  - `DiffSynth-Studio/`
  - `Ditto/`
  - `models/`
  - `datas/`
  - `out/`
  - `reference/`

前者是代码与流程定义。  
后者是实际运行、训练和实验所需的本地资源。
4. 在 reward 足够可信后，再引入 RL / GRPO


## 9. 分阶段项目计划

### Phase 0：原型基线

目标：

- 跑通 `planner -> edit -> critic -> memory` 闭环

当前状态：

- 当前仓库已经基本完成

产出：

- 单样本 smoke test
- planner / critic / edit / memory 的中间产物


### Phase 1：稳定闭环原型

目标：

- 让当前原型稳定、可复现、便于后续扩展

主要任务：

- 统一 planner / critic / memory 的输入输出 schema
- 明确支持的 model variant 和运行方式
- 在多个代表性样本上验证全链路可用性
- 建立 baseline 对比：
  - 无 agent 的 direct edit
  - 只有 planner 的 edit
  - 完整 planner + critic + reflection 闭环

交付物：

- 可稳定复现的内部基线系统


### Phase 2：形式化 low/high 规划维度

目标：

- 把 planner 从粗粒度 prompt 分解，升级为明确的 low/high 编辑维度规划器

主要任务：

- 定义 region / dimension taxonomy
- 支持主体 / 道具 / 背景 / 光照等并行规划
- 明确 preserve / rewrite / light-touch target 的定义
- 改进 planner prompt 和输出 schema

交付物：

- 与 low/high 编辑目标直接对齐的 planner 输出


### Phase 3：构建 low/high 对齐的 Critic

目标：

- 让 critic 真正对 low/high 编辑质量负责

主要任务：

- 定义 low/high 任务下的评分标准
- 设计 scalar 或 pairwise 的评价目标
- 加入 ID、一致性、时序和审美等维度
- 校准 accept / refine / replan 阈值

交付物：

- 更接近人类偏好的 critic 输出


### Phase 4：数据与标注管线

目标：

- 为训练和评估提供可扩展的数据基础

主要任务：

- 做批量化数据处理与清洗
- 整理 low/high 视频对或可编辑视频素材
- 构建 prompt 标注和 edit-dimension 标注
- 准备 planner 监督数据和 critic 偏好数据

交付物：

- 可用于训练和评估的数据资产


### Phase 5：训练 Planner 与 Critic

目标：

- 用可训练模块替换掉当前主要依赖 prompt engineering 的部分

主要任务：

- planner 的 SFT
- critic 的 SFT
- 在适合时加入 pairwise preference modeling
- 验证不同场景、不同风格下的泛化能力

交付物：

- 可训练、可迭代的 planner / critic 模块


### Phase 6：升级 Memory

目标：

- 让 memory 对后续样本的规划和修正产生实质性帮助

主要任务：

- 提升 memory retrieval 质量
- 将 memory 更深入地接入失败恢复流程
- 引入 preference-aware memory
- 测试 memory 是否能提升重复场景、重复风格下的表现

交付物：

- 能带来可测收益的经验复用机制


### Phase 7：验证 RL / GRPO

目标：

- 验证 planner/critic 的反馈是否能真正内化进生成模型

主要任务：

- 生成 grouped candidates
- 计算组内相对 reward
- 使用 GRPO 类目标优化 edit model 或 adapter
- 与 one-shot baseline、reflection-only baseline 进行对比

交付物：

- 对 RL 是否有效的实验性结论


## 10. 近期 To-Do

短期内建议优先推进以下事项：

1. 在多个 smoke sample 上跑通现有全链路。
2. 固定 planner / critic / memory 的 schema，避免接口频繁变化。
3. 更明确地定义 low/high 的规划维度。
4. 重构 critic 指标，补上 ID、姿态、时序和审美维度。
5. 把当前原型接到已有数据与批量化处理流程上。
6. 开始小规模标注，为 planner 监督和 critic 训练做准备。
7. 在引入 RL 之前，先验证 reflection 是否显著优于 one-shot baseline。


## 11. 仓库入口

常用脚本：

- `scripts/run_planner_and_edit.py`
- `scripts/test_video_planner.py`
- `scripts/test_edit_model.py`
- `scripts/test_critic.py`
- `scripts/test_memory.py`

核心源码目录：

- `src/planner/`
- `src/edit/`
- `src/critic/`
- `src/memory/`
- `src/loop/`

配置文件：

- `configs/planner_qwen.json`
- `configs/base_skills.json`


## 12. 环境安装与测试命令

### 12.1 安装依赖

建议在项目根目录执行：

```bash
pip install -r requirements.txt
```

这份依赖中已经包含本地 planner 所需的 `qwen-vl-utils`，否则运行 `local_qwen` 路径时会报错。

如果后续需要单独安装本地 DiffSynth 包，也可以执行：

```bash
pip install -e DiffSynth-Studio
```


### 12.2 检查可用编辑变体

在测试 edit model 之前，建议先检查当前本地可运行的 variant：

```bash
python scripts/test_edit_model.py --list-variants
```


### 12.3 测试 Ditto + Wan 版本

当前仓库中，`Ditto + Wan` 对应的 variant 名称为：

- `wan14b+dittolora`
- `wan14b+dittolora+4steplora`

示例命令如下。

测试 `Ditto + Wan`：

```bash
python scripts/test_edit_model.py \
  --variant wan14b+dittolora \
  --low-video smoke-data/1_low-cost.mp4 \
  --output-video out/test_ditto_wan/sample1_edited.mp4 \
  --prompt-file smoke-data/prompt1.txt \
  --device cuda
```

测试 `Ditto + Wan + 4step LoRA`：

```bash
python scripts/test_edit_model.py \
  --variant wan14b+dittolora+4steplora \
  --low-video smoke-data/1_low-cost.mp4 \
  --output-video out/test_ditto_wan/sample1_edited_4step.mp4 \
  --prompt-file smoke-data/prompt1.txt \
  --device cuda
```


### 12.4 从 planner 输出继续测试 edit model

如果已经先跑过 planner，也可以直接使用 planner 输出的 `normalized_prompt`：

```bash
python scripts/test_edit_model.py \
  --variant wan14b+dittolora \
  --low-video smoke-data/1_low-cost.mp4 \
  --output-video out/test_ditto_wan/from_planner.mp4 \
  --planner-response out/sample_1/planner.response.json \
  --device cuda
```


### 12.5 运行完整闭环

完整闭环入口如下：

```bash
python scripts/run_planner_and_edit.py \
  --sample-id 1 \
  --prompt-file smoke-data/prompt1.txt \
  --low-video smoke-data/1_low-cost.mp4 \
  --output-dir out/full_loop_sample1 \
  --variant wan14b+dittolora \
  --max-attempts 3
```

如果 `--output-dir` 已经存在，脚本会自动在该目录下创建一个新的时间戳子目录，例如：

```bash
out/full_loop_sample1/rerun_20260330_120501
```

这样重复运行不会覆盖旧的 attempt 结果。

`scripts/run_planner_and_edit.py` 现在默认会开启两项 GPU 常驻优化：

- `LOW_HIGH_AGENT_KEEP_VLM_ON_GPU=1`
- `LOW_HIGH_AGENT_KEEP_EDIT_ON_GPU=1`

这意味着在同一个 Python 进程里：

- 本地 Qwen VLM 会优先常驻 GPU，减少重复加载时间
- Ditto / DiffSynth 的 edit pipeline 会优先常驻 GPU，减少重复初始化和 LoRA 重挂载时间

如果想显式指定，也可以这样运行：

```bash
LOW_HIGH_AGENT_KEEP_VLM_ON_GPU=1 \
LOW_HIGH_AGENT_KEEP_EDIT_ON_GPU=1 \
python scripts/run_planner_and_edit.py
```

如果显存不够，想回到更保守的 offload 模式，可以关闭：

```bash
LOW_HIGH_AGENT_KEEP_VLM_ON_GPU=0 \
LOW_HIGH_AGENT_KEEP_EDIT_ON_GPU=0 \
python scripts/run_planner_and_edit.py
```

当前默认 memory 文件位置为：

```bash
out/memory/global_memory.json
```

如果不手动覆盖 `--memory-path`，闭环运行时会默认读写这个文件。


### 12.6 从 memory 导出可晋升 custom skills

如果希望从 memory 中筛选高质量 custom skills，并导出成中间 JSON 文件，可以执行：

```bash
python scripts/export_promotable_skills.py
```

默认行为：

- 从 `out/memory/global_memory.json` 读取 memory
- 从 `memory.custom_skill_records` 中筛选可晋升 skill
- 导出到 `out/promoted_skills/`

默认筛选规则是保守版：

- `accepted == true`
- `overall_score >= 0.78`
- `count >= 1`
- `skill_payload` 必填字段完整

例如，使用更严格的筛选条件：

```bash
python scripts/export_promotable_skills.py --min-count 2 --min-score 0.82
```


### 12.7 将导出的 promoted skill JSON 转成正式 skill 文件

如果确认导出的 skill 值得沉淀到正式 skill 库中，可以执行：

```bash
python scripts/promote_skill_json_to_template.py
```

默认行为：

- 从 `out/promoted_skills/` 读取导出的 JSON
- 转换为 `src/skills/templates/<style_family>/<skill_id>.py`

只晋升某个 style family：

```bash
python scripts/promote_skill_json_to_template.py --style-family sci_fi
```

如果需要覆盖已有同名 skill 文件：

```bash
python scripts/promote_skill_json_to_template.py --overwrite
```


### 12.8 导出 skill execution summaries

如果希望离线查看哪些 skill / carrier 在某个 `style_family + archetype` 下更稳定有效，可以执行：

```bash
python scripts/export_skill_execution_summaries.py
```

默认行为：

- 从 `out/memory/global_memory.json` 读取 memory
- 从 `memory.skill_execution_summaries` 读取聚合结果
- 导出到 `out/skill_execution_summaries/`

例如，只看 `sci_fi` 下的执行统计：

```bash
python scripts/export_skill_execution_summaries.py --style-family sci_fi
```

例如，只看某个 archetype：

```bash
python scripts/export_skill_execution_summaries.py --style-family sci_fi --archetype-id single_person_indoor_closeup
```


### 12.9 常用脚本汇总

规划器：

- `python scripts/test_video_planner.py`

编辑模型：

- `python scripts/test_edit_model.py --list-variants`
- `python scripts/test_edit_model.py --variant wan14b+dittolora --low-video smoke-data/1_low-cost.mp4 --output-video out/test_ditto_wan/sample1_edited.mp4 --prompt-file smoke-data/prompt1.txt --device cuda`

Critic：

- `python scripts/test_critic.py`

完整闭环：

- `python scripts/run_planner_and_edit.py`

memory 分析与回放：

- `python scripts/test_memory.py`

从 memory 导出可晋升 skill：

- `python scripts/export_promotable_skills.py`

导出 skill execution summaries：

- `python scripts/export_skill_execution_summaries.py`

将导出的 promoted skill 落入 skill 模板库：

- `python scripts/promote_skill_json_to_template.py`


## 13. Reference 资料目录

`reference/` 存放论文、外部项目快照和 benchmark 参考资料，不是运行时依赖。为了避免把大体积参考材料直接放进主仓库，它被拆到一个 private Hugging Face dataset repo，并在主仓库中记录为 submodule：

```text
https://huggingface.co/datasets/Ouzhang/low-high-reference
```

首次拉取主仓库后，如果需要本地展开 `reference/`，先准备有访问权限的 Hugging Face token：

```bash
export HF_TOKEN='your_hf_token'
```

推荐方式是按 submodule 拉取：

```bash
cd /path/to/low-high-new
GIT_LFS_SKIP_SMUDGE=1 git -c http.extraHeader="Authorization: Bearer ${HF_TOKEN}" \
  submodule update --init --depth 1 reference
git -C reference -c http.extraHeader="Authorization: Bearer ${HF_TOKEN}" lfs pull
```

如果当前机器的 Git 访问 Hugging Face 不稳定，也可以用 HF CLI 直接下载内容到同名目录：

```bash
cd /path/to/low-high-new
rm -rf reference
HF_TOKEN="${HF_TOKEN}" hf download Ouzhang/low-high-reference \
  --repo-type dataset \
  --local-dir reference
```

注意：

- `reference/` 只用于研究参考和资料查阅，训练、推理和评测脚本不依赖它。
- 这个 repo 是 private 的；没有 HF token 或权限时，主仓库仍然可以正常使用，只是 `reference/` 不会展开。
- 不要把 token 写进 `.gitmodules`、README 或任何提交文件里。


## 14. 当前阶段判断

当前仓库更适合被视为一个：

`面向 low/high 视频编辑的 agent 原型系统`

而不是一个已经完备的训练框架。

因此，现阶段最合理的推进顺序是：

- 先稳定闭环
- 再形式化 planner 和 critic 的目标
- 再接数据与训练
- 最后验证 RL / GRPO

这样可以避免在目标尚未定义清楚时，就提前投入大规模训练。


## 15. 总结

本项目的目标是构建一个自适应、agent 驱动的 low-cost 到 high-cost 视频生成/编辑系统。

它的核心贡献方向，是将一次性的视频编辑流程改造成一个包含以下模块的闭环：

- planning
- execution
- critique
- reflection
- memory

当前仓库已经具备这一思路的有效原型。下一阶段的重点，是把原型逐步推进成一个可训练、可扩展、可验证的研究系统。
