# outt/ 拷贝说明

来源：`/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/low-high-new`（原始目录未做任何改动）

用途：基于 Ditto / Bernini 做 **skill 框架代码 + benchmark + skill 优化**。
按你的要求：**训练相关的全部排除**，**模型权重用软链接**，其余需要的文件实拷。

---

## 1. 保留了什么

### 代码（实拷）
| 目录 | 说明 |
|---|---|
| `src/skills/` | skill 框架核心。`base.py` + `registry.py` 是旧的 style-family 模型；**`enhancement.py` + `enhancement_templates.py`（10 个编辑维度 skill）是当前在用的路径** |
| `src/skills/templates/` | 13 个风格模板（sci_fi 7 / anime 3 / historical_fantasy 3），legacy 但仍是 registry 的数据源 |
| `src/planner/` | `core/pipeline.py:754 _compose_edit_prompt()` 是真正的 skill→prompt 编译器；`core/skill_runtime.py` 做 skill 选择 + custom skill 合并；`core/video_planner.py` 是 5 阶段 VLM 规划 |
| `src/critic/` | 评估 + reflection（决定 accept / local_refine_prompt / global_replan） |
| `src/edit/` | edit model wrapper，含 variant 与模型路径常量 |
| `src/loop/` | 闭环 runner |
| `src/memory/` | **skill 优化的关键**：`evolver.py`（`_promotion_bucket` 决定 promotable/custom/quarantined）、`analyzer.py`（抽取 custom skill / recipe / failure pattern）、`retriever.py`（`retrieve_runtime_skills` 输出 boosted/penalized skill_ids） |
| `src/vlm/`、`src/utils/` | VLM 后端与 prompt budget（umT5 512 token 窗口） |
| `scripts/bench_skill/` | benchmark 生成入口 `run_bench.py`（21 个 arm）+ `env.sh` + 各 launch 脚本 |
| `scripts/eval/bench_v3/` | **最新 benchmark**（3 tier × 3 track，LLM judge + GT） |
| `scripts/eval/bench_v2/` | GT-referenced eval（`gt_metrics.py`、`build_manifest.py`、`data/` 下的 intents 与 top50_sources） |
| `scripts/eval/ditto_quant/` | reward / judge 工具，bench 仍在用 |
| `scripts/export_promotable_skills.py`、`promote_skill_json_to_template.py`、`replay_memory_from_attempts.py`、`rebuild_memory.py`、`run_memory_mode_experiments.py`、`summarize_memory_mode_experiments.py`、`cleanup_memory.py`、`export_skill_execution_summaries.py` | skill 优化 / promotion 闭环 |
| `configs/` | `memory.json` 是活的（promotion 阈值、retrieval 权重）。**注意：`base_skills.json` 和 `planner_qwen.json` 没有任何代码读取，是死配置** |

### 外部代码依赖（实拷）
- `Ditto/`（100M）— 带本地改动，**不要用上游原版替换**（含 `latent_memory_runtime.py` 等补丁）
- `third_party/Bernini/`（100M，已去掉其 `.git`）— Bernini arm 需要

### Benchmark 数据（实拷）
- `reports/top50_video_bundle_20260725_nodup/`（63M）— 50 组 source_low + target_high + `sample_index.json`（含 expert instruction）+ `scores.csv`。所有 launch 脚本都指向它
- `归档_extract/`（209M）— **原本在仓库外**（`dev/归档_extract`），内含 `source_low`/`target_high`/`methods`，是 bench_v3 三个 track 的 source_video 来源，已一并拷进来
- `smoke-data/` — smoke test 用
- `zyx/`（93M）— OPUSCAR 98 风格库，`styles.json` 里 98 条可复用 style prompt，是扩充 skill 模板的现成语料

### 实验产物（实拷）
- `out/bench_v3/`、`out/bench_v2/`、`out/bench_skill/`、`out/report/`、`out/judge_gpt56luna/`、`out/*.md`
- 已排除 `out/hf_eval_bundle/`（5.9G，HF 上传归档）

### 模型权重（软链接）
`models/` 下每个条目软链回原仓库，不占空间：
`Wan2.1-VACE-14B`(70G)、`Bernini-R-1.3B-Diffusers`(27G)、`Qwen3-VL-8B-Thinking`(17G)、`vjepa`(9.9G)、`hf_cache`(9.2G)、`Ditto_models`(235M)、`torch_cache`、`cache`

另外 `out/hf_checkpoints/.../step-6000.safetensors`（264M，"ours" variant 的推理权重，硬编码在 `src/edit/video_editor.py:44`）也做了软链。

---

## 2. 排除了什么

| 排除项 | 原因 |
|---|---|
| `.git/`（2.0G） | 按你的选择不拷历史 |
| `scripts/training/` | 训练 |
| `datas/critic-data/`、`datas/memory-data/`、`datas/ditto_face/`、`datas/ditto_face2/`（共 ~370M） | 训练数据与 teacher label |
| `datas/low_high_cost/` | 训练数据流水线 |
| `teacher_labels.checkpoint.jsonl` | 训练标注 |
| `out/hf_eval_bundle/` | HF 上传归档，纯产物 |
| `out/hf_checkpoints/`（权重本身） | 训练产物，仅保留一个推理用权重软链 |
| `CCEdit/`、`reference/`、`DiffSynth-Studio/` | 子模块，本地为空 |
| `.pytest_cache/`、`.cache/`、`.codex/` | 缓存 |

---

## 3. 在新目录跑起来之前必须改的路径

代码里几乎没有硬编码绝对路径（都从 `__file__` 推导），但**下面几处是绝对/外部路径，需要重新指向**：

1. `scripts/eval/bench_v3/eval_config.json` → `"HIGH_TOP50"`，当前指向
   `/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/归档_extract/target_high`
   改为 `归档_extract/target_high`（outt 内已有一份）

2. `scripts/eval/bench_v3/data/{track_a,style_track,ditto_track}.jsonl` 中每条的
   `source_video` / `target_video` 字段指向 `dev/归档_extract/...`，需要批量替换前缀

3. `scripts/eval/bench_v2/run_gt_all.sh` → `RELEASE_ROOT` 默认也是 `dev/归档_extract`

4. `scripts/bench_skill/bin/ffmpeg` → 指向 `/opt/venv/...` 的链接已失效（容器本地路径）。
   `env.sh` 会在装了 `imageio_ffmpeg` 时自动重建

5. 网络与 key：planner / critic / judge 走 litellm 网关 `http://litellm-sgp.woa.com`，
   需要 `LOW_HIGH_JUDGE_API_KEY` + `LOW_HIGH_JUDGE_BASE_URL`（见 `scripts/bench_skill/env.sh`）

---

## 4. skill 系统已合并为一套（2026-10-24）

原来三套并存的 skill 已合成一套。**风格迁移为主轴**，指令点名的保留要求作为逐维度覆盖层。

### 新增文件

| 文件 | 作用 |
|---|---|
| `src/skills/dimensions.py` | 11 个规范维度 + 共享的 `DimensionPolicy`（trigger / must_preserve / order / strength / keep_phrase / direction）。跨风格通用的东西都在这一份里 |
| `src/skills/dimension_parse.py` | 指令 → 哪些维度 keep / edit。确定性关键词，不经 VLM |
| `scripts/test_dimensions.py` | 回归测试，`python3 scripts/test_dimensions.py`，7/7 |
| `scripts/build_opuscar_templates.py` | `zyx/styles.json` → 98 个风格模板（英文 + 参考图） |
| `_deprecated/enhancement.py`、`enhancement_templates.py` | 已废弃，代码里无人引用。**可以删掉**（我用 `rm` 被权限拦了，用 `mv` 移到了这里） |

`src/skills/base.py` 的 `SkillTemplate` 从 8 个渲染字段扩到 11 个规范维度，另加 `reference_image`。

### 送达方式开关

新增环境变量，跟 `LOW_HIGH_AGENT_PROMPT_MODE` 同一套路：

```
LOW_HIGH_STYLE_DELIVERY = inline | reference | none
```

- `inline`（默认）— 风格按维度写进 prompt（`Target world:` 段）
- `reference` — 风格走 VACE 参考图，prompt 只留短指令 + 关键约束
- `none` — 对照 arm，不用 style

`src/utils/prompt_budget.py` 的 `fit_sections()` 加了 `style_lines` 参数：`instruction → plan → style → craft → preserve`，超限时依次裁 craft → plan → style，instruction 和 preserve 永不裁。

### 新增的 98 个风格

`src/skills/templates/styles_{2d,3d,25d}/opuscar_NN_*.py`，共 98 个（按 `kind` 分族），每个带 `reference_image` 指向 `zyx/img/opuscar98/NN.jpg`。registry 现在共 111 个模板。

**已知局限**：`prompt_en` 是按句号切分 + 关键词归类的启发式拆分，约一半子句匹配不到维度（进了 `prompt_compilation_notes`，不进 prompt），少数子句会误分类（相机/剪辑语言已过滤）。对 `reference` 模式无影响（不吃文本）；`inline` 模式的文本质量因此偏低——这正是要对比实验来判定的事。

### 还没做

- 13 个原有模板仍是 8 字段、7 个是中文（整个 sci_fi 族），没迁移。它们有 zyx 没有的元数据（archetype、carriers、failure modes）。
- `reference` 模式需要 renderer 侧真正把 `reference_image` 传给 VACE——目前只在 compose 层做了分支，通道接线（`run_bench.py` 的 `vace_reference`）没动。

---

## 5. benchmark 已合并为 bench_v3（2026-10-24）

`scripts/eval/` 现在只剩两个目录：

| 目录 | 角色 |
|---|---|
| `scripts/eval/bench_v3/` | **唯一的 benchmark**。3 track × 3 tier，LLM judge + GT |
| `scripts/eval/ditto_quant/` | 不是 benchmark，是 bench_v3 的 judge 库（`judges.py:44` 直接依赖它） |

`scripts/bench_skill/` 是**出片 runner**，不是评测，保留。

### 从 bench_v2 搬进来的

bench_v3 原本依赖 bench_v2 的四样东西，已全部搬入 `scripts/eval/bench_v3/`：

- `data/top50_sources.jsonl`、`data/intents_{specific,generic}.jsonl`、`data/release30_ids.txt`
- `gt_metrics.py`（GT 指标，run_eval.py 调用）
- `build_manifest.py`、`build_ablation_prompts.py`、`make_prompt_walkthrough.py`

源码里 22 处 `scripts/eval/bench_v2/` 路径已改为 `bench_v3/`。

### 归档到 `_deprecated/eval/` 的

`bench_v2/`（剩余）、`low_high_critic/`（训练相关）、`stage2_fastdata_top50/`（旧集群管线）、`report/`（展示用）。

**`out/bench_v2/` 没有动**——eval_config.json 里多个 arm 指向它下面的已生成结果（`planning/`、`ablation/`），那是产物不是代码。

### 注意

框架本身同时覆盖纯风格迁移和定点编辑：指令没点名任何维度就全走 style，点名了就走 style + 该维度执行细节，说要保就替换成 hold 约束。所以三个 track 都能跑，不需要为定点编辑单开一条路径。

---

## 6. critic 加了逐维度保留检查（2026-10-24）

原来 critic 只给 7 个整体分，保留这块只有一个 `structure_preservation`——发现"没保住"但说不出是哪个维度，skill 优化就无从下手（不知道该改模板的哪个字段）。

现在补上了：

- `VideoPlan` 新增 `dimension_overrides` 字段，`build_plan` 把 `parse_overrides()` 的结果存进去
- `evaluate_edit_result()` 自动从 `planner_payload["dimension_overrides"]` 取出 KEEP 的维度（也可显式传 `hold_requirements`）
- prompt 多一个 `HOLD_CHECK:` 段，要求每个维度单独一行报 `HELD` / `VIOLATED (哪里被改了)`
- 解析出的违规生成 `hold_violated:<维度>` 标签（如 `hold_violated:costume`），走 `to_dict()` 进 memory

**只在指令真的点名了要保什么时才要求这个段**，否则会白得一个 `evaluation_incomplete`。

**critic 没提到的维度记为 `reported=False` 但不打标签**——话少的回复不等于所有 hold 都崩了。

`run_loop.py` 不用改：它本来就把 `plan.model_dump()` 当 `planner_payload` 传给 critic。

### memory 侧已接上

违规信号现在能一路走到聚合层：

- `memory/contracts.py` 的 `build_critic_evaluation_signal` 保留结构化的 `hold_checks`（不只是扁平的 failure_tags）
- `memory/store.py` 新增 `hold_violation_summaries` 集合
- `memory/evolver.py` 新增 `hold_violation_summaries()`，按 **(模板, 维度)** 聚合违规率，写进 `rebuild_memory` 和 `evolve_memory`
- `_skill_quality_score` 对 hold 违规扣 0.15（跟 `artifact_heavy` 同档：指令被理解了但还是被覆盖）

聚合出来长这样：

```
strategy              dimension        违规/检查   比率
cyberpunk_dystopia    costume          4/6        0.6667
cyberpunk_dystopia    subject_surface  0/6        0.0
studio_ghibli         costume          0/5        0.0
```

同一个模板在 costume 上老违规、在 subject_surface 上没问题——这才是能动手修的信息（改模板的 `costume` 字段，而不是整个模板丢掉）。

测试：`scripts/test_critic_holds.py`，8/8。

**还没接的**：retriever 选 skill 时还没有读这份统计（即"避开 costume 违规率高的模板"这条还没生效）。目前聚合结果只进 memory 供人看和给 promotion 打分用。

顺带修了 `dimension_parse` 一个局限：`"keep the clothing and the face"` 以前只抓到 clothing。现在一个 hold 会管到它后面**下一个表示"改"的动词之前**的所有维度，所以并列列举能抓全，`"keep the face and make the lighting dramatic"` 也能正确分开。

---

## 7. 已知的坑

- `configs/base_skills.json`、`configs/planner_qwen.json` 无人读取，别当事实来源
- `models/Wan2.1-Distill-Loras` 在原仓库里**就不存在**，所以带 `4steplora` 的 variant 本来就跑不了
- `src/vlm/openrouter-key.txt` 里可能有真实 key，对外分享前先删
- 两套 skill 系统并存：legacy style-family（`SkillTemplate`）与在用的 enhancement（`EnhancementSkill`，关键词触发、最多选 6 个）。改 skill 之前先确认改的是哪一套
