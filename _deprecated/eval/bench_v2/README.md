# bench_v2: GT 参照的 low→high 评测

修正原 top50 bench 的三个问题,全部为新增文件,不改原有评测。

## 原 bench 的问题

1. **指标不逐条和 target_high 比。** CLIP-T / PickScore 比指令文本,CLIP-F / DINO-F 比自身相邻帧,
   CLIP Src-Edit 比 low,LAION / MUSIQ / NIQE / 美学 Reward 只看输出。target_high 只在 JEDi 里以集合分布出现。
   结果是"基本不改"也能拿到不错的分数。
2. **指令泄露答案。** 指令是 VLM 同时看 low 和 high 帧写的(见 `datas/low_high_cost/invert_high2low_prompts_api.py`
   的 `VIDEO_USER_TEMPLATE`),里面带着 target 的具体颜色和物件。它测的是"给定专家 prompt 的执行能力",
   不是 agent 写 prompt 的能力。
3. **任务混杂。** 50 条里有 11 条是 VHS / 胶片颗粒 / 黑白这类刻意降低制作感的滤镜,和 low→high 方向相反。

## 本目录

| 文件 | 作用 |
|---|---|
| `build_manifest.py` | 从 `归档.zip` 解压目录生成 manifest,补上 `category` / `story_aligned` / `in_release30` |
| `gt_metrics.py` | 逐条对 target_high 打分,自动带 `noop` / `oracle` 两个参照臂 |
| `run_gt_all.sh` | 对归档里 16 个已发表方法 + 我们的各臂全量重评 |

## 指标

按时间戳对齐取 K=8 帧(不同方法导出 17 / 49 / 73 / 原生帧数,按索引对齐会错位):

- `gt_dino` / `gt_clip`:输出和 target_high 的相似度
- `src_dino` / `src_clip`:输出和 low 的相似度(保留度,作为约束而不是目标)
- **`gain_dino` / `gain_clip`(主指标)** = `gt_*` − 同一时间窗口内 low 和 target 的相似度。
  原样返回输入的增益恰好为 0;往错误方向改会变成负数。

自检:`noop` 的增益必须是 0.0000,`oracle` 的 `gt_dino` 必须是 1.0000,否则说明流程有问题。

## 两条 track

- **执行 track**:low + 专家指令。和已发表结果可比,就是本目录目前跑的内容。
- **规划 track**:low + 用户一句话意图,这才是测 agent 的那条。manifest 里的 `user_intent` 留空,
  需要先固定、人工审核一版意图文件再生成。

## 运行

```bash
python3 scripts/eval/bench_v2/build_manifest.py \
  --release-root /path/to/归档_extract --output out/bench_v2/manifest.jsonl
bash scripts/eval/bench_v2/run_gt_all.sh          # GPU=0 默认
```

所需模型(本机 `models/hf_cache/hub` 已有):`openai/clip-vit-large-patch14`、`facebook/dino-vits16`。
