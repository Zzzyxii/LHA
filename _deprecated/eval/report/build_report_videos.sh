#!/usr/bin/env bash
# Build the comparison clips cited in out/report/组会报告.md.
# Every clip is <id>.mp4 + <id>.png under out/report/clips/. Re-run is idempotent.
set -eu
cd "$(dirname "$0")/../../.."

A=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/归档_extract
LOW=reports/top50_video_bundle_20260725_nodup/videos/source_low
HIGH=$A/target_high
PUB=$A/methods
T1D=out/bench_v2/planning/ours_intent_direct/generations
T1A=out/bench_v2/ablation/ours_abl_plan_only/generations
T1ORIG=out/bench_v2/planning/ours_intent_agent/generations
T3=out/bench_skill/run50_native/ours_raw/generations
T3B=out/bench_skill/run50_native/bernini_raw/generations
T3S=out/bench_skill/run50_native/ours_skill/generations
X=out/bench_v3/runs/track_a.t2_cross
S=out/bench_v3/runs/track_a.t2_same
ST=out/bench_v3/runs/style_track.t2
SEED=out/bench_v3/runs
OUT=out/report/clips
mk () { python3 scripts/eval/report/make_compare.py "$@"; }

# 1 原 bench:指令里写着答案
mk --out $OUT/01_leak_0133 --title "0133  专家指令写着:前景绿色刺绣袍+深蓝立领,后排米色袍+红花+红发带" \
  --panel "low 输入=$LOW/heldout1000_0133.mp4" --panel "high 标准答案=$HIGH/heldout1000_0133.mp4"
# 2 原 Reward 奖励改得漂亮,不是改对
mk --out $OUT/02_reward_0296 --title "0296  已发表视频:Bernini 旧美学分均值 3.77(排第 3),这一条 GT 增益 -0.54;ours +0.02" \
  --panel "low 输入=$LOW/heldout1000_0296.mp4" --panel "high 标准答案=$HIGH/heldout1000_0296.mp4" \
  --panel "Bernini(已发表)=$PUB/Bernini-R-1.3B/heldout1000_0296.mp4" --panel "ours(已发表)=$PUB/ours_full/heldout1000_0296.mp4"
# 3 几乎不改也能拿分
mk --out $OUT/03_noop_0549 --title "0549  Senorita 和原片相似度 0.996,基本没改;ours GT 增益 +0.14" \
  --panel "low 输入=$LOW/heldout1000_0549.mp4" --panel "high 标准答案=$HIGH/heldout1000_0549.mp4" \
  --panel "Senorita(已发表)=$PUB/Senorita/heldout1000_0549.mp4" --panel "ours(已发表)=$PUB/ours_full/heldout1000_0549.mp4"
# 4 judge 噪声:同一个文件
mk --out $OUT/04_samefile_0296 --title "0296  两个组输出的是字节相同的同一个文件,同一个评委一次给 2.5、一次给 2.0" \
  --panel "low 输入=$LOW/heldout1000_0296.mp4" --panel "ours 输出(同一文件)=$T3/heldout1000_0296.mp4"
# 5 专家指令下 ours vs Bernini(候选 0346 / 0644)
for id in 0346 0644; do
  mk --out $OUT/05_bernini_vs_ours_$id --title "$id  专家指令:Bernini 改得多、更好看,但动了不该动的;ours 更保守" \
    --panel "low 输入=$LOW/heldout1000_$id.mp4" --panel "high 标准答案=$HIGH/heldout1000_$id.mp4" \
    --panel "ours=$T3/heldout1000_$id.mp4" --panel "Bernini=$T3B/heldout1000_$id.mp4"
done
# 6 专家指令下 skill v1 只让编辑变保守
mk --out $OUT/06_skill_conservative_0903 --title "0903  专家指令:加 skill 后改动量 0.44 -> 0.10,GT 增益 +0.14 -> +0.01" \
  --panel "low 输入=$LOW/heldout1000_0903.mp4" --panel "high 标准答案=$HIGH/heldout1000_0903.mp4" \
  --panel "直接喂专家指令=$T3/heldout1000_0903.mp4" --panel "agent+skill(最多 3 轮修改)=$T3S/heldout1000_0903.mp4"
# 7 一句话直接喂会改坏,agent 保住(候选 0357 / 0236)
for id in 0357; do
  mk --out $OUT/07_oneliner_$id --title "$id  一句话意图:直接喂改坏(比不改离答案还远),agent 保住原片" \
    --panel "low 输入=$LOW/heldout1000_$id.mp4" --panel "high 标准答案=$HIGH/heldout1000_$id.mp4" \
    --panel "一句话直接喂=$T1D/heldout1000_$id.mp4" --panel "agent 扩写=$T1A/heldout1000_$id.mp4"
done
# 8a agent 改得不够
mk --out $OUT/08a_underedit_0236 --title "0236  一句话意图:直接喂换了人;agent 保住了脸,但答案要的头饰和蓝领一个没做" \
  --panel "low 输入=$LOW/heldout1000_0236.mp4" --panel "high 标准答案=$HIGH/heldout1000_0236.mp4" \
  --panel "一句话直接喂=$T1D/heldout1000_0236.mp4" --panel "agent 扩写=$T1A/heldout1000_0236.mp4"
# 8b judge 偏好改得多的
mk --out $OUT/08b_judge_bias_0977 --title "0977  agent 最接近答案,但 GPT 和 Claude 两个评委都判直接喂更符合要求、更好看" \
  --panel "low 输入=$LOW/heldout1000_0977.mp4" --panel "high 标准答案=$HIGH/heldout1000_0977.mp4" \
  --panel "一句话直接喂=$T1D/heldout1000_0977.mp4" --panel "agent 扩写=$T1A/heldout1000_0977.mp4"
# 9 消融:保护清单/工艺要求/参考图只让模型改得更少(候选 0764 / 0903)
for id in 0764; do
  mk --out $OUT/09_ablation_$id --title "$id  同一份方案:只给方案 vs 方案+工艺+保护清单+参考图" \
    --panel "low 输入=$LOW/heldout1000_$id.mp4" --panel "high 标准答案=$HIGH/heldout1000_$id.mp4" \
    --panel "只给方案=$T1A/heldout1000_$id.mp4" --panel "全套(原 agent)=$T1ORIG/heldout1000_$id.mp4"
done
# 10 跨片段参考图:直接喂被带偏,翻译成文字后保住
mk --out $OUT/10_crossref_0357 --title "0357  参考图来自 0133(另一个片段):GT 增益:直接塞图 -0.19,agent 翻译 -0.01,翻译后再给图 -0.10" \
  --panel "low 输入=$LOW/heldout1000_0357.mp4" --panel "参考图(0133)=out/bench_v3/refs/cross_src/heldout1000_0133.png" \
  --panel "参考图直接塞给模型=$X/ours_t2_direct_ref/generations/heldout1000_0357.mp4" \
  --panel "agent 翻译成文字=$X/ours_t2_agent/generations/heldout1000_0357.mp4" \
  --panel "翻译后再给图=$X/ours_t2_agent_ref/generations/heldout1000_0357.mp4"
# 11 同镜头参考图 = 泄题上限
mk --out $OUT/11_sameref_0272 --title "0272  参考图就是标准答案的一帧(泄题):翻译后再给图,GT 增益 +0.45" \
  --panel "low 输入=$LOW/heldout1000_0272.mp4" --panel "参考图(GT 帧)=out/bench_v3/refs/same/heldout1000_0272.png" \
  --panel "high 标准答案=$HIGH/heldout1000_0272.mp4" \
  --panel "翻译后再给图=$S/ours_t2_agent_ref/generations/heldout1000_0272.mp4"
# 12 电影风格参考图
mk --out $OUT/12_style_0445 --title "风格:《辛德勒的名单》剧照作参考;直接喂参考图 vs agent 翻译成文字" \
  --panel "low 输入=$LOW/heldout1000_0445.mp4" --panel "参考图=zyx/img/opuscar98/66.jpg" \
  --panel "参考图直接塞给模型=$ST/ours_t2_direct_ref/generations/s66_schindler_hard_bw__heldout1000_0445.mp4" \
  --panel "agent 翻译成文字=$ST/ours_t2_agent/generations/s66_schindler_hard_bw__heldout1000_0445.mp4"
# 13 seed 稳定
mk --out $OUT/13_seeds_0764 --title "0764  一句话意图,同一 prompt 换 seed(不带记忆的 agent):三个 seed 都比直接喂好" \
  --panel "一句话直接喂 s42=$T1D/heldout1000_0764.mp4" \
  --panel "agent s42=$SEED/track_a.t1_fresh/ours_t1_agent/generations/heldout1000_0764.mp4" \
  --panel "agent s1055=$SEED/track_a.t1_fresh_seed1055/ours_t1_agent/generations/heldout1000_0764.mp4" \
  --panel "agent s2068=$SEED/track_a.t1_fresh_seed2068/ours_t1_agent/generations/heldout1000_0764.mp4"
# 14 对抗对照
mk --out $OUT/14_adversarial_0133 --title "0133  对抗对照:只调色的假输出 / 换成别的样本的标准答案" \
  --panel "low 输入=$LOW/heldout1000_0133.mp4" --panel "只调色=out/bench_v3/eval/_adv_recolor/heldout1000_0133.mp4" \
  --panel "换样本的答案=out/bench_v3/eval/_adv_swap/heldout1000_0133.mp4"
# 15 17 帧协议
mk --out $OUT/15_protocol_0133 --title "0133  原 bench 协议 17 帧(0.85s) vs 本次按原片时长" \
  --panel "low 输入=$LOW/heldout1000_0133.mp4" --panel "ours 已发表(17帧)=$PUB/ours_full/heldout1000_0133.mp4" \
  --panel "ours 原生时长=$T3/heldout1000_0133.mp4"
# 16 闭环修正三轮(专家指令 + skill v1,critic 每轮决定怎么改)
L3=out/bench_skill/run50_native/ours_skill/loop
mk --out $OUT/16_rounds_regress_0856 --title "0856  修正三轮:GT 增益 第1轮 +0.14 -> 第3轮 +0.09,越改越远" \
  --panel "low 输入=$LOW/heldout1000_0856.mp4" --panel "high 标准答案=$HIGH/heldout1000_0856.mp4" \
  --panel "第1轮=$L3/heldout1000_0856/attempt_001/edited.mp4" --panel "第2轮=$L3/heldout1000_0856/attempt_002/edited.mp4" \
  --panel "第3轮=$L3/heldout1000_0856/attempt_003/edited.mp4"
mk --out $OUT/16_rounds_improve_0272 --title "0272  修正三轮:GT 增益 第1轮 +0.15 -> 第3轮 +0.21,少数变好的例子" \
  --panel "low 输入=$LOW/heldout1000_0272.mp4" --panel "high 标准答案=$HIGH/heldout1000_0272.mp4" \
  --panel "第1轮=$L3/heldout1000_0272/attempt_001/edited.mp4" --panel "第2轮=$L3/heldout1000_0272/attempt_002/edited.mp4" \
  --panel "第3轮=$L3/heldout1000_0272/attempt_003/edited.mp4"
# 08b 一句话档的应改项是"通用理解",标准答案本身也会丢分
mk --out $OUT/08b_checklist_0133 --title "0133  一句话档应改项:\"背景变成华丽宫殿\"。标准答案没换背景,判没做到;直接喂换了,判做到" \
  --panel "low 输入=$LOW/heldout1000_0133.mp4" --panel "high 标准答案=$HIGH/heldout1000_0133.mp4" \
  --panel "一句话直接喂=$T1D/heldout1000_0133.mp4"
# 08c 方案写偏了,模型没照做
mk --out $OUT/08c_plan_ignored_0977 --title "0977  agent 方案写的是科幻(神经植入物、战术长袍),模型基本没照做,结果反而最接近答案" \
  --panel "low 输入=$LOW/heldout1000_0977.mp4" --panel "high 标准答案=$HIGH/heldout1000_0977.mp4" \
  --panel "一句话直接喂=$T1D/heldout1000_0977.mp4" --panel "agent 扩写=$T1A/heldout1000_0977.mp4"
# 17 换成 Bernini,同一份 agent 方案让它改过头
BV=out/bench_v3/runs/track_a.t1_bernini_v3
mk --out $OUT/17_bernini_backbone_0316 --title "0316  同一份 agent 方案:ours GT 增益 +0.10;Bernini 只给一句话几乎不改(0.01),给方案后改动量 0.39、GT -0.03" \
  --panel "low 输入=$LOW/heldout1000_0316.mp4" --panel "high 标准答案=$HIGH/heldout1000_0316.mp4" \
  --panel "ours + agent=out/bench_v3/runs/track_a.t1_fresh/ours_t1_agent/generations/heldout1000_0316.mp4" \
  --panel "Bernini 一句话直接喂=$BV/bernini_t1_direct/generations/heldout1000_0316.mp4" \
  --panel "Bernini + agent=$BV/bernini_t1_agent/generations/heldout1000_0316.mp4"
