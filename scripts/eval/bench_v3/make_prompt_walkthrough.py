#!/usr/bin/env python3
"""Render a step-by-step prompt walkthrough for a few samples into markdown.

For each sample, every arm's prompt is reproduced verbatim from the run
directories: expert instruction, one-line intents, the skill-v1 rewrite (all
attempts), skill v2 (instruction + appended skill blocks), and the planning-track
agent arms with each planner stage's output. Commentary is passed in per sample so
the generated text stays tied to the files it describes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from skills.dimensions import compose_dimensions  # noqa: E402
from utils.prompt_budget import count_tokens  # noqa: E402

OUT = ROOT / "out"
V1 = OUT / "bench_skill/run50_native/ours_skill/loop"
V2 = OUT / "bench_skill/run30_v2/ours_skill/loop"
PLAN = OUT / "bench_v2/planning"


def _jsonl(path: Path) -> dict[str, dict]:
    return {r["sample_id"]: r for r in map(json.loads, open(path, encoding="utf-8")) if r}


def _field(text: str, name: str) -> str:
    match = re.search(rf"^{name}:\s*(.*?)(?=\n[A-Z_]{{4,}}:|\Z)", text, re.S | re.M)
    return match.group(1).strip() if match else ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _block(text: str) -> str:
    return "```text\n" + text.strip() + "\n```"


def _details(summary: str, text: str) -> str:
    return f"<details><summary>{summary}</summary>\n\n{_block(text)}\n\n</details>"


def _critic(attempt: Path) -> tuple[str, str]:
    try:
        score = json.loads(_read(attempt / "evaluation.response.json"))["scores"]["overall_score"]
    except (ValueError, KeyError):
        score = "?"
    try:
        action = json.loads(_read(attempt / "reflection.response.json")).get("action", "?")
    except ValueError:
        action = "?"
    return str(score), str(action)


def _selected(loop_dir: Path, final_video: Path) -> str:
    data = final_video.read_bytes() if final_video.exists() else b""
    for attempt in sorted(loop_dir.glob("attempt_*")):
        video = attempt / "edited.mp4"
        if video.exists() and video.read_bytes() == data:
            return attempt.name
    return "?"


def _sections(prompt: str) -> dict[str, str]:
    parts = re.split(r"\n\n(?=(?:Detailed edit plan|Craft direction|Preserve unchanged):\n)", prompt)
    out = {"Instruction": parts[0]}
    for part in parts[1:]:
        head, _, body = part.partition(":\n")
        out[head] = body
    return out


def _agent_section(arm: str, sample_id: str, label: str) -> list[str]:
    attempt = PLAN / arm / "loop" / sample_id / "attempt_001"
    calls = {name: _read(attempt / f"planner.call_{name}.response.txt") for name in
             ("001_family", "002a_grounding", "002b_skill_selection", "003_candidates", "004_final")}
    sent = _read(attempt / "edit.prompt.txt")
    sections = _sections(sent)
    intent = sections["Instruction"]
    plan = _field(calls["004_final"], "COMPILED_EDIT_PROMPT")
    fired = compose_dimensions(intent + " " + plan)
    all_craft = [policy.direction for policy in fired if policy.direction]
    sent_craft = [line[2:] for line in sections.get("Craft direction", "").splitlines() if line.startswith("- ")]
    dropped = [line for line in all_craft if line not in sent_craft]
    score, action = _critic(attempt)
    candidates = re.findall(r"strategy_id:\s*(\S+)", calls["003_candidates"])
    matched = " ".join(_field(calls["002b_skill_selection"], "MATCHED_SKILL_CANDIDATES").split()) or "(无)"

    lines = [f"#### {label}", ""]
    lines += [
        "| 步骤 | 模块 | 看到什么 | 产出(摘要) |",
        "|---|---|---|---|",
        f"| 1 | planner 风格类别 | 一句话 | `{_field(calls['001_family'], 'SELECTED_STYLE_FAMILY')}`,置信度 {_field(calls['001_family'], 'STYLE_FAMILY_CONFIDENCE')} |",
        f"| 2 | planner 画面识别 | 一句话 + 6 帧 low | {_field(calls['002a_grounding'], 'VIDEO_BRIEF')[:160]}… |",
        f"| 3 | planner 选模板 | 一句话 + 第 2 步结果 + 旧 skill 库 | 匹配:{matched};另起自定义方案 |",
        f"| 4 | planner 候选方案 | 一句话 + 6 帧 + 前面结果 | 候选 {', '.join(f'`{c}`' for c in candidates)},选中 `{_field(calls['004_final'], 'SELECTED_STRATEGY')}` |",
        f"| 5 | planner 最终方案 | 一句话 + 6 帧 + 选中候选 | 英文方案 {len(plan.split())} 词(全文见下) |",
        f"| 6 | skill 关键词匹配 | 一句话 + 方案的文字 | 触发 {len(composed.skills)} 个:{', '.join(s.skill_id for s in composed.skills)} |",
        f"| 7 | 拼接 + 512 token 裁剪 | 上面所有 | {count_tokens(sent)} token;工艺要求 {len(all_craft)} 条 → 保留 {len(sent_craft)} 条;方案未裁剪 |",
        f"| 8 | 编辑模型 ours | low + 最终 prompt | 视频 |",
        f"| 9 | critic | 3 帧 low + 3 帧输出 + 最终 prompt | 总分 {score};反思结论 `{action}`(只跑 1 轮,不生效) |",
        "",
        "**最终交给编辑模型的 prompt**(原样):",
        "",
        _block(sent),
        "",
    ]
    if dropped:
        lines += ["因超长被删掉的工艺要求:", ""] + [f"- ~~{line}~~" for line in dropped] + [""]
    return lines


def render(sample_id: str, notes: dict, gt: dict, pairwise: dict) -> list[str]:
    index = {r["sample_id"]: r for r in json.load(open(ROOT / "reports/top50_video_bundle_20260725_nodup/sample_index.json"))}
    specific = _jsonl(ROOT / "scripts/eval/bench_v3/data/intents_specific.jsonl")[sample_id]["intent"]
    generic = _jsonl(ROOT / "scripts/eval/bench_v3/data/intents_generic.jsonl")[sample_id]["intent"]
    expert = index[sample_id]["instruction"]
    cat = gt["ours_raw"][sample_id]["category"]

    lines = [f"## {notes['title']}", ""]
    lines += [f"样本 `{sample_id}`,类别 `{cat}`。{notes['scene']}", ""]
    lines += [
        f"对比视频:`out/bench_v2/grids/{notes['grid']}`",
        "",
        "### 三种输入",
        "",
        f"**专家指令**({len(expert.split())} 词,VLM 看着 low 和 high 写的):",
        "",
        _block(expert),
        "",
        f"**一句话意图**:`{specific}`",
        "",
        f"**通用一句话**:`{generic}`",
        "",
        "### 执行 track(输入 = 专家指令)",
        "",
        "#### ① ours_raw:直接喂",
        "",
        "prompt = 上面的专家指令,原样。不经过任何大模型。",
        "",
    ]

    v1_dir = V1 / sample_id
    selected = _selected(v1_dir, OUT / f"bench_skill/run50_native/ours_skill/generations/{sample_id}.mp4")
    lines += ["#### ② skill v1(原作者框架的原始版本):planner 改写,替换掉专家指令", ""]
    lines += [
        "流程:专家指令 + low → planner(gemini-3.1-pro,旧提示词)写出改写方案 → **改写方案直接当 prompt**,专家原文丢弃 "
        "→ 编辑 → critic 打分 → 规则反思 → 在 prompt 后追加修正要点重新规划 → 最多 3 轮,取 critic 分最高的一轮。",
        "",
    ]
    for attempt in sorted(v1_dir.glob("attempt_*")):
        score, action = _critic(attempt)
        prompt = _read(attempt / "edit.prompt.txt")
        mark = "  ← 最终选中" if attempt.name == selected else ""
        summary = f"{attempt.name}:{len(prompt.split())} 词,critic {score},反思 {action}{mark}"
        lines += [_details(summary, prompt), ""]

    v2_dir = V2 / sample_id / "attempt_001"
    if v2_dir.exists():
        dropped_plan = _field(_read(v2_dir / "planner.call_004_final.response.txt"), "COMPILED_EDIT_PROMPT")
        score, _ = _critic(v2_dir)
        sent = _read(v2_dir / "edit.prompt.txt")
        lines += [
            "#### ③ skill v2(交接文档的改进版):专家原文打头,后面追加 skill",
            "",
            "流程:专家指令 + low → planner 照样完整跑 6 步 → **planner 写的方案被丢弃** → "
            "专家原文 + skill 工艺要求 + 保持不变清单(通用 + skill + planner 输出的保持项)→ 编辑 → critic。1 轮。",
            "",
            _details(f"被丢弃的 planner 方案({len(dropped_plan.split())} 词)", dropped_plan),
            "",
            f"**最终交给编辑模型的 prompt**({count_tokens(sent)} token,超过 512 的部分模型看不到;critic {score}):",
            "",
            _block(sent),
            "",
        ]

    lines += [
        "### 规划 track(输入 = 一句话)",
        "",
        "#### ④ intent_direct / generic_direct:直接喂",
        "",
        f"prompt 分别就是 `{specific}` 和 `{generic}`,不经过任何大模型。",
        "",
    ]
    lines += _agent_section("ours_intent_agent", sample_id, "⑤ intent_agent:一句话 → planner 扩写 → 追加 skill")
    lines += _agent_section("ours_generic_agent", sample_id, "⑥ generic_agent:通用句 → planner 扩写 → 追加 skill")

    def row(name: str, arm: str, desc: str) -> str:
        r = gt[arm].get(sample_id)
        if not r:
            return f"| {name} | {desc} | — | — | — |"
        return f"| {name} | {desc} | {r['gain_dino']:+.3f} | {r['gain_clip']:+.3f} | {1 - r['src_dino']:.2f} |"

    lines += [
        "### 结果",
        "",
        "| 组 | prompt 来源 | gain_dino | gain_clip | 改动量(1−src_dino) |",
        "|---|---|---|---|---|",
        f"| 标准答案 high | — | {gt['oracle'][sample_id]['gain_dino']:+.3f} | {gt['oracle'][sample_id]['gain_clip']:+.3f} | {1 - gt['oracle'][sample_id]['src_dino']:.2f} |",
        row("① ours_raw", "ours_raw", "专家指令"),
        row("② skill v1(第 1 轮)", "v1_attempt1", "planner 改写"),
        row("② skill v1(最终)", "ours_skill_v1", f"planner 改写,{selected}"),
        row("③ skill v2", "ours_skill_v2", "专家指令 + skill"),
        row("④ intent_direct", "intent_direct", "一句话"),
        row("⑤ intent_agent", "intent_agent", "一句话 + 方案 + skill"),
        row("④ generic_direct", "generic_direct", "通用句"),
        row("⑥ generic_agent", "generic_agent", "通用句 + 方案 + skill"),
        "",
        f"成对 judge:intent_agent vs intent_direct → `{pairwise['intent'].get(sample_id)}`;"
        f"generic_agent vs generic_direct → `{pairwise['generic'].get(sample_id)}`",
        "",
        "### 这个例子说明什么",
        "",
        notes["analysis"],
        "",
        "---",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=Path, required=True, help="JSON: {intro, samples: [{sample_id, title, scene, grid, analysis}]}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notes = json.loads(args.notes.read_text(encoding="utf-8"))

    gt: dict[str, dict] = {}
    for folder, arms in (
        (OUT / "bench_v2/gt_all", ("ours_raw", "ours_skill_v1", "ours_skill_v2", "oracle")),
        (OUT / "bench_v2/gt_planning", ("intent_direct", "intent_agent", "generic_direct", "generic_agent")),
        (OUT / "bench_v2/gt_v1_attempt1", ("v1_attempt1",)),
    ):
        for arm in arms:
            gt[arm] = _jsonl(folder / f"{arm}.jsonl")
    pairwise = {
        form: {r["sample_id"]: r.get("verdict") for r in map(json.loads, open(
            OUT / f"bench_v2/pairwise_planning/{form}_agent__vs__{form}_direct.jsonl", encoding="utf-8"))}
        for form in ("intent", "generic")
    }

    lines = [notes["intro"], ""]
    for sample in notes["samples"]:
        lines += render(sample["sample_id"], sample, gt, pairwise)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
