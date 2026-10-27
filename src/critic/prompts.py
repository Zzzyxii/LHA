# from __future__ import annotations

import json


def build_critic_identity_prefix() -> str:
    """构建 critic 相关提示词共享的身份前缀。"""
    return "你是专业特效电影评价师，负责低质量到高质量转换流程的评估。\n"


def build_edit_evaluator_system_prompt(*, with_hold_check: bool = False) -> str:
    """构建编辑评估器的系统提示词"""
    hold_rule = (
        "HOLD_CHECK: 每个必须保持原样的维度各占一行，形如 '- <维度>: HELD' 或 "
        "'- <维度>: VIOLATED (一句说明哪里被改了)'，不要合并、不要省略。\n"
        if with_hold_check
        else ""
    )
    return (
        build_critic_identity_prefix()
        +
        "你负责低质量到高质量转换流程的评估。\n"
        "你需要对比低质量视频帧和编辑后视频帧，判断它们是否符合预期的编辑提示词。\n"
        "评判编辑是否遵循了提示词、是否保留了必要的身份/结构、是否做出了有意义的改变、\n"
        "效果是否附着在可见的载体上，以及目标世界是否得以实现。\n"
        "仅以纯文本形式回复，不要输出JSON，不要添加markdown代码块标记。\n"
        "使用用户提示词要求的精确标题格式。\n"
        "分数必须在[0,1]范围内。失败标签和反思提示应简洁明了。\n"
        "不要包含思维链、逐步推理、自我修正或重复的思考过程。\n"
        "不要详细解释分数，只返回简洁的最终判断。\n"
        "直接以PROMPT_ALIGNMENT:开头，不要在标题前添加任何前言或分析。\n"
        "每个分数标题必须在同一行只包含一个数值。\n"
        "FAILURE_TAGS必须是单行的逗号分隔的蛇形命名标签。\n"
        "REFLECTION_HINTS必须是单行，包含1-3个简短的建议，用逗号分隔。\n"
        "NOTES必须是单行，最多3个简短的事实观察，用逗号分隔。\n"
        + hold_rule
    )


def build_edit_evaluator_user_prompt(
    *,
    sample_id: str,
    prompt: str,
    planner_summary: dict,
    edit_summary: dict,
    metric_hints: dict,
    hold_requirements: dict[str, str] | None = None,
) -> str:
    """构建编辑评估器的用户提示词

    ``hold_requirements`` 是 {维度: 该维度的保留短语}，来自指令里点名要保的
    东西。给了它就会多要一个 HOLD_CHECK 段：逐维度报是否保住，而不是只给一个
    笼统的 structure_preservation 分。优化 skill 时需要知道是哪个维度被改了，
    一个总分指不出该改模板里的哪个字段。
    """
    holds = {k: v for k, v in (hold_requirements or {}).items() if k and v}
    hold_block = ""
    hold_heading = ""
    if holds:
        lines = "\n".join(f"- {dim}: {phrase}" for dim, phrase in holds.items())
        hold_block = (
            "必须保持原样的维度（逐个单独检查，不要只看整体印象）:\n"
            f"{lines}\n"
        )
        hold_heading = "HOLD_CHECK:\n"
    return (
        f"sample_id: {sample_id}\n"
        f"target_edit_prompt: {prompt}\n"
        f"planner_summary: {json.dumps(planner_summary, ensure_ascii=False)}\n"
        f"edit_summary: {json.dumps(edit_summary, ensure_ascii=False)}\n"
        f"metric_hints: {json.dumps(metric_hints, ensure_ascii=False)}\n"
        f"{hold_block}"
        "图像顺序:\n"
        "- 前半部分是低质量视频帧。\n"
        "- 后半部分是对应时刻的编辑后视频帧。\n"
        "按照以下含义进行评估:\n"
        "- prompt_alignment: 编辑是否符合预期的提示词。\n"
        "- structure_preservation: 是否需要保留的身份、姿态、构图和大致布局是否被保留。\n"
        "- transformation_strength: 改变的幅度是否足够强烈，让人觉得是有意为之。\n"
        "- carrier_grounding: 效果是否附着在可见的道具/主体/环境上，而不是漂浮的无意义效果。\n"
        "- world_realization: 目标世界/风格是否令人信服地实现。\n"
        "- temporal_coherence: 结果是否随时间保持稳定和连贯。\n"
        "- artifact_penalty: 数值越高表示视觉伪影越少、失真越少。\n"
        "失败标签应使用简短的蛇形命名标签，如: prompt_misalignment, under_transformation, weak_world_realization, identity_drift, detached_fx, artifact_heavy。\n"
        "格式规则:\n"
        "- 立即以PROMPT_ALIGNMENT:开头，不要引言、不要推理、不要自言自语。\n"
        "- 从PROMPT_ALIGNMENT到OVERALL_SCORE，每个标题后只放一个数字在同一行。\n"
        "- FAILURE_TAGS: 仅限逗号分隔的蛇形命名标签，不要解释，不要括号。\n"
        "- REFLECTION_HINTS: 仅限1到3个简短的祈使句提示。\n"
        "- NOTES: 最多3个简短的事实观察。\n"
        "- 不要在不同标题中重复相同的观点。\n"
        + (
            "- HOLD_CHECK: 上面列出的每个维度各占一行，'- <维度>: HELD' 或 "
            "'- <维度>: VIOLATED (哪里被改了)'。一个维度一行，不要合并成一句。\n"
            if holds
            else ""
        )
        + "返回以下精确的标题:\n"
        "PROMPT_ALIGNMENT:\n"
        "STRUCTURE_PRESERVATION:\n"
        "TRANSFORMATION_STRENGTH:\n"
        "CARRIER_GROUNDING:\n"
        "WORLD_REALIZATION:\n"
        "TEMPORAL_COHERENCE:\n"
        "ARTIFACT_PENALTY:\n"
        "OVERALL_SCORE:\n"
        f"{hold_heading}"
        "FAILURE_TAGS:\n"
        "REFLECTION_HINTS:\n"
        "NOTES:\n"
    )


def build_edit_evaluator_repair_system_prompt() -> str:
    """构建修复不完整编辑评估器输出的系统提示词"""
    return (
        build_critic_identity_prefix()
        +
        "你正在修复一个不完整的评估输出。\n"
        "不要从头重新评估，只填充缺失的标题。\n"
        "仅以纯文本形式回复，使用请求的精确标题格式。\n"
        "不要包含任何分析或前言。\n"
        "每个分数标题必须只包含一个数值在同一行。\n"
    )


def build_edit_evaluator_repair_user_prompt(
    *,
    raw_response: str,
    missing_headings: list[str],
) -> str:
    """构建修复不完整编辑评估器输出的用户提示词"""
    headings = "\n".join(f"{item}:" for item in missing_headings)
    return (
        "之前的评论家回复不完整。\n"
        f"missing_headings: {json.dumps(missing_headings, ensure_ascii=False)}\n"
        "之前的回复:\n"
        f"{raw_response[:10000]}\n\n"
        "只返回这些缺失的标题，每个一次，按此确切顺序:\n"
        f"{headings}\n"
    )
