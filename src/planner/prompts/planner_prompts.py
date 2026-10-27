from __future__ import annotations

import json


def _compact_retry_feedback(memory_context: dict, *, include_prompt_patch: bool) -> str:
    directives = [str(item).strip() for item in (memory_context.get("critic_replan_directives") or []) if str(item).strip()]
    hints = [str(item).strip() for item in (memory_context.get("critic_failure_hints") or []) if str(item).strip()]
    prompt_patch = [str(item).strip() for item in (memory_context.get("critic_prompt_patch") or []) if str(item).strip()]
    lines: list[str] = []
    if directives:
        lines.append(f"critic_replan_directives: {json.dumps(directives[:6], ensure_ascii=False)}")
    if hints:
        lines.append(f"critic_failure_hints: {json.dumps(hints[:6], ensure_ascii=False)}")
    if include_prompt_patch and prompt_patch:
        lines.append(f"critic_prompt_patch: {json.dumps(prompt_patch[:6], ensure_ascii=False)}")
    return "\n".join(lines)


def _compact_style_convertibility_priors(memory_context: dict) -> str:
    priors = [item for item in (memory_context.get("style_convertibility_priors") or []) if isinstance(item, dict)]
    lines: list[str] = []
    for item in priors[:3]:
        style_family = str(item.get("style_family", "")).strip()
        archetype_id = str(item.get("archetype_id", "")).strip()
        observation = str(item.get("observation", "")).strip()
        key_failure_tags = item.get("key_failure_tags") or item.get("failure_tags") or []
        summary = {
            "style_family": style_family,
            "archetype_id": archetype_id,
            "observation": observation,
            "key_failure_tags": key_failure_tags[:4] if isinstance(key_failure_tags, list) else [],
        }
        if style_family or archetype_id or observation:
            lines.append(json.dumps(summary, ensure_ascii=False))
    return "\n".join(lines)


def _compact_prompt_compilation_recipes(memory_context: dict) -> str:
    recipes = [item for item in (memory_context.get("prompt_compilation_recipes") or []) if isinstance(item, dict)]
    lines: list[str] = []
    for item in recipes[:3]:
        raw_when_prompt_contains = item.get("when_prompt_contains", [])
        if isinstance(raw_when_prompt_contains, list):
            when_prompt_contains = raw_when_prompt_contains[:6]
        elif isinstance(raw_when_prompt_contains, str) and raw_when_prompt_contains.strip():
            when_prompt_contains = [raw_when_prompt_contains.strip()]
        else:
            when_prompt_contains = []
        summary = {
            "recipe_id": str(item.get("recipe_id", "")).strip(),
            "archetype_id": str(item.get("archetype_id", "")).strip(),
            "when_prompt_contains": when_prompt_contains,
            "patches": item.get("patches", [])[:6] if isinstance(item.get("patches"), list) else [],
            "notes": str(item.get("notes", "")).strip(),
        }
        if summary["recipe_id"] or summary["archetype_id"] or summary["patches"]:
            lines.append(json.dumps(summary, ensure_ascii=False))
    return "\n".join(lines)


def _compact_custom_skill_records(memory_context: dict) -> str:
    records = [item for item in (memory_context.get("custom_skill_records") or []) if isinstance(item, dict)]
    lines: list[str] = []
    for item in records[:5]:
        skill_payload = item.get("skill_payload", {}) if isinstance(item.get("skill_payload", {}), dict) else {}
        summary = {
            "skill_id": str(item.get("skill_id", "")).strip(),
            "style_family": str(item.get("style_family", "")).strip(),
            "source": str(item.get("source", "")).strip(),
            "archetype_id": str(item.get("archetype_id", "")).strip(),
            "accepted": bool(item.get("accepted", False)),
            "overall_score": item.get("overall_score", 0.0),
            "title": str(skill_payload.get("title", "")).strip(),
            "description": str(skill_payload.get("description", "")).strip(),
            "environment": str(skill_payload.get("environment", "")).strip(),
            "prop": str(skill_payload.get("prop", "")).strip(),
            "lighting": str(skill_payload.get("lighting", "")).strip(),
            "costume": str(skill_payload.get("costume", "")).strip(),
            "subject_surface": str(skill_payload.get("subject_surface", "")).strip(),
            "subject_integration": str(skill_payload.get("subject_integration", "")).strip(),
            "atmosphere": str(skill_payload.get("atmosphere", "")).strip(),
            "material": str(skill_payload.get("material", "")).strip(),
        }
        if summary["skill_id"] or summary["title"]:
            lines.append(json.dumps(summary, ensure_ascii=False))
    return "\n".join(lines)


def _compact_attempt_adaptation_state(memory_context: dict) -> str:
    state = memory_context.get("attempt_adaptation_state") or {}
    if not isinstance(state, dict):
        return ""
    payload = {
        "last_prompt_patch": [str(item).strip() for item in (state.get("last_prompt_patch") or []) if str(item).strip()][:4],
        "last_selected_strategy": str(state.get("last_selected_strategy", "")).strip(),
        "last_good_carriers": [str(item).strip() for item in (state.get("last_good_carriers") or []) if str(item).strip()][:4],
        "last_bad_carriers": [str(item).strip() for item in (state.get("last_bad_carriers") or []) if str(item).strip()][:4],
        "last_preferred_candidate_strategies": [str(item).strip() for item in (state.get("last_preferred_candidate_strategies") or []) if str(item).strip()][:4],
        "last_disfavored_candidate_strategies": [str(item).strip() for item in (state.get("last_disfavored_candidate_strategies") or []) if str(item).strip()][:4],
        "last_failure_hints": [str(item).strip() for item in (state.get("last_failure_hints") or []) if str(item).strip()][:6],
        "last_rewrite_targets": [str(item).strip() for item in (state.get("last_rewrite_targets") or []) if str(item).strip()][:6],
        "adaptive_bias": state.get("adaptive_bias", {}) if isinstance(state.get("adaptive_bias", {}), dict) else {},
        "last_materialized_skill_id": str((state.get("last_materialized_skill") or {}).get("strategy_id", "")).strip(),
        "last_accepted_custom_skill_ids": [
            str(item.get("strategy_id", "")).strip()
            for item in (state.get("last_accepted_custom_skills") or [])
            if isinstance(item, dict) and str(item.get("strategy_id", "")).strip()
        ][:3],
    }
    if not any(payload.values()):
        return ""
    return json.dumps(payload, ensure_ascii=False)


def _compact_runtime_skill_records(memory_context: dict) -> str:
    records = [item for item in (memory_context.get("runtime_skill_records") or []) if isinstance(item, dict)]
    lines: list[str] = []
    for item in records[:4]:
        lineage = item.get("_lineage", {}) if isinstance(item.get("_lineage", {}), dict) else {}
        summary = {
            "strategy_id": str(item.get("strategy_id", "")).strip(),
            "style_family": str(item.get("style_family", "")).strip(),
            "title": str(item.get("title", "")).strip(),
            "description": str(item.get("description", "")).strip(),
            "environment": str(item.get("environment", "")).strip(),
            "prop": str(item.get("prop", "")).strip(),
            "lighting": str(item.get("lighting", "")).strip(),
            "recommended_carriers": item.get("recommended_carriers", [])[:4] if isinstance(item.get("recommended_carriers"), list) else [],
            "runtime_source": str(item.get("_runtime_source", "")).strip(),
            "runtime_score": item.get("_runtime_score", 0.0),
            "retrieval_reason": str(item.get("_retrieval_reason", "")).strip(),
            "lineage_origin": str(lineage.get("origin", "")).strip(),
            "parent_skill_ids": lineage.get("parent_skill_ids", [])[:3] if isinstance(lineage.get("parent_skill_ids"), list) else [],
            "generation": lineage.get("generation", 0),
        }
        if summary["strategy_id"]:
            lines.append(json.dumps(summary, ensure_ascii=False))
    return "\n".join(lines)


def _compact_execution_adaptive_bias(memory_context: dict) -> str:
    bias = memory_context.get("execution_adaptive_bias") or {}
    if not isinstance(bias, dict):
        return ""
    payload = {
        "preferred_carriers": [str(item).strip() for item in (bias.get("preferred_carriers") or []) if str(item).strip()][:4],
        "forbidden_carriers": [str(item).strip() for item in (bias.get("forbidden_carriers") or []) if str(item).strip()][:4],
        "prefer_environment_first": bool(bias.get("prefer_environment_first", False)),
        "avoid_face_heavy_rewrite": bool(bias.get("avoid_face_heavy_rewrite", False)),
        "boosted_skill_ids": [str(item).strip() for item in (bias.get("boosted_skill_ids") or []) if str(item).strip()][:4],
        "penalized_skill_ids": [str(item).strip() for item in (bias.get("penalized_skill_ids") or []) if str(item).strip()][:4],
    }
    if not any(payload.values()):
        return ""
    return json.dumps(payload, ensure_ascii=False)


def _compact_scene_archetype(video_analysis: dict) -> str:
    archetype = video_analysis.get("scene_archetype") or {}
    if not isinstance(archetype, dict):
        return ""
    archetype_id = str(archetype.get("archetype_id", "")).strip()
    if not archetype_id:
        return ""
    payload = {
        "archetype_id": archetype_id,
        "scene_hint": str(archetype.get("scene_hint", "")).strip(),
        "subject_count": int(archetype.get("subject_count", 0) or 0),
        "world_anchor_count": int(archetype.get("world_anchor_count", 0) or 0),
        "prop_count": int(archetype.get("prop_count", 0) or 0),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_family_selection_system_prompt() -> str:
    return (
        "你是一位特效大师，并擅长用户需求分析。根据用户的 prompt，分析他们到底想要哪一种特效风格。\n"
        "如果这种特效风格在给定的 style family 里面，直接给出对应风格；如果没有，就创建新的 style family；如果用户信息中没有明显风格需求，就在已有 style family 中随机选择一种。\n"
        "只根据 user_prompt 和 style family 列表做判断。\n"
        "禁止输出思维过程、分析过程、解释、示例、前言、结语、markdown、代码块、<think> 标签或任何额外文本。\n"
        "你只能输出最终结果，而且必须严格限制为 5 行，每行一个字段，字段顺序不能变。\n"
        "STYLE_FAMILY_CONFIDENCE 必须输出 0.00 到 1.00 的小数。\n"
        "IS_PROMPT_VAGUE 必须输出 true 或 false。\n"
    )


def build_family_selection_user_prompt(*, sample_id: str, prompt: str, family_options: list[dict]) -> str:
    return (
        f"sample_id: {sample_id}\n"
        f"user_prompt: {prompt}\n"
        f"style_family_options: {json.dumps(family_options, ensure_ascii=False)}\n\n"
        "请严格只输出以下 5 行，不能多一行，不能少一行，不能输出任何额外内容：\n"
        "用户需求分析: <一句话>\n"
        "SELECTED_STYLE_FAMILY: <style family 名称>\n"
        "STYLE_FAMILY_CONFIDENCE: <0.00-1.00 的数字>\n"
        "STYLE_FAMILY_REASON: <一句话理由>\n"
        "IS_PROMPT_VAGUE: <true 或 false>\n"
    )


def build_video_analysis_system_prompt() -> str:
    return (
        "你做第二阶段：视频分析 + skill 初筛。\n"
        "只输出最终结构，不要输出分析过程，不要输出 <think>，不要在 VIDEO_BRIEF: 之前写任何内容。\n"
        "目标：从已有 skill 中保留可复用项；如果 skill 不足或不合适，就在当前 style family 下补 1-3 个 custom skills。\n"
        "MATCHED_SKILL_CANDIDATES 与 CUSTOM_SKILL_CANDIDATES 可以同时存在。\n"
        "CUSTOM_SKILL_CANDIDATE 的 intensity 只能是 low/normal/high；world_rewrite_cost 只能是 low/medium/high；identity_risk 只能是 low/medium/high。\n"
    )


def build_video_grounding_system_prompt() -> str:
    return (
        "你做第二阶段A：视频 grounding。\n"
        "只提取视频中真实可见的事实，不做 skill 选择，不做 custom skill 设计，不复述任务。\n"
        "重点是把人物、道具、背景锚点和潜在特效载体提取干净，供下一步 skill 选择直接使用。\n"
        "不要输出任何创作约束、保守建议、'必须尊重/只能局部修改' 之类的 planning 指令。\n"
        "直接从 VIDEO_BRIEF: 开始输出，不要输出分析过程、<think>、说明、前言或任何额外文字。\n"
    )


def build_video_grounding_user_prompt(*, sample_id: str, prompt: str) -> str:
    return (
        f"sample_id: {sample_id}\n"
        f"user_prompt: {prompt}\n\n"
        "要求：\n"
        "- 只描述视频中真实可见的内容，不要把用户想要的特效写成已发生事实。\n"
        "- VIDEO_BRIEF 详细描述镜头内容和场景状态。\n"
        "- VISIBLE_PEOPLE 只写当前能稳定看到的人物及其最关键外观信息。\n"
        "- VISIBLE_PROPS 只写明确可见、后续可能被引用的具体物体。\n"
        "- VISIBLE_WORLD_ANCHORS 只写稳定存在的环境锚点。\n"
        "- PRIMARY_FX_CARRIERS 写最可能承载特效的真实可见载体；如果没有明确载体，写 none。\n"
        "- GROUNDING_NOTES 只写可见事实层面的补充或歧义说明；不要写创作边界、不要写保守约束；如果没有补充，写 no。\n"
        "输出字段：\n"
        "VIDEO_BRIEF:\n"
        "VISIBLE_PEOPLE:\n"
        "VISIBLE_PROPS:\n"
        "VISIBLE_WORLD_ANCHORS:\n"
        "PRIMARY_FX_CARRIERS:\n"
        "GROUNDING_NOTES:\n"
    )


def build_skill_selection_system_prompt() -> str:
    return (
        "你做第二阶段B：skill 选择与补造。\n"
        "输入是 user_prompt、selected_style_family、视频 grounding 结果、skill_briefs，以及可选的 memory 先验。\n"
        "你不再重新分析视频帧，只根据 grounding 结果做 skill 决策。\n"
        "只输出最终结构，不要输出分析过程，不要输出 <think>，不要在 MATCHED_SKILL_CANDIDATES: 之前写任何内容。\n"
        "先保留真正可复用的已有 skill；如果已有 skill 数量不足、覆盖面不足，或没有一个足够适合当前视频，就在当前 style family 下补 1-3 个 custom skills。\n"
        "如果 user_prompt 本身包含明确的风格偏好、艺术家参考、电影美术方向、世界观关键词或强审美指令，而现有 skill 库里没有对应模板，可以直接以这些用户偏好为核心创造 custom skill。\n"
        "skill 选择要看五个维度：convertibility、anchor reuse、rewrite cost、subject compatibility、effect leverage。\n"
        "MATCHED_SKILL_CANDIDATES 与 CUSTOM_SKILL_CANDIDATES 可以同时存在，不是互斥关系。\n"
        "custom skill 必须首先是可复用的风格模板或世界模板，而不是只针对当前视频某个单独道具的临时特效补丁。\n"
        "custom skill 要有清晰的环境逻辑、材质逻辑、光线逻辑、主体融合逻辑，并且应当可以迁移到相近视频；当前视频上的具体落点适配留给下一阶段 candidate planning。\n"
        "字段名必须与 skill template 一致。\n"
        "CUSTOM_SKILL_CANDIDATE 的 intensity 只能是 low/normal/high；world_rewrite_cost 只能是 low/medium/high；identity_risk 只能是 low/medium/high。\n"
    )


def build_skill_selection_user_prompt(
    *,
    sample_id: str,
    prompt: str,
    selected_style_family: str,
    video_grounding: dict,
    skill_briefs: list[dict],
    memory_context: dict | None = None,
) -> str:
    memory_context = memory_context or {}
    retry_feedback = _compact_retry_feedback(memory_context, include_prompt_patch=False)
    style_priors = _compact_style_convertibility_priors(memory_context)
    custom_skill_records = _compact_custom_skill_records(memory_context)
    attempt_adaptation = _compact_attempt_adaptation_state(memory_context)
    runtime_skill_records = _compact_runtime_skill_records(memory_context)
    execution_adaptive_bias = _compact_execution_adaptive_bias(memory_context)
    parts = [
        f"sample_id: {sample_id}\n",
        f"user_prompt: {prompt}\n",
        f"selected_style_family: {selected_style_family}\n",
        f"video_grounding: {json.dumps(video_grounding, ensure_ascii=False)}\n",
        f"skill_briefs: {json.dumps(skill_briefs, ensure_ascii=False)}\n\n",
    ]
    if retry_feedback:
        parts.append(f"retry_feedback:\n{retry_feedback}\n\n")
    if style_priors:
        parts.append(f"style_convertibility_priors:\n{style_priors}\n\n")
    if custom_skill_records:
        parts.append(f"retrieved_custom_skill_records:\n{custom_skill_records}\n\n")
    if attempt_adaptation:
        parts.append(f"attempt_adaptation_state:\n{attempt_adaptation}\n\n")
    if runtime_skill_records:
        parts.append(f"runtime_skill_records:\n{runtime_skill_records}\n\n")
    if execution_adaptive_bias:
        parts.append(f"execution_adaptive_bias:\n{execution_adaptive_bias}\n\n")
    parts.extend(
        [
            "规则：\n",
            "- 只输出下面的字段和块。\n",
            "- 不要写分析过程。\n",
            "- 输出必须从 MATCHED_SKILL_CANDIDATES: 开始。\n",
            "- 先基于 video_grounding 判断哪些已有 skill 真正可复用，再写入 MATCHED_SKILL_CANDIDATES。\n",
            "- 判断 skill 时重点看五个维度：convertibility、anchor reuse、rewrite cost、subject compatibility、effect leverage。\n",
            "- 如果已有 skill 不足、覆盖面不足，或者没有一个足够适合当前视频，补 1-3 个 CUSTOM_SKILL_CANDIDATE。\n",
            "- 如果已有 skill 只能部分匹配，也可以保留在 MATCHED_SKILL_CANDIDATES，同时再补 custom skill。\n",
            "- 如果 user_prompt 里已经有明确风格偏好、艺术家名字、电影/动画参考、材质审美或世界观关键词，而 skill_briefs 里没有对应模板，优先从这些用户偏好出发创造 custom skill。\n",
            "- 用户 prompt 提供的风格偏好可以直接成为 CUSTOM_SKILL_SEED 的核心来源，而不必退化成围绕当前单个道具的局部补丁。\n",
            "- custom skill 不是当前视频的局部 FX patch，不要围绕单个道具临时发明一次性特效方案。\n",
            "- custom skill 首先要像 master skill：是一个可复用的风格模板或世界模板，具有清晰的 environment / material / lighting / subject_integration 逻辑。\n",
            "- custom skill 可以是强风格名，但不能只有一个空洞标签；必须把世界、材质、光线、主体融合方式写具体。\n",
            "- 当前视频上该把风格落在哪个载体、如何局部适配，留给下一阶段 candidate planning；不要在这里把 custom skill 写成'给麦克风加光环'这类样本级补丁。\n",
            "- 优先参考 retry_feedback、style_convertibility_priors、retrieved_custom_skill_records，但不要复述它们。\n",
            "- 如果给了 attempt_adaptation_state，优先复用上一轮有效 carrier 和接近成功的 strategy skeleton；降低上一轮失败 carrier 的优先级。\n",
            "- 如果给了 runtime_skill_records，这些是上一轮在线检索补进来的 skill。优先复用 retrieval_reason 更强、runtime_score 更高、carrier 更贴合当前 grounding 的项。\n",
            "- 如果给了 execution_adaptive_bias，直接把 preferred_carriers / boosted_skill_ids 升权，把 forbidden_carriers / penalized_skill_ids 降权。\n",
            "- REJECTED_SKILL_CANDIDATES 只写明确不适合当前视频的 skill。\n",
            "- NEEDS_CUSTOM_SKILL 只写 true 或 false。\n",
            "- CUSTOM_SKILL_SEED 用一句话概括本轮 custom skill 的风格方向；如果不需要 custom skill，写 no。\n",
            "- strategy_id 要简短稳定，尽量用英文 snake_case。\n",
            "- tags、anchor_requirements、best_for_archetypes、high_risk_archetypes、forbidden_mixes、recommended_carriers 用逗号分隔的短列表。\n",
            "- intensity 只能填 low / normal / high。\n",
            "- world_rewrite_cost 只能填 low / medium / high。\n",
            "- identity_risk 只能填 low / medium / high。\n",
            "- 如果没有 custom skill，CUSTOM_SKILL_CANDIDATES 必须明确写 no，不要留空。\n",
            "输出格式：\n",
            "MATCHED_SKILL_CANDIDATES:\n",
            "REJECTED_SKILL_CANDIDATES:\n",
            "NEEDS_CUSTOM_SKILL:\n",
            "CUSTOM_SKILL_SEED:\n",
            "CUSTOM_SKILL_CANDIDATES: no\n",
            "[CUSTOM_SKILL_CANDIDATE]\n",
            "strategy_id:\nstyle_family:\ntitle:\ndescription:\nenvironment:\nprop:\nlighting:\ncostume:\nsubject_surface:\nsubject_integration:\natmosphere:\nmaterial:\ntags:\nintensity:\npros:\ncons:\nforbidden_mixes:\nanchor_requirements:\nbest_for_archetypes:\nhigh_risk_archetypes:\nworld_rewrite_cost:\nidentity_risk:\nprompt_compilation_notes:\ncommon_failure_modes:\nrecommended_carriers:\n",
            "[/CUSTOM_SKILL_CANDIDATE]\n",
            "如果创建了多个 custom skill，就先写 CUSTOM_SKILL_CANDIDATES:，然后重复多个 [CUSTOM_SKILL_CANDIDATE]。\n",
        ]
    )
    return "".join(parts)


def build_video_analysis_user_prompt(
    *,
    sample_id: str,
    prompt: str,
    selected_style_family: str,
    skill_briefs: list[dict],
    memory_context: dict | None = None,
) -> str:
    memory_context = memory_context or {}
    retry_feedback = _compact_retry_feedback(memory_context, include_prompt_patch=False)
    style_priors = _compact_style_convertibility_priors(memory_context)
    custom_skill_records = _compact_custom_skill_records(memory_context)
    parts = [
        f"sample_id: {sample_id}\n",
        f"user_prompt: {prompt}\n",
        f"selected_style_family: {selected_style_family}\n",
        f"skill_briefs: {json.dumps(skill_briefs, ensure_ascii=False)}\n\n",
    ]
    if retry_feedback:
        parts.append(f"retry_feedback:\n{retry_feedback}\n\n")
    if style_priors:
        parts.append(f"style_convertibility_priors:\n{style_priors}\n\n")
    if custom_skill_records:
        parts.append(f"retrieved_custom_skill_records:\n{custom_skill_records}\n\n")
    parts.extend(
        [
            "规则：\n",
            "- 只输出下面的字段和块。\n",
            "- 不要写分析过程。\n",
            "- 输出必须从 VIDEO_BRIEF: 开始。\n",
            "- 如果已有 skill 可复用，保留到 MATCHED_SKILL_CANDIDATES。\n",
            "- 如果已有 skill 不足或不够适合，补 1-3 个 CUSTOM_SKILL_CANDIDATE。\n",
            "- custom skill 要能直接给下一阶段 candidate planning 使用。\n",
            "- 优先参考 retry_feedback、style_convertibility_priors、retrieved_custom_skill_records，但不要复述它们。\n",
            "- intensity 只能填 low / normal / high。\n",
            "- world_rewrite_cost 只能填 low / medium / high。\n",
            "- identity_risk 只能填 low / medium / high。\n",
            "输出格式：\n",
            "VIDEO_BRIEF:\n",
            "VISIBLE_PEOPLE:\n",
            "VISIBLE_PROPS:\n",
            "VISIBLE_WORLD_ANCHORS:\n",
            "PRIMARY_FX_CARRIERS:\n",
            "GROUNDING_NOTES:\n",
            "MATCHED_SKILL_CANDIDATES:\n",
            "REJECTED_SKILL_CANDIDATES:\n",
            "NEEDS_CUSTOM_SKILL:\n",
            "CUSTOM_SKILL_SEED:\n",
            "CUSTOM_SKILL_CANDIDATES:\n",
            "[CUSTOM_SKILL_CANDIDATE]\n",
            "strategy_id:\nstyle_family:\ntitle:\ndescription:\nenvironment:\nprop:\nlighting:\ncostume:\nsubject_surface:\nsubject_integration:\natmosphere:\nmaterial:\ntags:\nintensity:\npros:\ncons:\nforbidden_mixes:\nanchor_requirements:\nbest_for_archetypes:\nhigh_risk_archetypes:\nworld_rewrite_cost:\nidentity_risk:\nprompt_compilation_notes:\ncommon_failure_modes:\nrecommended_carriers:\n",
            "[/CUSTOM_SKILL_CANDIDATE]\n",
            "如果创建了多个 custom skill，就重复多个 [CUSTOM_SKILL_CANDIDATE]。\n",
        ]
    )
    return "".join(parts)


def build_candidate_planning_system_prompt() -> str:
    return (
        "你是视频特效规划器的第三阶段：候选方案设计器。\n"
        "你的任务是生成 2 到 3 个候选方案，不是输出分析过程，不是输出打分表，也不是长篇比较报告。\n"
        "请严格按块标签输出，不要输出 JSON，不要输出 <think>，不要在第一个 [CANDIDATE_STRATEGY] 之前写任何内容。\n"
        "不要只复述 skill 原文，必须结合当前视频里的具体人物、道具、环境来写。\n"
        "如果只有 1 个强匹配 skill，不要硬凑弱模板；保留那个强匹配 skill，并额外补 1 到 2 个同风格家族、明显更贴合当前视频的 custom skill 方案。\n"
        "custom skill 必须是为了当前视频更好改造，不是泛泛空想。\n"
    )


def build_candidate_planning_user_prompt(
    *,
    sample_id: str,
    prompt: str,
    video_analysis: dict,
    full_skills: list[dict],
    memory_context: dict | None = None,
) -> str:
    memory_context = memory_context or {}
    custom_skill_records = _compact_custom_skill_records(memory_context)
    attempt_adaptation = _compact_attempt_adaptation_state(memory_context)
    runtime_skill_records = _compact_runtime_skill_records(memory_context)
    execution_adaptive_bias = _compact_execution_adaptive_bias(memory_context)
    scene_archetype = _compact_scene_archetype(video_analysis)
    extra = ""
    if custom_skill_records:
        extra = (
            f"retrieved_custom_skill_records:\n{custom_skill_records}\n\n"
            "这些是历史上验证过的 custom skills；如果其中有与当前 prompt 和视频上下文接近的 skill，可以直接复用、改写，或与已有 skill 组合成更强的候选池。\n\n"
        )
    if attempt_adaptation:
        extra += (
            f"attempt_adaptation_state:\n{attempt_adaptation}\n\n"
            "这是同一个 sample 上一轮 attempt 的在线适应信号。优先复用其中有效的 carrier、strategy skeleton 和 materialized skill；避免再次使用 last_bad_carriers 对应的绑定方式；把 last_preferred_candidate_strategies 视为更稳的候选骨架，把 last_disfavored_candidate_strategies 视为需要回避的方向。\n\n"
        )
    if runtime_skill_records:
        extra += (
            f"runtime_skill_records:\n{runtime_skill_records}\n\n"
            "这些 runtime skills 是上一轮在线检索补进来的高优先级 skill。优先复用 retrieval_reason 更强、runtime_score 更高、并且和当前 grounding 锚点更一致的项。\n\n"
        )
    if scene_archetype:
        extra += (
            f"scene_archetype:\n{scene_archetype}\n\n"
            "候选设计必须显式适配当前 scene_archetype，不要给出与当前镜头类型无关的通用方案。\n\n"
        )
    if execution_adaptive_bias:
        extra += (
            f"execution_adaptive_bias:\n{execution_adaptive_bias}\n\n"
            "候选排序和 carrier 选择必须直接尊重这些 execution bias。\n\n"
        )
    return (
        f"sample_id: {sample_id}\n"
        f"user_prompt: {prompt}\n"
        f"video_analysis: {json.dumps(video_analysis, ensure_ascii=False)}\n"
        f"full_skills: {json.dumps(full_skills, ensure_ascii=False)}\n\n"
        f"{extra}"
        "规则：\n"
        "1. 先保留真正强匹配的已有 skill。\n"
        "2. 如果已有 skill 不足以支撑 2 到 3 个高质量方案，可以结合 retrieved_custom_skill_records，或自己新建 custom skill。\n"
        "3. 不要为了凑数把明显不适合当前视频的弱模板硬塞进来。\n"
        "4. 如果 video_analysis 里有 critic_replan_directives 或 critic_failure_hints，本轮候选必须显式修复这些失败点，不要重新给出本质上相同的方案。\n"
        "5. 每个候选都必须回答清楚：主要改什么、依赖哪些真实锚点、对人物/道具/环境各怎么处理。\n"
        "6. custom skill 也必须写成完整的 [CANDIDATE_STRATEGY]，并给出新的 strategy_id、title、rationale。\n"
        "7. 不要输出分析过程、比较过程或打分过程。\n"
        "8. 如果 scene_archetype 是 single_person_indoor_closeup，候选应优先考虑身份安全、前景设备 carrier、背景光源 carrier，避免把大特效直接压到面部中心。\n"
        "9. 如果 scene_archetype 是 world_anchor_driven_scene，候选应优先做环境锚点驱动的 world-building，再决定人物局部变化。\n\n"
        "10. 当前输入的 full_skills 已经做过 candidate skeleton 预筛选和偏置排序；优先围绕这些高优先级 skeleton 展开，不要重新发散回历史上已经证明不稳的方向。\n\n"
        "请输出以下结构：\n"
        "先输出 2 个或 3 个候选方案。\n"
        "每个候选方案必须按以下顺序输出一组配对块：先输出一个 [CANDIDATE_STRATEGY]，紧接着输出与它一一对应的一个 [PLAN_CANDIDATE]。\n"
        "每组里的 candidate_id 必须唯一，并且 PLAN_CANDIDATE.candidate_id 必须与对应候选一致。\n"
        "不要输出空块，不要输出没有对应 PLAN_CANDIDATE 的候选。\n\n"
        "[CANDIDATE_STRATEGY]\n"
        "strategy_id:\nstyle_family:\ntitle:\nrationale:\nenvironment:\nprop:\nlighting:\ncostume:\nsubject_surface:\nsubject_integration:\natmosphere:\nmaterial:\npros:\ncons:\n"
        "[/CANDIDATE_STRATEGY]\n"
        "[PLAN_CANDIDATE]\n"
        "candidate_id:\nskill_basis:\nenvironment_plan:\nprop_plan:\nlighting_plan:\ncostume_plan:\nsubject_surface_plan:\nintegration_plan:\n"
        "[/PLAN_CANDIDATE]\n"
        "按上面的配对结构继续输出，直到总共得到 2 个或 3 个完整候选方案。\n\n"
        "最后输出：\n"
        "SELECTED_CANDIDATE_ID:\n"
        "SELECTED_STRATEGY:\n"
        "SELECTION_REASON:\n"
        "FOCUS_REGIONS:\n"
    )


def build_candidate_scoring_system_prompt() -> str:
    return (
        "你是视频特效规划器的 3.5 阶段：候选方案比较器。\n"
        "你的任务不是重新设计方案，而是比较已经存在的 candidate plans。\n"
        "不要输出 JSON。请严格按给定标题和块标签输出。\n"
        "请根据 prompt、视频分析、候选方案内容，比较它们的：prompt fit、video anchor fit、identity safety、physical plausibility、effect impact。\n"
        "如果某个候选已经明显更好，就直接给出更高分，不要为了平均而平均。\n"
        "不要输出长篇思考，不要重复候选方案原文，不要编造不存在的新候选，不要在第一个 [CANDIDATE_SCORE] 之前写任何内容。\n"
    )


def build_candidate_scoring_user_prompt(
    *,
    sample_id: str,
    prompt: str,
    video_analysis: dict,
    candidate_planning: dict,
    memory_context: dict | None = None,
) -> str:
    memory_context = memory_context or {}
    scene_archetype = _compact_scene_archetype(video_analysis)
    execution_adaptive_bias = _compact_execution_adaptive_bias(memory_context)
    parts = [
        f"sample_id: {sample_id}\n",
        f"user_prompt: {prompt}\n",
        f"video_analysis: {json.dumps(video_analysis, ensure_ascii=False)}\n",
        f"candidate_planning: {json.dumps(candidate_planning, ensure_ascii=False)}\n\n",
    ]
    if scene_archetype:
        parts.append(f"scene_archetype:\n{scene_archetype}\n\n")
    if execution_adaptive_bias:
        parts.append(f"execution_adaptive_bias:\n{execution_adaptive_bias}\n\n")
    parts.extend(
        [
            "请只做候选比较，不要重新生成 candidate strategy。\n",
            "如果给了 scene_archetype，比较时要考虑该镜头类型下什么样的候选更稳。\n",
            "如果给了 execution_adaptive_bias，比较时要把 preferred_carriers、forbidden_carriers、boosted_skill_ids、penalized_skill_ids 作为历史稳定性偏置。\n",
            "输出格式：\n",
            "[CANDIDATE_SCORE]\n",
            "candidate_id:\n",
            "prompt_fit:\n",
            "video_anchor_fit:\n",
            "identity_safety:\n",
            "physical_plausibility:\n",
            "effect_impact:\n",
            "total:\n",
            "why_selected_or_rejected:\n",
            "[/CANDIDATE_SCORE]\n",
            "对每个 candidate 都输出一次。\n\n",
            "最后输出：\n",
            "SELECTED_CANDIDATE_ID:\n",
            "SELECTED_STRATEGY:\n",
            "SELECTION_REASON:\n",
            "要求：\n",
            "- 分数范围是 0 到 1。\n",
            "- why_selected_or_rejected 必须简短具体。\n",
            "- 如果已有 selected_candidate_id 不合理，可以纠正。\n",
            "- 不要输出分析过程或额外说明。\n",
        ]
    )
    return "".join(parts)


def build_final_plan_system_prompt() -> str:
    return (
        "你是视频特效规划器的第四阶段：最终方案定稿器。\n"
        "你的任务不是复述 strategy，而是把它转成可直接执行的、对象级的、电影制作级的编辑说明。\n"
        "不要输出 JSON。请严格按给定标题和块标签输出，不要写额外解释。\n"
        "不要浪费 token 去写打分、泛泛评价或空洞总结；把篇幅优先留给怎么修改、保留什么、哪些对象可重写、特效绑定到哪个真实载体、不同区域改到什么程度。\n"
        "REGION_EDIT 必须结合视频里的具体人物、道具、环境来写；COMPILED_EDIT_PROMPT 必须像一个可以直接喂给视频编辑模型的最终提示，而不是摘要。\n"
        "禁止使用空泛词替代细节，例如“增强电影感”“科幻风格特效”“适度加强”“整体优化”。\n"
        "所有关键对象都要写具体：颜色、材质、几何形状、发光逻辑、粒子/能量类型、光照方向、服装层级、表面变化、约束边界。\n"
        "如果视频里有多个人，必须分别说明；所有特效都必须绑定到真实可见载体，禁止无来源漂浮大特效。\n"
        "如果输入分析缺失，你也要根据视频帧和已知候选方案主动补足，不要把缺失信息当成停止理由。\n"
        "第四阶段的首要任务不是重新设计方案，而是沿用 candidate_planning 中已选中的 candidate strategy 和 plan candidate，把它们扩写成最终 plan。\n"
        "selected_skill 是主模板，不是背景噪声；但你不能机械照抄 skill 原文，必须把 skill 的世界逻辑和材质逻辑翻译成当前视频真正可执行的改法。\n"
        "最终方案必须同时满足两点：一是优先吸收 selected_skill 的关键字段，二是结合当前视频内容做二次创作和 video-specific adaptation。\n"
        "扩写时必须更紧密地结合当前视频里具体可见的人物、道具、环境、空间层次和运动关系，做大胆但有锚点的设计与想象；不要保守缩回成泛泛描述。\n"
    )


def build_final_plan_user_prompt(
    *,
    sample_id: str,
    prompt: str,
    video_analysis: dict,
    candidate_planning: dict,
    selected_skill: dict,
    memory_context: dict | None = None,
) -> str:
    memory_context = memory_context or {}
    retry_feedback = _compact_retry_feedback(memory_context, include_prompt_patch=True)
    compilation_recipes = _compact_prompt_compilation_recipes(memory_context)
    attempt_adaptation = _compact_attempt_adaptation_state(memory_context)
    scene_archetype = _compact_scene_archetype(video_analysis)
    execution_adaptive_bias = _compact_execution_adaptive_bias(memory_context)
    parts = [
        f"sample_id: {sample_id}\n",
        f"user_prompt: {prompt}\n",
        f"video_analysis: {json.dumps(video_analysis, ensure_ascii=False)}\n",
        f"candidate_planning: {json.dumps(candidate_planning, ensure_ascii=False)}\n",
        f"selected_skill: {json.dumps(selected_skill, ensure_ascii=False)}\n\n",
    ]
    if retry_feedback:
        parts.append(f"retry_feedback:\n{retry_feedback}\n\n")
    if compilation_recipes:
        parts.append(f"prompt_compilation_recipes:\n{compilation_recipes}\n\n")
    if attempt_adaptation:
        parts.append(f"attempt_adaptation_state:\n{attempt_adaptation}\n\n")
    if scene_archetype:
        parts.append(f"scene_archetype:\n{scene_archetype}\n\n")
    if execution_adaptive_bias:
        parts.append(f"execution_adaptive_bias:\n{execution_adaptive_bias}\n\n")
    parts.extend(
        [
            "目标：\n",
            "1. 第四阶段优先扩写 candidate_planning 中已选中的 candidate strategy 和 plan candidate，不要重新发明另一套方案。\n",
            "2. 把 selected candidate 中的 environment / prop / lighting / costume / subject_surface / subject_integration / atmosphere / material 扩写成对象级编辑计划。\n",
            "3. selected_skill 是主模板：优先吸收它的 environment、prop、lighting、costume、subject_surface、subject_integration、atmosphere、material、prompt_compilation_notes。\n",
            "4. 在吸收 selected_skill 的同时，必须结合当前视频内容做自己的判断和改写，形成 video-specific adaptation，而不是生搬硬套 skill 原文。\n",
            "5. 不要写没有执行价值的评分或空泛判断。\n\n",
            "执行原则：\n",
            "1. COMPILED_EDIT_PROMPT 必须覆盖：整体世界观与氛围、核心道具/装置改造、环境改造、人物逐个处理、光源逻辑、特效载体逻辑、保留/重写/轻改边界。\n",
            "2. COMPILED_EDIT_PROMPT 要写成高密度、连续、可直接执行的最终提示，不要写成提纲摘要。\n",
            "2b. COMPILED_EDIT_PROMPT 必须用英文书写，不超过 180 个英文单词。视频编辑模型的文本编码器只读取前 512 个 token，超出部分会被直接截断、完全不起作用；"
            "宁可只写最重要的改动，也不要面面俱到。保留类约束（构图、身份、表情、机位）由系统另行追加，这里不必重复。\n",
            "3. selected candidate 已经给出了本轮方案骨架；environment_plan / prop_plan / lighting_plan / costume_plan / subject_surface_plan / integration_plan 应该成为 REGION_EDIT 和 COMPILED_EDIT_PROMPT 的直接上游，而不是被忽略。\n",
            "4. selected_skill 提供上游风格模板与风险约束，但不应覆盖已经在第三阶段确定的 candidate-specific adaptation，除非 candidate 与 skill 明显冲突。\n",
            "5. selected_skill 里的 environment / lighting / atmosphere / material 优先决定世界观、材质语言、光照方向和空气层；不要另起一套互相冲突的世界设定。\n",
            "6. selected_skill 里的 prop / costume / subject_surface / subject_integration 必须被翻译成当前视频里具体对象的改法，但要根据当前视频实际可见对象做适配，而不是照抄抽象槽位。\n",
            "7. selected_skill 的 prompt_compilation_notes 必须优先吸收进 COMPILED_EDIT_PROMPT 和 REGION_EDIT；不要把它们当可有可无的注释。\n",
            "8. selected_skill 的 common_failure_modes 要反向转成规避约束，写进 SELF_CHECK_NOTES、FORBIDDEN_CHANGES、preserve 约束或 COMPILED_EDIT_PROMPT 的反例规避中。\n",
            "9. selected_skill 的 recommended_carriers 要优先进入 PRIMARY_FX_CARRIERS、SECONDARY_FX_CARRIERS 和 why_this_carrier 的推理；只有当它们与当前视频不匹配时，才可以换成本视频更合理的真实载体。\n",
            "10. 扩写时要紧贴当前视频里的具体元素展开大胆设计：把已有的人物关系、服装轮廓、道具几何、背景层次、雾气/灯光/山体/家具等真实元素推向更强的电影化形态，而不是只做轻微润色。\n",
            "11. 这种大胆设计必须有锚点，不能凭空发散；每个新增结构、材质、能量效果都要明确附着在哪个真实对象或空间层次上。\n",
            "12. 如果 selected_skill 偏抽象，必须把它具体化，例如颜色、材质、几何、发光、服装轮廓、人物角色定位。\n",
            "13. 不要写“某某人变得更如何”，要写成具体服装、材质、结构、光效与外观描述。\n",
            "14. HARD_PRESERVE_CONSTRAINTS 和 SOFT_PRESERVE_CONSTRAINTS 要明确区分绝对不能动与只能轻改的内容。\n",
            "15. REWRITE_TARGETS 和 LIGHT_TOUCH_TARGETS 要明确区分大改区域与轻改区域，不要混写。\n",
            "16. 每个 [REGION_EDIT] 都必须回答清楚：改哪里、改什么、保留什么、为什么选这个载体、强度多大；其中 template_basis 说明继承了 selected candidate / selected_skill 的哪部分，video_specific_adaptation 说明你针对当前视频新增了什么想法、强化了哪些元素、做了哪些落地约束。\n",
            "17. 如果给了 retry_feedback，这些就是上一轮失败反馈；这轮必须显式修复，不能重复上一轮 compiled_edit_prompt。\n",
            "18. 如果给了 prompt_compilation_recipes，请把它们当成 actor-prompt 编译经验，优先吸收主视觉中心、材质/几何显式化、服装轮廓变化、FX 载体绑定、避免 generic glow 等经验。\n\n",
            "19. 如果给了 attempt_adaptation_state，优先继承上一轮有效的 carrier 绑定方式、rewrite 重点和 materialized skill 骨架；显式避开上一轮 bad carriers 和 failure hints 对应的失败模式；对 last_preferred_candidate_strategies 对应的方案骨架更积极复用，对 last_disfavored_candidate_strategies 对应的方向保持保守。\n",
            "20. 如果给了 scene_archetype，最终方案必须显式针对这个镜头类型写 preserve/rewrite/carrier 规则，不要套用别的场景的改法。\n",
            "21. 当 archetype 是 single_person_indoor_closeup 时，优先保护人物身份、面部与主体运动，把强特效绑定在前景设备或背景光源上，不要让大特效漂在脸前。\n",
            "22. 当 archetype 是 world_anchor_driven_scene 时，优先围绕环境锚点和空间层次做 world-building，再决定人物局部改造。\n\n",
            "23. 如果给了 execution_adaptive_bias，直接继承 preferred_carriers，避开 forbidden_carriers；对 boosted_skill_ids 对应的结构和 carrier 用法更积极复用，对 penalized_skill_ids 对应做法保持保守。\n\n",
            "特效编辑维度检查清单：\n",
            "1. 人物层：\n",
            "- 皮肤：基础色调、表面纹理、次表面散射、异化或机械化细节。\n",
            "- 服装：材质、结构轮廓、装饰、发光带、层级关系、局部动态。\n",
            "- 化妆与面部：眼妆、唇部、轮廓修饰、贴花/纹身、面部局部装饰。\n",
            "- 表情与眼部：哪些表情必须继承原视频，哪些只做轻微增强；虹膜纹理、瞳孔形状、反光点、巩膜色调如何处理。\n",
            "2. 环境层：\n",
            "- 光照：主光/辅光/背光方向、色温、强度、阴影硬度、体积光。\n",
            "- 氛围：雾气、烟尘、粒子、色彩基调、空气密度。\n",
            "- 装饰与背景：背景替换、远景元素、装饰物、空间层次。\n",
            "- 故事感：环境线索、时间感、情绪氛围、叙事暗示。\n",
            "3. 道具层：\n",
            "- 几何结构：只改表面还是改体积/轮廓；新增部件、刻线、凹槽、球体、装甲、外扩轮廓。\n",
            "- 材质与光学：基础材质、金属度/粗糙度、自发光、反射/折射、环境反光来源。\n",
            "- 动态与交互：如何随人物运动，是否有次级运动，是否与手部/身体/地面发生反馈。\n",
            "4. 动物层（如果有）：\n",
            "- 形态结构、皮肤/毛发/鳞片、装扮与植入物、行为与眼神。\n",
            "5. 特效层：\n",
            "- 粒子：发射源、数量、大小、颜色、寿命、速度、重力。\n",
            "- 能量与光效：电弧、能量场、护盾、光束、全息投影、机械展开。\n",
            "- 合成融合：遮挡关系、阴影投射、反射/折射、前中后景层级、与环境色温匹配。\n",
            "6. 约束与锚点层：\n",
            "- 绝对不可变区域、软约束区域、运动锁定、身份保留区、载体绑定。\n",
            "- 每个特效都必须绑定到真实可见的人物/道具/环境锚点，禁止无来源漂浮特效。\n",
            "7. 风格化层：\n",
            "- 边缘轮廓、色调映射、纹理风格、动态模糊/拖尾、重复纹理周期。\n\n",
            "展开规则：\n",
            "1. 对视频里实际存在的人物、道具、环境、动物分别展开；没有的层可以跳过，但不要凭空编造对象。\n",
            "2. 每个重要对象至少回答：怎么改、保留什么、材质/结构/光效怎么落地、特效绑定在哪个真实载体上。\n",
            "3. 上述维度必须真的吸收到 REGION_EDIT 与 COMPILED_EDIT_PROMPT，不要只停留在抽象形容词。\n\n",
            "请按以下格式输出：\n",
            "STYLE_FAMILY:\n",
            "SELECTED_STRATEGY:\n",
            "VISIBLE_PEOPLE:\n",
            "VISIBLE_PROPS:\n",
            "VISIBLE_WORLD_ANCHORS:\n",
            "CARRIER_CANDIDATES:\n",
            "IMMUTABLE_OBJECTS:\n",
            "MUTABLE_REGIONS:\n",
            "GROUNDING_NOTES:\n",
            "HARD_PRESERVE_CONSTRAINTS:\n",
            "SOFT_PRESERVE_CONSTRAINTS:\n",
            "REWRITE_TARGETS:\n",
            "LIGHT_TOUCH_TARGETS:\n",
            "FORBIDDEN_CHANGES:\n",
            "PRIMARY_FX_CARRIERS:\n",
            "SECONDARY_FX_CARRIERS:\n",
            "NORMALIZED_PROMPT:\n",
            "[REGION_EDIT]\n",
            "edit_id:\nregion:\ntarget_objects:\naction:\nnew_description:\ntemplate_basis:\nvideo_specific_adaptation:\npreserve:\nwhy_this_carrier:\nstrength:\n",
            "[/REGION_EDIT]\n",
            "重复至少 4 次。\n",
            "SELF_CHECK_NOTES:\n",
            "CONFLICT_RESOLUTIONS:\n",
            "COMPILED_EDIT_PROMPT:\n",
        ]
    )
    return "".join(parts)
