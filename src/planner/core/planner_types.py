from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EditAction = Literal["replace", "redesign", "enhance", "integrate", "light_touch"]


class StrategyCandidate(BaseModel):
    strategy_id: str
    style_family: str
    title: str
    rationale: str
    environment: str
    prop: str
    lighting: str
    costume: str
    subject_surface: str
    subject_integration: str
    atmosphere: str
    material: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class GroundingEvidence(BaseModel):
    visible_people: list[str] = Field(default_factory=list)
    visible_props: list[str] = Field(default_factory=list)
    visible_world_anchors: list[str] = Field(default_factory=list)
    carrier_candidates: list[str] = Field(default_factory=list)
    immutable_objects: list[str] = Field(default_factory=list)
    mutable_regions: list[str] = Field(default_factory=list)


class SelectionScore(BaseModel):
    strategy_id: str
    prompt_fit: float
    anchor_leverage: float
    physical_plausibility: float
    identity_safety: float
    total: float


class PlanCandidateTrace(BaseModel):
    candidate_id: str
    skill_basis: list[str] = Field(default_factory=list)
    environment_plan: str = ""
    prop_plan: str = ""
    lighting_plan: str = ""
    costume_plan: str = ""
    subject_surface_plan: str = ""
    integration_plan: str = ""


class CandidateComparisonTrace(BaseModel):
    candidate_id: str
    prompt_fit: float = 0.0
    video_anchor_fit: float = 0.0
    identity_safety: float = 0.0
    physical_plausibility: float = 0.0
    effect_impact: float = 0.0
    total: float = 0.0
    why_selected_or_rejected: str = ""


class RegionEdit(BaseModel):
    edit_id: str
    region: str
    target_objects: list[str] = Field(default_factory=list)
    action: EditAction
    new_description: str
    template_basis: str = ""
    video_specific_adaptation: str = ""
    preserve: list[str] = Field(default_factory=list)
    why_this_carrier: str
    strength: float = 0.8


class VideoPlan(BaseModel):
    sample_id: str
    prompt: str
    scene_archetype: dict = Field(default_factory=dict)
    style_family: str
    style_family_confidence: float
    user_intent_summary: str = ""
    style_family_candidates: list[str] = Field(default_factory=list)
    selected_style_family: str = ""
    style_family_reason: str = ""
    matched_skill_candidates: list[str] = Field(default_factory=list)
    rejected_skill_candidates: list[str] = Field(default_factory=list)
    needs_custom_skill: bool = False
    custom_skill_seed: str = ""
    grounding_notes: list[str] = Field(default_factory=list)
    plan_candidates: list[PlanCandidateTrace] = Field(default_factory=list)
    candidate_score_details: list[CandidateComparisonTrace] = Field(default_factory=list)
    selected_candidate_id: str = ""
    selection_reason: str = ""
    candidate_strategies: list[StrategyCandidate]
    selected_strategy: str
    materialized_selected_skill: dict = Field(default_factory=dict)
    selection_scores: list[SelectionScore] = Field(default_factory=list)
    grounding_evidence: GroundingEvidence
    hard_preserve_constraints: list[str] = Field(default_factory=list)
    soft_preserve_constraints: list[str] = Field(default_factory=list)
    rewrite_targets: list[str] = Field(default_factory=list)
    light_touch_targets: list[str] = Field(default_factory=list)
    forbidden_changes: list[str] = Field(default_factory=list)
    # {dimension: "keep"|"edit"} -- which edit dimensions the instruction named
    # and what it asked for. Consumed downstream by the critic, which checks each
    # "keep" dimension individually instead of inferring holds from prompt prose.
    dimension_overrides: dict[str, str] = Field(default_factory=dict)
    primary_fx_carriers: list[str] = Field(default_factory=list)
    secondary_fx_carriers: list[str] = Field(default_factory=list)
    normalized_prompt: str
    # The planner's own plan text, before _compose_edit_prompt adds the intent and
    # skill blocks. compiled_edit_prompt is the composed result; raw_vlm_response is
    # truncated JSON and cannot be parsed back reliably.
    planner_plan: str = ""
    compiled_edit_prompt: str = ""
    focus_regions: list[str] = Field(default_factory=list)
    region_edits: list[RegionEdit] = Field(default_factory=list)
    self_check_notes: list[str] = Field(default_factory=list)
    conflict_resolutions: list[str] = Field(default_factory=list)
    raw_vlm_response: str = ""
