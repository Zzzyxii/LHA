"""Candidate edit axes for the low->high task, as CLIP antonym probes.

Why antonym probes. bench_v4's headline (`gain_dino`) asked whether the output is
closer to the target clip than the source is. Two measurements killed it:

* every arm scored *negative* (direct_intent -0.036, grounded_v4 -0.026) against an
  oracle of +0.2136, i.e. all editing moved away from the target;
* on the axes that were attainable (instruction adherence, zero-shot production
  value) the single-pass baseline already sat *at* the oracle, so those could not
  separate two arms either.

A single scalar cannot be both attainable and discriminative here, so the axes are
explicit: one signed number per edit dimension, measured as a CLIP margin between
two contradictory captions. The edit is then a *vector* in that space and the
question becomes "did it push the right knobs", which is answerable even when the
exact target pixels are not.

Every axis must survive validate_axes.py before it is used: the low->high shift
(margin(target) - margin(source)) has to be consistent across the 50 pairs. Axes
that failed are recorded in REJECTED_AXES with the reason, so a rejection is part
of the record instead of a silent omission.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """One signed edit axis, measured as a CLIP antonym margin."""

    name: str
    high: str
    low: str
    # Canonical skill dimension from src/skills/dimensions.py that this tracks.
    dimension: str


AXES: tuple[Axis, ...] = (
    Axis("production_value", "a frame from an expensive big-budget film production",
         "a frame from a cheap low-budget amateur video", "global"),
    Axis("costume_ornament", "a person wearing an ornate, richly embroidered, elaborate costume",
         "a person wearing plain simple everyday clothing", "costume"),
    Axis("cinematic_light", "a shot with dramatic cinematic lighting, a directional key light and deep shadows",
         "a shot with flat evenly lit ambient lighting", "lighting"),
    Axis("color_grade", "a film still with rich cinematic color grading",
         "a raw ungraded video frame with flat color", "color_grade"),
    Axis("material_detail", "surfaces showing rich fine material detail and texture",
         "smooth plain surfaces with no detail", "material"),
    Axis("set_detail", "an elaborate detailed set with rich background detail",
         "a plain empty background", "environment"),
    Axis("makeup_grooming", "a person with polished professional makeup and styled hair",
         "a person with a plain bare everyday look", "subject_surface"),
    Axis("headwear", "a person wearing an elaborate ornate headpiece or headdress",
         "a person with plain simple hair and no headwear", "headwear"),
    Axis("jewelry", "a person wearing ornate jewelry and accessories",
         "a person wearing no jewelry or ornament", "jewelry"),
    Axis("atmosphere", "a moody atmospheric cinematic scene",
         "a flat plain scene with no atmosphere", "atmosphere"),
    Axis("depth_of_field", "a cinematic shot with shallow depth of field and blurred background",
         "a flat shot with everything in focus", "global"),
    Axis("skin_finish", "flawless polished skin with a professional finish",
         "plain untreated skin", "subject_surface"),
    Axis("texture_richness", "highly detailed rich textures",
         "plain flat textures", "material"),
    Axis("warm_key", "warm golden cinematic lamp light",
         "cool flat fluorescent ceiling light", "lighting"),
    Axis("wealth", "an opulent wealthy luxurious setting",
         "a modest plain setting", "environment"),
    Axis("craft", "a professionally produced cinematic frame",
         "an unpolished amateur frame", "global"),
)


# Tried and did not survive validation. Kept so a later reader does not re-add them
# and so the axis table has a documented provenance.
REJECTED_AXES: dict[str, str] = {
    "sharpness": "CLIP cannot read resolution; the antonym margin tracked content, not "
                 "sharpness, and the low->high shift was not consistent.",
    "musiq": "MUSIQ-SPAQ ranked the source above the target (oracle -1.40 on 10 samples).",
    "vbench_imaging_quality": "noop 0.696 vs oracle 0.664; rewards technical cleanliness, "
                              "which a cinematic grade lowers.",
}


def axis_names() -> list[str]:
    return [a.name for a in AXES]


def prompt_pairs() -> list[tuple[str, str]]:
    """(high, low) caption pairs, in AXES order."""
    return [(a.high, a.low) for a in AXES]
