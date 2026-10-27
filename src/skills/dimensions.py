"""Shared dimension table for the merged skill system.

Before this module there were three overlapping vocabularies for "which part of
the frame are we talking about":

* ``skills.enhancement`` -- 10 dimensions (color_grade, lighting, texture_detail,
  garment, hair_headwear, jewelry, makeup_face, background, props, atmosphere)
* ``skills.base.SkillTemplate`` -- 8 render fields (environment, prop, lighting,
  costume, subject_surface, subject_integration, atmosphere, material)
* ``emit_region_edits()`` -- 6 region names (environment, functional_scene_props,
  lighting_system_design, costume_design, skin_surface_design,
  subject_environment_integration)

Each was named for the thing that produced it, so the same concept had up to
three names and no single place said what must be preserved when it is edited.

This module is that single place. It defines 11 canonical dimensions and, for
each one, the facts that are **cross-style generic** -- i.e. true whether the
target world is cyberpunk or Studio Ghibli:

* ``trigger_keywords`` -- which words in an instruction put this dimension in play.
  Matching is deterministic, never model-mediated: the previous design asked a VLM
  to merge template and instruction and the model invented edits nobody asked for.
* ``must_preserve`` -- what editing this dimension must not disturb.
* ``order`` -- global-to-local emission order; also the order in which dimensions
  are dropped when the prompt overflows.
* ``default_strength`` -- how hard this dimension is normally pushed.
* ``direction`` -- the generic execution-quality clause, used when an instruction
  names this dimension explicitly.

What stays per-style (in ``SkillTemplate``) is only "what this dimension looks
like in that world". Splitting it this way is what keeps 98 styles from needing
98 copies of the preservation rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache


# Applied to every edit regardless of which dimensions fired. Taken verbatim from
# skills.enhancement; losing these clauses is what the original regression traced
# back to.
UNIVERSAL_PRESERVE: tuple[str, ...] = (
    "the subjects' facial identity and likeness",
    "facial expressions",
    "body pose and gesture",
    "camera framing and motion",
    "scene composition and subject placement",
    "the number of people and objects present",
)


@dataclass(frozen=True)
class DimensionPolicy:
    """Cross-style generic facts about one edit dimension."""

    dimension: str
    title: str
    trigger_keywords: tuple[str, ...] = ()
    must_preserve: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    order: int = 50
    default_strength: float = 0.8
    # Noun phrase for "keep <this> unchanged", emitted when an instruction names
    # this dimension as a hold. Omitting the style's text for a dimension is not
    # enough on its own: the style's pull is strong enough that the edit model
    # restyles it anyway unless told not to in as many words.
    keep_phrase: str = ""
    direction: str = ""
    notes: str = ""

    def spans(self, instruction: str) -> list[tuple[int, int]]:
        """Start/end offsets of every trigger-keyword occurrence, sorted."""
        lowered = (instruction or "").lower()
        found = [
            (match.start(), match.end())
            for keyword in self.trigger_keywords
            for match in _keyword_regex(keyword).finditer(lowered)
        ]
        return sorted(found)

    def matches(self, instruction: str) -> bool:
        """True when any trigger keyword starts a word in the instruction.

        Anchored at a word boundary rather than matched anywhere in the string.
        Keyword stems ("saturat", "illuminat", "atmospher") still fire on
        saturation / illuminate / atmospheric, but "grade" no longer fires on
        "upgrade", "prop" not on "proper", "lit" not on "split". Plain substring
        matching was affordable while a match only appended one craft clause;
        once a match can suppress a style's text for that dimension, a false
        positive silently drops part of the requested restyle.
        """
        lowered = (instruction or "").lower()
        return any(_keyword_regex(keyword).search(lowered) for keyword in self.trigger_keywords)


_CJK = re.compile(r"[㐀-鿿]")


@lru_cache(maxsize=None)
def _keyword_regex(keyword: str) -> re.Pattern[str]:
    """Compile a trigger keyword.

    A trailing ``*`` marks a stem and matches the rest of the word
    ("saturat*" -> saturation, "illuminat*" -> illuminate). Anything else must
    match a whole word: "prop" must not fire on "proper" and "lit" not on
    "split", because a false positive can now suppress a style's text for that
    dimension rather than merely appending a clause.

    CJK keywords get no word boundaries at all. Python's ``\\b`` treats Han
    characters as word characters, so ``\\b服装\\b`` never matches inside
    "保留服装" -- there is no boundary to anchor to. Substring matching is
    correct for Chinese, which has no word delimiters to respect.
    """
    if _CJK.search(keyword):
        return re.compile(re.escape(keyword))
    if keyword.endswith("*"):
        return re.compile(r"\b" + re.escape(keyword[:-1]))
    return re.compile(r"\b" + re.escape(keyword) + r"\b")


# key -> policy. Keys are the canonical dimension names used by SkillTemplate
# fields and by DimensionOverride.
DIMENSIONS: dict[str, DimensionPolicy] = {
    "color_grade": DimensionPolicy(
        dimension="color_grade",
        title="Cinematic colour grade",
        trigger_keywords=(
            "color", "colors", "colored", "colour", "tone", "grade", "grading", "graded",
            "saturat*", "warm", "cool", "palette", "hue", "vibrant", "rich",
            "调色", "色彩", "色调", "饱和",
        ),
        must_preserve=("the original white balance of skin tones",),
        order=10,
        default_strength=0.8,
        keep_phrase="the original colour balance",
        direction=(
            "Grade cinematically: controlled saturation, filmic highlight rolloff, "
            "deep but detailed shadows, and one coherent warm-cool balance across the frame"
        ),
        notes="Highest-frequency dimension in the bench (28/30).",
    ),
    "lighting": DimensionPolicy(
        dimension="lighting",
        title="Motivated directional lighting",
        trigger_keywords=(
            "lighting", "light", "lights", "highlight", "shadow", "candela", "candle",
            "lantern", "backlit", "rim light", "glow", "illuminat*", "lit", "luminous",
            "光照", "灯光", "打光", "逆光",
        ),
        must_preserve=("the direction the existing light comes from",),
        order=20,
        default_strength=0.90,
        keep_phrase="the existing lighting",
        direction=(
            "Light with a motivated directional key, soft falloff, gentle rim separation "
            "from the background, and shadows that keep detail rather than crushing to black"
        ),
    ),
    "material": DimensionPolicy(
        dimension="material",
        title="Surface micro-detail",
        trigger_keywords=(
            "texture", "detail", "intricate", "ornate", "refined", "pattern", "material",
            "embroider", "fine", "crisp", "sharp",
            "材质", "质感", "纹理", "细节",
        ),
        order=30,
        default_strength=0.8,
        keep_phrase="the existing materials",
        direction=(
            "Resolve surface micro-detail: visible weave in fabric, individual strands in hair, "
            "crisp edges on small ornaments, and material grain that holds up at full resolution"
        ),
    ),
    "costume": DimensionPolicy(
        dimension="costume",
        title="Garment construction and drape",
        trigger_keywords=(
            "robe", "clothing", "garment", "embroider", "collar", "dress", "costume",
            "fabric", "outfit", "attire", "sleeve", "silk", "gown",
            "服装", "衣服", "服饰", "长袍",
        ),
        must_preserve=("the silhouette and coverage of the original clothing",),
        order=40,
        default_strength=0.82,
        keep_phrase="the clothing",
        direction=(
            "Render garments as real cloth: structured drape that follows the body, raised "
            "embroidery catching light at a grazing angle, and clean collar and seam lines"
        ),
    ),
    "headwear": DimensionPolicy(
        dimension="headwear",
        title="Hair ornament craft",
        trigger_keywords=(
            "headpiece", "hair accessor", "hairpin", "updo", "headband", "hairstyle",
            "crown", "tiara", "hair ornament", "hairdo",
            "头饰", "发饰", "发型", "头冠",
        ),
        must_preserve=("the original hairline and head shape",),
        order=50,
        default_strength=0.8,
        keep_phrase="the hair and headwear",
        direction=(
            "Give hair ornaments distinct metal and stone materials, seated correctly on the "
            "hair with believable weight and consistent scale and position across frames"
        ),
    ),
    "jewelry": DimensionPolicy(
        dimension="jewelry",
        title="Jewellery material response",
        trigger_keywords=(
            "earring", "necklace", "gemstone", "jade", "pendant", "bracelet", "jewel",
            "sapphire", "emerald", "pearl", "gold", "silver",
            "首饰", "珠宝", "项链", "耳环",
        ),
        order=60,
        default_strength=0.8,
        keep_phrase="the jewellery",
        direction=(
            "Give jewellery a distinct metal and gemstone response to light -- specular on "
            "metal, internal refraction in stones -- at a believable physical scale"
        ),
    ),
    "subject_surface": DimensionPolicy(
        dimension="subject_surface",
        title="Makeup refinement",
        trigger_keywords=(
            "makeup", "eyeliner", "lipstick", "eyeshadow", "contour", "complexion",
            "eyebrow", "lip", "blush",
            # "keep the face" is the most common way an instruction asks for this
            # dimension to be held. Safe under word-boundary matching: it does not
            # fire on "surface".
            "face", "faces", "facial", "identity", "likeness",
            "妆容", "妆", "皮肤", "面部",
        ),
        must_preserve=(
            "the original facial proportions and bone structure",
            "the original eye and mouth shape",
        ),
        order=70,
        default_strength=0.62,
        keep_phrase="the face and skin",
        direction=(
            "Refine makeup with clean edges and natural blending, strictly within the "
            "original facial structure"
        ),
        notes="Highest identity risk of any dimension; its preserve clauses are the point.",
    ),
    "environment": DimensionPolicy(
        dimension="environment",
        title="Background depth and dressing",
        trigger_keywords=(
            "background", "interior", "backdrop", "wall", "set dressing", "room",
            "scenery", "environment", "behind",
            "背景", "场景", "环境", "室内",
        ),
        must_preserve=("the spatial layout and depth ordering of the scene",),
        order=80,
        default_strength=0.95,
        keep_phrase="the background",
        direction=(
            "Build a depth-separated background with believable set dressing, held slightly "
            "softer than the subject so focus stays where the shot puts it"
        ),
    ),
    "props": DimensionPolicy(
        dimension="props",
        title="Prop material and contact",
        trigger_keywords=(
            "prop", "furniture", "object", "candle", "screen", "laptop", "duvet",
            "table", "chair", "vase",
            "道具", "物件", "家具",
        ),
        must_preserve=("the position and scale of existing objects",),
        order=90,
        default_strength=0.93,
        keep_phrase="the props",
        direction=(
            "Give props a consistent material identity and correct contact shadows where "
            "they meet surfaces, so they sit in the scene rather than on top of it"
        ),
    ),
    "atmosphere": DimensionPolicy(
        dimension="atmosphere",
        title="Atmospheric depth",
        trigger_keywords=(
            "mist", "fog", "haze", "dust", "smoke", "bokeh", "depth of field",
            "soft-focus", "soft focus", "blur", "ethereal", "dreamy", "atmospher*",
            "雾", "氛围", "景深",
        ),
        must_preserve=("subject legibility and edge definition",),
        order=100,
        default_strength=0.8,
        keep_phrase="the atmosphere",
        direction=(
            "Add restrained atmospheric depth -- light haze with distance, gentle falloff -- "
            "without washing out contrast or veiling the subject"
        ),
    ),
    "subject_integration": DimensionPolicy(
        dimension="subject_integration",
        title="Subject integration into the world",
        trigger_keywords=("integrat*", "belong", "contact light", "reflection", "融合", "融入"),
        order=110,
        default_strength=0.78,
        keep_phrase="how the subject sits in the scene",
        notes=(
            "No enhancement-system counterpart: this one exists only as a SkillTemplate "
            "field and an emit_region_edits() region. Carry the strength across so the "
            "merged table covers every field a style template can declare."
        ),
    ),
}


def sorted_dimensions() -> list[DimensionPolicy]:
    """All dimensions in emission order (global -> local)."""
    return sorted(DIMENSIONS.values(), key=lambda item: (item.order, item.dimension))


def match_dimensions(instruction: str) -> list[DimensionPolicy]:
    """Dimensions whose trigger keywords appear in the instruction, in order."""
    return [policy for policy in sorted_dimensions() if policy.matches(instruction)]


def compose_dimensions(instruction: str, max_dimensions: int = 6) -> list[DimensionPolicy]:
    """Dimensions that fire on this instruction, in emission order, capped.

    Greedy in order, skipping any dimension blocked by an earlier one's
    ``conflicts_with`` -- the conflict machinery in the old registry had no data
    behind it, but it costs nothing to carry and the table can now be given some.

    The cap matters: every direction clause competes with the instruction itself
    for text-encoder attention, and the measured failure was not "too few craft
    clauses" but "the wider the prompt, the less the model changed".
    """
    chosen: list[DimensionPolicy] = []
    blocked: set[str] = set()
    for policy in sorted_dimensions():
        if len(chosen) >= max_dimensions:
            break
        if policy.dimension in blocked or not policy.matches(instruction):
            continue
        chosen.append(policy)
        blocked.update(policy.conflicts_with)
    return chosen


def policy_for(dimension: str) -> DimensionPolicy | None:
    return DIMENSIONS.get(dimension)
