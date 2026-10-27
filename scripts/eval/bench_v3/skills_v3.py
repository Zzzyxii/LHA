"""Skill cards v3: carrier-gated domain knowledge for the edit-spec writer.

Why this exists. The v2 agent (a VLM writing a free-form plan) already names
concrete things -- "liangbatou headdress, diancui filigree, chenyi robe" -- so
adding more vocabulary was not the missing piece. Two measured failures were:

* 36% of the plan's content was detail the plan could not have known: it
  specified elements whose carriers are not visible in the low frames (a
  specific background behind a wall that fills the frame, a garment under a
  table), and those were the items the judges marked wrong.
* Prompts ran ~300-2000 tokens, and on ours the wider the prompt the less the
  model changed (ablation 7.2: change 0.21 -> 0.17 -> 0.12 as sections were
  added), while the edits still did not land.

So a skill card does three things a free-form plan cannot:

1. **Carrier gating** -- every card lists the carriers this category actually
   edits (headwear, upper garment, background wall, light source) and says what
   to do when a carrier is NOT visible (leave it alone; adjust grade/light
   globally instead). This is a rule about *when not to be specific*.
2. **Category vocabulary** -- the nouns a domain expert would use, so an element
   that IS visible can be named precisely instead of described vaguely.
3. **Model priors** -- how this edit model responds: short imperative clauses
   with one change each land; preservation boilerplate and long prose make it
   under-edit.

The card is knowledge, not text to paste. The writer uses it to decide what to
say, then emits a short spec; that is what keeps this from being "append more
adjectives to the prompt".
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillCard:
    key: str
    title: str
    # (carrier, what it looks like in frames, what an expert does with it)
    carriers: tuple[tuple[str, str, str], ...]
    vocabulary: tuple[str, ...]
    # What to do for carriers that are not visible.
    off_carrier_policy: str
    model_priors: tuple[str, ...]
    failure_modes: tuple[str, ...]
    must_preserve: tuple[str, ...] = (
        "each person's facial identity",
        "facial expressions",
        "body pose and action",
        "composition and camera framing",
    )

    def to_prompt_block(self) -> str:
        lines = [f"CATEGORY: {self.title}", "CARRIERS (only edit the ones you can see):"]
        for name, looks, action in self.carriers:
            lines.append(f"- {name} | looks like: {looks} | do: {action}")
        lines.append("VOCABULARY (use these nouns only for visible carriers): " + ", ".join(self.vocabulary))
        lines.append("IF A CARRIER IS NOT VISIBLE: " + self.off_carrier_policy)
        lines.append("MODEL BEHAVIOUR: " + " ".join(self.model_priors))
        lines.append("KNOWN FAILURES TO AVOID: " + " ".join(self.failure_modes))
        lines.append("ALWAYS KEEP: " + "; ".join(self.must_preserve))
        return "\n".join(lines)


_COMMON_PRIORS = (
    "This edit model under-edits when the prompt is long: keep the whole prompt under 90 English words.",
    "One change per clause, verb first; put the most important change first.",
    "Do not write preservation boilerplate; the model already receives a preserve list.",
)
_COMMON_FAILURES = (
    "Do not invent props or rooms that are not visible.",
    "Do not swap a person's face or change how many people there are.",
    "Do not describe camera moves; the source motion is reused.",
)

CARDS: dict[str, SkillCard] = {
    "chinese_period": SkillCard(
        key="chinese_period",
        title="Chinese historical / period drama (Qing, Han, wuxia, palace)",
        carriers=(
            ("headwear on each visible head", "bun, hairpins, floral ornament, bare hair",
             "upgrade to period headdress: liangbatou / diancui filigree / jade pins / tassels, sized to the head"),
            ("upper garment of each visible person", "t-shirt, sweater, blouse, modern collar",
             "replace with period robe: chenyi / changyi / qipao, standing collar, gold-thread embroidery"),
            ("background wall or set behind the people", "plain wall, mural, window, foliage, LED panel",
             "replace with palace interior: carved rosewood lattice, silk partition, bronze lanterns, palace wall"),
            ("light sources in frame", "flat daylight, ceiling lights, glowing orbs",
             "relight as warm directional lantern light with soft falloff; remove modern glowing fixtures"),
            ("makeup and skin", "modern natural makeup",
             "refine to period makeup: defined liner, red lip, even complexion"),
        ),
        vocabulary=(
            "liangbatou headdress", "diancui kingfisher-feather filigree", "nephrite hairpin",
            "silk tassel", "chenyi robe", "changyi robe", "standing collar", "gold-thread embroidery",
            "embroidered rank badge", "carved rosewood lattice", "painted silk partition",
            "bronze court lantern", "palace wall", "warm candlelight",
        ),
        off_carrier_policy=(
            "leave that region untouched; carry the period look through grade and light only "
            "(warm amber shadows, lower saturation in the greens)"
        ),
        model_priors=_COMMON_PRIORS,
        failure_modes=_COMMON_FAILURES + (
            "Do not name a dynasty or rank the frame does not support.",
            "Do not turn one headdress into a crown, or one robe into armour.",
        ),
    ),
    "western_period_fantasy": SkillCard(
        key="western_period_fantasy",
        title="Western period, fantasy or magical realism",
        carriers=(
            ("headwear / hair", "cap, scarf, bare head",
             "period hat, styled hair with metal or leather ornament"),
            ("upper garment", "t-shirt, jacket, hoodie",
             "period garment: wool coat, linen shirt, brocade vest, leather straps"),
            ("background set", "park, plain wall, office, modern interior",
             "period room or fantasy interior: stone hearth, wooden beams, arched window, candle sconces"),
            ("light sources", "overcast daylight, office lighting",
             "motivated practical light: firelight, candle, window shaft, enchanted glow"),
            ("props at hand", "cup, book, screen, phone",
             "period or fantasy counterpart of the same prop in the same place"),
        ),
        vocabulary=(
            "wool coat", "linen shirt", "brocade vest", "leather bracer", "stone hearth",
            "wooden beam", "arched window", "candle sconce", "enchanted ember", "aged parchment",
            "carved oak", "iron chandelier",
        ),
        off_carrier_policy="leave it alone and shift the grade instead: warmer highlights, deeper shadows",
        model_priors=_COMMON_PRIORS,
        failure_modes=_COMMON_FAILURES + ("Do not add creatures or effects the request did not ask for.",),
    ),
    "sci_fi_modern": SkillCard(
        key="sci_fi_modern",
        title="Sci-fi, futuristic or modern character styling",
        carriers=(
            ("headwear / eyewear", "bare head, glasses, earbud",
             "technical accessory: micro-comm nodule, visor, eye patch with strap, matte ear stud"),
            ("upper garment", "sweater, jacket, t-shirt",
             "technical garment: tactical tunic, seam-lined shell, ribbed knit, matte leather"),
            ("background set", "office, street, interior",
             "same space with technical dressing: panels, cable runs, cool practicals"),
            ("light sources", "office light, daylight",
             "cool key with rim separation; practical panel glow"),
            ("small props", "laptop, cup, phone",
             "technical counterpart in the same position"),
        ),
        vocabulary=(
            "tactical tunic", "carbon-fibre implant", "titanium micro-comm", "ribbed knit",
            "matte shell", "halo rim light", "panel glow", "cable run", "holo interface",
        ),
        off_carrier_policy="keep the region as-is; add the look through contrast and cool grade",
        model_priors=_COMMON_PRIORS,
        failure_modes=_COMMON_FAILURES + ("Do not turn the scene into a spaceship or add crowds.",),
    ),
    "style_filter": SkillCard(
        key="style_filter",
        title="Whole-frame style transfer (film stock, monochrome, illustration, retro video)",
        carriers=(
            ("the whole frame", "any content",
             "grade and texture the entire image; do not restyle individual objects"),
            ("highlight response", "flat digital highlights",
             "halation, gentle rolloff, film-like shoulder"),
            ("shadow response", "crushed or lifted blacks",
             "filmic toe: deep but detailed shadows"),
            ("texture layer", "clean digital surface",
             "grain, vignette, gate weave or paper/ink texture as the reference implies"),
        ),
        vocabulary=(
            "high-contrast monochrome", "selective colour retention", "halation", "film grain",
            "gate weave", "sepia wash", "crosshatch etching", "ink line", "pastel wash",
            "soft bloom", "vignette",
        ),
        off_carrier_policy="this category edits the whole frame; keep objects, faces and motion intact",
        model_priors=_COMMON_PRIORS + ("For this category the change is global, so state the grade first and the texture second.",),
        failure_modes=(
            "Do not change what objects are in frame.",
            "Do not flatten faces into illustration unless the request says so.",
        ),
    ),
    "nature_landscape": SkillCard(
        key="nature_landscape",
        title="Nature, landscape and documentary scenery",
        carriers=(
            ("sky and atmosphere", "flat sky, haze, clouds",
             "aerial perspective: layered haze, god rays, colour-graded sky"),
            ("ground and vegetation", "field, road, trees",
             "richer material response: dew, warm rim on foliage, defined texture"),
            ("light sources", "overcast or flat light",
             "golden-hour key with long shadows and warm-cool split"),
            ("distant structure", "houses, poles, roads",
             "keep them; let haze set them back in depth"),
        ),
        vocabulary=(
            "aerial perspective", "golden-hour key", "warm-cool split", "long shadows",
            "layered haze", "dew sheen", "volumetric light", "deep focus",
        ),
        off_carrier_policy="grade globally; do not invent landmarks",
        model_priors=_COMMON_PRIORS,
        failure_modes=("Do not add animals, people or buildings that are not there.",),
    ),
}

# Router: first matching rule wins. Keys are lowercase substrings of the intent.
_ROUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("style_filter", (
        "black and white", "black-and-white", "monochrome", "film grain", "vhs", "home video",
        "home movie", "sepia", "etching", "noir", "glitch", "storybook", "pastel", "dream sequence",
        "retro", "vintage", "1970", "1980", "1950", "1940", "19th-century", "17th-century",
        "still life", "illustration",
    )),
    ("chinese_period", (
        "chinese", "qing", "dynasty", "historical palace", "palace drama", "period drama",
        "hanfu", "wuxia", "imperial", "court", "ancient chinese", "traditional chinese", "costume drama",
    )),
    ("western_period_fantasy", ("period", "historical", "fantasy", "medieval", "victorian", "magical", "scholar")),
    ("sci_fi_modern", ("sci-fi", "scifi", "futuristic", "cyberpunk", "robot", "holographic", "neon", "tactical", "implant")),
    ("nature_landscape", ("landscape", "nature", "field", "mountain", "forest", "sky", "seascape")),
)


def route(intent: str) -> str:
    text = (intent or "").lower()
    for key, keywords in _ROUTES:
        if any(word in text for word in keywords):
            return key
    return "chinese_period" if "period" in text else "style_filter"


def card_for(intent: str) -> SkillCard:
    return CARDS[route(intent)]


# Added for the decisive arm. The base card makes the writer skip carriers it
# cannot see, which cut invented detail but also thinned the edit: on ours the
# plan-only arm already changed less than the expert prompt did (0.20 vs 0.24),
# and change volume correlates with getting closer to the target. These priors
# push the opposite way on the carriers that ARE visible.
DECISIVE_PRIORS = (
    "Make every requested replacement unmistakable: the new element must be clearly "
    "present in the first frame and stay for the whole clip, not a tint or a blend.",
    "Replace, do not overlay: the old garment/background/ornament must be gone, not visible underneath.",
    "If a carrier is visible and the user's request touches it, change it fully; do not hedge.",
)


def card_block(intent: str, *, decisive: bool = False, execution_only: bool = False) -> str:
    card = CARDS[route(intent)]
    if execution_only:
        # Measured: handing the writer free vocabulary made it name elements the
        # target did not contain (liangbatou/diancui where the target shows a
        # floral gold headpiece), doubling change volume and losing 0.06 GT. So
        # this variant may use the vocabulary only to describe HOW a requested
        # change is executed, never WHAT to add.
        return "\n".join([
            f"CATEGORY: {card.title}",
            "CARRIERS VISIBLE OR NOT: you may only change what the user asked for. "
            "Do not introduce an element, garment, ornament or set piece the request does not mention.",
            "IF THE USER'S REQUEST NAMES A THING, name its execution with these words: " + ", ".join(card.vocabulary),
            "IF THE USER'S REQUEST ONLY NAMES A MOOD OR LOOK, express it through grade, light and texture only: "
            + card.off_carrier_policy,
            "MODEL BEHAVIOUR: " + " ".join(card.model_priors),
            "KNOWN FAILURES TO AVOID: " + " ".join(card.failure_modes),
            "ALWAYS KEEP: " + "; ".join(card.must_preserve),
        ])
    block = card.to_prompt_block()
    if decisive:
        block += "\nINTENSITY: " + " ".join(DECISIVE_PRIORS)
    return block
