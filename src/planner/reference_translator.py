"""Reference-to-instruction translation for the reference-image input tier (bench_v3 T2).

The user gives a one-line intent plus one or more reference stills ("make it feel
like this"). The edit model takes text, so the agent's job here is to translate
what is visible in the reference into a concrete edit instruction for *this*
source video: which garments, headwear, props, set dressing, lighting and grade
to adopt, anchored to people and objects that are actually in the source.

The output is a plain instruction meant to be fed to the edit model verbatim, in
the same register as the expert instructions (which, fed directly, measured best
on the execution track). It deliberately carries no appended skill blocks: the
planning-track ablation showed those only damp how much the model changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from vlm import extract_frames, resolve_vision_model

SYSTEM_PROMPT = """You write instructions for a video edit model that restyles an existing live-action clip.
You are shown SOURCE frames from the clip to edit and REFERENCE images showing the look the user wants.
Translate the reference into concrete, visible edits on the source. Return strict JSON only."""

USER_TEMPLATE = """User request: {intent}

Images: the first {n_source} images are SOURCE frames of the clip to edit (in time order). The remaining {n_ref} image(s) are REFERENCE images of the desired look.

Write one edit instruction that makes the SOURCE clip match the REFERENCE's production design, within the user's request:
- Name concrete changes and anchor each to a person or object visible in the SOURCE (e.g. "the woman on the left", "the back wall", "the table").
- Cover what the reference actually shows: wardrobe (garment type, colour, material, trim), hair and headwear, makeup, props, set dressing and background, lighting direction and quality, colour grade, atmosphere.
- Use the reference's specific colours and materials when they are visible; do not invent details the reference does not show.
- If the reference is a stylised illustration, animation still or render rather than live-action footage, transfer only its visual medium and look (grade, palette, contrast, texture, grain, linework, print or paint technique), never its characters, costumes or objects.
- Do NOT copy the reference's people, faces, poses, framing or layout. Keep every SOURCE person's identity, expression, pose and action, and keep the composition and camera motion.
- English, one paragraph, at most 150 words.

Return JSON: {{"instruction": "...", "adopted_from_reference": ["short item", ...]}}"""


@dataclass
class ReferenceTranslation:
    instruction: str
    adopted_from_reference: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def translate_reference(
    *,
    intent: str,
    low_video: str,
    reference_images: list[str],
    n_source_frames: int = 4,
    model=None,
) -> ReferenceTranslation:
    if not reference_images:
        raise ValueError("translate_reference needs at least one reference image")
    source_frames = extract_frames(low_video, max_frames=n_source_frames)
    if not source_frames:
        raise RuntimeError(f"could not extract frames from {low_video}")
    model = model or resolve_vision_model(role="planner")
    payload = model.predict_json(
        system_prompt=SYSTEM_PROMPT,
        user_text=USER_TEMPLATE.format(intent=intent.strip(), n_source=len(source_frames), n_ref=len(reference_images)),
        image_paths=[*source_frames, *reference_images],
    )
    instruction = " ".join(str(payload.get("instruction", "")).split())
    if not instruction:
        raise RuntimeError(f"reference translation returned no instruction: {payload!r}"[:500])
    adopted = [str(x) for x in payload.get("adopted_from_reference") or []]
    return ReferenceTranslation(instruction=instruction, adopted_from_reference=adopted, raw=payload)


def load_or_translate(*, cache_dir: Path, sample_id: str, intent: str, low_video: str,
                      reference_images: list[str], model=None) -> dict:
    """The cached translation for ``sample_id``, translating and caching it on a miss.

    The one cache both the precompute step and the renderer use, so arms that render
    the same sample share one instruction. A cached entry made for a different
    intent or reference set is an error rather than silently reused.
    """
    path = Path(cache_dir) / f"{sample_id}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("intent") != intent or cached.get("reference_images") != list(reference_images):
            raise RuntimeError(f"translation cache {path} was made for different inputs")
        return cached
    result = translate_reference(intent=intent, low_video=low_video, reference_images=reference_images, model=model)
    record = {
        "intent": intent,
        "reference_images": list(reference_images),
        "instruction": result.instruction,
        "adopted_from_reference": result.adopted_from_reference,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return record
