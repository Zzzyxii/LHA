#!/usr/bin/env python3
"""Build skill templates from the 98 OPUSCAR styles in zyx/styles.json.

Emits one JSON per style shaped for ``promote_skill_json_to_template.py``, which
turns them into ``src/skills/templates/<family>/<id>.py``. Going through that
script rather than writing .py files directly keeps one code path for "a style
became a template", whether it arrived from the memory promotion loop or from
this corpus.

Why this corpus: the 13 hand-written templates are half Chinese (the whole
sci_fi family) and the edit model's text encoder is umT5 on a mostly-English
checkpoint, so they reach it only after a translation pass. ``prompt_en`` here is
already English and already structured -- palette, grading, texture, camera,
references, an explicit avoid-list. Each record also ships a still
(``img/opuscar98/NN.jpg``), which is what the ``reference`` delivery mode needs
and what the existing templates have no equivalent of.

What does not survive the mapping, and where it goes instead:

* cutting rhythm and shot duration -- no timeline concept in SkillTemplate.
* the "Reference: <film, photographer>" clause -- it is provenance, not
  direction.
* the "Avoid: ..." clause -- becomes ``common_failure_modes``, which is the
  closest field, since ``forbidden_mixes`` means "styles that must not be
  combined" rather than "renders to avoid".
* camera rigging language -- kept as a compilation note.

All three land in ``prompt_compilation_notes`` or ``common_failure_modes``, which
the composer never emits, so they inform the planner without spending prompt
budget.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

DEFAULT_STYLES = "zyx/styles.json"
DEFAULT_IMAGES = "zyx/img/opuscar98"
DEFAULT_OUT = "out/promoted_skills/opuscar"

# Splitting a free-form style prompt into the 11 canonical dimensions. Wider than
# DimensionPolicy.trigger_keywords on purpose: those fire off what a *user*
# writes, this reads what a *style author* writes ("halation", "gate weave",
# "aerial perspective"), and the two vocabularies barely overlap. Checked in
# canonical order so a clause goes to the most global dimension it mentions.
_SPLIT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "color_grade": (
        "palette", "grade", "grading", "graded", "tint", "tone", "saturat",
        "colour", "color", "contrast", "black level", "highlights", "shadows",
        "hex", "#", "warm", "cool", "monochrome", "sepia", "wash", "exposure",
    ),
    "lighting": (
        "light", "lit", "glow", "rim", "key light", "practical", "lamp",
        "candle", "fire", "sun", "falloff", "flare", "bloom", "spotlight",
    ),
    "material": (
        "grain", "texture", "scratch", "weave", "ink", "paper", "surface",
        "halation", "vignette", "dust", "flicker", "speckle", "streak",
        "brushwork", "crosshatch", "etching", "line work",
    ),
    "costume": (
        "garment", "cloth", "costume", "dress", "robe", "fabric", "wardrobe",
        "cloak", "uniform", "sequin", "fringe", "collar",
    ),
    "headwear": ("hat", "hair", "headwear", "cap", "helmet", "crown", "boater"),
    "jewelry": ("jewel", "metal", "gold", "silver", "gem", "pendant", "bronze"),
    "subject_surface": (
        "skin", "face", "complexion", "makeup", "eye", "lip", "figure",
        "portrait", "hands",
    ),
    "environment": (
        "background", "sky", "ground", "room", "interior", "set", "scenery",
        "landscape", "horizon", "street", "wall", "architecture", "proscenium",
        "panorama", "field", "trench", "stage",
    ),
    "props": (
        "prop", "object", "furniture", "vehicle", "chair", "table", "machine",
        "microphone", "lantern", "chandelier",
    ),
    "atmosphere": (
        "haze", "fog", "mist", "smoke", "depth", "aerial", "atmospher", "blur",
        "bokeh", "wind", "rain", "cloud", "god rays", "volumetric",
    ),
    "subject_integration": (
        "contact", "reflection", "integrat", "belong", "silhouette against",
        "reflected",
    ),
}

_CANONICAL_ORDER = tuple(_SPLIT_KEYWORDS)
# Longest-first so "key light" wins over "light" and "black level" over "black".
_SPLIT_PATTERNS = {
    dimension: re.compile(
        r"\b(" + "|".join(sorted((re.escape(k) for k in keywords), key=len, reverse=True)) + r")"
    )
    for dimension, keywords in _SPLIT_KEYWORDS.items()
}

# Sentence split only. Splitting on ";" too produces fragments like "#D97757);"
# from the hex-palette lists, and those fragments are what ends up in the prompt.
_CLAUSE_SPLIT = re.compile(r"(?<=[.])\s+")

# Camera and cutting language. A style prompt describes these constantly and they
# do belong to the look, but they are not directions for any render dimension --
# and the source motion is reused anyway, so a "push in" clause is noise here.
# Kept as a compilation note rather than dropped.
_NON_DIRECTION = re.compile(
    r"\b(cut every|camera|shot|rig in frame|push in|pull back|truck along|pov|"
    r"pan along|pan past|fps|frames per second|iris|signature shot)\b"
)

# Leading fragments left by a semicolon inside a sentence: "(e.g. #D97757); every
# lamp ..." splits nowhere, so trim the head down to the first real word.
_LEADING_FRAGMENT = re.compile(r"^[^A-Za-z]+")

# A semicolon inside a sentence leaves a hex-palette fragment at the head of the
# next clause: "(e.g. #D97757); every lamp stays grey" -> "D97757); every lamp ...".
_LEADING_HEX_FRAGMENT = re.compile(r"^[#A-Za-z0-9]{3,12}\)[\s;]*")

MIN_CLAUSE_CHARS = 25


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def _family_for(kind: str) -> str:
    return {"2D": "styles_2d", "3D": "styles_3d", "2.5D": "styles_25d"}.get(kind, "styles_2d")


def _split_prompt(prompt: str) -> tuple[dict[str, str], list[str], list[str]]:
    """Split a style prompt into (dimension -> text, avoid clauses, notes)."""
    dimensions: dict[str, list[str]] = {}
    notes: list[str] = []
    avoid: list[str] = []

    for clause in _CLAUSE_SPLIT.split(prompt or ""):
        clause = _LEADING_HEX_FRAGMENT.sub("", clause.strip())
        clause = _LEADING_FRAGMENT.sub("", clause)
        if len(clause) < MIN_CLAUSE_CHARS:
            continue
        lowered = clause.lower()
        if lowered.startswith("avoid"):
            avoid.append(clause.split(":", 1)[-1].strip())
            continue
        if lowered.startswith("reference"):
            notes.append(clause)
            continue
        if _NON_DIRECTION.search(lowered):
            notes.append(clause)
            continue
        for dimension in _CANONICAL_ORDER:
            if _SPLIT_PATTERNS[dimension].search(clause.lower()):
                dimensions.setdefault(dimension, []).append(clause)
                break
        else:
            notes.append(clause)

    # One clause per dimension, the longest one. Several clauses in a row often
    # describe the same thing from different angles ("Tint: ...; blacks stay
    # black"), and emitting all of them spends prompt budget on restatement.
    return {k: max(v, key=len) for k, v in dimensions.items()}, avoid, notes


def build_payload(record: dict, images_rel: str) -> dict:
    no = int(record["no"])
    style_en = str(record.get("style_en", "")).strip()
    family = _family_for(str(record.get("kind", "")))
    dimensions, avoid, notes = _split_prompt(str(record.get("prompt_en", "")))

    # A record whose prose matched no dimension at all would emit nothing in
    # inline mode. Fall back to the whole prompt rather than to silence.
    if not dimensions:
        dimensions["environment"] = str(record.get("prompt_en", "")).strip()
        notes.append("split fallback: no clause matched a canonical dimension")

    payload = {
        "strategy_id": f"opuscar_{no:02d}_{_slug(style_en)[:48]}",
        "style_family": family,
        "title": style_en or f"OPUSCAR {no:02d}",
        "description": str(record.get("desc_en", "")).strip(),
        "reference_image": f"{images_rel}/{no:02d}.jpg",
        "tags": [str(item) for item in (record.get("blend_en") or [])][:6],
        "intensity": "normal",
        "world_rewrite_cost": "high" if family != "styles_2d" else "medium",
        "identity_risk": "low",
        "common_failure_modes": avoid[:4],
        "prompt_compilation_notes": notes[:6],
        "pros": [],
        "cons": [],
        "forbidden_mixes": [],
        "anchor_requirements": [],
        "best_for_archetypes": [],
        "high_risk_archetypes": [],
        "recommended_carriers": [],
    }
    payload.update({dimension: "" for dimension in _CANONICAL_ORDER})
    payload.update(dimensions)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--styles-json", default=DEFAULT_STYLES)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = PROJECT_ROOT
    records = json.loads((root / args.styles_json).read_text(encoding="utf-8"))
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    stats = {"matched": 0, "unmatched_clauses": 0, "fallback_records": 0, "no_image": 0}

    for record in records:
        no = int(record["no"])
        if not (root / args.images_dir / f"{no:02d}.jpg").exists():
            stats["no_image"] += 1
        payload = build_payload(record, args.images_dir)
        if len(payload.get("prompt_compilation_notes") or []) and "split fallback" in " ".join(
            payload["prompt_compilation_notes"]
        ):
            stats["fallback_records"] += 1
        stats["matched"] += len([d for d in _CANONICAL_ORDER if payload.get(d)])
        stats["unmatched_clauses"] += len(
            [n for n in (payload.get("prompt_compilation_notes") or []) if not n.startswith(("Reference", "reference"))]
        )

        out_path = out_dir / f"{payload['strategy_id']}.json"
        if out_path.exists() and not args.overwrite:
            continue
        out_path.write_text(
            json.dumps(
                {
                    "skill_id": payload["strategy_id"],
                    "style_family": payload["style_family"],
                    "source": args.styles_json,
                    "memory_only": False,
                    "skill_payload": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written += 1

    print(json.dumps({"written": written, "out_dir": str(out_dir), "stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
