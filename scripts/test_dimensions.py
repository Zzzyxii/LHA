"""Regression tests for the merged skill system.

Covers the three things that are easy to break and hard to notice:

1. the dimension table (every dimension reachable, no duplicate orders);
2. override parsing -- especially that a hold binds to the dimension named
   *after* the signal word, which plain nearest-by-distance gets backwards;
3. prompt composition -- that a held dimension is dropped from the world block
   and turned into a preserve line instead, and that the composer never returns
   an empty string (run_loop.py silently falls back to the normalized prompt if
   it does, which drops every craft and preserve block without erroring).

Run:  PYTHONPATH=src python scripts/test_dimensions.py
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# The composer imports planner.core.pipeline, which pulls in the VLM backends.
# None of them are exercised here.
for _name, _attrs in [
    ("qwen_vl_utils", ["process_vision_info"]),
    ("transformers", ["AutoTokenizer", "AutoModelForCausalLM", "AutoProcessor", "AutoConfig"]),
    ("litellm", ["completion"]),
]:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        for _attr in _attrs:
            setattr(_mod, _attr, lambda *a, **k: None)
        sys.modules[_name] = _mod

from planner.core.pipeline import _compose_edit_prompt  # noqa: E402
from skills.dimension_parse import KEEP, EDIT, parse_overrides  # noqa: E402
from skills.dimensions import DIMENSIONS, UNIVERSAL_PRESERVE, sorted_dimensions  # noqa: E402

WORLD = {
    "environment": "neon skyline with rain-slick streets",
    "lighting": "cyan and magenta practicals",
    "costume": "technical tactical gear",
    "atmosphere": "volumetric haze",
}


def test_table() -> None:
    assert len(DIMENSIONS) == 11, len(DIMENSIONS)
    orders = [policy.order for policy in DIMENSIONS.values()]
    assert len(set(orders)) == len(orders), f"duplicate order values: {orders}"
    for name, policy in DIMENSIONS.items():
        assert policy.dimension == name, name
        assert policy.keep_phrase, f"{name} has no keep_phrase"
        assert policy.trigger_keywords, f"{name} can never fire"
    assert len(UNIVERSAL_PRESERVE) == 6
    assert [p.dimension for p in sorted_dimensions()][0] == "color_grade"


def test_keyword_matching() -> None:
    # Word-boundary matching: a keyword must not fire from inside another word.
    assert "color_grade" not in parse_overrides("upgrade the robe"), (
        "'upgrade' contains 'grade' -- stem matching must stay anchored"
    )
    assert "props" not in parse_overrides("a proper room"), "'prop' inside 'proper'"
    # ...but stems must still fire.
    assert "atmosphere" in parse_overrides("an atmospheric haze")
    assert "lighting" in parse_overrides("illuminated from the left")
    # An instruction that names no dimension names none.
    assert parse_overrides("turn the whole scene into a cyberpunk city") == {}


def test_override_direction() -> None:
    # The regression: "keep" is nearer to "robe" than to "face" by absolute
    # distance, but the thing being held is the face.
    assert parse_overrides("upgrade the robe, keep the face") == {
        "costume": EDIT,
        "subject_surface": KEEP,
    }
    assert parse_overrides("keep the background unchanged, upgrade the costume") == {
        "environment": KEEP,
        "costume": EDIT,
    }
    assert parse_overrides("make the robe richer but keep the face") == {
        "costume": EDIT,
        "subject_surface": KEEP,
    }
    assert parse_overrides("把视频变成赛博朋克风格，保留服装") == {"costume": KEEP}
    # A hold reaches a whole "A and B" list, not just its first noun -- and stops
    # at the next verb that talks about a change.
    assert parse_overrides("keep the clothing and the face") == {
        "costume": KEEP, "subject_surface": KEEP,
    }
    assert parse_overrides("keep the face and make the lighting dramatic") == {
        "subject_surface": KEEP, "lighting": EDIT,
    }
    # hard_preserve is a hold with nothing to disambiguate.
    assert parse_overrides("cyberpunk", ["preserve the background"]) == {"environment": KEEP}


def test_compose_delivery_modes() -> None:
    instruction = "turn the whole scene into a cyberpunk city, but keep the clothing"
    overrides = parse_overrides(instruction)
    assert overrides == {"costume": KEEP}

    os.environ["LOW_HIGH_STYLE_DELIVERY"] = "inline"
    inline = _compose_edit_prompt(
        user_prompt=instruction, planner_prompt="", hard_preserve=[],
        style_fields=WORLD, overrides=overrides,
    )
    assert "Target world:" in inline
    assert "technical tactical gear" not in inline, (
        "a held dimension must not contribute its world text"
    )
    assert "neon skyline" in inline, "unheld dimensions still contribute"
    assert "keep the clothing unchanged" in inline, (
        "dropping the text is not protection -- say it in as many words"
    )
    assert "the silhouette and coverage of the original clothing" in inline

    # The world travels as an image in reference mode, so no world text.
    os.environ["LOW_HIGH_STYLE_DELIVERY"] = "reference"
    reference = _compose_edit_prompt(
        user_prompt=instruction, planner_prompt="", hard_preserve=[],
        style_fields=WORLD, overrides=overrides,
    )
    assert "Target world:" not in reference
    assert "keep the clothing unchanged" in reference, (
        "holds survive even when the world is carried by an image"
    )

    os.environ["LOW_HIGH_STYLE_DELIVERY"] = "none"
    none = _compose_edit_prompt(
        user_prompt=instruction, planner_prompt="", hard_preserve=[],
        style_fields=WORLD, overrides=overrides,
    )
    assert "Target world:" not in none
    assert "neon skyline" not in none
    os.environ.pop("LOW_HIGH_STYLE_DELIVERY")


def test_hold_on_undeclared_dimension() -> None:
    """A hold binds on the instruction, not on what the style happens to declare.

    The Art Deco template declares no subject_surface text, so "keep the face"
    must still produce a constraint -- the style's overall pull repaints faces
    whether or not it has a dedicated clause for them.
    """
    instruction = "restage this as an art deco poster, but keep the face"
    overrides = parse_overrides(instruction)
    assert overrides == {"subject_surface": KEEP}

    # No subject_surface key at all -- the case that used to drop the hold.
    world = {"color_grade": "black and gold palette", "costume": "Erté gowns"}
    for mode in ("inline", "reference"):
        os.environ["LOW_HIGH_STYLE_DELIVERY"] = mode
        out = _compose_edit_prompt(
            user_prompt=instruction, planner_prompt="", hard_preserve=[],
            style_fields=world, overrides=overrides,
        )
        assert "keep the face and skin unchanged" in out, mode
    os.environ.pop("LOW_HIGH_STYLE_DELIVERY")


def test_compose_never_empty() -> None:
    """run_loop.py falls back to normalized_prompt on an empty result."""
    for mode in ("inline", "reference", "none"):
        os.environ["LOW_HIGH_STYLE_DELIVERY"] = mode
        for user_prompt in ("make it cyberpunk", "", "   "):
            for style_fields in (WORLD, {}, None):
                out = _compose_edit_prompt(
                    user_prompt=user_prompt,
                    planner_prompt="a planner rewrite",
                    hard_preserve=[],
                    style_fields=style_fields,
                    overrides={},
                )
                assert out.strip(), (
                    f"empty prompt for mode={mode} prompt={user_prompt!r} "
                    f"style={style_fields!r} -- run_loop would silently drop "
                    f"every craft and preserve block"
                )
    os.environ.pop("LOW_HIGH_STYLE_DELIVERY")


def test_low_confidence_skips_style() -> None:
    os.environ["LOW_HIGH_STYLE_DELIVERY"] = "inline"
    out = _compose_edit_prompt(
        user_prompt="make it cyberpunk", planner_prompt="", hard_preserve=[],
        style_fields=WORLD, overrides={}, style_family_confidence=0.1,
    )
    assert "Target world:" not in out, (
        "a near-random family match should not restyle the whole frame"
    )
    os.environ.pop("LOW_HIGH_STYLE_DELIVERY")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
