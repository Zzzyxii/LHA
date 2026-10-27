"""Turn an instruction into per-dimension overrides.

The merged skill system needs to know, for each of the 11 dimensions, whether the
instruction left it alone, asked for it to change, or explicitly asked for it to
be **kept**. Only the third one changes the plan materially: a dimension the user
said to keep must not receive the style's text for that dimension, because the
style's job is to restyle and the instruction's job here is to opt out.

Deterministic by design. The previous design asked a VLM to merge template and
instruction; the model invented edits nobody requested (a 174-character brief came
back 987 characters long, specifying a duvet as a bounce card). Rules cannot
hallucinate and can be tested offline.

How it decides
--------------
A preserve signal ("keep", "unchanged", "保留") is bound to the nearest dimension
named *after* it, not to the whole sentence. Judging the sentence globally is
wrong on the shapes that actually occur:

* "keep the background unchanged, upgrade the costume" -- one hold, one change.
* "make the robe richer but keep the face" -- the hold does not apply to the robe
  even though "keep" is 19 characters away.
* "upgrade the robe, keep the face" -- "keep" is 6 characters *after* "robe" and
  9 before "face". Plain nearest-by-distance binds the robe, which is backwards;
  looking ahead first binds the face.

Tie-breaking is deliberately asymmetric. ``KEEP`` suppresses the style's text for
that dimension, so a false KEEP silently drops part of the requested restyle --
invisible in the output. A false EDIT only adds one execution clause. So a
dimension is only ever marked KEEP when a signal word binds to it, and every
other named dimension falls through to EDIT.
"""

from __future__ import annotations

import re

from skills.dimensions import DIMENSIONS, DimensionPolicy

KEEP = "keep"
EDIT = "edit"

# How far a preserve signal may reach for the dimension it binds to. Wide enough
# for "keep the background" (9 chars), tight enough that both signals in
# "keep the background unchanged, upgrade the costume" bind to "background".
MAX_BIND_DISTANCE = 60

# Verbs that start talking about a change. A hold reaches every dimension named
# after it *until the next one of these*: "keep the clothing and the face" holds
# both, while "keep the clothing, upgrade the face" holds only the clothing.
# Without this, nearest-first binding takes the first noun of an "A and B" list
# and silently drops the rest.
_EDIT_VERBS = (
    "change", "changes", "changed", "replace", "replaces", "replaced",
    "upgrade", "upgrades", "upgraded", "modify", "modifies", "alter", "alters",
    "swap", "swaps", "restyle", "restage", "redesign", "transform", "convert",
    "turn", "turns", "make", "makes", "add", "adds", "remove", "removes",
    "render", "rebuild", "rework",
)
_EDIT_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(v) for v in _EDIT_VERBS), key=len, reverse=True)) + r")"
)
_LEADING_PRESERVE_SIGNALS = (
    "keep", "keeps", "keeping", "preserve", "preserved", "preserving",
    "maintain", "maintained", "retain", "retained", "leave", "stay", "stays",
    "保留", "保持", "维持", "不要改", "别改", "不要动",
)
# Signals that describe the noun they FOLLOW: "the background unchanged".
# Binding these forwards is what makes "keep the background unchanged, upgrade
# the costume" mark the costume as held -- "unchanged" sits after "background"
# and before "costume", so it reaches the wrong one.
_TRAILING_PRESERVE_SIGNALS = (
    "unchanged", "untouched", "not change", "don't change", "do not change",
    "without changing", "same", "不变", "不动", "依旧", "原样",
)


def _signal_spans(text: str) -> list[tuple[int, bool]]:
    """(offset, is_trailing) for every preserve signal, in text order.

    English entries match whole words so that "leave" does not fire on "leaves".
    Chinese entries have no word boundaries to respect -- see
    ``skills.dimensions._keyword_regex``.
    """
    spans: list[tuple[int, bool]] = []
    for signal in _LEADING_PRESERVE_SIGNALS:
        spans.extend(
            (m.start(), False)
            for m in re.finditer(r"\b" + re.escape(signal) + r"\b", text)
        )
        spans.extend((m.start(), False) for m in re.finditer(re.escape(signal), text))
    for signal in _TRAILING_PRESERVE_SIGNALS:
        spans.extend(
            (m.start(), True)
            for m in re.finditer(r"\b" + re.escape(signal) + r"\b", text)
        )
        spans.extend((m.start(), True) for m in re.finditer(re.escape(signal), text))
    return sorted(spans)


def _dimension_spans(text: str) -> list[tuple[int, str]]:
    """(offset, dimension) for every trigger-keyword occurrence, in text order."""
    spans = [
        (start, dimension)
        for dimension, policy in DIMENSIONS.items()
        for start, _end in policy.spans(text)
    ]
    return sorted(spans)


def _bind_signals(
    signal_spans: list[tuple[int, bool]],
    dimension_spans: list[tuple[int, str]],
    text: str,
) -> set[str]:
    """Each preserve signal holds the nearest dimension named *after* it.

    Directional on purpose. In "upgrade the robe, keep the face" the word "keep"
    sits 6 characters after "robe" and 9 before "face", so nearest-by-absolute-
    distance binds the robe -- exactly backwards. English puts what is being held
    after the verb, so look ahead first and only fall back to looking behind when
    nothing follows the signal.
    """
    held: set[str] = set()
    edits = sorted(match.start() for match in _EDIT_PATTERN.finditer(text))

    for signal_start, trailing in signal_spans:
        if trailing:
            pairs = [
                (signal_start - offset, dimension)
                for offset, dimension in dimension_spans
                if offset < signal_start
            ]
        else:
            # Everything named after the hold, up to the next change verb.
            after = [start for start in edits if start > signal_start]
            horizon = min(after) if after else len(text)
            pairs = [
                (offset - signal_start, dimension)
                for offset, dimension in dimension_spans
                if signal_start <= offset < horizon
            ]
            if not pairs:
                pairs = [
                    (signal_start - offset, dimension)
                    for offset, dimension in dimension_spans
                    if offset < signal_start
                ]
        pairs = [
            (distance, dimension)
            for distance, dimension in pairs
            if distance <= MAX_BIND_DISTANCE
        ]
        if not pairs:
            continue
        if trailing:
            # A trailing signal describes one thing, the nearest before it.
            held.add(min(pairs)[1])
        else:
            # A leading signal holds the whole list it introduces.
            held.update(dimension for _distance, dimension in pairs)
    return held


def parse_overrides(
    instruction: str,
    preserve_texts: list[str] | None = None,
) -> dict[str, str]:
    """Map each named dimension to ``KEEP`` or ``EDIT``.

    ``preserve_texts`` is the planner's free-text ``hard_preserve_constraints``
    list. Every dimension named there is a hold: that list exists precisely to
    record what must not change, so there is nothing to disambiguate.

    Dimensions absent from the result were not named at all and fall back to the
    style's own text for that dimension.
    """
    lowered = (instruction or "").lower()
    dimension_spans = _dimension_spans(lowered)

    held = _bind_signals(_signal_spans(lowered), dimension_spans, lowered)

    for text in preserve_texts or []:
        held |= {dimension for _offset, dimension in _dimension_spans(str(text).lower())}

    named = {dimension for _offset, dimension in dimension_spans}
    edit = named - held
    return {dimension: KEEP for dimension in sorted(held)} | {
        dimension: EDIT for dimension in sorted(edit)
    }


def keep_dimensions(instruction: str, preserve_texts: list[str] | None = None) -> list[str]:
    """Dimensions the instruction explicitly asked to hold, in emission order."""
    overrides = parse_overrides(instruction, preserve_texts)
    return sorted(
        (dimension for dimension, verdict in overrides.items() if verdict == KEEP),
        key=lambda dimension: DIMENSIONS[dimension].order,
    )


def policy_for(dimension: str) -> DimensionPolicy | None:
    return DIMENSIONS.get(dimension)
