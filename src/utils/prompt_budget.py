"""Fit an edit prompt to the edit model's text-encoder window.

Wan2.1's umT5 prompter pads and truncates every prompt to 512 tokens
(``WanPrompter(text_len=512)``); anything past that never reaches the model.
Prompt composition appends sections after the user's instruction, so overflow
always costs the *last* section first -- which was the "Preserve unchanged" block,
the one part whose loss caused the original skill regression. Measured: 16/30
skill-v2 prompts overflowed, and a gpt-6.1-sol planner plan alone ran to 2,176
tokens, leaving only the intent and the first quarter of the plan visible.

``fit_sections`` makes the trade-off explicit instead of letting truncation pick.
The instruction and the preserve block are never cut; generic craft direction is
dropped line by line first, then the planner's plan is shortened sentence by
sentence. Counting uses the edit model's own tokenizer, so "fits" means fits.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = PROJECT_ROOT / "models" / "Wan2.1-VACE-14B" / "google" / "umt5-xxl"

_SENTENCE_END = re.compile(r"(?<=[.!?。！？；;])\s*")


def token_limit() -> int:
    try:
        return int(os.environ.get("LOW_HIGH_EDIT_PROMPT_TOKEN_LIMIT", "512"))
    except ValueError:
        return 512


@lru_cache(maxsize=1)
def _tokenizer():
    path = os.environ.get("LOW_HIGH_EDIT_TOKENIZER", str(DEFAULT_TOKENIZER))
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is not None:
        return len(tok(text).input_ids)
    # Conservative fallback when the tokenizer is unavailable: umT5 spends roughly
    # one token per CJK character and ~1.3 per English word.
    cjk = len(re.findall(r"[㐀-鿿]", text))
    words = len(re.findall(r"[A-Za-z0-9]+", text))
    return int(cjk + words * 1.3 + len(re.findall(r"[^\w\s]", text)) * 0.5) + 1


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(text.strip()) if part.strip()]


def _join_sentences(parts: list[str]) -> str:
    out = ""
    for part in parts:
        if not out:
            out = part
        elif re.search(r"[㐀-鿿]$", out) or re.match(r"[㐀-鿿]", part):
            out += part
        else:
            out += " " + part
    return out


def fit_sections(
    *,
    instruction: str,
    plan: str = "",
    style_lines: list[str] | None = None,
    craft_lines: list[str] | None = None,
    preserve_lines: list[str] | None = None,
    limit: int | None = None,
) -> tuple[str, dict]:
    """Assemble the prompt within ``limit`` tokens; return (prompt, report).

    Section order is instruction, plan, style, craft, preserve. Cut order is
    craft (from the last line), then plan (from the last sentence), then style
    (from the last line). Instruction and preserve are kept whole; if they alone
    exceed the limit the prompt is returned as-is and the report says so, because
    silently cutting either would reintroduce the exact failure this exists to
    prevent.

    Style is cut last of the three rather than first because it carries the
    target world: for a style-transfer task that is the point of the prompt,
    whereas craft clauses are only execution detail. A caller that is not doing
    style transfer leaves ``style_lines`` empty and gets the previous behaviour.
    """
    limit = token_limit() if limit is None else limit
    style = list(style_lines or [])
    craft = list(craft_lines or [])
    preserve = list(preserve_lines or [])
    plan_parts = _sentences(plan) if plan else []
    report = {
        "limit": limit,
        "dropped_craft_lines": 0,
        "dropped_style_lines": 0,
        "dropped_plan_sentences": 0,
    }

    def render() -> str:
        sections = [instruction]
        if plan_parts:
            sections.append("Detailed edit plan:\n" + _join_sentences(plan_parts))
        if style:
            sections.append("Target world:\n" + "\n".join(f"- {line}" for line in style))
        if craft:
            sections.append("Craft direction:\n" + "\n".join(f"- {line}" for line in craft))
        if preserve:
            sections.append("Preserve unchanged:\n" + "\n".join(f"- {line}" for line in preserve))
        return "\n\n".join(sections)

    text = render()
    report["tokens_before"] = count_tokens(text)
    while count_tokens(text) > limit and craft:
        craft.pop()
        report["dropped_craft_lines"] += 1
        text = render()
    while count_tokens(text) > limit and plan_parts:
        plan_parts.pop()
        report["dropped_plan_sentences"] += 1
        text = render()
    while count_tokens(text) > limit and style:
        style.pop()
        report["dropped_style_lines"] += 1
        text = render()
    report["tokens_after"] = count_tokens(text)
    report["fits"] = report["tokens_after"] <= limit
    return text, report
