#!/usr/bin/env python3
"""Build the bench_v3 task-type-B (style / look transfer) dataset.

Two tracks, one row per sample, each row carrying the three input tiers of
out/TASK_DEFINITION_DRAFT.md:

  B_opuscar : 6 OPUSCAR 98 film looks (zyx/styles.json) x 8 live-action source
              clips = 48 rows. t1 = film title, t2 = the OPUSCAR still,
              t3 = the style prompt rewritten by an LLM into a look-only spec.
  B_ditto   : the 13 Ditto-1M rows of top50. t1 = the bench_v2 one-line intent,
              t2 = a mid-clip frame of target_high, t3 = the original instruction.

LLM calls (gemini via LiteLLM) are cached under out/bench_v3/style_track/cache/,
keyed by a hash of the request, so a rerun with a warm cache makes no calls and
writes byte-identical output.

    source /root/.low_high_judge.env
    python3 scripts/eval/bench_v3/build_style_track.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/eval/ditto_quant"))

ZYX = ROOT / "zyx"
TOP50 = ROOT / "reports/top50_video_bundle_20260725_nodup"
SOURCES = ROOT / "scripts/eval/bench_v3/data/top50_sources.jsonl"
INTENTS = ROOT / "scripts/eval/bench_v3/data/intents_specific.jsonl"
MANIFEST_V2 = ROOT / "out/bench_v2/manifest.jsonl"
OUT_DIR = ROOT / "out/bench_v3/style_track"
DATA_DIR = ROOT / "scripts/eval/bench_v3/data"

# (OPUSCAR no, style_id, why it is in the subset). Chosen so a video edit model
# applied to live-action footage could plausibly realise the look, and so the six
# span tinted B&W, hard B&W, warm film stock, modern neon grade, paint and print.
STYLES = [
    (1, "s01_wings_tinted_ortho_bw",
     "Tinted silent-era B&W: orthochromatic contrast with amber-tinted whites and projector wear; pure grade + texture."),
    (66, "s66_schindler_hard_bw",
     "High-contrast cold-neutral B&W with heavy silver grain; the hard B&W counterpart to the tinted one."),
    (49, "s49_rocky_70s_35mm",
     "Warm 1970s 35mm stock: warm highlights, cool shadows, fine grain; a colour film-stock look on live action."),
    (78, "s78_crash_sodium_neon",
     "Modern orange/cyan night grade (sodium orange vs cyan neon, bloom, moving grain): the teal-orange slot."),
    (45, "s45_godfather_rembrandt_oil",
     "Painterly: Rembrandt chiaroscuro oil paint with impasto and amber varnish, readable on faces and interiors."),
    (42, "s42_warhol_silkscreen",
     "Print/poster: photo-silkscreen halftone over offset flat colour blocks on silver; works on photographic sources."),
]

# 8 of the 37 low-high-cost low clips, picked from a mid-frame contact sheet for
# shot variety and clean frames (clips with visible mask/inpaint blobs such as
# 0346, 0272, 0666, 0856, 0977, 0357 were skipped).
SOURCE_IDS = [
    ("heldout1000_0133", "two women, close two-shot, warm interior"),
    ("heldout1000_0526", "two women, medium shot, modern interior with shelves"),
    ("heldout1000_0445", "man in a cap, single close-up, outdoor bokeh"),
    ("heldout1000_0374", "woman lying down, soft-light profile close-up"),
    ("heldout1000_0679", "man in a dark office, low-key close-up"),
    ("heldout1000_0747", "wide interior, a group of people in a living room"),
    ("heldout1000_0754", "outdoor garden, three people, medium-wide, greens"),
    ("heldout1000_0264", "outdoor street, several people walking, daylight"),
]

PRESERVE = [
    "the people's identity and likeness",
    "facial expressions",
    "body pose and actions",
    "scene composition and subject placement",
    "camera framing and motion",
]
# Appended by the script (not written by the LLM) so the LLM text can be checked
# for camera words while the final spec still tells the model to keep the camera.
KEEP_SENTENCE = (
    "Keep the people, their identity, expressions and actions, the composition, "
    "and the camera framing and motion of the source clip unchanged."
)

BANNED = re.compile(
    r"\b(camera|cameras|cut|cuts|cutting|shot|shots|pan|pans|panning|push(?:es|ing)?|"
    r"pull(?:s|ing)?(?:-back)?|zoom|orbit|steadicam|tilt|dolly|handheld|tracking|iris|"
    r"transition|transitions|title|titles|intertitle|intertitles|clawd|hero|story|pov|"
    r"signature|repeat(?:ed|s|ing)?|repetition|side by side)\b",
    re.I,
)
MAX_WORDS = 120

SPEC_SYSTEM = """You turn a film-style description written for an animated short into a look-only grading/texture spec for editing an EXISTING live-action video clip.
Return strict JSON only."""

SPEC_USER = """Film: {title} ({year}). Style name: {style}.

ORIGINAL STYLE PROMPT (written for a different, animated piece):
{prompt}

Write a LOOK-ONLY spec that a video editing model can apply to an existing live-action clip of real people.
Keep: colour grade, palette (hex values are fine), contrast curve, black/white levels, tint, saturation, grain, film/print texture, wear, bloom/halation, vignette, rendering medium (e.g. oil paint, halftone print) and its surface texture.
Drop entirely: camera moves and lens choices, shots, cuts, edits, transitions, iris openings, title cards or text, staging, sets, props, weather, story content, any named character, the word "hero", any rule about one subject keeping colour (apply the look to the whole frame), and any rule that repeats, duplicates or rearranges figures.
Do not mention the camera at all; a fixed sentence about preserving the source will be appended separately.
Write it as a direct instruction ("Grade the clip as ..."), at most 90 English words.

Also write a checklist of 4-7 short, visually verifiable statements about the edited frames that would be true if the look was applied (e.g. "the image is black and white", "whites are tinted toward pale amber", "visible film grain"). Look items only, no preservation items, no camera words.

Return JSON: {{"spec": "...", "checklist": ["...", "..."]}}"""

DITTO_SYSTEM = """You extract verifiable visual checks from a video style-editing instruction. Return strict JSON only."""

DITTO_USER = """EDIT INSTRUCTION:
{instruction}

Write 3-6 short, visually verifiable statements about the edited frames that would be true if this instruction was followed (look/style items only: palette, tint, grain, texture, medium, lighting mood). No preservation items, no camera words.

Return JSON: {{"checklist": ["...", "..."]}}"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    return json.loads(match.group(0))


class CachedLLM:
    """LiteLLM judge wrapper that caches raw responses on disk by request hash."""

    def __init__(self, cache_dir: Path, model: str, reasoning_effort: str, offline: bool) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.offline = offline
        self._judge = None
        self.calls = 0

    def _client(self):
        if self._judge is None:
            from evaluate_aesthetic_reward_litellm import LiteLLMJudge

            api_key = os.environ.get("LOW_HIGH_JUDGE_API_KEY", "").strip()
            if not api_key:
                raise SystemExit("cache miss and LOW_HIGH_JUDGE_API_KEY is not set (source /root/.low_high_judge.env)")
            self._judge = LiteLLMJudge(
                base_url=os.environ.get("LOW_HIGH_JUDGE_BASE_URL", "http://litellm-sgp.woa.com"),
                api_key=api_key, model=self.model, reasoning_effort=self.reasoning_effort,
                temperature=0.0, max_completion_tokens=8000,
            )
        return self._judge

    def complete(self, system: str, user: str) -> str:
        key = hashlib.sha256(json.dumps([self.model, self.reasoning_effort, system, user]).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        if self.offline:
            raise SystemExit(f"--offline but cache miss for {key}")
        response = self._client().complete(system_prompt=system, user_text=user, images_b64=[])
        self.calls += 1
        path.write_text(json.dumps({"model": self.model, "system": system, "user": user, "response": response},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        return response


def check_look_text(text: str, what: str) -> None:
    hit = BANNED.search(text)
    assert hit is None, f"{what}: banned word {hit.group(0)!r} in {text!r}"


def build_spec(llm: CachedLLM, style: dict, attempts: int = 3) -> tuple[str, list[str]]:
    user = SPEC_USER.format(title=style["title_en"], year=style["year"], style=style["style_en"], prompt=style["prompt_en"])
    feedback = ""
    for attempt in range(attempts):
        data = _parse_json(llm.complete(SPEC_SYSTEM, user + feedback))
        spec = " ".join(str(data.get("spec", "")).split())
        checklist = _clean_items(data.get("checklist", []))
        problems = _checklist_problems(checklist, 4, 7)
        if BANNED.search(spec):
            problems.append(f"the spec contains the forbidden word {BANNED.search(spec).group(0)!r}")
        full = f"{spec} {KEEP_SENTENCE}"
        if len(full.split()) > MAX_WORDS:
            problems.append(f"spec is {len(spec.split())} words, keep it under {MAX_WORDS - len(KEEP_SENTENCE.split())}")
        if not problems:
            return full, checklist
        feedback = _rejection(attempt, problems)
    raise SystemExit(f"style {style['no']}: no valid spec after {attempts} attempts: {problems}")


def build_ditto_checklist(llm: CachedLLM, sample_id: str, instruction: str, attempts: int = 3) -> list[str]:
    # Words the instruction itself uses are fair game in its checklist ("VHS
    # tracking lines" is a look, not a camera move); only new ones are banned.
    allowed = {m.group(0).lower() for m in BANNED.finditer(instruction)}
    user = DITTO_USER.format(instruction=instruction)
    feedback = ""
    for attempt in range(attempts):
        checklist = _clean_items(_parse_json(llm.complete(DITTO_SYSTEM, user + feedback)).get("checklist", []))
        problems = _checklist_problems(checklist, 3, 6, allowed)
        if not problems:
            return checklist
        feedback = _rejection(attempt, problems)
    raise SystemExit(f"{sample_id}: no valid checklist after {attempts} attempts: {problems}")


def _clean_items(items) -> list[str]:
    return [" ".join(str(x).split()) for x in items if str(x).strip()]


def _banned_hits(text: str, allowed: set[str] = frozenset()) -> list[str]:
    return [m.group(0) for m in BANNED.finditer(text) if m.group(0).lower() not in allowed]


def _checklist_problems(checklist: list[str], low: int, high: int, allowed: set[str] = frozenset()) -> list[str]:
    problems = []
    bad = {c: _banned_hits(c, allowed) for c in checklist if _banned_hits(c, allowed)}
    if bad:
        # The ban is lexical, so "light cuts through haze" trips it too; asking for
        # a rewording is cheaper than a context-aware filter and keeps one rule.
        problems.append(f"checklist items use forbidden words; reword them: {bad}")
    if not low <= len(checklist) <= high:
        problems.append(f"checklist has {len(checklist)} items, need {low}-{high}")
    return problems


def _rejection(attempt: int, problems: list[str]) -> str:
    return f"\n\nYour previous answer (attempt {attempt + 1}) was rejected: " + "; ".join(problems) + ". Fix it."


def mid_frame_png(video: Path, out: Path) -> Path:
    import av

    if out.exists():
        return out
    with av.open(str(video)) as container:
        frames = [f for f in container.decode(video=0)]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[len(frames) // 2].to_image().convert("RGB").save(out, format="PNG")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LOW_HIGH_JUDGE_MODEL", "gemini-3.8-flash"))
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--offline", action="store_true", help="fail on any cache miss instead of calling the LLM")
    args = parser.parse_args()

    llm = CachedLLM(OUT_DIR / "cache", args.model, args.reasoning_effort, args.offline)
    styles = {s["no"]: s for s in json.loads((ZYX / "styles.json").read_text(encoding="utf-8"))}
    low_high = {r["sample_id"] for r in _jsonl(SOURCES) if r["source_dataset"] == "low_high_cost"}
    ditto_ids = [r["sample_id"] for r in _jsonl(SOURCES) if r["source_dataset"] == "ditto"]
    v2 = {r["sample_id"]: r for r in _jsonl(MANIFEST_V2)}

    # --- B_opuscar -----------------------------------------------------------
    opuscar_rows = []
    for no, style_id, _why in STYLES:
        style = styles[no]
        still = (ZYX / style["frame"]).resolve()
        assert still.is_file(), still
        spec, checklist = build_spec(llm, style)
        check_look_text(spec.replace(KEEP_SENTENCE, ""), f"style {no} spec")
        for item in checklist:
            check_look_text(item, f"style {no} checklist")
        year = re.match(r"\d{4}", style["year"]).group(0)
        for source_id, _desc in SOURCE_IDS:
            assert source_id in low_high, f"{source_id} is not a low-high-cost sample"
            source_video = (TOP50 / "videos/source_low" / f"{source_id}.mp4").resolve()
            assert source_video.is_file(), source_video
            opuscar_rows.append({
                "sample_id": f"{style_id}__{source_id}",
                "track": "B_opuscar",
                "source_id": source_id,
                "source_video": str(source_video),
                "style_no": no,
                "style_name": style["style_en"],
                "t1": f"Make this video look like {style['title_en']} ({year}).",
                "t2_reference_images": [str(still)],
                "t3": spec,
                "checklist": checklist,
                "preserve": list(PRESERVE),
            })

    # --- B_ditto -------------------------------------------------------------
    intents = {r["sample_id"]: r["intent"] for r in _jsonl(INTENTS)}
    instructions = {r["sample_id"]: r["instruction"] for r in json.loads((TOP50 / "sample_index.json").read_text(encoding="utf-8"))}
    ditto_rows = []
    for sid in ditto_ids:
        row = v2[sid]
        target = Path(row["target_high"])
        ref = mid_frame_png(target, OUT_DIR / "refs/ditto" / f"{sid}.png")
        checklist = build_ditto_checklist(llm, sid, instructions[sid])
        allowed = {m.group(0).lower() for m in BANNED.finditer(instructions[sid])}
        for item in checklist:
            assert not _banned_hits(item, allowed), f"{sid} checklist: {item!r}"
        ditto_rows.append({
            "sample_id": sid,
            "track": "B_ditto",
            "source_video": row["source_low"],
            "target_video": row["target_high"],
            "t1": intents[sid],
            "t2_reference_images": [str(ref.resolve())],
            "t3": instructions[sid],
            "checklist": checklist,
            "preserve": list(PRESERVE),
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("style_track.jsonl", opuscar_rows), ("ditto_track.jsonl", ditto_rows)):
        (DATA_DIR / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"B_opuscar rows={len(opuscar_rows)} (styles={len(STYLES)} x sources={len(SOURCE_IDS)})")
    print(f"B_ditto rows={len(ditto_rows)}")
    print(f"LLM calls this run={llm.calls}")


if __name__ == "__main__":
    main()
