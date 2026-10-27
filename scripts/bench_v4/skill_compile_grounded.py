#!/usr/bin/env python3
"""Grounded skill compiler: intent + source frames -> edit prompt, via a local VLM.

Two things the deterministic compiler (skill_compile.py) cannot do, both of which
bench_v3 measured as the difference between the agent arm and the direct arm:

1. it cannot see the clip, so it emits carrier clauses for things that may not be
   in frame -- the single largest source of "invented detail" failures;
2. it cannot tell that these clips are already *in* the target genre. `source_low`
   is a cheap version of the same shot, so "restyle as a period drama" asks for a
   change the target does not make; what the target does is raise production
   value. The writer is told to upgrade, not to restyle.

Grounding runs on Qwen3-VL-8B-Thinking locally (the litellm gateway is out of
quota), temperature 0, so the arm is still reproducible from (intent, frames, card).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval" / "bench_v3"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio_ffmpeg  # noqa: E402
from PIL import Image  # noqa: E402
from skills_v3 import card_block  # noqa: E402
from skill_compile import OFF_CARRIER, HOLD, _words, _global_clause  # noqa: E402

os.environ["PATH"] = f"{ROOT / 'scripts' / 'bench_skill' / 'bin'}:{os.environ.get('PATH', '')}"

SYSTEM = """You are a prompt engineer writing the final prompt for a video edit model.

Step 1: list what you actually see in the frames -- how many people, their garments,
the setting, the light sources. Maximum 10 short items.
Step 2: write the edit prompt using ONLY those items.

The task is a low-cost -> high-cost upgrade: the source is a cheap version of the
same shot, and the target is the same people doing the same thing with real
production value. So upgrade what is there -- richer materials, finer detail,
motivated cinematic light, cleaner background -- instead of restyling the clip into
a different genre or swapping what the people wear for a different outfit.

Rules for the prompt:
- English, at most 90 words.
- Verb first, one change per clause, most important first.
- Never mention a person, garment, room or object that is not in your Step 1 list.
- No camera moves. No "preserve" boilerplate.

Return ONLY the final prompt as plain English text: no JSON, no headings, no
explanation, no preamble.
"""

WORD_BUDGET = 90
# The Thinking model reasons before it answers and the answer is lost if the
# budget runs out first; 768 lost half the samples. 2048 with a retry above it.
MAX_NEW_TOKENS = 2048
# The measured prior from bench_v3 is "longer prompt -> less change", but the
# grounded arm now changes *less* than the oracle does (0.195 vs 0.214 DINO) while
# still landing short of the target, so the missing ingredient is not restraint.
# --rich spends the extra budget on production value that does not move content:
# light, grade, material detail. That is the part the target actually adds.
RICH_BUDGET = 105
# --faithful is the arm the data asked for. Across the 37 low-high clips the
# per-sample change in `gain_dino` correlates -0.733 with the change in DINO
# change: on the 28 clips where the method changed *less* than the baseline it
# gained +0.0351 (21/7), on the 9 where it changed more it lost -0.0230 (3/6).
# The overshoot is content replacement the target never makes, so this recipe
# forbids swapping elements and spends the edit on fidelity instead.
FAITHFUL_BUDGET = 95
FAITHFUL = ("Do not swap or add any garment, object or background element: raise "
            "the fidelity of what is already there -- fabric weave and thread detail, "
            "skin and makeup finish, set materials, cinematic key and rim light, filmic "
            "grade, shallow depth of field.")
PRODUCTION_VALUE = ("Render it with real production value: motivated key and rim light, "
                    "high dynamic range filmic grade, fine material detail, clean detailed "
                    "background.")


def extract_frames(video: Path, count: int, cache_dir: Path) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / video.stem
    out.mkdir(parents=True, exist_ok=True)
    expected = [out / f"{i}.png" for i in range(count)]
    if all(p.exists() for p in expected):
        return [str(p) for p in expected]
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ff, "-y", "-v", "error", "-i", str(video), "-vf", "fps=1,scale=512:-1",
         "-frames:v", str(count), str(out / "%d.png")],
        capture_output=True, check=False,
    )
    got = sorted(out.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not got:
        raise RuntimeError(f"no frames from {video}")
    for i, dst in enumerate(expected):
        src = got[min(i, len(got) - 1)]
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
    return [str(p) for p in expected]


REASONING_MARKERS = (
    "Got it", "let's ", "Let's ", "Step 1", "Step 2", "First, ", "The key is",
    "carrier:", "carriers:", "Wait,", "Hmm", "check each", "break down",
    "user's request", "user request", "So ", "Now ", "Therefore",
)


def looks_like_reasoning(text: str) -> bool:
    """True when the writer never closed its thinking block.

    With max_new_tokens too small the answer is cut off inside the reasoning, and
    whatever is left is prose about the task instead of a prompt for the edit
    model. That is not a cosmetic problem: 26/50 of the first grounded run were
    this, they were sent to the edit model verbatim, and they are the likeliest
    explanation for that run's numbers (rambling -> the model under-edits ->
    `gain_dino` goes up). Anything that trips this check is rejected, not shipped.
    """
    return any(m in text for m in REASONING_MARKERS)


def _parse(raw: str) -> tuple[list[str], str]:
    """The prompt is whatever follows the model's reasoning block.

    Asking a Thinking model for JSON mostly gets prose with braces in it, and the
    8B writer drifts out of the schema; asking for plain text works and is what
    the edit model needs anyway. `visible` is left empty -- it was logging only.
    """
    text = raw
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    text = re.sub(r"^(final prompt|prompt)\s*[:\-]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    return [], text


class Writer:
    def __init__(self, model_dir: Path, device: str = "cuda") -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        import qwen_vl_utils  # noqa: F401
        self.torch = torch
        self.qwen_vl_utils = sys.modules["qwen_vl_utils"]
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map=device)
        self.model.eval()

    def __call__(self, intent: str, frames: list[str], card: str,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> tuple[list[str], str]:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": f} for f in frames]
            + [{"type": "text", "text": f"User request: {intent}\n\n"
                                        f"The images are frames of the source video.\n\n{card}\n\n"
                                        f"Write the final edit prompt."}],
        }]
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _ = self.qwen_vl_utils.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        raw = self.processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _parse(raw), raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3-VL-8B-Thinking"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default=str(ROOT / "out" / "bench_v4" / "vlm_frames"))
    ap.add_argument("--rich", action="store_true",
                    help="append the production-value clause and allow a longer prompt")
    ap.add_argument("--faithful", action="store_true",
                    help="fidelity-only recipe: no element may be swapped, raise how real "
                         "and well-lit the existing ones look")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.intents).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    # resume: keep rows already produced
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[str(r["sample_id"])] = r

    pending = [r for r in rows if str(r["sample_id"]) not in done]
    if not pending:
        print("nothing to do", file=sys.stderr)
        return 0

    raw_cache = Path(args.cache_dir) / "raw"
    raw_cache.mkdir(parents=True, exist_ok=True)

    writer = Writer(Path(args.model), device=args.device)
    with out_path.open("a", encoding="utf-8") as fh:
        for row in pending:
            sid = str(row["sample_id"])
            intent = str(row.get("intent") or row.get("instruction") or "")
            video = Path(args.sources) / f"{sid}.mp4"
            if not video.exists():
                print(f"missing source {video}", file=sys.stderr)
                continue
            frames = extract_frames(video, args.frames, cache_dir)
            # decisive=True adds "Replace, do not tint / change it fully", which directly
            # contradicts the faithful constraint and won: the writer emitted "Swap upper
            # garments" while the appended clause never made it into the budget. Faithful
            # needs the plain card and the constraint up front, where the model reads first.
            card = card_block(intent, decisive=not args.faithful)
            if args.rich:
                card += ("\nMake the upgrade unmistakable and full: the finished result must look "
                         "shot on a real set with real lighting, not a filtered version of the source.")
            if args.faithful:
                card += ("\nThe reference keeps the same people, garments, objects and place as the "
                         "source. Never introduce or swap an element: only raise how real, detailed "
                         "and well-lit the existing ones look.")
            raw_file = raw_cache / (sid + ".json")
            visible, prompt, raw = [], "", ""
            if raw_file.exists():
                raw = json.loads(raw_file.read_text(encoding="utf-8"))["raw"]
                visible, prompt = _parse(raw)
            if not prompt or looks_like_reasoning(prompt):
                for attempt, budget_tokens in enumerate((MAX_NEW_TOKENS, MAX_NEW_TOKENS * 2)):
                    (visible, prompt), raw = writer(intent, frames, card, max_new_tokens=budget_tokens)
                    if prompt and not looks_like_reasoning(prompt):
                        break
                raw_file.write_text(json.dumps({"raw": raw}, ensure_ascii=False), encoding="utf-8")
            if not prompt:
                # deterministic fallback keeps the arm complete and flags the hole
                prompt = _global_clause(intent) + " " + HOLD + " " + OFF_CARRIER
                print(f"{sid}: VLM parse failed, used fallback", file=sys.stderr)
                print("RAW_TAIL:" + raw[-300:].replace("\n", " "), file=sys.stderr)
            # Guardrails are appended rather than asked of the VLM: raw mode gets
            # no automatic preserve list (only the planner path did), and asking a
            # Thinking model for them produced prose instead of a usable prompt.
            budget = RICH_BUDGET if args.rich else (FAITHFUL_BUDGET if args.faithful else WORD_BUDGET)
            if args.faithful:
                prompt = (FAITHFUL + " " + prompt).strip()
                tail = [HOLD, OFF_CARRIER]
            else:
                tail = ([PRODUCTION_VALUE] if args.rich else []) + [HOLD, OFF_CARRIER]
            for sentence in tail:
                trial = (prompt + " " + sentence).strip()
                if _words(trial) <= budget:
                    prompt = trial
            if _words(prompt) > budget:
                prompt = " ".join(prompt.split()[:budget])
            fh.write(json.dumps(
                {"sample_id": sid, "instruction": prompt, "intent": intent,
                 "visible": visible, "words": _words(prompt)},
                ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{sid}: {len(visible)} visible, {_words(prompt)} words", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
