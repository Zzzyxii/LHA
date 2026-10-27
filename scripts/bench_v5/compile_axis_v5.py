"""Turn the measured axis residual into prompt edits.

Two arms come out of this file, and the difference between them is the point of
the paper:

* `--mode prior`    rewrite the grounded prompt against the *global* low->high
                    profile (which axes such an upgrade moves, estimated on the
                    calibration split). No information about this clip's own render.
* `--mode feedback` the same, but the residual is `profile - what this clip's
                    round-0 render actually moved`. That is a closed loop: render,
                    measure the axes, rewrite, render again.

Neither mode reads the target clip. The profile is a task-level prior from a frozen
calibration split; the feedback residual is computed from (source, output) only. So
an arm built here can be scored against `target_high` without that target ever
having influenced the prompt.

Why clauses are dropped as well as added: the edit model does less the longer the
prompt gets (measured in bench_v3: change 0.21 -> 0.17 -> 0.12 as sections were
added), so budget spent on an axis that already overshot is budget taken away from
one that under-delivered.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes  # noqa: E402
from audit_prompts import CONTAMINATION_MARKERS, MAX_WORDS  # noqa: E402

WORD_BUDGET = 100

# Which words put a clause on which axis. Only used to attribute an existing clause
# so it can be dropped or upgraded; new clauses come from CLAUSE_BANK.
AXIS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "costume_ornament": ("garment", "robe", "costume", "clothing", "dress", "sleeve",
                         "embroidery", "fabric", "chenyi", "outfit"),
    "headwear": ("headwear", "headdress", "hat", "hairpin", "headpiece", "crown",
                 "liangbatou", "tassel"),
    "jewelry": ("jewelry", "necklace", "earring", "pendant", "ornament", "jade", "bead"),
    "color_grade": ("grade", "grading", "palette", "filmic", "colour", "color"),
    "set_detail": ("background", "wall", "room", "interior", "set", "lattice",
                   "architecture", "partition", "lantern"),
    "production_value": ("production value", "big-budget", "dynamic range",
                         "cinematic exposure", "high-budget"),
    "warm_key": ("warm", "candlelight", "candle", "golden", "lamp"),
    "cinematic_light": ("relight", "key light", "rim light", "lighting", "falloff",
                        "shadow", "motivated"),
    "atmosphere": ("atmosphere", "haze", "mist", "aura", "motes", "glow"),
    "material_detail": ("material", "weave", "grain", "texture", "thread", "surface"),
    "texture_richness": ("texture", "detail", "patina", "carving"),
    "wealth": ("silk", "brocade", "lacquer", "brass", "gold", "carved", "opulent",
               "luxurious", "rich"),
    "skin_finish": ("skin", "complexion", "makeup", "liner", "lip", "pore", "retouch",
                    "flawless", "polished"),
}

# One clause per axis. Written to be addable to any prompt without knowing the shot,
# so they are carriers-neutral: they push the axis, they do not invent props.
CLAUSE_BANK: dict[str, str] = {
    "costume_ornament": "Replace the garments with richly embroidered period costume in heavy woven fabric.",
    "headwear": "Replace the hair accessories with an ornate period headpiece.",
    "jewelry": "Add ornate period jewelry at the neck and in the hair.",
    "color_grade": "Grade with a rich filmic palette and deep shadows.",
    "set_detail": "Dress the background as a detailed period interior.",
    "production_value": "Render it as a big-budget production: fine material detail and deep cinematic exposure.",
    "warm_key": "Motivate the light as warm practical sources.",
    "cinematic_light": "Light with a single motivated key and real falloff into shadow.",
    "atmosphere": "Add a light atmospheric haze between subject and background.",
    "material_detail": "Raise surface detail until fabric weave and wood grain are visible.",
    "texture_richness": "Increase texture detail throughout.",
    "wealth": "Make every visible surface read as expensive.",
    "skin_finish": "Keep skin natural and unretouched, with real texture and no beauty smoothing.",
}

# Axes whose profile term is negative: the clause bank above would push the wrong
# way, so for these the "add" action installs the NEGATIVE clause instead.
NEGATIVE_CLAUSE_BANK: dict[str, str] = {
    "skin_finish": "Keep skin natural and unretouched, with real texture and no beauty smoothing.",
}

# Strengthening happens by intensifying the clause's own verb, not by prefixing a
# stock phrase: "Replace outright, do not tint: replace headwear with ..." reads as
# a stutter and the edit model sees a duplicated verb.
_VERB_UPGRADE = {"replace": "Fully replace", "relight": "Relight fully",
                 "refine": "Refine decisively", "add": "Add clearly",
                 "dress": "Dress fully", "render": "Render fully",
                 "grade": "Grade decisively", "keep": "Keep exactly",
                 "make": "Make unmistakably", "increase": "Increase sharply",
                 "raise": "Raise sharply", "motivate": "Motivate clearly",
                 "light": "Light decisively"}


def strengthen(clause: str) -> str:
    first = clause.split()[0].lower().rstrip(",:")
    strong = _VERB_UPGRADE.get(first)
    if not strong:
        return "Decisively " + clause[0].lower() + clause[1:]
    return strong + clause[len(first):]


def _sentence_case(clause: str) -> str:
    """Re-capitalise a clause that was split mid-sentence ("remove modern lighting")."""
    if not clause:
        return clause
    return clause[0].upper() + clause[1:]


def words(text: str) -> int:
    return len([w for w in text.replace("\n", " ").split() if w])


def split_clauses(prompt: str) -> list[str]:
    """Split on sentence and semicolon boundaries, re-terminating each clause once.

    Splitting on `;` and then appending "." produced "soft falloff;. remove modern
    lighting", which is not a sentence the edit model should be asked to read.
    """
    text = prompt.strip()
    parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", text) if p.strip()]
    return [p.rstrip(";").rstrip(".").rstrip() + "." for p in parts]


# Guardrails are never rewritten. "carry the look through grade and light instead"
# contains the trigger words for color_grade and cinematic_light, so a keyword-only
# rule would strengthen a clause whose entire job is to restrain the edit.
GUARDRAIL_MARKERS = ("only change what is visible", "keep faces", "keep the faces",
                     "not in frame", "unchanged", "number of people", "preserve",
                     "expressions", "poses", "framing", "maintain")


def is_guardrail(clause: str) -> bool:
    low = clause.lower()
    return any(m in low for m in GUARDRAIL_MARKERS)


def clause_axes(clause: str) -> set[str]:
    low = clause.lower()
    return {a for a, kws in AXIS_KEYWORDS.items() if any(k in low for k in kws)}


def looks_contaminated(text: str) -> str:
    """Why this text is not an edit instruction, or "" if it is one.

    The grounded writer is a Thinking VLM: when it runs out of tokens before
    closing <think> it returns reasoning ("Got it, let's break down ..."), which
    reads like a prompt and renders like one. 26/50 of the first grounded arm and
    27/50 of the first axis_v5 / loop_r2 arms were that, and both were rendered
    before anyone noticed (LESSONS.md, "The contamination bug"). Compiling on top
    of a contaminated base silently propagates it, so the check runs on the base
    *and* on the output of every compile.
    """
    hits = [m for m in CONTAMINATION_MARKERS if m in text]
    if hits:
        return "reasoning markers: " + ", ".join(hits[:4])
    if len(text.split()) > MAX_WORDS:
        return f"{len(text.split())} words (> {MAX_WORDS})"
    if not text.strip():
        return "empty"
    return ""


def compile_one(prompt: str, residual: np.ndarray, axis_names: list[str],
                budget: int = WORD_BUDGET, top_add: int = 3, top_drop: int = 2) -> dict:
    """Rewrite `prompt` so the clause set matches the residual's sign and size."""
    clauses = split_clauses(prompt)
    by_axis = {a: [] for a in axis_names}
    untagged = []
    guard = []
    for c in clauses:
        if is_guardrail(c):
            guard.append(c)
            continue
        hits = clause_axes(c)
        if not hits:
            untagged.append(c)
        for a in hits:
            by_axis[a].append(c)

    order = np.argsort(-np.abs(residual))
    want_add = [axis_names[i] for i in order if residual[i] > 0][:top_add]
    want_drop = [axis_names[i] for i in order if residual[i] < 0][:top_drop]

    # The instruction's own subject ("Replace headwear with liangbatou ...") is what
    # grounds the edit in this shot; never drop the whole clause, only the axes the
    # residual says are already overshot, and only if some other clause stays.
    dropped = []
    kept = []
    for c in clauses:
        if c in guard:
            continue
        hits = clause_axes(c)
        if hits and hits <= set(want_drop) and len(untagged) + len(kept) > 2:
            dropped.append(c)
        else:
            kept.append(c)

    for a in want_add:
        if by_axis.get(a):
            # already addressed: strengthen in place rather than spend budget
            for i, c in enumerate(kept):
                if c in by_axis[a]:
                    kept[i] = strengthen(c)
                    break
        else:
            new = CLAUSE_BANK.get(a)
            if new:
                kept.append(new)

    # An axis the profile wants moved *down* still needs saying. Dropping the clause
    # that pushed it up is not enough: the edit model fills the gap with its own
    # default, which is a beautified, smoothed look. Say what to do instead.
    for a in want_drop:
        neg = NEGATIVE_CLAUSE_BANK.get(a)
        if neg and by_axis.get(a):
            kept.append(neg)

    out, used = [], 0
    # Guardrails last and never sacrificed to the budget: they are what keeps a
    # harder push on costume from rewriting the person wearing it.
    for c in kept:
        w = words(c)
        if used + w > budget:
            continue
        out.append(_sentence_case(c))
        used += w
    for c in guard:
        out.append(_sentence_case(c))
    return dict(prompt=" ".join(out), words=used, added=want_add, dropped=want_drop,
                n_dropped=len(dropped))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-prompts", required=True, help="jsonl {sample_id, instruction}")
    ap.add_argument("--profile", required=True, help="axis_profile_calib.json")
    ap.add_argument("--mode", choices=("prior", "feedback"), required=True)
    ap.add_argument("--feedback-scores", default="", help="scores jsonl with <arm>__cv")
    ap.add_argument("--feedback-arm", default="grounded_v4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=WORD_BUDGET)
    ap.add_argument("--top-add", type=int, default=3)
    ap.add_argument("--top-drop", type=int, default=2)
    args = ap.parse_args()

    prof = json.loads(Path(args.profile).read_text())
    axis_names = prof["axis_names"]
    prior = np.array(prof["mean_tp"])

    achieved = {}
    if args.mode == "feedback":
        if not args.feedback_scores:
            raise SystemExit("--feedback-scores is required in feedback mode")
        for l in Path(args.feedback_scores).read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            key = args.feedback_arm + "__cv"
            if key in r:
                v = np.array(r[key])
                if len(v) != len(axis_names):
                    raise SystemExit(
                        f"feedback vector for {r['sample_id']} has {len(v)} axes but the "
                        f"profile has {len(axis_names)}: the scores file was written with a "
                        f"different validated axis set. Re-score with "
                        f"--axis-validation that matches axis_profile_calib.json")
                achieved[r["sample_id"]] = v
        print(f"feedback: residuals for {len(achieved)} samples", file=sys.stderr)

    rows = [json.loads(l) for l in Path(args.base_prompts).read_text().splitlines() if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as fh:
        for r in rows:
            sid = r["sample_id"]
            base = str(r.get("instruction") or r.get("intent") or "")
            why = looks_contaminated(base)
            if why:
                raise SystemExit(f"contaminated base prompt for {sid}: {why}\n"
                                 f"  {base[:200]}")
            if args.mode == "feedback" and sid in achieved:
                residual = prior - achieved[sid]
            else:
                residual = prior.copy()
            res = compile_one(base, residual, axis_names, budget=args.budget,
                              top_add=args.top_add, top_drop=args.top_drop)
            why = looks_contaminated(res["prompt"])
            if why:
                raise SystemExit(f"compiler produced a contaminated prompt for "
                                 f"{sid}: {why}\n  {res['prompt'][:200]}")
            fh.write(json.dumps({"sample_id": sid, "instruction": res["prompt"],
                                 "intent": r.get("intent", ""), "words": res["words"],
                                 "added": res["added"], "dropped": res["dropped"],
                                 "n_dropped_clauses": res["n_dropped"]},
                                ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} prompts -> {out_path} (mode={args.mode})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
