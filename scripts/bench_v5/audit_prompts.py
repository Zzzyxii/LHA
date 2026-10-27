#!/usr/bin/env python3
"""Reject prompt files that are not prompts.

A Thinking VLM that runs out of `max_new_tokens` before closing its `<think>`
block returns its reasoning, and the reasoning looks enough like an edit
instruction to survive every downstream check: it is English, it is the right
length, it mentions the right nouns. 26/50 prompts of the `grounded_v4` arm were
this. They were rendered, the arm scored well, and the win was an artefact --
rambling input makes this edit model under-edit, and `gain_dino` rewards
under-editing.

Run this on every generated prompt file before rendering it:

    python3 scripts/bench_v5/audit_prompts.py scripts/bench_v4/data/*.jsonl \
        scripts/bench_v5/data/*.jsonl

Exit code 1 if any file has contaminated rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONTAMINATION_MARKERS = (
    "Got it", "let's ", "Let's ", "Step 1", "Step 2", "First, ", "The key is",
    "carrier:", "carriers:", "Wait,", "Hmm", "check each", "break down",
    "user's request", "user request", "Therefore", "Now ", "So ",
)

# A real edit prompt is imperative and short. These are cheap structural checks
# that catch contamination the marker list misses.
MAX_WORDS = 110

MARKERS = CONTAMINATION_MARKERS  # backwards-compatible alias


def audit(path: Path) -> list[tuple[str, str]]:
    bad: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id") or "?")
        text = str(row.get("instruction") or row.get("prompt") or "")
        hits = [m for m in CONTAMINATION_MARKERS if m in text]
        words = len(text.split())
        reason = ""
        if hits:
            reason = "reasoning markers: " + ", ".join(hits[:4])
        elif words > MAX_WORDS:
            reason = f"{words} words (> {MAX_WORDS})"
        elif not text:
            reason = "empty"
        if reason:
            bad.append((sid, reason))
    return bad


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    failed = 0
    for arg in sys.argv[1:]:
        for path in sorted(Path().glob(arg)) if any(c in arg for c in "*?") else [Path(arg)]:
            if not path.exists():
                continue
            bad = audit(path)
            status = "OK" if not bad else f"{len(bad)} CONTAMINATED"
            print(f"{status:<20} {path}")
            for sid, reason in bad[:5]:
                print(f"    {sid}: {reason}")
            failed += bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
