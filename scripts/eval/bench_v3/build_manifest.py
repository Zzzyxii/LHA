#!/usr/bin/env python3
"""Build the corrected bench manifest from the archived 50-sample release.

The archived bench (``归档.zip``) is kept as the source of truth for videos and
expert instructions; this only adds the fields the original manifest lacked:

* ``category`` -- what kind of edit the sample asks for. The original release
  mixes content rewrites with deliberately lo-fi style filters (VHS, film grain,
  black-and-white). Those pull in opposite directions for a low-cost to
  high-cost story, so results are reported per category instead of pooled.
* ``story_aligned`` -- False for the lo-fi filters, whose "high" target is a
  degradation of production value rather than an increase.
* ``in_release30`` -- membership in the 30-sample ``manifest.jsonl`` subset,
  so v2 numbers can be read on the same samples as earlier results.

Categorisation is a keyword heuristic over the instruction text. It is recorded
with its rule so a reviewer can override individual rows rather than trust it.

``user_intent`` is left empty here; the planning track fills it from a separate,
reviewed file so that the intents are fixed before any arm is generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LOFI_KEYWORDS = (
    "vhs", "home video", "home movie", "film grain", "grainy", "retro", "sepia",
    "faded", "1950", "1970", "1980", "vintage", "black and white", "black-and-white",
    "monochrome", "noir", "tracking lines", "flicker",
)
FILTER_KEYWORDS = (
    "style the entire", "transform the entire", "apply a", "convert the scene",
    "color grade", "tint", "aesthetic", "dream sequence", "pastel",
)
CONTENT_KEYWORDS = (
    "replace", "robe", "headpiece", "garment", "costume", "backdrop", "background",
    "makeup", "earring", "collar", "embroider", "add ",
)


def categorize(instruction: str) -> tuple[str, str]:
    text = instruction.lower()
    lofi = [k for k in LOFI_KEYWORDS if k in text]
    content = [k for k in CONTENT_KEYWORDS if k in text]
    filters = [k for k in FILTER_KEYWORDS if k in text]
    if len(content) >= 2 and not lofi:
        return "content_rewrite", f"content keywords {content[:3]}"
    if lofi:
        return "lofi_filter", f"lo-fi keywords {lofi[:3]}"
    if filters:
        return "style_filter", f"filter keywords {filters[:3]}"
    return "other", "no rule matched"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bench_v2 manifest.")
    parser.add_argument("--release-root", type=Path, required=True,
                        help="extracted 归档.zip (has sample_index.json, manifest.jsonl, source_low/, target_high/)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.release_root
    index = json.loads((root / "sample_index.json").read_text(encoding="utf-8"))
    release30 = {
        json.loads(line)["sample_id"]
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    rows = []
    for item in index:
        sample_id = item["sample_id"]
        low = root / "source_low" / f"{sample_id}.mp4"
        high = root / "target_high" / f"{sample_id}.mp4"
        if not low.exists() or not high.exists():
            raise FileNotFoundError(f"{sample_id}: missing low or high video under {root}")
        category, rule = categorize(item["instruction"])
        rows.append(
            {
                "sample_id": sample_id,
                "source_low": str(low.resolve()),
                "target_high": str(high.resolve()),
                "expert_instruction": item["instruction"],
                "user_intent": "",
                "category": category,
                "category_rule": rule,
                "story_aligned": category != "lofi_filter",
                "in_release30": sample_id in release30,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter

    counts = Counter(row["category"] for row in rows)
    print(f"wrote {len(rows)} rows -> {args.output}")
    print("categories:", dict(counts))
    print("story_aligned:", sum(row["story_aligned"] for row in rows), "/", len(rows))
    print("in_release30:", sum(row["in_release30"] for row in rows))


if __name__ == "__main__":
    main()
