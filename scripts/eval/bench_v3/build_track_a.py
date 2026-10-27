#!/usr/bin/env python3
"""Task file for bench_v3 track A (production-design upgrade, low-high-cost data).

One row per low-high-cost sample with all three input tiers:
  t1 : one-line intent (scripts/eval/bench_v3/data/intents_specific.jsonl)
  t2 : the intent plus a reference still, in two variants
       - same-shot : a mid-clip frame of this sample's own target_high (the answer
                     is reachable, so GT metrics stay meaningful)
       - cross-shot: a mid-clip frame of the *other* sample whose
                     transformation_tags overlap most (closer to real use: "make it
                     feel like this other drama")
  t3 : the expert instruction
plus the checklist (transformation_tags) and a fixed preserve list for the judges.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[3]
PRESERVE = [
    "facial identity of every person",
    "facial expressions",
    "body pose and action",
    "composition and subject placement",
    "camera framing and motion",
]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mid_frame(video: Path, out: Path) -> Path:
    """Save the frame nearest the clip's midpoint as PNG (skips if present)."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        frames = list(container.decode(video=0))
    frames[len(frames) // 2].to_image().convert("RGB").save(out)
    return out


def tag_tokens(tags: list[str]) -> set[str]:
    return {tok for tag in tags for tok in re.split(r"[_\s-]+", tag.lower()) if len(tok) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "scripts/eval/bench_v3/data/track_a.jsonl")
    parser.add_argument("--ref-dir", type=Path, default=ROOT / "out/bench_v3/refs")
    args = parser.parse_args()

    manifest = {r["sample_id"]: r for r in _jsonl(ROOT / "out/bench_v2/manifest.jsonl")}
    sources = [r for r in _jsonl(ROOT / "scripts/eval/bench_v3/data/top50_sources.jsonl") if r["source_dataset"] == "low_high_cost"]
    intents = {r["sample_id"]: r["intent"] for r in _jsonl(ROOT / "scripts/eval/bench_v3/data/intents_specific.jsonl")}
    tokens = {r["sample_id"]: tag_tokens(r["transformation_tags"]) for r in sources}

    rows = []
    for r in sources:
        sid = r["sample_id"]
        m = manifest[sid]
        same = mid_frame(Path(m["target_high"]), args.ref_dir / "same" / f"{sid}.png")
        others = sorted((jaccard(tokens[sid], tokens[o]), o) for o in tokens if o != sid)
        score, partner = others[-1]
        cross = mid_frame(Path(manifest[partner]["target_high"]), args.ref_dir / "cross_src" / f"{partner}.png")
        rows.append({
            "sample_id": sid,
            "track": "A",
            "source_video": m["source_low"],
            "target_video": m["target_high"],
            "t1": intents[sid],
            "t2_same_reference_images": [str(same)],
            "t2_cross_reference_images": [str(cross)],
            "t2_cross_from": partner,
            "t2_cross_tag_jaccard": round(score, 3),
            "t3": m["expert_instruction"],
            "checklist": r["transformation_tags"],
            "preserve": PRESERVE,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    js = sorted(x["t2_cross_tag_jaccard"] for x in rows)
    print(f"{args.output}: {len(rows)} rows; cross-shot tag jaccard median={js[len(js) // 2]} min={js[0]}")


if __name__ == "__main__":
    main()
