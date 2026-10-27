#!/usr/bin/env python3
"""Write a run_bench.py --instruction-file (and sample index) for one input tier of a bench_v3 track.

Tiers map to track-file fields:
  t1        -> intent = t1
  t2_same   -> intent = t1, reference_images = t2_same_reference_images   (track A)
  t2_cross  -> intent = t1, reference_images = t2_cross_reference_images  (track A)
  t2        -> intent = t1, reference_images = t2_reference_images        (track B)
  t3        -> intent = t3
The sample index written next to it carries source_video, so run_bench can be
called without --source-dir for tracks whose sample ids are not source names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REF_FIELD = {
    "t2_same": "t2_same_reference_images",
    "t2_cross": "t2_cross_reference_images",
    "t2": "t2_reference_images",
}


def build(rows: list[dict], tier: str) -> list[dict]:
    out = []
    for r in rows:
        item = {"sample_id": r["sample_id"], "source_video": r["source_video"]}
        if tier == "t3":
            item["intent"] = r["t3"]
        else:
            item["intent"] = r["t1"]
        if tier in REF_FIELD:
            refs = r.get(REF_FIELD[tier]) or []
            if not refs:
                raise SystemExit(f"{r['sample_id']}: tier {tier} needs {REF_FIELD[tier]}")
            item["reference_images"] = refs
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--tier", required=True, choices=["t1", "t2_same", "t2_cross", "t2", "t3"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--intent-file", type=Path, default=None,
                        help="jsonl {sample_id, intent} overriding the tier's intent (e.g. precomputed agent prompts); "
                        "written as <stem>.<tier>_<name> where name is --name")
    parser.add_argument("--name", default="", help="suffix for the output stem when --intent-file is used")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.track.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = build(rows, args.tier)
    stem = f"{args.track.stem}.{args.tier}"
    if args.intent_file:
        lines = args.intent_file.read_text(encoding="utf-8").splitlines()
        override = {r["sample_id"]: r["intent"] for r in (json.loads(line) for line in lines if line.strip())}
        missing = [x["sample_id"] for x in items if x["sample_id"] not in override]
        if missing:
            raise SystemExit(f"--intent-file lacks {len(missing)} samples, e.g. {missing[:3]}")
        items = [{**x, "intent": override[x["sample_id"]]} for x in items]
        stem += f"_{args.name or 'override'}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{stem}.instructions.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    (args.output_dir / f"{stem}.index.jsonl").write_text(
        "".join(json.dumps({"sample_id": x["sample_id"], "source_video": x["source_video"]}) + "\n" for x in items),
        encoding="utf-8")
    print(f"{stem}: {len(items)} rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
