"""Build the IVEBench *attribute editing* subset: the point-editing half of the
external benchmark.

The style subset tests restyling a whole clip; the project's task description also
includes 定点编辑 (localised, point edits), and IVEBench's `attribute_editing`
category is exactly that: change one object's colour, size or position and leave
everything else alone. It is also the harder test of a skill system, because the
same "be decisive" advice that helps a restyle will wreck a point edit by restyling
the whole frame -- so `compile_ive.py` gets a separate `--mode skill_attr`.

Selection: evenly spaced within each subcategory, deterministic, so the subset is
reproducible without a random seed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

QUOTA = {"color adjustment": 18, "subject scaling": 12, "position change": 10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="ivebenchdb_prompt_short.json")
    ap.add_argument("--out", default="subset_attr.json")
    ap.add_argument("--info-out", default="ive_info_attr.json")
    args = ap.parse_args()

    rows = [r for r in json.loads(Path(args.prompts).read_text())
            if r["category"] == "attribute_editing"]
    by_sub: dict[str, list] = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["id"]):
        by_sub[r["subcategory"]].append(r)

    picked = []
    for sub, want in QUOTA.items():
        have = by_sub.get(sub, [])
        if not have:
            continue
        step = len(have) / want
        idx = sorted({min(len(have) - 1, int(i * step)) for i in range(want)})
        picked.extend(have[i] for i in idx)
    picked.sort(key=lambda x: x["id"])

    Path(args.out).write_text(json.dumps(picked, ensure_ascii=False, indent=1))
    info = [{"id": i + 1, "src_video_name": r["src_video_name"], "category": r["category"],
             "subcategory": r["subcategory"], "source_prompt": r["source_prompt"],
             "edit_prompt": r["edit_prompt"], "target_prompt": r["target_prompt"],
             "target_span": r["target_span"], "target_phrase": r["target_phrase"]}
            for i, r in enumerate(picked)]
    Path(args.info_out).write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"wrote {len(picked)} clips -> {args.out} (+{args.info_out})")
    for sub, n in sorted((s, sum(1 for p in picked if p['subcategory'] == s))
                         for s in QUOTA):
        print(f"  {sub:20s} {n}")
    print("\n".join("  " + p["src_video_name"] for p in picked[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
