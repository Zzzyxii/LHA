#!/usr/bin/env python3
"""Manifest for re-scoring every arm with the original paper metrics.

One row per (arm, sample) in the format the original metric scripts read
(sample_id, method, source_video, target_video, edited_video, instruction).
Every arm gets the *expert* instruction as its text, as in the original table, so
CLIP-T / PickScore compare all arms against the same text regardless of what each
arm was fed. Two anchors are added: ``noop`` (the low clip itself) and ``oracle``
(the high clip itself).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "out"
# The archive bundle under out/hf_eval_bundle holds dangling symlinks to the
# original host; the unpacked release archive has the real files.
PUB = Path("/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/归档_extract/methods")

ARMS = {
    # published videos, 17-frame protocol: calibration against the original table
    "pub_ours_full": PUB / "ours_full",
    "pub_Bernini": PUB / "Bernini-R-1.3B",
    "pub_Ditto": PUB / "Ditto",
    "pub_VACE": PUB / "VACE",
    # our runs, native duration
    "expert_ours_raw": OUT / "bench_skill/run50_native/ours_raw/generations",
    "expert_bernini_raw": OUT / "bench_skill/run50_native/bernini_raw/generations",
    "expert_skill_v1": OUT / "bench_skill/run50_native/ours_skill/generations",
    "expert_skill_v2": OUT / "bench_skill/run30_v2/ours_skill/generations",
    "intent_direct": OUT / "bench_v2/planning/ours_intent_direct/generations",
    "intent_agent": OUT / "bench_v2/planning/ours_intent_agent/generations",
    "generic_direct": OUT / "bench_v2/planning/ours_generic_direct/generations",
    "generic_agent": OUT / "bench_v2/planning/ours_generic_agent/generations",
    "abl_agentprompt": OUT / "bench_v2/ablation/ours_abl_agentprompt/generations",
    "abl_plan_craft": OUT / "bench_v2/ablation/ours_abl_plan_craft/generations",
    "abl_plan_only": OUT / "bench_v2/ablation/ours_abl_plan_only/generations",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=OUT / "bench_v2/manifest.jsonl")
    parser.add_argument("--output", type=Path, default=OUT / "bench_v2/orig_metrics/manifest_all.jsonl")
    args = parser.parse_args()

    base = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for item in base:
        common = {
            "sample_id": item["sample_id"],
            "source_video": item["source_low"],
            "target_video": item["target_high"],
            "instruction": item["expert_instruction"],
        }
        rows.append({**common, "method": "noop", "edited_video": item["source_low"]})
        rows.append({**common, "method": "oracle", "edited_video": item["target_high"]})
        for arm, folder in ARMS.items():
            video = folder / f"{item['sample_id']}.mp4"
            if video.exists():
                rows.append({**common, "method": arm, "edited_video": str(video.resolve())})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["method"]] = counts.get(r["method"], 0) + 1
    print(args.output, len(rows), counts)


if __name__ == "__main__":
    main()
