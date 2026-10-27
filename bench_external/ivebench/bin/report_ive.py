"""Assemble the IVEBench results into one table, with the anchors next to them.

IVEBench's own numbers are computed per arm into a summary.json by eval_ive.sh.
This puts the source anchor (what an untouched clip scores) and the published
reference row beside each arm, because on this benchmark the absolute numbers are
close together -- compliance starts at 0.236 and the best published method reaches
0.25 -- and an arm's score is meaningless without the anchor it started from.

    python3 bench_external/ivebench/bin/report_ive.py --out bench_external/ivebench/RESULTS_IVEBENCH_TABLE.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("overall_semantic_consistency_score", "phrase_semantic_consistency_score",
           "semantic_fidelity_score", "temporal_flickering_score")
SHORT = {"overall_semantic_consistency_score": "OSC",
         "phrase_semantic_consistency_score": "PSC",
         "semantic_fidelity_score": "SF",
         "temporal_flickering_score": "TF"}

# Published short-subset numbers (PUBLISHED.md), all categories. Not a same-subset
# control: context only, and labelled as such in the output.
PUBLISHED = {"Ditto": {"OSC": 0.25, "PSC": 0.24, "SF": 0.89, "TF": 0.97},
             "VACE": {"OSC": 0.23, "PSC": 0.22, "SF": 0.97, "TF": 0.98}}

ARMS = [
    ("style", "src_anchor", "source (anchor)"),
    ("style", "direct", "direct (benchmark prompt)"),
    ("style", "skill", "skill"),
    ("attr", "src_attr_anchor", "source (anchor)"),
    ("attr", "attr_direct", "direct (benchmark prompt)"),
    ("attr", "attr_skill", "skill_attr"),
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {SHORT[k]: v for k, v in d["metrics"].items() if k in SHORT}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root",
                    default="/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt/bench_external/ivebench")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.eval_root)

    rows = []
    for kind, name, label in ARMS:
        d = load(root / f"eval_{name}" / "summary.json")
        if d is None:
            rows.append((kind, label, None))
            continue
        rows.append((kind, label, d))

    anchors = {kind: d for kind, label, d in rows if d and label.startswith("source")}
    L = ["# IVEBench results (generated)\n",
         "Metrics available on this pod: OSC/PSC (compliance, VideoCLIP-XL), "
         "SF (fidelity), TF (quality). No IVEBench total is quoted -- 8 of 12 "
         "metrics need checkpoints that cannot be fetched here.\n"]
    for kind, title in (("style", "Style editing (48 clips)"),
                        ("attr", "Attribute / point editing (40 clips)")):
        L.append(f"\n## {title}\n")
        header = ["arm", "n"] + list(SHORT.values()) + ["dOSC vs source"]
        body = []
        for k, label, d in rows:
            if k != kind:
                continue
            if d is None:
                body.append([label, "-"] + ["-"] * len(SHORT) + ["-"])
                continue
            delta = (f"{d['OSC'] - anchors[kind]['OSC']:+.4f}" if kind in anchors else "-")
            body.append([label, ""] + [f"{d[m]:.4f}" for m in SHORT.values()] + [delta])
        L.append("| " + " | ".join(header) + " |")
        L.append("|" + "|".join("---" for _ in header) + "|")
        for r in body:
            L.append("| " + " | ".join(str(c) for c in r) + " |")
        if kind in anchors:
            L.append("")
            L.append("Published Ditto / VACE (short subset, **all categories** -- "
                     "context, not a same-subset control): "
                     + ", ".join(f"{m}: Ditto {PUBLISHED['Ditto'][m]:.2f} / "
                                 f"VACE {PUBLISHED['VACE'][m]:.2f}" for m in ("OSC", "PSC", "SF", "TF")))
    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
