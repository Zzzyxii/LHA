#!/usr/bin/env python3
"""Freeze and score a balanced 30-clip IVEBench slice from completed runs.

Selection is determined only by prompt metadata: 15 style clips spread across
subcategories and 15 attribute clips (5 each for color, scale, and position).
The evaluator outputs are joined by video filename; score values never affect
selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = {
    "overall_semantic_consistency_score": "OSC",
    "phrase_semantic_consistency_score": "PSC",
    "semantic_fidelity_score": "SF",
    "temporal_flickering_score": "TF",
}
STYLE_N = 15
ATTR_QUOTAS = {
    "color adjustment": 5,
    "subject scaling": 5,
    "position change": 5,
}
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numeric_id(row: dict) -> int:
    name = row.get("src_video_name") or row.get("video_name") or row.get("sample_id", "")
    return int(Path(name).stem.split("_")[-1])


def video_name(row: dict) -> str:
    name = row.get("src_video_name") or row.get("video_name") or row.get("sample_id", "")
    return name if Path(name).suffix else f"{name}.mp4"


def stratified_evenly(rows: list[dict], n: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["subcategory"]].append(row)
    for group in groups.values():
        group.sort(key=numeric_id)
    picked = []
    keys = sorted(groups)
    while len(picked) < n:
        progressed = False
        for key in keys:
            if groups[key]:
                picked.append(groups[key].pop(0))
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            raise ValueError(f"requested {n} rows but only {len(picked)} available")
    return picked


def evenly_spaced(rows: list[dict], n: int) -> list[dict]:
    ordered = sorted(rows, key=numeric_id)
    if n > len(ordered):
        raise ValueError(f"requested {n} rows but only {len(ordered)} available")
    indices = [int((i + 0.5) * len(ordered) / n) for i in range(n)]
    if len(set(indices)) != n:
        raise ValueError("even-spacing selection produced duplicate indices")
    return [ordered[i] for i in indices]


def parse_arm_specs(specs: list[str] | None, defaults: list[tuple[str, str]]) -> dict[str, str]:
    arms = dict(defaults)
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"arm must be LABEL=EVALUATION_DIR, got: {spec}")
        label, folder = spec.split("=", 1)
        label, folder = label.strip(), folder.strip()
        if not label or not folder:
            raise ValueError(f"arm must have non-empty label and directory: {spec}")
        arms[label] = folder
    if "direct" not in arms or "skill" not in arms:
        raise ValueError("each track must include the direct and skill arms")
    return arms


def load_scores(eval_root: Path, evaluation_dir: str) -> dict[str, dict[str, float]]:
    folder = Path(evaluation_dir)
    if not folder.is_absolute():
        folder = eval_root / folder
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing evaluator summary: {summary_path}")
    csv_path = Path(json.loads(summary_path.read_text())["csv"])
    if not csv_path.exists():
        candidates = [
            eval_root / csv_path,
            folder / csv_path.name,
        ] + sorted(folder.glob("*_eval_results.csv"), reverse=True)
        csv_path = next((candidate for candidate in candidates if candidate.exists()), csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing per-video scores: {csv_path}")
    out = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["video_name"]
            if name in out:
                raise ValueError(f"duplicate video in {csv_path}: {name}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError(f"invalid/sentinel metric for {name} in {csv_path}")
            out[name] = values
    return out


def sign_test_p(values: list[float]) -> float:
    non_ties = [v for v in values if v != 0]
    n = len(non_ties)
    if n == 0:
        return 1.0
    wins = sum(v > 0 for v in non_ties)
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def bootstrap_ci(values: list[float], seed: int = 20261026, reps: int = 20000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(reps))
    return samples[int(0.025 * reps)], samples[min(reps - 1, int(0.975 * reps))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--manifest-out", type=Path, default=None)
    ap.add_argument("--report-out", type=Path, default=None)
    ap.add_argument(
        "--style-arm", action="append", metavar="LABEL=EVAL_DIR",
        help="add/override a style-track evaluated method; repeatable",
    )
    ap.add_argument(
        "--point-arm", action="append", metavar="LABEL=EVAL_DIR",
        help="add/override a point-edit-track evaluated method; repeatable",
    )
    args = ap.parse_args()
    root = args.root
    manifest_out = args.manifest_out or root / "FROZEN30_MANIFEST.jsonl"
    report_out = args.report_out or root / "FROZEN30_SCORECARD.md"
    style_arms = parse_arm_specs(
        args.style_arm, [("direct", "eval_direct"), ("skill", "eval_skill")]
    )
    point_arms = parse_arm_specs(
        args.point_arm, [("direct", "eval_attr_direct"), ("skill", "eval_attr_skill")]
    )

    styles = read_jsonl(root / "ive_direct.jsonl")
    attrs = read_jsonl(root / "ive_prompts_attr.jsonl")
    if len(styles) != 48 or len(attrs) != 40:
        raise ValueError(f"expected current evaluated prompt sets (48 style, 40 attr), got {len(styles)}, {len(attrs)}")
    style_picked = stratified_evenly(styles, STYLE_N)
    attr_picked = []
    for category, quota in ATTR_QUOTAS.items():
        group = [row for row in attrs if row["subcategory"] == category]
        attr_picked.extend(evenly_spaced(group, quota))
    selected = (
        [{"track": "style", **row, "src_video_name": video_name(row)} for row in style_picked]
        + [{"track": "point_edit", **row, "src_video_name": video_name(row)}
           for row in sorted(attr_picked, key=numeric_id)]
    )
    if len({(row["track"], video_name(row)) for row in selected}) != 30:
        raise ValueError("frozen selection contains duplicate clip keys")

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    if manifest_out.exists():
        frozen_rows = read_jsonl(manifest_out)
        if frozen_rows != selected:
            raise ValueError(
                f"source prompts/selection no longer match frozen manifest {manifest_out}; "
                "do not silently replace the benchmark sample set"
            )
    else:
        manifest_out.write_text(manifest_text)
    for track, filename in (("style", "FROZEN30_STYLE.jsonl"),
                            ("point_edit", "FROZEN30_POINT_EDIT.jsonl")):
        track_rows = [row for row in selected if row["track"] == track]
        prompt_rows = [
            {key: row[key] for key in (
                "sample_id", "instruction", "category", "subcategory", "source_prompt",
                "target_prompt", "target_span", "target_phrase"
            )}
            for row in track_rows
        ]
        (root / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompt_rows)
        )
    for track, source_file, filename in (
        ("style", "ive_skill.jsonl", "FROZEN30_STYLE_SKILL.jsonl"),
        ("point_edit", "ive_attr_skill.jsonl", "FROZEN30_POINT_EDIT_SKILL.jsonl"),
    ):
        source_rows = read_jsonl(root / source_file)
        by_id = {row["sample_id"]: row for row in source_rows}
        selected_rows = [row for row in selected if row["track"] == track]
        missing = {row["sample_id"] for row in selected_rows} - by_id.keys()
        if missing:
            raise ValueError(
                f"{source_file} lacks frozen skill prompts for: {', '.join(sorted(missing))}"
            )
        skill_rows = []
        for selected_row in selected_rows:
            skill_row = by_id[selected_row["sample_id"]]
            if skill_row["category"] != selected_row["category"] or (
                skill_row["subcategory"] != selected_row["subcategory"]
            ):
                raise ValueError(
                    f"{source_file} metadata mismatch for {selected_row['sample_id']}"
                )
            for key in ("source_prompt", "target_prompt", "target_span", "target_phrase"):
                if skill_row.get(key) != selected_row.get(key):
                    raise ValueError(
                        f"{source_file} target/source metadata mismatch for "
                        f"{selected_row['sample_id']} ({key})"
                    )
            skill_rows.append(skill_row)
        (root / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skill_rows)
        )

    lines = [
        "# Frozen 30-clip IVEBench scorecard",
        "",
        "Frozen selection: 15 style-editing + 15 point-editing clips. Style selection "
        "round-robins sorted subcategories; point-edit selection takes five evenly "
        "spaced sample IDs in each of color adjustment, subject scaling, and position "
        "change. Selection uses prompt metadata only, never scores. The selection is "
        "retrospective (the 48/40 runs already existed), so this is a compact "
        "exploratory reporting slice, not an independent confirmatory test set.",
        "",
        "All arms use the local IVEBench metrics on the same selected clips. For each "
        "method, the table shows its mean, paired difference against direct, and the "
        "skill/compiler-minus-method difference. The 95% intervals are paired "
        "bootstrap percentile intervals (20,000 resamples); p values are two-sided "
        "exact sign tests of direction, unadjusted across method/metric/track tests. "
        "Intervals estimate mean differences; sign tests are not tests of those means.",
        "",
        f"Frozen manifest: `{manifest_out.name}`.",
        "",
    ]
    for track, label, rows in (
        ("style", "Style transfer (n=15)", style_picked),
        ("point_edit", "Point editing (n=15)", attr_picked),
    ):
        names = {video_name(row) for row in rows}
        arm_specs = style_arms if track == "style" else point_arms
        score_cache = {
            method: load_scores(root, evaluation_dir)
            for method, evaluation_dir in arm_specs.items()
        }
        for method, scores in score_cache.items():
            missing = names - scores.keys()
            if missing:
                raise ValueError(
                    f"{track} method {method} is missing selected clip scores: "
                    + ", ".join(sorted(missing))
                )
        direct_scores = score_cache["direct"]
        skill_scores = score_cache["skill"]
        lines.extend([
            f"## {label}", "",
            "| Metric | Method | Mean | Δ vs direct (95% CI) | p | Skill−method (95% CI) | p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for metric, short in METRICS.items():
            direct_values = [direct_scores[name][metric] for name in sorted(names)]
            skill_values = [skill_scores[name][metric] for name in sorted(names)]
            skill_delta = [s - d for s, d in zip(skill_values, direct_values)]
            skill_ci = bootstrap_ci(skill_delta)
            for method, method_scores in score_cache.items():
                values = [method_scores[name][metric] for name in sorted(names)]
                vs_direct = [v - d for v, d in zip(values, direct_values)]
                skill_vs_method = [s - v for s, v in zip(skill_values, values)]
                direct_ci = bootstrap_ci(vs_direct)
                skill_method_ci = bootstrap_ci(skill_vs_method)
                if method == "direct":
                    direct_cell, direct_p = "—", "—"
                else:
                    direct_cell = f"{mean(vs_direct):+.4f} [{direct_ci[0]:+.4f}, {direct_ci[1]:+.4f}]"
                    direct_p = f"{sign_test_p(vs_direct):.4f}"
                if method == "skill":
                    skill_cell, skill_p = "0 (self)", "—"
                else:
                    skill_cell = (
                        f"{mean(skill_vs_method):+.4f} "
                        f"[{skill_method_ci[0]:+.4f}, {skill_method_ci[1]:+.4f}]"
                    )
                    skill_p = f"{sign_test_p(skill_vs_method):.4f}"
                lines.append(
                    f"| {short} | {method} | {mean(values):.4f} | "
                    f"{direct_cell} | {direct_p} | {skill_cell} | {skill_p} |"
                )
        lines.append("")
        lines.append("Selected IDs: " + ", ".join(sorted(names)) + ".")
        lines.append("")

    extra_style = sorted(set(style_arms) - {"direct", "skill"})
    extra_point = sorted(set(point_arms) - {"direct", "skill"})
    if extra_style or extra_point:
        extra_status = (
            "Additional same-sample evaluated methods are included by track: "
            f"style={', '.join(extra_style) or 'none'}, "
            f"point_edit={', '.join(extra_point) or 'none'}. Their protocol and "
            "model identity must be verified from each render manifest."
        )
    else:
        extra_status = (
            "Independent open-source methods (for example Bernini-R or VACE) do not "
            "yet have outputs on these same clips, so they are absent. Published "
            "all-category IVEBench rows are not substituted for paired results."
        )
    lines.extend([
        "## Scope and next comparison",
        "",
        "This scorecard compares our skill/compiler arm with direct Ditto prompting "
        "and any additional evaluated methods supplied on the same frozen clips.",
        extra_status,
        "",
        "Full 48-style / 40-point paired results remain in `RESULTS_IVEBENCH.md`; "
        "this frozen 30 is a compact, balanced reporting slice, not a replacement "
        "for the full results.",
    ])
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines) + "\n")
    print(f"wrote manifest {manifest_out} ({len(selected)} clips)")
    print(f"wrote scorecard {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
