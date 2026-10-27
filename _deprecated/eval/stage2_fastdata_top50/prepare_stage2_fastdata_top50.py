#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FAMILIES = [
    (
        "easyvfx",
        "stage2_exp1_step7000_ours_easyvfx_full_only_8gpu_save1k_fastdata",
        "none",
    ),
    (
        "textmem",
        "stage2_exp1_step7000_ours_text_hint_mem_only_8gpu_save1k_fastdata",
        "text+vace_hint",
    ),
    (
        "auxall",
        "stage2_exp1_step7000_ours_text_hint_mem_auxall_8gpu_save1k_fastdata",
        "text+vace_hint",
    ),
    (
        "allnew",
        "stage2_exp1_step7000_ours_all_new_methods_8gpu_save1k_fastdata",
        "text+vace_hint",
    ),
]

PATH_MAP = [
    ("/mnt/si002961ale4/default/lgy/shiying", "/home/l/liyj/shiying"),
    ("/mnt/a100b/default/lgy/shiying", "/home/l/liyj/shiying"),
    ("/home/l/liyj/shiying", "/mnt/si002961ale4/default/lgy/shiying"),
    ("/mnt/scratch/l/liyj", "/mnt/si002961ale4/default/lgy/shiying"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def remap_path(value: str) -> str:
    original = Path(value).expanduser()
    if original.exists():
        return str(original)
    result = value
    for old, new in PATH_MAP:
        if result.startswith(old):
            candidate = new + result[len(old) :]
            if Path(candidate).expanduser().exists():
                return candidate
            result = candidate
    return result


def resolve_sample_video(row: dict[str, Any], key: str, sample_index: Path) -> Path:
    sample_id = str(row["sample_id"])
    raw = Path(remap_path(str(row[key]))).expanduser()
    if raw.is_file():
        return raw.resolve()

    local_subdir = "source_low" if key == "source_video" else "target_high"
    local = sample_index.parent / "videos" / local_subdir / f"{sample_id}.mp4"
    if local.is_file():
        return local.resolve()

    return raw


def step_key(path: Path) -> int:
    match = re.search(r"step-(\d+)\.safetensors$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=50)
    args = parser.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    samples = read_json(args.sample_index)
    if not isinstance(samples, list):
        raise TypeError(f"expected list in {args.sample_index}")
    if len(samples) != args.expected_samples:
        raise RuntimeError(f"expected {args.expected_samples} samples, found {len(samples)}")

    eval_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in samples:
        sample_id = str(row["sample_id"])
        source = resolve_sample_video(row, "source_video", args.sample_index)
        target = resolve_sample_video(row, "target_video", args.sample_index)
        if not source.is_file():
            missing.append(f"source {sample_id}: {source}")
        if not target.is_file():
            missing.append(f"target {sample_id}: {target}")
        base = {
            "sample_id": sample_id,
            "instruction": row["instruction"],
            "source_video": str(source),
            "target_video": str(target),
        }
        eval_rows.append(base)
        high_rows.append({**base, "method": "high", "edited_video": str(target)})
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing Top50 videos ({len(missing)}):\n{preview}")

    write_jsonl(args.run_root / "eval_inputs.top50.jsonl", eval_rows)
    write_jsonl(args.run_root / "high_manifest.top50.jsonl", high_rows)

    methods: list[dict[str, Any]] = []
    ckpt_base = args.checkpoint_root / "checkpoints" / "stage2_exp1_step7000_fastdata"
    missing_families: list[str] = []
    for short, directory, latent_mode in FAMILIES:
        ckpt_dir = ckpt_base / directory
        checkpoints = sorted(ckpt_dir.glob("step-*.safetensors"), key=step_key)
        if not checkpoints:
            missing_families.append(directory)
            continue
        for ckpt in checkpoints:
            step = step_key(ckpt)
            methods.append(
                {
                    "method": f"{short}_s{step}",
                    "family": short,
                    "source_dir": directory,
                    "step": step,
                    "latent_memory_mode": latent_mode,
                    "checkpoint": str(ckpt),
                }
            )

    if not methods:
        raise RuntimeError(f"no checkpoints found under {ckpt_base}")

    (args.run_root / "methods.json").write_text(
        json.dumps(methods, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (args.run_root / "methods.tsv").open("w", encoding="utf-8") as handle:
        for item in methods:
            handle.write(
                "\t".join(
                    [
                        item["method"],
                        item["family"],
                        item["latent_memory_mode"],
                        item["checkpoint"],
                    ]
                )
                + "\n"
            )
    (args.run_root / "expected_methods.txt").write_text(str(len(methods)) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "samples": len(eval_rows),
                "methods": len(methods),
                "missing_families": missing_families,
                "run_root": str(args.run_root),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
