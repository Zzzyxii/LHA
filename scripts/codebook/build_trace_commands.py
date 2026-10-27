#!/usr/bin/env python3
"""Build shell commands for vace_hint routing trace runs from metadata CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
import json
import shlex
from pathlib import Path


def infer_category(prompt: str) -> str:
    text = prompt.lower()
    if any(x in text for x in ["sci-fi", "sci_fi", "science fiction", "cyberpunk", "futuristic", "wormhole", "portal", "quantum"]):
        return "sci_fi"
    if any(x in text for x in ["ancient", "hanfu", "wuxia", "xianxia", "imperial", "palace", "period", "costume drama"]):
        return "period_fantasy"
    if any(x in text for x in ["costume", "wardrobe", "dress", "jewelry", "accessory", "armor", "makeup"]):
        return "costume_appearance"
    if any(x in text for x in ["lighting", "cinematic", "atmosphere", "color grading", "grand", "epic", "background"]):
        return "scene_lighting"
    return "unknown"


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_ditto_prompt_remap(json_paths: list[Path], low_roots: list[str]) -> dict[str, deque[dict[str, str]]]:
    if not json_paths:
        return {}
    if len(low_roots) == 1 and len(json_paths) > 1:
        low_roots = low_roots * len(json_paths)
    if len(low_roots) != len(json_paths):
        raise ValueError("--ditto-low-roots must contain one root or the same number of roots as --ditto-filtered-json")

    remap: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for json_path, low_root in zip(json_paths, low_roots):
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        for row in rows:
            instruction = str(row.get("instruction", "")).strip()
            source_path = str(row.get("source_path", "")).strip()
            if not instruction or not source_path:
                continue
            mapped_low = str(Path(low_root) / Path(source_path).name)
            remap[instruction].append(
                {
                    "low_video": mapped_low,
                    "source_path": source_path,
                    "ditto_json": str(json_path),
                }
            )
    return remap


def remap_low_video(
    low_video: str,
    prompt: str,
    remap_by_prompt: dict[str, deque[dict[str, str]]],
    repo_root: Path,
) -> tuple[str, dict[str, str] | None]:
    if not remap_by_prompt:
        return low_video, None
    if (repo_root / low_video).exists() or Path(low_video).is_absolute() and Path(low_video).exists():
        return low_video, None
    candidates = remap_by_prompt.get(prompt.strip())
    if not candidates:
        return low_video, None

    # Prefer a mapped file that exists on this machine; otherwise consume the
    # next candidate deterministically so the generated script still points at
    # the materialized hash-style dataset layout used on the remote.
    for _ in range(len(candidates)):
        candidate = candidates.popleft()
        candidates.append(candidate)
        mapped = candidate["low_video"]
        mapped_path = Path(mapped)
        if (mapped_path.is_absolute() and mapped_path.exists()) or (repo_root / mapped).exists():
            candidates.remove(candidate)
            return mapped, candidate

    candidate = candidates.popleft()
    return candidate["low_video"], candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output-sh", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint-path", default="models/Wan2.1-VACE-14B")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--latent-memory-mode", choices=["vace_hint", "vace_context", "text"], default="vace_hint")
    parser.add_argument("--trace-dir", type=Path, default=Path("out/codebook/traces"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--ditto-filtered-json",
        type=Path,
        action="append",
        default=[],
        help="Optional ditto_filtered_*.json files used to remap old metadata video names to hash-style low video paths by matching prompt/instruction.",
    )
    parser.add_argument(
        "--ditto-low-roots",
        default="datas/ditto_face/low,datas/ditto_face2/low",
        help="Comma-separated low-video roots corresponding to --ditto-filtered-json files.",
    )
    args = parser.parse_args()

    rows = read_metadata(args.metadata_csv)
    selected = rows[args.offset : args.offset + args.limit if args.limit > 0 else None]
    remap_by_prompt = load_ditto_prompt_remap(args.ditto_filtered_json, split_csv(args.ditto_low_roots))

    args.output_sh.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {q(args.repo_root)}",
        "",
    ]
    manifest = []
    for idx, row in enumerate(selected, start=args.offset):
        prompt = row["prompt"]
        category = infer_category(prompt)
        sample_id = f"trace_{idx:05d}"
        output_video = args.trace_dir / f"{sample_id}.mp4"
        trace_output = args.trace_dir / f"{sample_id}.trace.json"
        low_video = row["vace_video"]
        low_video_original = low_video
        low_video, remap_info = remap_low_video(low_video, prompt, remap_by_prompt, args.repo_root)
        high_video = row["video"]
        cmd = [
            "python3",
            "scripts/codebook/trace_latent_memory_routing.py",
            "--mode",
            args.latent_memory_mode,
            "--sample-id",
            sample_id,
            "--category",
            category,
            "--input-video",
            low_video,
            "--control-video",
            low_video,
            "--output-video",
            str(output_video),
            "--checkpoint-path",
            args.checkpoint_path,
            "--lora-path",
            args.lora_path,
            "--prompt",
            prompt,
            "--trace-output",
            str(trace_output),
            "--device-id",
            str(args.device_id),
            "--num-frames",
            str(args.num_frames),
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
        ]
        lines.append(" ".join(q(part) for part in cmd))
        lines.append("")
        manifest.append(
            {
                "sample_id": sample_id,
                "category": category,
                "low_video": low_video,
                "low_video_original": low_video_original,
                "high_video": high_video,
                "prompt": prompt,
                "trace_output": str(trace_output),
                "remap_info": remap_info,
            }
        )

    args.output_sh.write_text("\n".join(lines), encoding="utf-8")
    args.output_sh.chmod(0o755)
    manifest_path = args.output_sh.with_suffix(args.output_sh.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    remapped = sum(1 for item in manifest if item["remap_info"] is not None)
    print(
        json.dumps(
            {
                "commands": len(selected),
                "output_sh": str(args.output_sh),
                "manifest": str(manifest_path),
                "remapped_low_videos": remapped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
