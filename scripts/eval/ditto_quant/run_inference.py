#!/usr/bin/env python3
"""Run Ditto inference for a fixed eval set across multiple LoRA checkpoints.

This script completes the edit-model evaluation loop:

eval inputs (source_video + instruction)
-> generated videos for each checkpoint
-> quant manifest for ``scripts/eval/ditto_quant/evaluate.py``
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DITTO_INFER_SCRIPT = REPO_ROOT / "Ditto" / "inference" / "infer_ditto.py"


@dataclass(frozen=True)
class EvalInput:
    sample_id: str
    source_video: Path
    instruction: str
    control_video: Path | None
    reference_image: Path | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class CheckpointSpec:
    method: str
    lora_path: Path | None
    lora_alpha: float | None


def _fail(message: str) -> None:
    raise SystemExit(f"[ditto_quant.run_inference] {message}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            payload.setdefault("_line_no", line_no)
            rows.append(payload)
    return rows


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _path_from_row(row: dict[str, Any], keys: tuple[str, ...], *, required: bool, base_dir: Path) -> Path | None:
    raw = _first_value(row, keys)
    if raw in (None, ""):
        if required:
            raise ValueError(f"missing required path field; expected one of {', '.join(keys)}")
        return None
    return _resolve_path(Path(str(raw)).expanduser(), base_dir)


def _instruction_from_row(row: dict[str, Any]) -> str:
    raw = _first_value(row, ("instruction", "prompt", "edit_instruction", "compiled_edit_prompt"))
    text = str(raw or "").strip()
    if not text:
        raise ValueError("missing instruction field; expected instruction, prompt, edit_instruction, or compiled_edit_prompt")
    return text


def load_eval_inputs(path: Path) -> list[EvalInput]:
    rows = _read_jsonl(path)
    base_dir = path.parent.resolve()
    items: list[EvalInput] = []
    for index, row in enumerate(rows):
        source_video = _path_from_row(
            row,
            ("source_video", "input_video", "low_video", "low_video_path", "video"),
            required=True,
            base_dir=base_dir,
        )
        control_video = _path_from_row(
            row,
            ("control_video", "depth_video", "vace_video", "vace_video_path"),
            required=False,
            base_dir=base_dir,
        )
        reference_image = _path_from_row(
            row,
            ("reference_image", "reference_image_path", "vace_reference_image"),
            required=False,
            base_dir=base_dir,
        )
        assert source_video is not None
        items.append(
            EvalInput(
                sample_id=str(row.get("sample_id") or row.get("id") or f"case_{index:06d}"),
                source_video=source_video,
                instruction=_instruction_from_row(row),
                control_video=control_video,
                reference_image=reference_image,
                raw=row,
            )
        )
    return items


def parse_checkpoint_spec(raw: str, *, base_dir: Path, default_alpha: float) -> CheckpointSpec:
    text = str(raw).strip()
    if not text:
        raise ValueError("empty checkpoint spec")
    if "=" in text:
        method, value = text.split("=", 1)
        method = method.strip()
        value = value.strip()
    else:
        path = Path(text).expanduser()
        method = path.stem if path.name else "checkpoint"
        value = text
    if not method:
        raise ValueError(f"invalid checkpoint spec without method name: {raw}")

    alpha = default_alpha
    if "@" in value:
        value, alpha_text = value.rsplit("@", 1)
        value = value.strip()
        alpha = float(alpha_text.strip())

    if value.lower() in {"none", "base", "no_lora", "nolora"}:
        return CheckpointSpec(method=method, lora_path=None, lora_alpha=None)
    lora_path = _resolve_path(Path(value).expanduser(), base_dir)
    return CheckpointSpec(method=method, lora_path=lora_path, lora_alpha=alpha)


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value.strip())
    return cleaned or "sample"


def _build_command(args: argparse.Namespace, item: EvalInput, checkpoint: CheckpointSpec, output_video: Path) -> list[str]:
    command = [
        str(args.python_executable),
        str(DITTO_INFER_SCRIPT),
        "--input_video",
        str(item.source_video),
        "--output_video",
        str(output_video),
        "--checkpoint_path",
        str(args.checkpoint_path),
        "--device_id",
        str(args.device_id),
        "--prompt",
        item.instruction,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--cfg_scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--quality",
        str(args.quality),
        "--default_lora_alpha",
        str(args.default_lora_alpha),
    ]
    if checkpoint.lora_path is not None:
        command.extend(["--lora_path", str(checkpoint.lora_path)])
        if checkpoint.lora_alpha is not None:
            command.extend(["--lora_alpha", str(checkpoint.lora_alpha)])
    if args.latent_memory_mode != "none":
        command.extend(
            [
                "--latent_memory_mode",
                args.latent_memory_mode,
                "--latent_memory_tokens",
                str(args.latent_memory_tokens),
                "--latent_memory_hidden_dim",
                str(args.latent_memory_hidden_dim),
                "--latent_memory_scale",
                str(args.latent_memory_scale),
                "--latent_memory_init_std",
                str(args.latent_memory_init_std),
            ]
        )
        if args.latent_memory_checkpoint:
            command.extend(["--latent_memory_checkpoint", str(args.latent_memory_checkpoint)])
    if args.negative_prompt:
        command.extend(["--negative_prompt", args.negative_prompt])
    if item.control_video is not None:
        command.extend(["--control_video", str(item.control_video)])
    if item.reference_image is not None:
        command.extend(["--reference_image", str(item.reference_image)])
    return command


def _validate_args(args: argparse.Namespace, checkpoints: list[CheckpointSpec]) -> None:
    if not DITTO_INFER_SCRIPT.exists():
        _fail(f"Ditto inference script not found: {DITTO_INFER_SCRIPT}")
    if not args.checkpoint_path.exists():
        _fail(f"--checkpoint-path not found: {args.checkpoint_path}")
    if not args.python_executable.exists():
        _fail(f"--python-executable not found: {args.python_executable}")
    for checkpoint in checkpoints:
        if checkpoint.lora_path is not None and not checkpoint.lora_path.exists():
            _fail(f"LoRA checkpoint for method={checkpoint.method} not found: {checkpoint.lora_path}")


def _validate_items(items: list[EvalInput]) -> None:
    missing: list[str] = []
    for item in items:
        if not item.source_video.is_file():
            missing.append(f"source_video sample_id={item.sample_id}: {item.source_video}")
        if item.control_video is not None and not item.control_video.is_file():
            missing.append(f"control_video sample_id={item.sample_id}: {item.control_video}")
        if item.reference_image is not None and not item.reference_image.is_file():
            missing.append(f"reference_image sample_id={item.sample_id}: {item.reference_image}")
    if missing:
        preview = "\n".join(missing[:20])
        _fail(f"missing inference input files ({len(missing)}):\n{preview}")


def run_inference(args: argparse.Namespace, items: list[EvalInput], checkpoints: list[CheckpointSpec]) -> list[dict[str, Any]]:
    generations_dir = args.output_dir / "generations"
    manifest_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        method_dir = generations_dir / _safe_name(checkpoint.method)
        method_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(items, start=1):
            output_video = method_dir / f"{_safe_name(item.sample_id)}.mp4"
            print(
                f"[ditto_quant.run_inference] {checkpoint.method} {index}/{len(items)} "
                f"sample_id={item.sample_id} -> {output_video}",
                flush=True,
            )
            if output_video.exists() and args.skip_existing:
                print("[ditto_quant.run_inference] skip existing output", flush=True)
            else:
                command = _build_command(args, item, checkpoint, output_video)
                if args.dry_run:
                    print(shlex.join(command), flush=True)
                else:
                    try:
                        subprocess.run(command, check=True)
                    except subprocess.CalledProcessError as exc:
                        if args.fail_fast:
                            raise
                        print(
                            f"[ditto_quant.run_inference] command failed method={checkpoint.method} "
                            f"sample_id={item.sample_id} exit={exc.returncode}",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                if not output_video.is_file():
                    message = (
                        f"[ditto_quant.run_inference] inference command exited successfully but did not create "
                        f"output_video={output_video}"
                    )
                    if args.fail_fast:
                        raise RuntimeError(message)
                    print(message, file=sys.stderr, flush=True)
                    continue

            manifest_rows.append(
                {
                    "sample_id": item.sample_id,
                    "method": checkpoint.method,
                    "source_video": str(item.source_video),
                    "edited_video": str(output_video),
                    "instruction": item.instruction,
                    "lora_checkpoint": str(checkpoint.lora_path) if checkpoint.lora_path else "",
                    "lora_alpha": checkpoint.lora_alpha,
                    "raw_eval_input": item.raw,
                }
            )
    return manifest_rows


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    manifests_dir = args.output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "quant_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = {
        "eval_inputs": str(args.eval_inputs),
        "checkpoint_path": str(args.checkpoint_path),
        "output_dir": str(args.output_dir),
        "num_rows": len(rows),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "fps": args.fps,
        "quality": args.quality,
        "latent_memory_mode": args.latent_memory_mode,
        "latent_memory_tokens": args.latent_memory_tokens,
        "latent_memory_hidden_dim": args.latent_memory_hidden_dim,
        "latent_memory_scale": args.latent_memory_scale,
        "latent_memory_init_std": args.latent_memory_init_std,
        "latent_memory_checkpoint": str(args.latent_memory_checkpoint) if args.latent_memory_checkpoint else "",
    }
    (args.output_dir / "run_inference_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ditto_quant.run_inference] wrote quant manifest: {manifest_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ditto inference across LoRA checkpoints for a fixed eval set.")
    parser.add_argument("--eval-inputs", required=True, type=Path, help="JSONL with source_video and instruction.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated videos and quant manifest.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint spec. Use method=/path/to/lora.safetensors or method=/path@alpha. Use method=none for no LoRA.",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Local Wan2.1-VACE-14B checkpoint directory.")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable), help="Python executable for Ditto inference.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=73)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--negative-prompt", default="", help="Override Ditto inference negative prompt.")
    parser.add_argument("--default-lora-alpha", type=float, default=1.0)
    parser.add_argument(
        "--latent-memory-mode",
        choices=["none", "text", "vace_context", "vace_hint", "text+vace_hint", "text+vace_context"],
        default="none",
        help="Attach latent memory adapter during inference.",
    )
    parser.add_argument("--latent-memory-tokens", type=int, default=4)
    parser.add_argument("--latent-memory-hidden-dim", type=int, default=0)
    parser.add_argument("--latent-memory-scale", type=float, default=1.0)
    parser.add_argument("--latent-memory-init-std", type=float, default=0.02)
    parser.add_argument("--latent-memory-checkpoint", type=Path, default=None, help="Optional explicit checkpoint for latent memory weights.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not regenerate existing output videos.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifest without running inference.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on the first failed inference command.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        _fail("provide at least one --checkpoint, for example --checkpoint step-5000=/path/to/step-5000.safetensors")
    args.eval_inputs = args.eval_inputs.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.python_executable = args.python_executable.expanduser().resolve()
    if args.latent_memory_checkpoint is not None:
        args.latent_memory_checkpoint = args.latent_memory_checkpoint.expanduser().resolve()
    checkpoints = [
        parse_checkpoint_spec(raw, base_dir=Path.cwd(), default_alpha=args.default_lora_alpha)
        for raw in args.checkpoint
    ]
    _validate_args(args, checkpoints)
    items = load_eval_inputs(args.eval_inputs)
    if not items:
        _fail("eval inputs are empty")
    _validate_items(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_inference(args, items, checkpoints)
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
