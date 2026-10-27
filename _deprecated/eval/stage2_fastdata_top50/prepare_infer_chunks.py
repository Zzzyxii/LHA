#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def absolutize_eval_row(row: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    result = dict(row)
    for key in ("source_video", "target_video", "control_video", "reference_image"):
        value = result.get(key)
        if value in (None, ""):
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        result[key] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()

    methods = json.loads((args.run_root / "methods.json").read_text(encoding="utf-8"))
    inputs_path = args.run_root / "eval_inputs.top50.jsonl"
    inputs = [
        absolutize_eval_row(row, base_dir=inputs_path.parent.resolve())
        for row in read_jsonl(inputs_path)
    ]
    tasks_path = args.run_root / "infer_tasks.tsv"
    chunks_root = args.run_root / "eval_input_chunks"

    lines: list[str] = []
    task_id = 0
    for item in methods:
        method = item["method"]
        for start in range(0, len(inputs), args.chunk_size):
            chunk_rows = inputs[start : start + args.chunk_size]
            chunk_id = f"chunk_{start // args.chunk_size:03d}"
            chunk_path = chunks_root / method / f"{chunk_id}.jsonl"
            write_jsonl(chunk_path, chunk_rows)
            output_dir = args.run_root / "infer_chunks" / method / chunk_id
            lines.append(
                "\t".join(
                    [
                        str(task_id),
                        method,
                        item["latent_memory_mode"],
                        item["checkpoint"],
                        str(chunk_path),
                        str(output_dir),
                    ]
                )
            )
            task_id += 1
    tasks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": task_id, "chunk_size": args.chunk_size, "tasks_path": str(tasks_path)}, indent=2))


if __name__ == "__main__":
    main()
