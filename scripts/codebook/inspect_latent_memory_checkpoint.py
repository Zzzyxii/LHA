#!/usr/bin/env python3
"""Inspect latent-memory weights in an edit-model checkpoint.

This is a lightweight check that does not load Wan/Ditto. It reports memory
shape, per-layer/per-token norms, selector parameter shapes, and token cosine
similarities. Use it before runtime tracing to verify what was actually saved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_state(path: Path):
    import torch

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu")


def find_key(state, suffix: str) -> str | None:
    for key in state:
        if key == suffix or key.endswith("." + suffix):
            return key
    return None


def cosine(left, right) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=0).item())


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=["text", "vace_context", "vace_hint"], default="vace_hint")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    state = load_state(args.checkpoint)
    if args.mode == "text":
        memory_key = find_key(state, "latent_memory_text.memory") or find_key(state, "memory")
        selector_prefix = "latent_memory_text.gate"
    elif args.mode == "vace_context":
        memory_key = find_key(state, "latent_memory_context.memory") or find_key(state, "memory")
        selector_prefix = "latent_memory_context.selector"
    else:
        memory_key = find_key(state, "latent_memory_hint.memory") or find_key(state, "memory")
        selector_prefix = "latent_memory_hint.selector"

    if memory_key is None:
        available = [k for k in state if "latent_memory" in k or k.endswith("memory")]
        raise RuntimeError(f"Could not find latent memory tensor. Candidate keys: {available[:50]}")

    memory = state[memory_key].float()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "memory_key": memory_key,
        "memory_shape": list(memory.shape),
        "state_keys": len(state),
        "latent_memory_keys": [k for k in state if "latent_memory" in k],
        "selector_keys": [k for k in state if selector_prefix in k or ".selector." in k or ".gate." in k],
    }

    norm_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    if args.mode == "vace_hint" and memory.ndim == 3:
        num_layers, num_tokens, _dim = memory.shape
        for layer in range(num_layers):
            layer_memory = memory[layer]
            for token in range(num_tokens):
                norm_rows.append(
                    {
                        "layer": layer,
                        "token": token,
                        "memory_norm": float(layer_memory[token].norm().item()),
                    }
                )
            for i in range(num_tokens):
                for j in range(i + 1, num_tokens):
                    cosine_rows.append(
                        {
                            "layer": layer,
                            "token_i": i,
                            "token_j": j,
                            "cosine": cosine(layer_memory[i], layer_memory[j]),
                        }
                    )
    elif memory.ndim == 2:
        num_tokens, _dim = memory.shape
        for token in range(num_tokens):
            norm_rows.append({"layer": -1, "token": token, "memory_norm": float(memory[token].norm().item())})
        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                cosine_rows.append({"layer": -1, "token_i": i, "token_j": j, "cosine": cosine(memory[i], memory[j])})
    else:
        raise RuntimeError(f"Unexpected memory shape: {tuple(memory.shape)}")

    write_csv(args.output_dir / "memory_norms.csv", norm_rows, ["layer", "token", "memory_norm"])
    write_csv(args.output_dir / "token_cosines.csv", cosine_rows, ["layer", "token_i", "token_j", "cosine"])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
