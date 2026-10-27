#!/usr/bin/env python3
"""Run one Ditto inference and trace vace_hint routing weights/delta norms.

The script monkey-patches LatentMemoryHintAdapter.forward inside the current
process, then calls Ditto/inference/infer_ditto.py. It records per-call routing
weights, entropy, memory delta norm, and hint norm. Use a tiny inference setting
for analysis, e.g. 17 frames and 1-4 denoising steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DITTO_ROOT = PROJECT_ROOT / "Ditto"
if str(DITTO_ROOT) not in sys.path:
    sys.path.insert(0, str(DITTO_ROOT))


TRACE_EVENTS: list[dict[str, Any]] = []


def tensor_list(tensor) -> list[float]:
    return [float(x) for x in tensor.detach().float().cpu().reshape(-1).tolist()]


def entropy(weights) -> float:
    w = weights.detach().float().clamp_min(1e-12)
    return float((-(w * w.log()).sum(dim=-1)).mean().item())


def patch_latent_memory_hint() -> None:
    import torch
    import diffsynth.models.latent_memory as lm

    original_forward = lm.LatentMemoryHintAdapter.forward

    def traced_forward(self, hint, context, hint_index=0, block_id=None):
        hint_index_int = int(hint_index)
        pooled, _ = lm._mean_pool_sequence(context)
        self.selector.to(device=pooled.device, dtype=pooled.dtype)
        logits = self.selector(pooled.to(device=pooled.device, dtype=pooled.dtype))
        weights = torch.softmax(logits, dim=-1).to(dtype=hint.dtype)
        memory = self.memory[hint_index_int].to(dtype=hint.dtype, device=hint.device)
        delta = weights @ memory
        delta = lm._match_batch(delta, hint.shape[0])
        output = hint + delta.unsqueeze(1) * self.scale

        weights_mean = weights.detach().float().mean(dim=0)
        delta_norm = delta.detach().float().norm(dim=-1).mean()
        hint_norm = hint.detach().float().norm(dim=-1).mean()
        TRACE_EVENTS.append(
            {
                "hint_index": hint_index_int,
                "block_id": int(block_id) if block_id is not None else None,
                "weights": tensor_list(weights_mean),
                "weight_entropy": entropy(weights),
                "top_token": int(weights_mean.argmax().item()),
                "top_weight": float(weights_mean.max().item()),
                "delta_norm": float(delta_norm.item()),
                "hint_norm": float(hint_norm.item()),
                "delta_to_hint_norm": float((delta_norm / hint_norm.clamp_min(1e-12)).item()),
                "num_tokens": int(weights_mean.numel()),
                "scale": float(self.scale),
            }
        )
        return output

    lm.LatentMemoryHintAdapter.forward = traced_forward
    # Keep a reference for debuggers; not used at runtime.
    lm.LatentMemoryHintAdapter._original_forward_for_trace = original_forward


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_layer.setdefault(int(event["hint_index"]), []).append(event)
    rows = []
    for layer, layer_events in sorted(by_layer.items()):
        num_tokens = int(layer_events[0]["num_tokens"])
        avg_weights = [0.0 for _ in range(num_tokens)]
        for event in layer_events:
            for i, value in enumerate(event["weights"]):
                avg_weights[i] += float(value)
        avg_weights = [value / len(layer_events) for value in avg_weights]
        rows.append(
            {
                "hint_index": layer,
                "calls": len(layer_events),
                "avg_weights": avg_weights,
                "top_token": int(max(range(num_tokens), key=lambda i: avg_weights[i])),
                "top_weight": max(avg_weights),
                "avg_entropy": sum(float(e["weight_entropy"]) for e in layer_events) / len(layer_events),
                "avg_delta_norm": sum(float(e["delta_norm"]) for e in layer_events) / len(layer_events),
                "avg_hint_norm": sum(float(e["hint_norm"]) for e in layer_events) / len(layer_events),
                "avg_delta_to_hint_norm": sum(float(e["delta_to_hint_norm"]) for e in layer_events) / len(layer_events),
            }
        )
    return {"num_events": len(events), "layers": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--control-video", default="")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--checkpoint-path", default="models/Wan2.1-VACE-14B")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--latent-memory-tokens", type=int, default=4)
    parser.add_argument("--latent-memory-hidden-dim", type=int, default=0)
    parser.add_argument("--latent-memory-scale", type=float, default=1.0)
    parser.add_argument("--latent-memory-init-std", type=float, default=0.02)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--quality", type=int, default=8)
    args = parser.parse_args()

    patch_latent_memory_hint()
    from inference import infer_ditto

    infer_args = SimpleNamespace(
        input_video=args.input_video,
        output_video=args.output_video,
        checkpoint_path=args.checkpoint_path,
        control_video=args.control_video,
        reference_image="",
        lora_path=[args.lora_path],
        device_id=args.device_id,
        prompt=args.prompt,
        negative_prompt=infer_ditto.DEFAULT_NEGATIVE_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        lora_alpha=[],
        default_lora_alpha=1.0,
        latent_memory_mode="vace_hint",
        latent_memory_tokens=args.latent_memory_tokens,
        latent_memory_hidden_dim=args.latent_memory_hidden_dim,
        latent_memory_scale=args.latent_memory_scale,
        latent_memory_init_std=args.latent_memory_init_std,
        latent_memory_checkpoint=args.lora_path,
        fps=args.fps,
        quality=args.quality,
    )
    infer_ditto.main(infer_args)

    payload = {
        "sample_id": args.sample_id,
        "category": args.category,
        "input_video": args.input_video,
        "control_video": args.control_video,
        "output_video": args.output_video,
        "lora_path": args.lora_path,
        "prompt": args.prompt,
        "num_inference_steps": args.num_inference_steps,
        "num_frames": args.num_frames,
        "events": TRACE_EVENTS,
        "summary": summarize_events(TRACE_EVENTS),
    }
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trace_output": str(args.trace_output), "events": len(TRACE_EVENTS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
