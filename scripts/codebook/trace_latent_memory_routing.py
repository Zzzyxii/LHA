#!/usr/bin/env python3
"""Run one Ditto inference and trace latent-memory routing for text/context/hint modes."""

from __future__ import annotations

import argparse
import json
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


def patch_latent_memory(mode: str) -> None:
    import torch
    import diffsynth.models.latent_memory as lm

    if mode == "text":
        original_forward = lm.LatentMemoryTextAdapter.forward

        def traced_text_forward(self, context):
            context, squeezed = lm._as_batched_sequence(context)
            pooled = context.mean(dim=1)
            self.gate.to(device=context.device, dtype=context.dtype)
            pooled = pooled.to(device=context.device, dtype=context.dtype)
            gate = self.gate(pooled).to(dtype=context.dtype).unsqueeze(-1)
            memory = self.memory.to(dtype=context.dtype, device=context.device)
            expanded_memory = memory.unsqueeze(0).expand(context.shape[0], -1, -1)
            injected = expanded_memory * gate * self.scale
            output = torch.cat([context, injected], dim=1)

            gate_mean = gate.detach().float().squeeze(-1).mean(dim=0)
            injected_norm = injected.detach().float().norm(dim=-1).mean()
            context_norm = context.detach().float().norm(dim=-1).mean()
            TRACE_EVENTS.append(
                {
                    "mode": "text",
                    "slot": 0,
                    "block_id": None,
                    "weights": tensor_list(gate_mean),
                    "weight_entropy": None,
                    "top_token": int(gate_mean.argmax().item()),
                    "top_weight": float(gate_mean.max().item()),
                    "delta_norm": float(injected_norm.item()),
                    "base_norm": float(context_norm.item()),
                    "delta_to_base_norm": float((injected_norm / context_norm.clamp_min(1e-12)).item()),
                    "num_tokens": int(gate_mean.numel()),
                    "scale": float(self.scale),
                    "routing_type": "sigmoid_gate",
                }
            )
            return output.squeeze(0) if squeezed else output

        lm.LatentMemoryTextAdapter.forward = traced_text_forward
        lm.LatentMemoryTextAdapter._original_forward_for_trace = original_forward
        return

    if mode == "vace_context":
        original_forward = lm.LatentMemoryContextAdapter.forward

        def traced_context_forward(self, vace_tokens, context):
            pooled, _ = lm._mean_pool_sequence(context)
            self.selector.to(device=pooled.device, dtype=pooled.dtype)
            logits = self.selector(pooled.to(device=pooled.device, dtype=pooled.dtype))
            weights = torch.softmax(logits, dim=-1).to(dtype=vace_tokens.dtype)
            memory = self.memory.to(dtype=vace_tokens.dtype, device=vace_tokens.device)
            delta = weights @ memory
            delta = lm._match_batch(delta, vace_tokens.shape[0])
            output = vace_tokens + delta.unsqueeze(1) * self.scale

            weights_mean = weights.detach().float().mean(dim=0)
            delta_norm = delta.detach().float().norm(dim=-1).mean()
            base_norm = vace_tokens.detach().float().norm(dim=-1).mean()
            TRACE_EVENTS.append(
                {
                    "mode": "vace_context",
                    "slot": 0,
                    "block_id": None,
                    "weights": tensor_list(weights_mean),
                    "weight_entropy": entropy(weights),
                    "top_token": int(weights_mean.argmax().item()),
                    "top_weight": float(weights_mean.max().item()),
                    "delta_norm": float(delta_norm.item()),
                    "base_norm": float(base_norm.item()),
                    "delta_to_base_norm": float((delta_norm / base_norm.clamp_min(1e-12)).item()),
                    "num_tokens": int(weights_mean.numel()),
                    "scale": float(self.scale),
                    "routing_type": "softmax_selector",
                }
            )
            return output

        lm.LatentMemoryContextAdapter.forward = traced_context_forward
        lm.LatentMemoryContextAdapter._original_forward_for_trace = original_forward
        return

    if mode == "vace_hint":
        original_forward = lm.LatentMemoryHintAdapter.forward

        def traced_hint_forward(self, hint, context, hint_index=0, block_id=None):
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
            base_norm = hint.detach().float().norm(dim=-1).mean()
            TRACE_EVENTS.append(
                {
                    "mode": "vace_hint",
                    "slot": hint_index_int,
                    "hint_index": hint_index_int,
                    "block_id": int(block_id) if block_id is not None else None,
                    "weights": tensor_list(weights_mean),
                    "weight_entropy": entropy(weights),
                    "top_token": int(weights_mean.argmax().item()),
                    "top_weight": float(weights_mean.max().item()),
                    "delta_norm": float(delta_norm.item()),
                    "base_norm": float(base_norm.item()),
                    "hint_norm": float(base_norm.item()),
                    "delta_to_base_norm": float((delta_norm / base_norm.clamp_min(1e-12)).item()),
                    "delta_to_hint_norm": float((delta_norm / base_norm.clamp_min(1e-12)).item()),
                    "num_tokens": int(weights_mean.numel()),
                    "scale": float(self.scale),
                    "routing_type": "softmax_selector",
                }
            )
            return output

        lm.LatentMemoryHintAdapter.forward = traced_hint_forward
        lm.LatentMemoryHintAdapter._original_forward_for_trace = original_forward
        return

    raise ValueError(f"unsupported mode: {mode}")


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_slot: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_slot.setdefault(int(event["slot"]), []).append(event)
    rows = []
    for slot, slot_events in sorted(by_slot.items()):
        num_tokens = int(slot_events[0]["num_tokens"])
        avg_weights = [0.0 for _ in range(num_tokens)]
        for event in slot_events:
            for i, value in enumerate(event["weights"]):
                avg_weights[i] += float(value)
        avg_weights = [value / len(slot_events) for value in avg_weights]
        entropies = [float(e["weight_entropy"]) for e in slot_events if e.get("weight_entropy") is not None]
        block_ids = sorted({e.get("block_id") for e in slot_events if e.get("block_id") is not None})
        rows.append(
            {
                "slot": slot,
                "hint_index": slot if slot_events[0].get("mode") == "vace_hint" else None,
                "block_ids": block_ids,
                "calls": len(slot_events),
                "avg_weights": avg_weights,
                "top_token": int(max(range(num_tokens), key=lambda i: avg_weights[i])),
                "top_weight": max(avg_weights),
                "avg_entropy": sum(entropies) / len(entropies) if entropies else None,
                "avg_delta_norm": sum(float(e["delta_norm"]) for e in slot_events) / len(slot_events),
                "avg_base_norm": sum(float(e["base_norm"]) for e in slot_events) / len(slot_events),
                "avg_delta_to_base_norm": sum(float(e["delta_to_base_norm"]) for e in slot_events) / len(slot_events),
            }
        )
    return {"num_events": len(events), "slots": rows, "layers": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vace_context", "vace_hint"], required=True)
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

    patch_latent_memory(args.mode)
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
        latent_memory_mode=args.mode,
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
        "mode": args.mode,
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
