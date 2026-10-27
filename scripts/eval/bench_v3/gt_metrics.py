#!/usr/bin/env python3
"""GT-referenced metrics for bench_v2.

Every metric in the original bench scored the edited video against something
other than the target: the instruction text (CLIP-T, PickScore), its own adjacent
frames (CLIP-F, DINO-F), the source (CLIP Src-Edit), or nothing at all (LAION,
MUSIQ, NIQE, the aesthetic reward). ``target_high`` was touched only by JEDi, and
only as a set-level distribution. So the bench could not tell an edit that
reached the target look from one that changed nothing -- and on content-rewrite
samples whose low input is already attractive TV footage, changing nothing scores
well on most of those metrics.

This script scores each edited clip against its own target:

* ``gt_dino`` / ``gt_clip`` -- similarity to ``target_high`` on time-aligned
  frames. DINO tracks structure and appearance, CLIP tracks semantics and style.
* ``src_dino`` / ``src_clip`` -- similarity to ``source_low`` on the same frames.
  A preservation signal, read as a constraint rather than a goal.
* ``gain_dino`` / ``gain_clip`` -- ``gt_*`` minus the same similarity for the
  untouched low clip over the same time window. This is the headline number: an
  arm that returns its input unchanged scores exactly 0, so under-editing can no
  longer pass for quality, and an arm that edits in the wrong direction goes
  negative.

Frames are aligned by timestamp, not index. The archived methods were exported
at 17, 49, 73 and native frame counts; sampling K timestamps across each edited
clip's own duration and fetching the nearest target/source frame at each keeps a
17-frame clip compared against the first 0.85 s of its target rather than
against frames it never covered.

Two reference arms are always included: ``noop`` (the low clip itself; gain 0 by
construction) and ``oracle`` (the high clip itself; the ceiling). If either reads
anything other than its expected value, the pipeline is broken.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any

import av
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _probe(path: Path) -> tuple[int, float]:
    """Frame count and fps. Counts by decoding: container headers proved unreliable here."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else 16.0
        count = sum(1 for _ in container.decode(video=0))
    return count, fps


def _decode_indices(path: Path, wanted: set[int]):
    from PIL import Image

    out: dict[int, Image.Image] = {}
    if not wanted:
        return out
    last = max(wanted)
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                out[index] = frame.to_image().convert("RGB")
            if index >= last:
                break
    return out


class Embedder:
    """CLIP-L and DINO ViT-S/16 image embeddings, cached per (video, frame index)."""

    def __init__(self, *, cache_dir: str, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        kwargs = {"cache_dir": cache_dir, "local_files_only": True}
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", **kwargs).to(device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", **kwargs)
        self.dino = AutoModel.from_pretrained("facebook/dino-vits16", **kwargs).to(device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained("facebook/dino-vits16", **kwargs)
        self._cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._probe_cache: dict[str, tuple[int, float]] = {}

    def probe(self, path: Path) -> tuple[int, float]:
        key = str(path)
        if key not in self._probe_cache:
            self._probe_cache[key] = _probe(path)
        return self._probe_cache[key]

    def embed(self, path: Path, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (clip, dino) unit vectors for the given frame indices, in order."""
        key = str(path)
        missing = sorted({i for i in indices if (key, i) not in self._cache})
        if missing:
            frames = _decode_indices(path, set(missing))
            present = [i for i in missing if i in frames]
            if present:
                images = [frames[i] for i in present]
                with self.torch.inference_mode():
                    clip_in = self.clip_proc(images=images, return_tensors="pt").to(self.device)
                    clip_vec = self.clip.get_image_features(**clip_in)
                    dino_in = self.dino_proc(images=images, return_tensors="pt").to(self.device)
                    dino_vec = self.dino(**dino_in).last_hidden_state[:, 0]
                clip_vec = self.torch.nn.functional.normalize(clip_vec.float(), dim=-1).cpu().numpy()
                dino_vec = self.torch.nn.functional.normalize(dino_vec.float(), dim=-1).cpu().numpy()
                for row, index in enumerate(present):
                    self._cache[(key, index)] = (clip_vec[row], dino_vec[row])
        clip = np.stack([self._cache[(key, i)][0] for i in indices])
        dino = np.stack([self._cache[(key, i)][1] for i in indices])
        return clip, dino


def _aligned_indices(edited: tuple[int, float], other: tuple[int, float], k: int) -> tuple[list[int], list[int]]:
    """K timestamps across the edited clip; nearest frame index in each video."""
    n_e, fps_e = edited
    n_o, fps_o = other
    duration = n_e / fps_e
    times = [(j + 0.5) / k * duration for j in range(k)]
    # One rounding rule for both sides: with mixed rules, noop (edited == low)
    # picked neighbouring frames and read a spurious nonzero gain.
    idx_e = [min(n_e - 1, int(t * fps_e)) for t in times]
    idx_o = [min(n_o - 1, int(t * fps_o)) for t in times]
    return idx_e, idx_o


def _cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum(axis=1).mean())


def score_sample(emb: Embedder, edited: Path, low: Path, high: Path, k: int) -> dict[str, float]:
    p_e, p_l, p_h = emb.probe(edited), emb.probe(low), emb.probe(high)
    idx_e, idx_h = _aligned_indices(p_e, p_h, k)
    _, idx_l = _aligned_indices(p_e, p_l, k)
    e_clip, e_dino = emb.embed(edited, idx_e)
    h_clip, h_dino = emb.embed(high, idx_h)
    l_clip, l_dino = emb.embed(low, idx_l)
    gt_dino, gt_clip = _cos_mean(e_dino, h_dino), _cos_mean(e_clip, h_clip)
    base_dino, base_clip = _cos_mean(l_dino, h_dino), _cos_mean(l_clip, h_clip)
    return {
        "gt_dino": gt_dino,
        "gt_clip": gt_clip,
        "src_dino": _cos_mean(e_dino, l_dino),
        "src_clip": _cos_mean(e_clip, l_clip),
        "noop_gt_dino": base_dino,
        "noop_gt_clip": base_clip,
        "gain_dino": gt_dino - base_dino,
        "gain_clip": gt_clip - base_clip,
        "edited_frames": p_e[0],
        "edited_seconds": round(p_e[0] / p_e[1], 3),
    }


def _bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


METRICS = ("gain_dino", "gain_clip", "gt_dino", "gt_clip", "src_dino", "src_clip")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for metric in METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if not values:
            continue
        out[metric] = statistics.fmean(values)
        if metric.startswith("gain"):
            lo, hi = _bootstrap_ci(values)
            out[f"{metric}_ci95"] = [lo, hi]
    return out


def _parse_arm(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--arm must be name=dir, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="GT-referenced bench_v2 metrics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[], help="name=dir containing <sample_id>.mp4; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--hf-cache", default=os.environ.get("HF_HUB_CACHE", "models/hf_cache/hub"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = _read_jsonl(args.manifest)
    arms: list[tuple[str, str]] = [("noop", "@source_low"), ("oracle", "@target_high")]
    arms += [_parse_arm(spec) for spec in args.arm]

    emb = Embedder(cache_dir=args.hf_cache, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for name, where in arms:
        rows = []
        for item in manifest:
            sid = item["sample_id"]
            low, high = Path(item["source_low"]), Path(item["target_high"])
            if where == "@source_low":
                edited = low
            elif where == "@target_high":
                edited = high
            else:
                edited = Path(where) / f"{sid}.mp4"
            if not edited.exists():
                continue
            metrics = score_sample(emb, edited, low, high, args.frames)
            rows.append(
                {
                    "sample_id": sid,
                    "arm": name,
                    "category": item["category"],
                    "story_aligned": item["story_aligned"],
                    "in_release30": item["in_release30"],
                    **metrics,
                }
            )
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        entry = {"all": summarize(rows)}
        entry["release30"] = summarize([r for r in rows if r["in_release30"]])
        entry["story_aligned"] = summarize([r for r in rows if r["story_aligned"]])
        for category in sorted({r["category"] for r in rows}):
            entry[f"cat:{category}"] = summarize([r for r in rows if r["category"] == category])
        summary[name] = entry
        a = entry["all"]
        print(
            f"[gt_metrics] {name:<18} n={a['n']:>2}  gain_dino={a.get('gain_dino', float('nan')):+.4f}  "
            f"gt_dino={a.get('gt_dino', float('nan')):.4f}  src_dino={a.get('src_dino', float('nan')):.4f}",
            flush=True,
        )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
