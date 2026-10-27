#!/usr/bin/env python3
"""
Shared dataset loader for smoke-data, ditto, ditto_face, and custom formats.

Unified format (pair.json / manifest.json):
    {
        "sample_id": "...",
        "prompt": "...",
        "low": "low/xxxx.mp4",          # relative to pair-dir root
        "high": "high/xxxx_1.mp4"        # relative to pair-dir root
    }

manifest.json additionally has:
    "low_materialized": "/absolute/path/to/low/xxxx.mp4"
    "high_materialized": "/absolute/path/to/high/xxxx_1.mp4"
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataSample:
    sample_id: str
    prompt: str
    low_video: Path  # absolute path
    high_video: Path | None = None  # absolute path (may be None for smoke-data)

    @property
    def prompt_text(self) -> str:
        return self.prompt


def _load_pair_json(pair_dir: Path) -> list[dict]:
    path = pair_dir / "pair.json"
    if not path.exists():
        raise FileNotFoundError(f"pair.json not found in {pair_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_json(pair_dir: Path) -> list[dict]:
    path = pair_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found in {pair_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(pair_dir: Path) -> list[DataSample]:
    """
    Load all samples from a pair directory.
    Supports both pair.json and manifest.json formats.
    """
    if (pair_dir / "pair.json").exists():
        records = _load_pair_json(pair_dir)
    elif (pair_dir / "manifest.json").exists():
        records = _load_manifest_json(pair_dir)
    else:
        raise FileNotFoundError(
            f"Neither pair.json nor manifest.json found in {pair_dir}. "
            "Available files: " + ", ".join(sorted(p.name for p in pair_dir.iterdir()))
        )

    samples = []
    for record in records:
        # Resolve low video path
        low_rel = record.get("low") or record.get("low_rel") or ""
        low_materialized = record.get("low_materialized", "")

        if low_materialized and Path(low_materialized).exists():
            low_path = Path(low_materialized)
        else:
            low_path = pair_dir / low_rel
            if not low_path.exists():
                low_path = Path(low_rel)  # absolute path already

        # Resolve high video path (optional)
        high_rel = record.get("high") or record.get("high_rel") or ""
        high_materialized = record.get("high_materialized", "")
        high_path: Path | None = None

        if high_rel or high_materialized:
            if high_materialized and Path(high_materialized).exists():
                high_path = Path(high_materialized)
            else:
                high_path = pair_dir / high_rel
                if not high_path.exists():
                    high_path = Path(high_rel)

        samples.append(DataSample(
            sample_id=str(record.get("sample_id", "")),
            prompt=str(record.get("prompt", "")),
            low_video=low_path,
            high_video=high_path,
        ))

    return samples


def get_sample_by_index(pair_dir: Path, index: int) -> DataSample:
    samples = load_samples(pair_dir)
    if not samples:
        raise ValueError(f"No samples found in {pair_dir}")
    return samples[index]


def get_sample_by_id(pair_dir: Path, sample_id: str) -> DataSample:
    samples = load_samples(pair_dir)
    for s in samples:
        if s.sample_id == sample_id:
            return s
    raise KeyError(f"Sample '{sample_id}' not found in {pair_dir}. Available: {[s.sample_id for s in samples]}")


def smoke_sample(index: int = 0) -> DataSample:
    """
    Load a sample from smoke-data directory (smoke-data/prompt{N}.txt + smoke-data/{N}_low-cost.mp4).
    Index 0 -> prompt1.txt / 1_low-cost.mp4
    """
    n = index + 1
    prompt_file = PROJECT_ROOT / "smoke-data" / f"prompt{n}.txt"
    low_video = PROJECT_ROOT / "smoke-data" / f"{n}_low-cost.mp4"
    prompt = prompt_file.read_text(encoding="utf-8").strip() if prompt_file.exists() else ""
    return DataSample(
        sample_id=str(n),
        prompt=prompt,
        low_video=low_video,
        high_video=None,
    )
