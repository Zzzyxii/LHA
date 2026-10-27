"""CLIP feature extraction shared by every bench_v5 script.

One model, one frame-sampling rule and one cache, so that no two scripts in this
directory can disagree about what a video's axis vector is.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes  # noqa: E402

MODEL_NAME = "ViT-L-14"
# openai's `clip` spells the same checkpoint with a slash.
CLIP_PKG_NAME = "ViT-L/14"
PRETRAINED = "openai"
N_FRAMES = int(os.environ.get("LH_N_FRAMES", "16"))
# Matches bench_v4 so the two benchmarks stay comparable.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model = None
_preprocess = None
_tokenizer = None
_device = None
_backend = None


def _load():
    """Load ViT-L/14 from whichever CLIP package this machine has.

    Two backends because they live on different machines: open_clip on the
    workstation, openai's `clip` on the GPU pod (which has the weights cached and
    cannot reach pypi). Same weights, so the numbers are comparable across
    machines -- but never mix clips extracted on different backends in one
    comparison, and never mix frame counts either.
    """
    global _model, _preprocess, _tokenizer, _device, _backend
    if _model is not None:
        return
    _device = torch.device(os.environ.get("LH_DEVICE")
                           or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        import open_clip
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=_device)
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        _backend = "open_clip"
    except ImportError:
        import clip as _clip
        # openai's package spells it with a slash, open_clip with dashes.
        _model, _preprocess = _clip.load(CLIP_PKG_NAME, device=str(_device))
        _tokenizer = _clip.tokenize
        _backend = "clip"
    _model.eval()
    # From the environment, never os.cpu_count(): a pool of N workers each grabbing
    # 32 intra-op threads oversubscribes the box by N*32 and every worker slows
    # down by more than N.
    torch.set_num_threads(int(os.environ.get("LH_THREADS", "2")))


def text_matrix() -> np.ndarray:
    """(2K, D) L2-normalised text embeddings: [high_0, low_0, high_1, low_1, ...]."""
    _load()
    caps = [c for pair in axes.prompt_pairs() for c in pair]
    with torch.no_grad():
        tok = _tokenizer(caps).to(_device)
        emb = _model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()


def read_frames(path: str | Path, n_frames: int = N_FRAMES) -> list[np.ndarray]:
    """n_frames RGB frames spread evenly through the clip, in [0,1] float32."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise ValueError(f"no frames in {path}")
    idx = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    cap.release()
    if not out:
        raise ValueError(f"could not decode any frame from {path}")
    return out


def image_matrix(frames: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    """(n_frames, D) L2-normalised image embeddings."""
    _load()
    import torchvision.transforms.functional as TF
    batch = []
    for f in frames:
        t = torch.from_numpy(f).permute(2, 0, 1)
        t = TF.resize(t, [224, 224], antialias=True)
        t = TF.normalize(t, torch.from_numpy(IMAGENET_MEAN), torch.from_numpy(IMAGENET_STD))
        batch.append(t)
    out = []
    with torch.no_grad():
        for i in range(0, len(batch), batch_size):
            x = torch.stack(batch[i:i + batch_size]).to(_device)
            e = _model.encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.float().cpu().numpy())
    return np.concatenate(out, 0)


def axis_vector(frames_or_path, n_frames: int = N_FRAMES) -> np.ndarray:
    """(K,) mean CLIP antonym margin over the clip's frames."""
    tx = text_matrix()
    im = image_matrix(frames_or_path if isinstance(frames_or_path, list)
                      else read_frames(frames_or_path, n_frames))
    sims = im @ tx.T                       # (F, 2K)
    K = tx.shape[0] // 2
    margin = sims[:, 0::2] - sims[:, 1::2]  # (F, K)
    return margin.mean(0)


def clip_embedding(frames_or_path, n_frames: int = N_FRAMES) -> np.ndarray:
    """(D,) mean normalised image embedding, for similarity and adherence."""
    im = image_matrix(frames_or_path if isinstance(frames_or_path, list)
                      else read_frames(frames_or_path, n_frames))
    v = im.mean(0)
    return v / (np.linalg.norm(v) + 1e-9)
