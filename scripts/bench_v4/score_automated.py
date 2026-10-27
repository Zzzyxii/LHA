#!/usr/bin/env python3
"""Automated, LLM-free scoring for bench_v4.

Why this exists: every headline number in bench_v3 came from an LLM judge, and
the judges disagreed with each other (see LESSONS.md: gemini favoured the agent
arms on fidelity/quality, gpt-5.6-luna favoured the direct arms, kappa 0.11).
A judge disagreement of that size means the headline metric was measuring the
judge. bench_v4 therefore puts the primary numbers on metrics that are pure
functions of (source frames, output frames, instruction text) and can be
recomputed bit-for-bit by anyone.

Metrics, and what each is compared against:

  change        1 - mean cosine(clip_img(out_f), clip_img(src_f))   vs SOURCE
                How far the output moved away from the low-cost source.
                noop must be exactly 0.
  preservation  mean cosine over frames, identical to 1 - change    vs SOURCE
                Kept as its own column because "did not break it" and "got it
                right" look the same on a target-referenced metric otherwise.
  clip_t_out    mean cosine(clip_img(out_f), clip_txt(instruction)) vs INSTRUCTION
  clip_t_gain   clip_t_out - clip_t_src                             vs INSTRUCTION
                Did the edit move the clip toward what was asked.
  temporal      mean cosine(clip_img(out_f), clip_img(out_f+1))     vs ITSELF
                Flicker / frame-to-frame coherence of the output.
  musiq         mean MUSIQ-SPAQ over frames                          vs NOTHING
  musiq_gain    musiq(out) - musiq(src)                              vs SOURCE
                Did it get higher-cost (perceptual quality).

Reported separately, never merged: change (how much), clip_t_gain (right
direction), musiq_gain (quality), preservation (did not break it).

Anchors: `noop` (source copied as output) must give change=0, clip_t_gain=0,
musiq_gain=0 exactly; a nonzero value there is a pipeline bug. `oracle`
(target_high as output) gives the ceiling for musiq_gain and change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP_CACHE = os.environ.get("BENCH_V4_CLIP_CACHE", "/root/.cache/clip")
FRAME_CACHE = PROJECT_ROOT / "out" / "bench_v4" / "frames"
# VBench ships the LAION aesthetic head and the CLIP ViT-L/14 weights it needs; both
# are already on this filesystem, which is the only reason this works offline.
VBENCH_CACHE = Path(os.environ.get(
    "BENCH_V4_VBENCH_CACHE", "/apdcephfs_szcf/share_304169095/junjhzhu/vbench_cache"))
# DINO ViT-B/16, same family as bench_v2/v3's dino-vits16 and the only reason
# gain_dino has any headroom: CLIP puts source and target at 0.93 cosine, so a
# CLIP "closer to target" gain is bounded at 0.07 and says any edit moved away.
# DINO puts them at ~0.73, which leaves room for an arm to actually score.
DINO_REPO = Path(os.environ.get(
    "BENCH_V4_DINO_REPO",
    "/apdcephfs_szcf/share_304169095/junjhzhu/vbench_cache/dino_model/facebookresearch_dino_main"))
DINO_WEIGHTS = Path(os.environ.get(
    "BENCH_V4_DINO_WEIGHTS",
    "/apdcephfs_szcf/share_304169095/junjhzhu/vbench_cache/dino_model/dino_vitbase16_pretrain.pth"))
HIGH_COST_PROMPT = ("a frame from a high-budget cinematic film, professional lighting, "
                    "rich material detail, polished color grading")
LOW_COST_PROMPT = ("a frame from a low-budget amateur video, flat lighting, cheap looking, "
                   "plain and unpolished")
NUM_FRAMES = int(os.environ.get("BENCH_V4_FRAMES", "8"))


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _key(path: Path) -> str:
    st = path.stat()
    raw = f"{path}|{st.st_size}|{int(st.st_mtime)}|{NUM_FRAMES}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def extract_frames(video: Path) -> list[Path]:
    """Uniformly sample NUM_FRAMES frames, cached under out/bench_v4/frames.

    ffmpeg's image2 numbering starts at 1, so the raw files are kept in a
    subdirectory and the canonical 0-based names are written afterwards; padding
    a short clip then can never collide with a name ffmpeg already used.
    """
    out_dir = FRAME_CACHE / _key(video)
    canonical = [out_dir / ("%02d.png" % i) for i in range(NUM_FRAMES)]
    if all(f.exists() for f in canonical):
        return canonical
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob("*.png"):
        stale.unlink()
    cmd = [
        _ffmpeg(), "-y", "-v", "error", "-i", str(video),
        "-vf", "fps=1,scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2",
        str(raw_dir / "%03d.png"),
    ]
    subprocess.run(cmd, check=False, capture_output=True)
    raws = sorted(raw_dir.glob("*.png"))
    if not raws:
        raise RuntimeError("no frames extracted from %s" % video)
    picks = raws[:NUM_FRAMES]
    while len(picks) < NUM_FRAMES:
        picks.append(raws[-1])
    for raw, dst in zip(picks, canonical):
        if raw.resolve() == dst.resolve():
            continue
        shutil.copyfile(raw, dst)
    return canonical


class Scorer:
    def __init__(self, device: str = "cuda") -> None:
        import clip
        import torch
        import pyiqa
        self.torch = torch
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device, download_root=CLIP_CACHE)
        self.clip_model.eval()
        self.musiq = pyiqa.create_metric("musiq-spaq", device=device)
        self.aesthetic = self._load_aesthetic(device)
        self.dino, self.dino_tf = self._load_dino(device)

    def _load_dino(self, device: str):
        import sys
        import torch
        from torchvision import transforms
        sys.path.insert(0, str(DINO_REPO))
        import vision_transformer as vits  # noqa: E402
        model = vits.vit_base(patch_size=16, num_classes=0)
        state = torch.load(str(DINO_WEIGHTS), map_location="cpu")
        missing = model.load_state_dict(state, strict=False)
        if missing.missing_keys:
            print("dino missing keys:", missing.missing_keys[:3], file=sys.stderr)
        model.eval()
        model.to(device)
        tf = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return model, tf

    def _load_aesthetic(self, device: str):
        import torch
        import torch.nn as nn
        head = nn.Linear(768, 1)
        path = VBENCH_CACHE / "aesthetic_model" / "emb_reader" / "sa_0_4_vit_l_14_linear.pth"
        if not path.exists():
            return None
        head.load_state_dict(torch.load(str(path), map_location="cpu"))
        head.eval()
        return head.to(device)

    @staticmethod
    def _cos(a, b) -> float:
        import torch.nn.functional as F
        return float(F.cosine_similarity(a, b, dim=-1).mean())

    def image_features(self, frames: list[Path]):
        import torch
        from PIL import Image
        imgs = torch.stack([
            self.preprocess(Image.open(p).convert("RGB")) for p in frames
        ]).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def text_feature(self, text: str):
        import clip
        import torch
        tok = clip.tokenize([text[:200]], truncate=True).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_text(tok)
        return feat / feat.norm(dim=-1, keepdim=True)

    def dino_features(self, frames: list[Path]):
        import torch
        from PIL import Image
        batch = torch.stack([self.dino_tf(Image.open(f).convert("RGB")) for f in frames]).to(self.device)
        with torch.no_grad():
            feats = self.dino(batch)
        if feats.dim() == 3:  # (B, 1+patches, C) -> class token
            feats = feats[:, 0]
        return feats / feats.norm(dim=-1, keepdim=True)

    def _aesthetic(self, feats) -> float:
        import torch
        with torch.no_grad():
            raw = self.aesthetic(feats.float()).squeeze(-1)
        return float(raw.mean() / 10.0)

    def musiq_score(self, frames: list[Path]) -> float:
        import torch
        from PIL import Image
        import torchvision.transforms as T
        tf = T.Compose([T.ToTensor()])
        vals = []
        for p in frames:
            img = Image.open(p).convert("RGB")
            x = tf(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vals.append(float(self.musiq(x)))
        return sum(vals) / len(vals)

    def score_pair(self, src: Path, out: Path, instruction: str, target: Path | None = None) -> dict:
        src_frames = extract_frames(src)
        out_frames = extract_frames(out)
        f_src = self.image_features(src_frames)
        f_out = self.image_features(out_frames)
        t = self.text_feature(instruction)

        pair_cos = self._cos(f_out, f_src)
        temporal_src = self._cos(f_src[:-1], f_src[1:])
        temporal_out = self._cos(f_out[:-1], f_out[1:])
        t_out = self._cos(f_out, t)
        t_src = self._cos(f_src, t)
        m_out = self.musiq_score(out_frames)
        m_src = self.musiq_score(src_frames)
        row = {
            "preservation": round(pair_cos, 4),
            "change": round(1.0 - pair_cos, 4),
            "clip_t_out": round(t_out, 4),
            "clip_t_src": round(t_src, 4),
            "clip_t_gain": round(t_out - t_src, 4),
            "temporal_out": round(temporal_out, 4),
            "temporal_src": round(temporal_src, 4),
            "musiq_out": round(m_out, 3),
            "musiq_src": round(m_src, 3),
            "musiq_gain": round(m_out - m_src, 3),
        }
        if self.aesthetic is not None:
            a_out = self._aesthetic(f_out)
            a_src = self._aesthetic(f_src)
            row.update({
                "aes_out": round(a_out, 4),
                "aes_src": round(a_src, 4),
                "aes_gain": round(a_out - a_src, 4),
            })
        if self.dino is not None:
            d_out = self.dino_features(out_frames)
            d_src = self.dino_features(src_frames)
            row["dino_change"] = round(1.0 - self._cos(d_out, d_src), 4)
            if target is not None and target.exists():
                d_tgt = self.dino_features(extract_frames(target))
                gt = self._cos(d_out, d_tgt)
                base = self._cos(d_src, d_tgt)
                row.update({
                    "gt_dino": round(gt, 4),
                    "src_dino": round(base, 4),
                    "gain_dino": round(gt - base, 4),
                })
        if target is not None and target.exists():
            f_tgt = self.image_features(extract_frames(target))
            t_out = self._cos(f_out, f_tgt)
            t_src = self._cos(f_src, f_tgt)
            row.update({
                "target_sim_out": round(t_out, 4),
                "target_sim_src": round(t_src, 4),
                "target_gain": round(t_out - t_src, 4),
            })
        t_hi = self.text_feature(HIGH_COST_PROMPT)
        t_lo = self.text_feature(LOW_COST_PROMPT)
        z_out = self._cos(f_out, t_hi) - self._cos(f_out, t_lo)
        z_src = self._cos(f_src, t_hi) - self._cos(f_src, t_lo)
        row.update({
            "zsq_out": round(z_out, 4),
            "zsq_src": round(z_src, 4),
            "zsq_gain": round(z_out - z_src, 4),
        })
        return row


def iter_arm(dirpath: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(dirpath.glob("*.mp4"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="comma list of NAME=DIR (dir holds <sample_id>.mp4)")
    ap.add_argument("--sources", required=True, help="dir of <sample_id>.mp4 sources")
    ap.add_argument("--targets", default=None, help="optional dir of <sample_id>.mp4 targets (GT-referenced metrics)")
    ap.add_argument("--instructions", required=True, help="jsonl {sample_id, instruction|intent}")
    ap.add_argument("--out", required=True, help="output jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    instructions: dict[str, str] = {}
    for line in Path(args.instructions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        instructions[str(item["sample_id"])] = str(item.get("instruction") or item.get("intent") or "")

    arms = []
    for spec in args.arms.split(","):
        name, _, dirpath = spec.partition("=")
        arms.append((name.strip(), Path(dirpath.strip())))

    scorer = Scorer(device=args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["arm"], row["sample_id"]))

    with out_path.open("a", encoding="utf-8") as fh:
        for arm_name, arm_dir in arms:
            videos = iter_arm(arm_dir)
            n = 0
            for sample_id, out_video in videos.items():
                if (arm_name, sample_id) in done:
                    continue
                src = Path(args.sources) / f"{sample_id}.mp4"
                tgt = Path(args.targets) / f"{sample_id}.mp4" if args.targets else None
                if not src.exists() or sample_id not in instructions:
                    print(f"skip {arm_name}/{sample_id}: missing source or instruction", file=sys.stderr)
                    continue
                try:
                    row = scorer.score_pair(src, out_video, instructions[sample_id], tgt)
                except Exception as exc:  # keep going; report the hole
                    print(f"FAIL {arm_name}/{sample_id}: {type(exc).__name__} {exc}", file=sys.stderr)
                    continue
                row.update({"arm": arm_name, "sample_id": sample_id})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                if args.limit and n >= args.limit:
                    break
            print(f"{arm_name}: scored {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
