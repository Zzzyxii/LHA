#!/usr/bin/env python3
"""Side-by-side comparison clip plus a still sheet, for reports and slides.

    python3 scripts/eval/report/make_compare.py --out out/report/videos/c1_0133 \
        --title "0133  指令里写着 target 的颜色" \
        --panel "low 输入=path/low.mp4" --panel "high 标准答案=path/high.mp4"

Writes <out>.mp4 (panels play in lockstep) and <out>.png (rows = 15% / 50% / 85%
of the clip). Panels are aligned by timestamp, not frame index: the published
baselines were exported at 17 frames (0.85 s) while ours follow the source
duration, so a short clip holds its last frame and its label says so. A panel
path ending in an image suffix is shown as a static frame (reference images).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 416, 240
LABEL_H = 26
TITLE_H = 34
OUT_FPS = 16
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# This image ships no CJK font. The Noto CJK copy below lives on the shared disk;
# REPORT_FONT overrides it.
FONT_CANDIDATES = (
    os.environ.get("REPORT_FONT", ""),
    "/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/StoryTelling/VGen/assets/fonts/NotoSansCJKsc-Regular.otf",
)
FONT_DIRS = ("/usr/share/fonts/cjk", "/usr/share/fonts")


def _draws_cjk(font) -> bool:
    """True only if two different hanzi render differently.

    A font without CJK glyphs draws the same .notdef box for every character, which
    has a nonzero bbox -- so checking the bbox alone accepts fonts that print boxes.
    """
    def ink(ch: str) -> np.ndarray:
        image = Image.new("L", (40, 40), 0)
        ImageDraw.Draw(image).text((2, 2), ch, fill=255, font=font)
        return np.asarray(image)

    a, b = ink("中"), ink("国")
    return bool(a.sum()) and not np.array_equal(a, b)


def _font(size: int):
    paths = [Path(p) for p in FONT_CANDIDATES if p]
    for root in FONT_DIRS:
        for pattern in ("*.tt[cf]", "*.otf"):
            paths.extend(sorted(Path(root).rglob(pattern)))
    for candidate in paths:
        if not candidate.is_file():
            continue
        try:
            font = ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
        if _draws_cjk(font):
            return font
    raise SystemExit("no CJK-capable font found; set REPORT_FONT to a .ttf/.otf with Chinese glyphs")


class Source:
    def __init__(self, label: str, path: Path) -> None:
        self.label = label
        self.path = path
        if path.suffix.lower() in IMAGE_SUFFIXES:
            self.frames = [np.asarray(Image.open(path).convert("RGB"))]
            self.fps = None
            self.duration = None
        else:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                self.fps = float(stream.average_rate or 16)
                self.frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            self.duration = len(self.frames) / self.fps

    def at(self, t: float) -> np.ndarray:
        if self.fps is None:
            return self.frames[0]
        return self.frames[min(len(self.frames) - 1, int(t * self.fps))]


def _fit(frame: np.ndarray) -> Image.Image:
    """Letterbox into W x H so 832x480 and 1280x720 sources share one layout."""
    image = Image.fromarray(frame)
    scale = min(W / image.width, H / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(resized, ((W - resized.width) // 2, (H - resized.height) // 2))
    return canvas


def _compose(sources: list[Source], t: float, title: str, label_font, title_font, master: float) -> np.ndarray:
    width = W * len(sources)
    sheet = Image.new("RGB", (width, TITLE_H + LABEL_H + H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), title, fill=(0, 0, 0), font=title_font)
    for i, src in enumerate(sources):
        label = src.label
        # Only flag genuinely truncated clips (the 17-frame published exports); a
        # few frames lost to 4n+1 rounding is not worth a label.
        if src.duration is not None and src.duration < master - 0.5:
            label += f"(仅 {src.duration:.2f}s)"
        draw.text((i * W + 6, TITLE_H + 3), label, fill=(0, 0, 0), font=label_font)
        sheet.paste(_fit(src.at(t)), (i * W, TITLE_H + LABEL_H))
    return np.asarray(sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lockstep comparison clip + still sheet.")
    parser.add_argument("--out", type=Path, required=True, help="output stem (writes .mp4 and .png)")
    parser.add_argument("--title", default="")
    parser.add_argument("--panel", action="append", required=True, help='"label=path"; repeat, max 5')
    args = parser.parse_args()

    sources = []
    for spec in args.panel:
        label, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"missing panel file: {p}")
        sources.append(Source(label.strip(), p))
    durations = [s.duration for s in sources if s.duration is not None]
    if not durations:
        raise SystemExit("need at least one video panel")
    # The first video panel (normally the low input) sets the clock: a longer
    # unrelated clip, e.g. the swapped-sample adversarial control, must not make
    # the reference panels look truncated.
    master = durations[0]

    label_font, title_font = _font(17), _font(19)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    times = [i / OUT_FPS for i in range(max(1, int(master * OUT_FPS)))]
    writer = imageio.get_writer(str(args.out.with_suffix(".mp4")), fps=OUT_FPS, quality=8, macro_block_size=1)
    try:
        for t in times:
            writer.append_data(_compose(sources, t, args.title, label_font, title_font, master))
    finally:
        writer.close()

    rows = [_compose(sources, f * master, args.title if i == 0 else "", label_font, title_font, master)
            for i, f in enumerate((0.15, 0.5, 0.85))]
    rows = [rows[0]] + [r[TITLE_H:] for r in rows[1:]]
    Image.fromarray(np.vstack(rows)).save(args.out.with_suffix(".png"))
    print(f"wrote {args.out.with_suffix('.mp4')} and .png ({master:.2f}s, {len(sources)} panels)")


if __name__ == "__main__":
    main()
