# OPUSCAR 98 · The 98 Styles

Source: https://lemomo-ai.github.io/lemo-opuscar/opuscar98/#styles (fetched 2026-10-17)

A 6 min 31 s film. One Clawd walks through all 98 Best Picture winners (1927–2025), and each film is redrawn in a
style that fits it. Per the page, every frame, note and cut was written in code by Claude Opus 5.5, with no video model.
The #styles section gives, for each film, what the style is, a reusable style prompt, and the shot list.

| File | Content |
|---|---|
| `styles.json` | 98 records. See fields below |
| `styles.csv` | Same data flattened to one row per film, for a spreadsheet. Shots are joined into one column |
| `img/opuscar98/01–98.jpg` | One 960×400 still per film, as on the page |
| `img/feature_opuscar98.jpg` | The page's cover image |
| `films/opuscar98.mp4` | The full film, 89 MB |
| `opuscar98/index.html` | The page as saved. Opening it locally shows the stills; fonts load from Google |
| `extract_styles.py` | Rebuilds `styles.json` and `styles.csv` from `index.html` |

## Fields in `styles.json`

| Field | Meaning |
|---|---|
| `no`, `year`, `title_en`, `title_zh` | Film number 1–98, award year, title |
| `decade` | C1–C10, the page's ten decade chapters (C1 = 1927–38 … C10 = 2020–25) |
| `kind` | `2D` (77) / `2.5D` (4) / `3D` (17). 3D means a real 3D scene and camera, mostly three.js |
| `tone` | Colour note, e.g. `B&W`. Blank for most |
| `t0_seconds`, `duration_seconds` | Where this film's segment starts in `opuscar98.mp4`, and how long it lasts |
| `style_en/zh` | Style name, e.g. "Silent-era 3D Dogfight, Tinted Ortho B&W" |
| `blend_en/zh` | The ingredients of the style, e.g. three.js 3D, orthochromatic B&W, projector wear |
| `desc_en/zh` | What the style is and why it suits the film |
| `prompt_en/zh` | The reusable style prompt: palette (hex), grading, texture, camera, cutting rhythm, references, what to avoid |
| `shots` | Shot list: `name`, `seconds`, `text_en`, `text_zh` |
| `direction_en/zh` | How the shots are cut together and hand over to the next film |
| `references` | Films, photographers and artworks the style is modelled on |
| `frame` | The matching still |

To cut one film's segment out of the full video:

```bash
ffmpeg -ss <t0_seconds> -t <duration_seconds> -i films/opuscar98.mp4 -c copy no01.mp4
```
