"""Extract the 98-style catalogue on OPUSCAR 98's #styles section into styles.json and styles.csv.

    python3 extract_styles.py

Reads opuscar98/index.html (the saved page). One record per film: title, year, decade, kind (2D / 2.5D / 3D),
tone, the style name and its ingredients, the description, the reusable style prompt, the shot list with
per-shot durations, the direction note, the references, and the matching still in img/opuscar98/.
Every text field comes in English and Chinese, as the page shows both.
"""
import csv
import html
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
PAGE = (HERE / "opuscar98" / "index.html").read_text(encoding="utf-8")


def clean(fragment: str) -> str:
    """Visible text of an HTML fragment: tags dropped, entities decoded, whitespace collapsed."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def by_lang(fragment: str, tag: str = "span") -> dict:
    """The en / zh-CN pair of `<tag lang=...>` elements directly in a fragment."""
    out = {}
    for lang, key in (("en", "en"), ("zh-CN", "zh")):
        m = re.search(rf'<{tag} lang="{lang}">(.*?)</{tag}>', fragment, re.S)
        out[key] = clean(m.group(1)) if m else ""
    return out


def first(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.S)
    return m.group(1) if m else default


def film(no: int, art: str) -> dict:
    attr = lambda k: html.unescape(first(rf'data-{k}="([^"]*)"', art))
    style = first(r'<p class="style">(.*?)</p>', art)
    # the kind badge is the first span; the style name is the en/zh pair after it
    style_name = by_lang(re.sub(r'<span class="kind".*?</span>\s*</small>\s*</span>', "", style, flags=re.S))
    blend = first(r'<ul class="blend">(.*?)</ul>', art)
    prompt = first(r'<div class="prompt">(.*?)<button', art)
    shots = [{
        "name": clean(first(r"<b>(.*?)</b>", li)),
        "seconds": float(first(r"<i>([\d.]+)s</i>", li, "0")),
        **{f"text_{k}": v for k, v in by_lang(li, "p").items()},
    } for li in re.findall(r"<li>(.*?)</li>", first(r'<ol class="shots">(.*?)</ol>', art), re.S)]
    title = re.search(r"<h4>(.*?)<small>(.*?)</small></h4>", art, re.S)
    return {
        "no": no,
        "year": clean(first(r'class="no"[^>]*>.*?</a><span>(.*?)</span>', art)),
        "title_en": clean(title.group(1)) if title else "",
        "title_zh": clean(title.group(2)) if title else "",
        "decade": attr("ch"),
        "kind": attr("k"),
        "tone": attr("tone"),
        "t0_seconds": float(attr("t0") or 0),
        "duration_seconds": float(attr("dur") or 0),
        "style_en": style_name["en"], "style_zh": style_name["zh"],
        "blend_en": [clean(x) for x in re.findall(r'<span lang="en">(.*?)</span>', blend, re.S)],
        "blend_zh": [clean(x) for x in re.findall(r'<span lang="zh-CN">(.*?)</span>', blend, re.S)],
        **{f"desc_{k}": v for k, v in by_lang(first(r'<p class="desc">(.*?)</p>', art)).items()},
        **{f"prompt_{k}": v for k, v in by_lang(prompt, "p").items()},
        "shots": shots,
        **{f"direction_{k}": v for k, v in by_lang(first(r'<p class="dir">(.*?)</p>', art)).items()},
        "references": clean(re.sub(r"<span lang=[^>]*>.*?</span>", "", first(r'<p class="refs">(.*?)</p>', art), flags=re.S)),
        "frame": f"img/opuscar98/{no:02d}.jpg",
    }


def main() -> None:
    films = [film(int(no), art) for no, art in
             re.findall(r'<article class="film" id="no-(\d+)"(.*?)</article>', PAGE, re.S)]
    (HERE / "styles.json").write_text(json.dumps(films, ensure_ascii=False, indent=1), encoding="utf-8")
    flat = ["no", "year", "title_en", "title_zh", "decade", "kind", "tone", "t0_seconds", "duration_seconds",
            "style_en", "style_zh", "blend_en", "blend_zh", "desc_en", "desc_zh", "prompt_en", "prompt_zh",
            "direction_en", "direction_zh", "references", "frame"]
    with (HERE / "styles.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([*flat, "shots"])
        for r in films:
            writer.writerow([*("; ".join(r[k]) if isinstance(r[k], list) else r[k] for k in flat),
                             " | ".join(f"{s['name']} {s['seconds']}s: {s['text_en']}" for s in r["shots"])])
    print(f"{len(films)} films -> styles.json, styles.csv")


if __name__ == "__main__":
    main()
