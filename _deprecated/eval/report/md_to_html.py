#!/usr/bin/env python3
"""Render the group-meeting report to a self-contained HTML page with playable clips.

    python3 scripts/eval/report/md_to_html.py out/report/组会报告.md

Writes the .html next to the .md. Handles only the Markdown the report uses
(headings, rules, pipe tables, bullet lists, paragraphs, bold, inline code,
links, images). An image pointing at clips/<name>.png becomes a <video> of
clips/<name>.mp4 when that file exists, so the page plays every example in the
browser; video paths stay relative, so keep the .html next to clips/.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

CSS = """
body{font-family:-apple-system,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif;
max-width:1500px;margin:24px auto;padding:0 24px;color:#222;line-height:1.6}
h1{font-size:28px;border-bottom:2px solid #333;padding-bottom:6px}
h2{font-size:22px;margin-top:36px;border-bottom:1px solid #ccc;padding-bottom:4px}
h3{font-size:18px;margin-top:26px}
table{border-collapse:collapse;margin:10px 0}
th,td{border:1px solid #ccc;padding:4px 10px;text-align:left;vertical-align:top}
th{background:#f3f3f3}
code{background:#f3f3f3;padding:1px 4px;border-radius:3px;font-size:90%}
video{width:100%;max-width:1500px;display:block;margin:8px 0;background:#000}
hr{border:none;border-top:1px solid #ddd;margin:28px 0}
"""


def inline(text: str, base: Path) -> str:
    out = html.escape(text, quote=False)

    def image(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        mp4 = Path(src).with_suffix(".mp4")
        if src.startswith("clips/") and (base / mp4).exists():
            return (f'<video controls loop preload="auto" src="{html.escape(str(mp4))}"'
                    f' title="{html.escape(alt)}"></video>')
        return f'<img src="{html.escape(src)}" alt="{html.escape(alt)}" style="max-width:100%">'

    out = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def render(md: str, base: Path) -> str:
    lines = md.splitlines()
    body: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            body.append(f"<h{level}>{inline(heading.group(2), base)}</h{level}>")
            i += 1
            continue
        if stripped == "---":
            body.append("<hr>")
            i += 1
            continue
        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, rest = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            body.append("<table><tr>" + "".join(f"<th>{inline(c, base)}</th>" for c in head) + "</tr>")
            body.extend("<tr>" + "".join(f"<td>{inline(c, base)}</td>" for c in r) + "</tr>" for r in rest)
            body.append("</table>")
            continue
        if re.match(r"^(-|\d+\.)\s+", stripped):
            ordered = bool(re.match(r"^\d+\.", stripped))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines) and re.match(r"^(-|\d+\.)\s+", lines[i].strip()):
                items.append(re.sub(r"^(-|\d+\.)\s+", "", lines[i].strip()))
                i += 1
            body.append(f"<{tag}>" + "".join(f"<li>{inline(x, base)}</li>" for x in items) + f"</{tag}>")
            continue
        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||-|\d+\.\s|---)", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        content = inline(" ".join(para), base)
        # A standalone image renders as a block video, not inside a paragraph.
        body.append(content if content.startswith("<video") else f"<p>{content}</p>")
    return "\n".join(body)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: md_to_html.py <report.md>")
    md_path = Path(sys.argv[1])
    base = md_path.parent
    title = next((l.lstrip("# ").strip() for l in md_path.read_text(encoding="utf-8").splitlines() if l.startswith("# ")), md_path.stem)
    page = (f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body>\n{render(md_path.read_text(encoding='utf-8'), base)}\n</body></html>\n")
    out = md_path.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  videos={page.count('<video')}")


if __name__ == "__main__":
    main()
