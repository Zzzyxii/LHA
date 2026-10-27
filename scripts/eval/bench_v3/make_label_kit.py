#!/usr/bin/env python3
"""Human-label kit for calibrating the pairwise judge.

Build a self-contained HTML page with N (source, left, right) pairs; which arm is
left/right is randomized per pair and recorded only in answer_key.json:

    python3 scripts/eval/bench_v3/make_label_kit.py build \
      --tasks tasks.jsonl --arm-a ours_raw=dir_a --arm-b intent_direct=dir_b \
      -n 30 --output-dir out/bench_v3/label_kit

The labeller opens index.html, answers three questions per pair and clicks
"Export answers", which downloads answers.json. Score the judge against them:

    python3 scripts/eval/bench_v3/make_label_kit.py score \
      --key out/bench_v3/label_kit/answer_key.json --answers answers.json \
      --judge out/bench_v3/pairwise/ours_raw__vs__intent_direct.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judges  # noqa: E402

QUESTION_TEXT = {
    "fidelity_to_requirement": "哪个更好地完成了要求?(改对了吗)",
    "preservation": "和原视频比,哪个更好地保住了不该变的东西(人物身份、表情、动作、构图、机位)?",
    "visual_quality": "不管要求,哪个看起来更像专业高制作的画面?",
}

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>bench_v3 标注</title>
<style>
body{{font-family:sans-serif;margin:20px;max-width:1500px}}
.pair{{border:1px solid #ccc;border-radius:8px;padding:12px;margin:18px 0}}
.vids{{display:flex;gap:8px}} .vids div{{flex:1}} video{{width:100%}}
.req{{background:#f6f6f6;padding:8px;white-space:pre-wrap;font-size:13px}}
.q{{margin:6px 0}} .done{{border-color:#3a3}}
</style></head><body>
<h2>bench_v3 成对标注({n} 对)</h2>
<p>每对有三段视频:原视频、左、右。每个问题独立回答,可以一题选左、一题选右。答案会自动保存在浏览器里;全部做完后点最下面的「导出答案」。</p>
{body}
<button onclick="exportAnswers()" style="font-size:18px;padding:8px 20px">导出答案 answers.json</button>
<script>
const PAIRS = {pairs_json};
const QS = {questions_json};
const STORE = "bench_v3_labels_{token}";
let answers = JSON.parse(localStorage.getItem(STORE) || "{{}}");
function save(pid, q, v) {{
  answers[pid] = answers[pid] || {{}}; answers[pid][q] = v;
  localStorage.setItem(STORE, JSON.stringify(answers)); mark(pid);
}}
function mark(pid) {{
  const done = answers[pid] && QS.every(q => answers[pid][q]);
  document.getElementById(pid).classList.toggle("done", !!done);
}}
function playAll(pid) {{
  document.querySelectorAll("#" + pid + " video").forEach(v => {{ v.currentTime = 0; v.play(); }});
}}
function exportAnswers() {{
  const blob = new Blob([JSON.stringify(answers, null, 2)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "answers.json"; a.click();
}}
PAIRS.forEach(p => {{
  QS.forEach(q => {{
    const v = answers[p.pair_id] && answers[p.pair_id][q];
    if (v) {{ const el = document.querySelector(`input[name="${{p.pair_id}}_${{q}}"][value="${{v}}"]`); if (el) el.checked = true; }}
  }});
  mark(p.pair_id);
}});
</script></body></html>
"""


def _src(path: Path, page_dir: Path, absolute: bool) -> str:
    return path.resolve().as_uri() if absolute else os.path.relpath(path.resolve(), page_dir.resolve())


def render_page(pairs: list[dict], tasks: dict[str, dict], dirs: dict[str, Path], page_dir: Path,
                absolute: bool = False) -> str:
    blocks = []
    for p in pairs:
        task = tasks[p["sample_id"]]
        vids = [("原视频", Path(task["source_video"])), ("左", dirs[p["left"]] / f"{p['sample_id']}.mp4"),
                ("右", dirs[p["right"]] / f"{p['sample_id']}.mp4")]
        video_html = "".join(
            f'<div><b>{label}</b><video src="{html.escape(_src(v, page_dir, absolute))}" muted loop controls></video></div>'
            for label, v in vids
        )
        refs = "".join(
            f'<img src="{html.escape(_src(Path(r), page_dir, absolute))}" style="height:120px;margin-right:6px">'
            for r in task.get("reference_images") or []
        )
        questions = "".join(
            f'<div class="q">{html.escape(text)} '
            + "".join(
                f'<label><input type="radio" name="{p["pair_id"]}_{q}" value="{v}" '
                f'onchange="save(\'{p["pair_id"]}\',\'{q}\',\'{v}\')"> {label}</label> '
                for v, label in (("left", "左"), ("tie", "差不多"), ("right", "右"))
            )
            + "</div>"
            for q, text in QUESTION_TEXT.items()
        )
        blocks.append(
            f'<div class="pair" id="{p["pair_id"]}"><b>{p["pair_id"]}</b> '
            f'<button onclick="playAll(\'{p["pair_id"]}\')">同步播放</button>'
            f'<div class="req">{html.escape(task["requirement"])}</div>{refs}'
            f'<div class="vids">{video_html}</div>{questions}</div>'
        )
    public = [{"pair_id": p["pair_id"]} for p in pairs]
    # Stable across rebuilds of the same pairs, so saved browser answers survive a re-render.
    token = hashlib.sha1("".join(p["pair_id"] + p["sample_id"] for p in pairs).encode()).hexdigest()[:8]
    return PAGE.format(n=len(pairs), body="\n".join(blocks), pairs_json=json.dumps(public),
                       questions_json=json.dumps(list(QUESTION_TEXT)), token=token)


def build(args: argparse.Namespace) -> None:
    (arm_a, dir_a), (arm_b, dir_b) = judges.parse_arm(args.arm_a), judges.parse_arm(args.arm_b)
    tasks = {t["sample_id"]: t for t in judges.read_jsonl(args.tasks)}
    usable = [sid for sid in tasks if (dir_a / f"{sid}.mp4").exists() and (dir_b / f"{sid}.mp4").exists()]
    pairs = judges.make_label_pairs(usable, arm_a, arm_b, args.n, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page = render_page(pairs, tasks, {arm_a: dir_a, arm_b: dir_b}, args.output_dir, args.absolute_paths)
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    (args.output_dir / "answer_key.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"{len(pairs)} pairs -> {args.output_dir / 'index.html'} (key: answer_key.json; do not show the labeller)")


def score(args: argparse.Namespace) -> None:
    key = json.loads(args.key.read_text(encoding="utf-8"))
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    result = judges.score_human_agreement(key, answers, judges.read_jsonl(args.judge))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tasks", type=Path, required=True)
    b.add_argument("--arm-a", required=True)
    b.add_argument("--arm-b", required=True)
    b.add_argument("-n", type=int, default=30)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--absolute-paths", action="store_true", help="file:// URIs instead of paths relative to the page")
    s = sub.add_parser("score")
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--answers", type=Path, required=True)
    s.add_argument("--judge", type=Path, required=True, help="run_pairwise.py output jsonl for the same two arms")
    args = parser.parse_args()
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
