"""Tests for the bench_v3 judges. No network: a scripted fake client stands in."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import judges  # noqa: E402
import make_label_kit  # noqa: E402
import run_checklist  # noqa: E402


# ------------------------------------------------------------------ fixtures


def _write_video(path: Path, n: int, fps: int, value: int = 0, size=(64, 48)) -> Path:
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), (value + i * 3) % 255, np.uint8), format="rgb24")
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


@pytest.fixture()
def videos(tmp_path):
    src = _write_video(tmp_path / "src.mp4", 40, 10)          # 4.0 s
    out = _write_video(tmp_path / "out.mp4", 17, 20, 100)     # 0.85 s
    return {"src": src, "out": out, "dir": tmp_path}


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, *, system_prompt, user_text, images_b64):
        self.calls.append({"user": user_text, "n_images": len(images_b64)})
        reply = self.replies.pop(0)
        return reply(user_text) if callable(reply) else reply


def _checklist_reply(change, preserve, extra=()):
    return json.dumps({
        "change": [{"id": i + 1, "label": l, "evidence": "e"} for i, l in enumerate(change)],
        "preserve": [{"id": i + 1, "label": l, "evidence": "e"} for i, l in enumerate(preserve)],
        "unrequested_changes": list(extra),
    })


# ------------------------------------------------------------------- parsing


def test_parse_json_fenced_and_dirty():
    assert judges.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert judges.parse_json("<think>hmm {not json}</think>Sure! {\"a\": 2} trailing") == {"a": 2}
    assert judges.parse_json('prefix {"a": {"b": [1, 2]}} suffix') == {"a": {"b": [1, 2]}}
    with pytest.raises(judges.JudgeOutputError):
        judges.parse_json("no json here")


def test_validate_checklist_rejects_wrong_counts_and_labels():
    ok = json.loads(_checklist_reply(["done"], ["kept"]))
    assert judges.validate_checklist(ok, 1, 1)["change"][0]["label"] == "done"
    with pytest.raises(ValueError):
        judges.validate_checklist(ok, 2, 1)
    bad = json.loads(_checklist_reply(["maybe"], ["kept"]))
    with pytest.raises(ValueError):
        judges.validate_checklist(bad, 1, 1)


def test_validate_pairwise_normalizes_case():
    out = judges.validate_pairwise({"fidelity_to_requirement": "a", "preservation": "Tie", "visual_quality": "B"})
    assert out["fidelity_to_requirement"] == "A" and out["preservation"] == "tie" and out["visual_quality"] == "B"
    with pytest.raises(ValueError):
        judges.validate_pairwise({"fidelity_to_requirement": "C", "preservation": "A", "visual_quality": "A"})


# ---------------------------------------------------------------- aggregation


def test_score_checklist():
    result = judges.validate_checklist(json.loads(_checklist_reply(
        ["done", "partial", "not_done", "done"], ["kept", "changed"], ["new hat"])), 4, 2)
    s = judges.score_checklist(result)
    assert s["fidelity"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert s["preservation"] == 0.5
    assert (s["n_done"], s["n_partial"], s["n_not_done"], s["n_unrequested"]) == (2, 1, 1, 1)


def test_summarize_checklist_ignores_errors_and_uses_latest():
    good = {"arm": "x", "sample_id": "s1", "error": "", "scores": judges.score_checklist(
        judges.validate_checklist(json.loads(_checklist_reply(["done"], ["kept"])), 1, 1))}
    failed = {"arm": "x", "sample_id": "s2", "error": "boom"}
    superseded = {"arm": "x", "sample_id": "s1", "error": "earlier failure"}
    summary = judges.summarize_checklist([superseded, good, failed])["x"]
    assert summary["n"] == 2 and summary["errors"] == 1
    assert summary["fidelity"] == 1.0 and summary["preservation"] == 1.0


def test_combine_orders_disagreement_is_tie():
    fwd = {"fidelity_to_requirement": "A", "preservation": "A", "visual_quality": "tie"}
    rev = {"fidelity_to_requirement": "B", "preservation": "A", "visual_quality": "tie"}
    v = judges.combine_orders(fwd, rev, "x", "y")
    # forward A=x, reverse B=x -> agree on x; preservation: forward x, reverse y -> tie
    assert v == {"fidelity_to_requirement": "x", "preservation": "tie", "visual_quality": "tie"}


def test_summarize_pairwise_first_slot_and_sign():
    rows = []
    for i in range(6):
        rows.append({"sample_id": f"s{i}", "error": "", "identical_bytes": False,
                     "verdict": {q: "x" for q in judges.QUESTIONS},
                     "forward": {q: "A" for q in judges.QUESTIONS},
                     "reverse": {q: "B" for q in judges.QUESTIONS}})
    s = judges.summarize_pairwise(rows, "x", "y")["fidelity_to_requirement"]
    assert s["wins_x"] == 6 and s["wins_y"] == 0 and s["ties"] == 0
    assert s["first_slot_win_rate"] == 0.5
    assert s["sign_test_p"] == pytest.approx(2 / 64)


# -------------------------------------------------------------- frames / calls


def test_aligned_frames_follow_anchor_duration(videos):
    times, (out_f, src_f) = judges.aligned_frames(videos["out"], [videos["src"]], 4)
    assert len(times) == 4 and times[-1] < 0.85
    assert len(out_f) == len(src_f) == 4
    # source frames come from the first 0.85 s (indices <= 8 at 10 fps), i.e. dark values
    assert max(np.asarray(f).mean() for f in src_f) < 40


def test_judge_checklist_retries_once_then_scores(videos):
    task = {"sample_id": "s", "source_video": str(videos["src"]), "requirement": "make it golden",
            "checklist": ["gold", "silk"], "preserve": ["face"]}
    client = FakeClient(["garbage", _checklist_reply(["done", "not_done"], ["kept"])])
    r = judges.judge_checklist(client, task, videos["out"], frames=3)
    assert r["calls"] == 2 and r["scores"]["fidelity"] == 0.5
    assert client.calls[0]["n_images"] == 6  # 3 source + 3 output, interleaved
    assert "SOURCE t=" in client.calls[0]["user"] and "OUTPUT t=" in client.calls[0]["user"]
    assert "previous answer was invalid" in client.calls[1]["user"]


def test_judge_checklist_fails_after_second_bad_answer(videos):
    task = {"sample_id": "s", "source_video": str(videos["src"]), "requirement": "r", "checklist": ["a"], "preserve": ["b"]}
    with pytest.raises(judges.JudgeOutputError):
        judges.judge_checklist(FakeClient(["nope", "still nope"]), task, videos["out"], frames=2)


def test_pairwise_identical_files_skip_the_call(videos):
    copy = videos["dir"] / "copy.mp4"
    shutil.copy(videos["out"], copy)
    client = FakeClient([])
    task = {"sample_id": "s", "source_video": str(videos["src"]), "requirement": "r"}
    r = judges.judge_pairwise(client, task, videos["out"], copy, "x", "y")
    assert r["identical_bytes"] and client.calls == []
    assert set(r["verdict"].values()) == {"tie"}


def test_pairwise_swaps_order(videos):
    other = _write_video(videos["dir"] / "other.mp4", 17, 20, 200)
    reply = json.dumps({"fidelity_to_requirement": "A", "preservation": "B", "visual_quality": "A", "reason": "r"})
    client = FakeClient([reply, reply])  # always "A" -> position bias -> must become a tie
    task = {"sample_id": "s", "source_video": str(videos["src"]), "requirement": "r"}
    r = judges.judge_pairwise(client, task, videos["out"], other, "x", "y", frames=2)
    assert len(client.calls) == 2 and client.calls[0]["n_images"] == 6
    assert r["verdict"] == {"fidelity_to_requirement": "tie", "preservation": "tie", "visual_quality": "tie"}


# ------------------------------------------------------------------- resume


def test_pending_skips_done_retries_failed(tmp_path):
    jobs = [{"arm": "x", "sample_id": f"s{i}"} for i in range(3)]
    done = [{"arm": "x", "sample_id": "s0", "error": ""}, {"arm": "x", "sample_id": "s1", "error": "boom"}]
    todo = judges.pending(jobs, done, ("arm", "sample_id"))
    assert [j["sample_id"] for j in todo] == ["s1", "s2"]


def test_runner_appends_and_records_errors(tmp_path):
    out = tmp_path / "o.jsonl"

    def work(job):
        if job["sample_id"] == "bad":
            raise RuntimeError("x")
        return {"scores": {"fidelity": 1.0}}

    judges.Runner(out, 2).run([{"arm": "a", "sample_id": "ok"}, {"arm": "a", "sample_id": "bad"}], work, log=lambda _: None)
    rows = {r["sample_id"]: r for r in judges.read_jsonl(out)}
    assert rows["ok"]["error"] == "" and rows["bad"]["error"].startswith("RuntimeError")
    assert [j["sample_id"] for j in judges.pending([{"arm": "a", "sample_id": "ok"}, {"arm": "a", "sample_id": "bad"}],
                                                   judges.read_jsonl(out), ("arm", "sample_id"))] == ["bad"]


def test_build_jobs_reports_missing(tmp_path):
    d = tmp_path / "arm"
    d.mkdir()
    (d / "s1.mp4").write_bytes(b"x")
    jobs, missing = run_checklist.build_jobs([{"sample_id": "s1"}, {"sample_id": "s2"}], [("a", d)])
    assert [j["sample_id"] for j in jobs] == ["s1"] and missing == ["a/s2"]


# ---------------------------------------------------------------- label kit


def test_label_pairs_deterministic_and_balanced():
    ids = [f"s{i}" for i in range(40)]
    p1 = judges.make_label_pairs(ids, "x", "y", 30, seed=0)
    p2 = judges.make_label_pairs(ids, "x", "y", 30, seed=0)
    assert p1 == p2 and len(p1) == 30
    assert {(p["left"], p["right"]) for p in p1} == {("x", "y"), ("y", "x")}


def test_label_kit_round_trip(tmp_path, videos):
    ids = ["s1", "s2", "s3"]
    dirs = {}
    for arm in ("x", "y"):
        dirs[arm] = tmp_path / arm
        dirs[arm].mkdir()
        for sid in ids:
            shutil.copy(videos["out"], dirs[arm] / f"{sid}.mp4")
    tasks = {sid: {"sample_id": sid, "source_video": str(videos["src"]), "requirement": f"req {sid}"} for sid in ids}
    pairs = judges.make_label_pairs(ids, "x", "y", 3, seed=1)
    page = make_label_kit.render_page(pairs, tasks, dirs, tmp_path)
    assert all(p["pair_id"] in page for p in pairs)
    # the page must not reveal which arm is left/right
    assert '"x"' not in page.split("const PAIRS")[1].split(";")[0]

    # human says the arm "x" wins everything; judge says x on fidelity, y on preservation, tie on quality
    answers = {p["pair_id"]: {q: ("left" if p["left"] == "x" else "right") for q in judges.QUESTIONS} for p in pairs}
    judge_rows = [{"sample_id": sid, "error": "", "verdict": {
        "fidelity_to_requirement": "x", "preservation": "y", "visual_quality": "tie"}} for sid in ids]
    agree = judges.score_human_agreement(pairs, answers, judge_rows)
    assert agree["fidelity_to_requirement"] == {"n": 3, "agreement": 1.0}
    assert agree["preservation"]["agreement"] == 0.0
    assert agree["visual_quality"]["agreement"] == 0.0
