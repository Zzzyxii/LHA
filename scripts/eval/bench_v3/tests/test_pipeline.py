"""Tests for the bench_v3 data/run plumbing (no network, no GPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "scripts/bench_skill"))
sys.path.insert(0, str(ROOT / "src"))

import aggregate  # noqa: E402
import build_track_a  # noqa: E402
import make_instruction_file  # noqa: E402
import run_bench  # noqa: E402
import run_eval  # noqa: E402


def _write_video(path: Path, n: int = 12, fps: int = 6) -> Path:
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 10 % 255, np.uint8), format="rgb24")
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


# ------------------------------------------------------------- build_track_a


def test_tag_tokens_and_jaccard():
    a = build_track_a.tag_tokens(["period_headpiece", "cinematic_lighting"])
    b = build_track_a.tag_tokens(["ornate_headpiece", "dramatic lighting"])
    assert a == {"period", "headpiece", "cinematic", "lighting"}
    assert build_track_a.jaccard(a, b) == pytest.approx(2 / 6)
    assert build_track_a.jaccard(set(), set()) == 0.0


def test_mid_frame_is_cached(tmp_path):
    video = _write_video(tmp_path / "v.mp4")
    out = build_track_a.mid_frame(video, tmp_path / "f.png")
    stamp = out.stat().st_mtime_ns
    assert build_track_a.mid_frame(video, out) == out and out.stat().st_mtime_ns == stamp


# ------------------------------------------------------- make_instruction_file

ROW = {"sample_id": "s1", "source_video": "/x/s1.mp4", "t1": "one line", "t3": "expert text",
       "t2_same_reference_images": ["/r/same.png"], "t2_cross_reference_images": ["/r/cross.png"],
       "t2_reference_images": ["/r/still.png"]}


@pytest.mark.parametrize("tier,intent,refs", [
    ("t1", "one line", None),
    ("t3", "expert text", None),
    ("t2_same", "one line", ["/r/same.png"]),
    ("t2_cross", "one line", ["/r/cross.png"]),
    ("t2", "one line", ["/r/still.png"]),
])
def test_instruction_tiers(tier, intent, refs):
    (item,) = make_instruction_file.build([ROW], tier)
    assert item["intent"] == intent and item["source_video"] == "/x/s1.mp4"
    assert item.get("reference_images") == refs


def test_reference_tier_requires_references():
    with pytest.raises(SystemExit):
        make_instruction_file.build([{**ROW, "t2_same_reference_images": []}], "t2_same")


# ----------------------------------------------------------------- run_bench


def test_read_index_json_and_jsonl(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([{"sample_id": "x"}]))
    (tmp_path / "b.jsonl").write_text('{"sample_id": "y"}\n\n{"sample_id": "z"}\n')
    assert run_bench._read_index(tmp_path / "a.json") == [{"sample_id": "x"}]
    assert [r["sample_id"] for r in run_bench._read_index(tmp_path / "b.jsonl")] == ["y", "z"]


def test_apply_instruction_file_attaches_references(tmp_path):
    path = tmp_path / "i.jsonl"
    path.write_text(json.dumps({"sample_id": "s1", "intent": " hi ", "reference_images": ["/r.png"]}) + "\n"
                    + json.dumps({"sample_id": "s2", "intent": "yo"}) + "\n")
    rows = run_bench.apply_instruction_file([{"sample_id": "s1", "instruction": "expert"}, {"sample_id": "s2"}], path)
    assert rows[0]["instruction"] == "hi" and rows[0]["reference_images"] == ["/r.png"]
    assert rows[1]["reference_images"] == []


def test_apply_instruction_file_refuses_partial_coverage(tmp_path):
    path = tmp_path / "i.jsonl"
    path.write_text(json.dumps({"sample_id": "s1", "intent": "hi"}) + "\n")
    with pytest.raises(SystemExit):
        run_bench.apply_instruction_file([{"sample_id": "s1"}, {"sample_id": "missing"}], path)


def test_reference_arms_are_raw_or_translate():
    for name, spec in run_bench.ARMS.items():
        if spec.get("vace_reference"):
            assert spec["mode"] in {"raw", "translate"}, name


# ---------------------------------------------------------- reference translator


def test_translator_orders_images_and_cleans_text(monkeypatch):
    import planner.reference_translator as rt

    monkeypatch.setattr(rt, "extract_frames", lambda video, max_frames: [f"/f{i}.png" for i in range(max_frames)])

    class Fake:
        def predict_json(self, *, system_prompt, user_text, image_paths):
            self.images, self.text = image_paths, user_text
            return {"instruction": " Make  it\nred. ", "adopted_from_reference": ["red robe"]}

    fake = Fake()
    out = rt.translate_reference(intent="x", low_video="/v.mp4", reference_images=["/ref.png"], model=fake)
    assert fake.images == ["/f0.png", "/f1.png", "/f2.png", "/f3.png", "/ref.png"]  # source first, then reference
    assert "first 4 images are SOURCE" in fake.text
    assert out.instruction == "Make it red." and out.adopted_from_reference == ["red robe"]


def test_translation_cache_shared_and_input_checked(tmp_path, monkeypatch):
    import planner.reference_translator as rt

    calls = []
    monkeypatch.setattr(rt, "translate_reference",
                        lambda **kw: calls.append(kw) or rt.ReferenceTranslation(instruction="go red", adopted_from_reference=["red"]))
    args = dict(cache_dir=tmp_path, sample_id="s1", intent="x", low_video="/v.mp4", reference_images=["/r.png"])
    first = rt.load_or_translate(**args)
    second = rt.load_or_translate(**args)
    assert first == second and first["instruction"] == "go red" and len(calls) == 1
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(RuntimeError):
        rt.load_or_translate(**{**args, "reference_images": ["/other.png"]})


def test_translator_rejects_empty_instruction(monkeypatch):
    import planner.reference_translator as rt

    monkeypatch.setattr(rt, "extract_frames", lambda video, max_frames: ["/f.png"])

    class Fake:
        def predict_json(self, **_):
            return {"instruction": "   "}

    with pytest.raises(RuntimeError):
        rt.translate_reference(intent="x", low_video="/v.mp4", reference_images=["/r.png"], model=Fake())
    with pytest.raises(ValueError):
        rt.translate_reference(intent="x", low_video="/v.mp4", reference_images=[], model=Fake())


# ------------------------------------------------------------ eval plumbing


def test_ready_arms_skips_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "task_ids", lambda tier: ["a", "b"])
    full, part = tmp_path / "full", tmp_path / "part"
    full.mkdir(), part.mkdir()
    for d, ids in ((full, "ab"), (part, "a")):
        for s in ids:
            (d / f"{s}.mp4").write_bytes(b"")
    assert set(run_eval.ready_arms("t", {"full": str(full), "part": str(part), "gone": str(tmp_path / "x")})) == {"full"}


def test_eval_config_arms_resolve():
    cfg = run_eval.load_config()
    for tier, spec in cfg["tiers"].items():
        assert "{" not in "".join(spec["arms"].values()), tier
        for a, b in spec.get("pairwise", []):
            assert a in spec["arms"] and b in spec["arms"], (tier, a, b)


def test_sign_test_and_paired():
    assert aggregate.sign_p(0, 0) == 1.0
    assert aggregate.sign_p(10, 0) == pytest.approx(2 / 1024)
    data = {"a": {str(i): {"x": 1.0 + (i % 2)} for i in range(6)}, "b": {str(i): {"x": 1.0} for i in range(6)}}
    p = aggregate.paired(data, "a", "b", "x")
    assert p["n"] == 6 and p["wins"] == 3 and p["losses"] == 0 and p["delta"] == pytest.approx(0.5)
    assert aggregate.paired(data, "a", "missing", "x") is None


def test_judge_dirs_and_pairwise_consensus(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "EVAL", tmp_path)
    monkeypatch.setattr(aggregate, "JUDGES", ["j1", "j2"])
    assert aggregate.judge_dir("t", "pairwise", "j1") == tmp_path / "t.pairwise"
    assert aggregate.judge_dir("t", "pairwise", "j2") == tmp_path / "t.pairwise@j2"
    verdicts = {"j1": {"s1": "a", "s2": "a", "s3": "b"}, "j2": {"s1": "a", "s2": "tie", "s3": "b"}}
    for judge, per in verdicts.items():
        d = aggregate.judge_dir("t", "pairwise", judge)
        d.mkdir()
        (d / "a__vs__b.jsonl").write_text("".join(
            json.dumps({"sample_id": s, "verdict": {q: w for q in aggregate.QUESTIONS}}) + "\n" for s, w in per.items()))
    got = aggregate.pairwise_verdicts("t", "a", "b")
    assert set(got) == {"j1", "j2"} and got["j2"]["s2"]["preservation"] == "tie"


def test_tally_consensus_requires_every_judge():
    q = "preservation"
    verdicts = {"j1": {"s1": {q: "a"}, "s2": {q: "a"}, "s3": {q: "b"}, "s4": {q: "a"}},
                "j2": {"s1": {q: "a"}, "s2": {q: "tie"}, "s3": {q: "b"}}}  # s4 unseen by j2
    per = aggregate.tally_question(verdicts, q, "a", "b")
    assert per["j1"] == {"a": 3, "b": 1, "tie": 0} and per["j2"] == {"a": 1, "b": 1, "tie": 1}
    assert per["consensus"]["a"] == 1 and per["consensus"]["b"] == 1 and per["consensus"]["n"] == 3
    assert "consensus" not in aggregate.tally_question({"j1": verdicts["j1"]}, q, "a", "b")


def test_paired_lower_is_better_counts_fewer_as_win():
    data = {"a": {str(i): {"unrequested_x": 0.0} for i in range(4)}, "b": {str(i): {"unrequested_x": 2.0} for i in range(4)}}
    p = aggregate.paired(data, "a", "b", "unrequested_x")
    assert p["delta"] == pytest.approx(-2.0) and p["wins"] == 4 and p["losses"] == 0


def test_load_tier_ignores_unconfigured_arms(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "EVAL", tmp_path)
    monkeypatch.setattr(aggregate, "JUDGES", ["j1"])
    monkeypatch.setattr(aggregate, "task_ids", lambda tier: {"s1"})
    (tmp_path / "t.gt").mkdir()
    for arm in ("keep", "stale"):
        (tmp_path / "t.gt" / f"{arm}.jsonl").write_text(json.dumps(
            {"sample_id": "s1", "gain_dino": 0.1, "gain_clip": 0.2, "src_dino": 0.9}) + "\n")
    assert set(aggregate.load_tier("t", {"keep"})) == {"keep"}


def test_launch_queue_parse_jobs():
    import launch_queue

    jobs = launch_queue.parse_jobs(["arm:in/x:out/y", "arm2:in/z:out/w:1055"], shards=2)
    assert jobs == [("arm", "in/x", "out/y", "", 0), ("arm", "in/x", "out/y", "", 1),
                    ("arm2", "in/z", "out/w", "1055", 0), ("arm2", "in/z", "out/w", "1055", 1)]
    with pytest.raises(SystemExit):
        launch_queue.parse_jobs(["only:two"], shards=1)


# ------------------------------------------------------- stale-result detection


def test_pending_rejudges_when_inputs_change(tmp_path):
    import judges

    video = tmp_path / "v.mp4"
    video.write_bytes(b"one")
    task = {"sample_id": "s1", "source_video": "/s.mp4", "requirement": "r", "checklist": ["x"], "preserve": ["p"]}
    job = {"arm": "a", "sample_id": "s1", "input_digest": judges.input_digest(task, video)}
    done = [{**job, "error": ""}]
    key = ("arm", "sample_id")
    assert judges.pending([job], done, key) == []
    assert judges.current_rows(done, [job], key) == done
    changed_task = {**job, "input_digest": judges.input_digest({**task, "checklist": ["y"]}, video)}
    assert judges.pending([changed_task], done, key) == [changed_task]
    assert judges.current_rows(done, [changed_task], key) == []
    legacy = [{"arm": "a", "sample_id": "s1", "error": ""}]  # no digest: stale until migrated
    assert judges.pending([job], legacy, key) == [job]


def test_migration_stamps_only_matching_rows(tmp_path, monkeypatch):
    import migrate_input_digest as mig

    monkeypatch.chdir(tmp_path)  # old rows store repo-relative paths, jobs absolute ones
    task = {"sample_id": "s1", "checklist": ["x"], "preserve": ["p"]}
    job = {"arm": "a", "sample_id": "s1", "video": str(tmp_path / "v.mp4"), "input_digest": "d1"}
    good = {"arm": "a", "sample_id": "s1", "video": "v.mp4", "error": "",
            "change": [{"item": "x"}], "preserve": [{"item": "p"}]}
    stale = {**good, "arm": "b", "change": [{"item": "old"}]}
    moved = {**good, "arm": "c", "video": "elsewhere/v.mp4"}
    path = tmp_path / "c.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (good, stale, moved)))
    jobs = [job, {**job, "arm": "b"}, {**job, "arm": "c"}]
    assert mig.migrate(path, jobs, ("arm", "sample_id"), {"s1": task}, True) == (1, 2)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["input_digest"] == "d1" and "input_digest" not in rows[1] and "input_digest" not in rows[2]


def test_consensus_can_exclude_a_judge():
    q = "visual_quality"
    verdicts = {"gem": {"s1": {q: "a"}, "s2": {q: "a"}}, "x": {"s1": {q: "b"}, "s2": {q: "a"}}, "y": {"s1": {q: "b"}, "s2": {q: "a"}}}
    per = aggregate.tally_question(verdicts, q, "a", "b", consensus_of=["x", "y"])
    assert per["consensus"] == {"a": 1, "b": 1, "n": 2, "p": 1.0, "judges": ["x", "y"]}
    assert "consensus" not in aggregate.tally_question(verdicts, q, "a", "b", consensus_of=["x", "missing"])


def test_oracle_check_flags_arms_above_ceiling():
    means = {"oracle": {"fidelity_j": 0.8}, "good": {"fidelity_j": 0.7}, "bad": {"fidelity_j": 0.9}}
    notes = aggregate.oracle_check(means, ["j"])
    assert len(notes) == 1 and "bad" in notes[0] and "good" not in notes[0]
    assert aggregate.oracle_check({"x": {}}, ["j"]) == []
