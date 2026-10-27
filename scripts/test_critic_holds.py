"""Regression tests for the critic's per-dimension hold checking.

A style template declares a world; an instruction can name dimensions of that
world it wants left alone. The composer turns those into explicit "keep X
unchanged" lines, but until the critic checks them one by one there is no signal
back to the skill loop -- a low ``structure_preservation`` score says something
was not preserved without saying which dimension, so there is nothing to fix.

These tests cover that loop: requirements from overrides, the prompt section that
asks for the check, parsing the reply, and the per-dimension failure tags.

Run:  python3 scripts/test_critic_holds.py
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# The critic imports the VLM backends and torch on module load. None of them are
# exercised here -- hold parsing is pure text handling -- so stub them rather than
# require a GPU stack to run a string test.
_STUB_ROOTS = {
    "qwen_vl_utils", "transformers", "litellm", "torch", "peft", "accelerate",
    "modelscope", "diffsynth", "einops", "safetensors", "PIL", "cv2",
}


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        return None


sys.meta_path.append(_StubFinder())

from critic.evaluator import (  # noqa: E402
    _hold_requirements_from,
    _parse_evaluation_text,
    _parse_hold_checks,
    hold_failure_tags,
)
from critic.prompts import (  # noqa: E402
    build_edit_evaluator_system_prompt,
    build_edit_evaluator_user_prompt,
)
from memory.contracts import build_critic_evaluation_signal  # noqa: E402
from memory.evolver import (  # noqa: E402
    _hold_violation_dims,
    _skill_quality_score,
    hold_violation_summaries,
)
from memory.store import default_global_memory  # noqa: E402

REPLY = """PROMPT_ALIGNMENT: 0.80
STRUCTURE_PRESERVATION: 0.45
TRANSFORMATION_STRENGTH: 0.75
CARRIER_GROUNDING: 0.80
WORLD_REALIZATION: 0.85
TEMPORAL_COHERENCE: 0.80
ARTIFACT_PENALTY: 0.90
OVERALL_SCORE: 0.70
HOLD_CHECK:
- costume: VIOLATED (the subject now wears a cyberpunk jacket)
- subject_surface: HELD
FAILURE_TAGS: identity_drift
REFLECTION_HINTS: reduce the wardrobe change
NOTES: grading landed well
"""


def test_requirements_from_overrides() -> None:
    # "edit" verdicts are not holds and must not be asked about.
    reqs = _hold_requirements_from({"costume": "keep", "lighting": "edit"})
    assert reqs == {"costume": "keep the clothing unchanged"}, reqs
    assert _hold_requirements_from({}) == {}
    assert _hold_requirements_from(None) == {}
    # An unknown dimension still produces a usable phrase rather than nothing.
    assert _hold_requirements_from({"nonexistent": "keep"}) == {
        "nonexistent": "keep nonexistent unchanged"
    }


def test_prompt_only_asks_when_holds_exist() -> None:
    base = dict(
        sample_id="s1", prompt="make it cyberpunk", planner_summary={},
        edit_summary={}, metric_hints={},
    )
    plain = build_edit_evaluator_user_prompt(**base)
    assert "HOLD_CHECK:" not in plain, "no holds means no hold section"
    assert "must be kept" not in plain.lower()

    with_hold = build_edit_evaluator_user_prompt(
        **base, hold_requirements={"costume": "keep the clothing unchanged"}
    )
    assert "HOLD_CHECK:" in with_hold
    assert "costume: keep the clothing unchanged" in with_hold
    # The dimension has to be named in the reply format too, or the parser gets
    # prose instead of per-dimension verdicts.
    assert "'- <维度>: HELD'" in with_hold

    assert "HOLD_CHECK" not in build_edit_evaluator_system_prompt(with_hold_check=False)
    assert "HOLD_CHECK" in build_edit_evaluator_system_prompt(with_hold_check=True)


def test_parse_hold_checks() -> None:
    checks = _parse_hold_checks(REPLY, ["costume", "subject_surface"])
    by_dim = {c.dimension: c for c in checks}
    assert by_dim["costume"].held is False
    assert by_dim["costume"].reported is True
    assert "jacket" in by_dim["costume"].note
    assert by_dim["subject_surface"].held is True

    # A dimension the critic never mentioned is recorded, not assumed violated:
    # a terse reply must not look like every hold failed.
    missing = _parse_hold_checks(REPLY, ["costume", "lighting"])
    by_dim = {c.dimension: c for c in missing}
    assert by_dim["lighting"].reported is False
    assert by_dim["lighting"].held is True
    assert hold_failure_tags(missing) == ["hold_violated:costume"]


def test_end_to_end_tags() -> None:
    result = _parse_evaluation_text(
        REPLY,
        sample_id="s1",
        variant="wan14b",
        metric_hints={},
        hold_dimensions=["costume", "subject_surface"],
    )
    assert not result.evaluation_incomplete, result.missing_headings
    assert "identity_drift" in result.failure_tags
    assert "hold_violated:costume" in result.failure_tags, result.failure_tags
    assert "hold_violated:subject_surface" not in result.failure_tags
    # Survives to_dict, which is what reaches memory.
    payload = result.to_dict()
    assert payload["hold_checks"][0]["dimension"] == "costume"
    assert payload["hold_checks"][0]["held"] is False


def test_no_holds_keeps_old_behaviour() -> None:
    """A run with no holds must not be flagged incomplete for a missing section."""
    result = _parse_evaluation_text(
        REPLY, sample_id="s1", variant="wan14b", metric_hints={},
    )
    assert not result.evaluation_incomplete, result.missing_headings
    assert result.hold_checks == []
    assert not any(tag.startswith("hold_violated") for tag in result.failure_tags)


def test_memory_keeps_hold_checks() -> None:
    """The structured rows have to survive into memory, not just the tags."""
    result = _parse_evaluation_text(
        REPLY, sample_id="s1", variant="wan14b", metric_hints={},
        hold_dimensions=["costume", "subject_surface"],
    )
    signal = build_critic_evaluation_signal(result.to_dict())
    assert signal["hold_checks"][0]["dimension"] == "costume"
    assert signal["hold_checks"][0]["held"] is False
    assert "hold_violated:costume" in signal["failure_tags"]
    # A run with no holds stores an empty list rather than a missing key.
    assert build_critic_evaluation_signal({})["hold_checks"] == []
    assert "hold_violation_summaries" in default_global_memory()


def test_violation_summary_aggregates_per_dimension() -> None:
    """The whole point: name the field, not just the template."""

    def rec(strategy: str, violated: dict[str, bool]) -> dict:
        return {
            "selected_strategy": strategy,
            "hold_checks": [
                {"dimension": d, "held": (not v), "reported": True, "note": ""}
                for d, v in violated.items()
            ],
            "failure_tags": [f"hold_violated:{d}" for d, v in violated.items() if v],
        }

    records = [
        rec("cyberpunk_dystopia", {"costume": True, "subject_surface": False}),
        rec("cyberpunk_dystopia", {"costume": True, "subject_surface": False}),
        rec("cyberpunk_dystopia", {"costume": False, "subject_surface": False}),
        rec("studio_ghibli", {"costume": False, "subject_surface": False}),
    ]
    summaries = {(s["strategy_id"], s["dimension"]): s for s in hold_violation_summaries(records)}
    costume = summaries[("cyberpunk_dystopia", "costume")]
    assert (costume["violations"], costume["checks"]) == (2, 3)
    # rate is rounded to 4 places on the way out.
    assert abs(costume["rate"] - 2 / 3) < 1e-3, costume
    # Same template, different dimension, no violations -- that is the whole
    # reason to aggregate per dimension instead of per template.
    assert summaries[("cyberpunk_dystopia", "subject_surface")]["violations"] == 0
    assert summaries[("studio_ghibli", "costume")]["violations"] == 0
    # Worst first.
    assert hold_violation_summaries(records)[0]["dimension"] == "costume"

    assert _hold_violation_dims(rec("x", {"costume": True})) == ["costume"]
    assert _hold_violation_dims({"failure_tags": ["identity_drift"]}) == []
    assert _hold_violation_dims({}) == []


def test_quality_score_penalises_hold_violation() -> None:
    clean = {"overall_score": 0.8, "accepted": True, "failure_tags": []}
    violated = {"overall_score": 0.8, "accepted": True, "failure_tags": ["hold_violated:costume"]}
    assert _skill_quality_score(violated) < _skill_quality_score(clean)
    # Comparable to artifact_heavy: the instruction was understood and overridden.
    artifact = {"overall_score": 0.8, "accepted": True, "failure_tags": ["artifact_heavy"]}
    assert _skill_quality_score(violated) == _skill_quality_score(artifact)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
