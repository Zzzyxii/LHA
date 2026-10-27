#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from critic import evaluate_edit_result, load_critic_artifact_summary, reflect_edit_result
from memory import build_attempt_memory_contract
from _dataset import DataSample, get_sample_by_id, get_sample_by_index, smoke_sample


DEFAULT_SAMPLE_ID = "1"
DEFAULT_LOW_VIDEO = str(PROJECT_ROOT / "smoke-data" / "1_low-cost.mp4")
DEFAULT_EDITED_VIDEO = str(PROJECT_ROOT / "out" / "test_ditto_wan" / "sample1_edited_4step-prompt2.mp4")
DEFAULT_PROMPT_FILE = str(PROJECT_ROOT / "smoke-data" / "prompt1.txt")
DEFAULT_PLANNER_RESPONSE = str(PROJECT_ROOT / "out" / "sample_1" / "planner.response.json")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "out" / "test_ditto_wan" / "critic_sample1_4step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an edited video and run the critic reflector.")
    parser.add_argument("--pair-dir", default="", type=str,
                        help="Path to a data directory with pair.json or manifest.json.")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index of the sample to use (when --pair-dir is set). 0-based.")
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID,
                        help="Sample ID (when --pair-dir is set). Overrides --sample-index.")
    parser.add_argument("--low-video", default="")
    parser.add_argument("--edited-video", default=DEFAULT_EDITED_VIDEO)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--planner-response", default="")
    parser.add_argument("--edit-response", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve_sample_and_prompt(args: argparse.Namespace) -> tuple[DataSample, str]:
    """Resolve sample + prompt from args. Supports --pair-dir or smoke-data fallback."""
    if args.pair_dir:
        pair_dir = (PROJECT_ROOT / args.pair_dir).resolve()
        if args.sample_id and args.sample_id != DEFAULT_SAMPLE_ID:
            sample = get_sample_by_id(pair_dir, args.sample_id)
        else:
            sample = get_sample_by_index(pair_dir, args.sample_index)
        prompt = args.prompt.strip() or sample.prompt_text
        return sample, prompt
    # smoke-data
    n = int(args.sample_id) if args.sample_id.isdigit() else 1
    sample = smoke_sample(index=n - 1)
    prompt = (
        args.prompt.strip()
        or (Path(args.prompt_file).read_text(encoding="utf-8").strip() if args.prompt_file else "")
        or sample.prompt_text
    )
    return sample, prompt


def _resolve_prompt_from_args_or_file(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.planner_response:
        payload = json.loads(Path(args.planner_response).read_text(encoding="utf-8"))
        return str(payload.get("normalized_prompt", "") or payload.get("prompt", "")).strip()
    raise ValueError("Provide --prompt, --prompt-file, or --planner-response")


def main() -> None:
    args = parse_args()
    sample, prompt = _resolve_sample_and_prompt(args)
    low_video = args.low_video or str(sample.low_video)
    planner_payload = json.loads(Path(args.planner_response).read_text(encoding="utf-8")) if args.planner_response else None
    edit_payload = json.loads(Path(args.edit_response).read_text(encoding="utf-8")) if args.edit_response else None
    artifact_summary = load_critic_artifact_summary()

    evaluation = evaluate_edit_result(
        sample_id=sample.sample_id,
        prompt=prompt,
        low_video=low_video,
        edited_video=args.edited_video,
        planner_payload=planner_payload,
        edit_payload=edit_payload,
    )
    reflection = reflect_edit_result(evaluation)

    evaluation_payload = evaluation.to_dict()
    reflection_payload = reflection.to_dict()
    contract_variant = str((edit_payload or {}).get("variant", "") or evaluation_payload.get("variant", ""))
    contract_critic_mode = str(evaluation_payload.get("memory_mode", "") or reflection_payload.get("memory_mode", ""))
    memory_contract_payload = build_attempt_memory_contract(
        sample_id=sample.sample_id,
        attempt_index=1,
        variant=contract_variant,
        prompt=prompt,
        planner_payload=planner_payload or {},
        critic_query={},
        critic_memory_mode=contract_critic_mode,
        critic_evaluation_payload=evaluation_payload,
        critic_reflection_payload=reflection_payload,
        edit_memory_mode="",
        edit_controls=dict((edit_payload or {}).get("implicit_controls", {}) or {}),
        edit_payload=edit_payload or {},
        edit_hidden_report=dict((edit_payload or {}).get("implicit_hidden_memory", {}) or {}),
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evaluation.response.json").write_text(json.dumps(evaluation_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "reflection.response.json").write_text(json.dumps(reflection_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "memory.contract.json").write_text(json.dumps(memory_contract_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "critic.artifacts.json").write_text(json.dumps(artifact_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "critic_runtime": artifact_summary.get("evaluator_runtime", {}),
                "evaluation": evaluation_payload,
                "reflection": reflection_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
