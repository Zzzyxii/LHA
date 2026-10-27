#!/usr/bin/env python3
"""Generate top50 bench videos for one arm on one GPU shard.

Arms are either "skill" arms, which run the full agent closed loop
(planner -> skill-compiled prompt -> edit -> critic -> reflection -> memory), or
"raw" arms, which feed the bench instruction straight to the edit model in a
single pass. The raw arm is the protocol-matched control that separates the
skill's contribution from export-protocol differences.

One process handles one (arm, shard) pair so the resident Wan pipeline is loaded
once and reused across that shard's samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "Ditto"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed face, extra limbs, identity drift, cartoon, anime, "
    "oversaturated neon, broken geometry"
)

ARMS: dict[str, dict] = {
    "ours_skill": {"variant": "wan14b+ours", "mode": "skill"},
    "ours_raw": {"variant": "wan14b+ours", "mode": "raw"},
    "ours_bestof": {"variant": "wan14b+ours", "mode": "bestof"},
    # Planning track (bench_v2): input is a one-line intent via --instruction-file.
    # "direct" feeds it straight to the edit model; "agent" lets the planner expand it.
    "ours_intent_direct": {"variant": "wan14b+ours", "mode": "raw"},
    "ours_intent_agent": {"variant": "wan14b+ours", "mode": "skill"},
    "ours_generic_direct": {"variant": "wan14b+ours", "mode": "raw"},
    "ours_generic_agent": {"variant": "wan14b+ours", "mode": "skill"},
    # Planning-track ablation: the intent_agent prompts with sections removed
    # (scripts/eval/bench_v3/build_ablation_prompts.py), rendered without the loop.
    "ours_abl_agentprompt": {"variant": "wan14b+ours", "mode": "raw"},
    "ours_abl_plan_craft": {"variant": "wan14b+ours", "mode": "raw"},
    "ours_abl_plan_only": {"variant": "wan14b+ours", "mode": "raw"},
    # bench_v3 reference tier (T2): --instruction-file rows carry "reference_images".
    # "vace_reference" hands the first reference to VACE as its reference image;
    # "translate" has the agent turn the reference into a text instruction first.
    "ours_t2_direct_ref": {"variant": "wan14b+ours", "mode": "raw", "vace_reference": True},
    "ours_t2_agent": {"variant": "wan14b+ours", "mode": "translate"},
    "ours_t2_agent_ref": {"variant": "wan14b+ours", "mode": "translate", "vace_reference": True},
    # bench_v3 text tiers: the instruction file already holds the final prompt
    # (t1 intent, t1 intent + precomputed plan, or t3 spec), rendered as-is.
    "ours_t1_direct": {"variant": "wan14b+ours", "mode": "raw"},
    "ours_t1_agent": {"variant": "wan14b+ours", "mode": "raw"},
    "ours_t3_direct": {"variant": "wan14b+ours", "mode": "raw"},
    # Cross-backbone check: render the same t1 / t1_agent prompts with Bernini to
    # see whether the agent's gain survives a different edit model.
    "bernini_t1_direct": {"variant": "bernini1.3b", "mode": "raw"},
    "bernini_t1_agent": {"variant": "bernini1.3b", "mode": "raw"},
    "ditto_skill": {"variant": "wan14b+dittolora", "mode": "skill"},
    "ditto_raw": {"variant": "wan14b+dittolora", "mode": "raw"},
    "bernini_skill": {"variant": "bernini1.3b", "mode": "skill"},
    "bernini_raw": {"variant": "bernini1.3b", "mode": "raw"},
}


def _log(message: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _read_index(path: Path) -> list[dict]:
    """A JSON array (sample_index.json) or JSONL of rows with at least sample_id."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON array in {path}")
    return payload


def _accepted_video(output_dir: Path, result: dict) -> Path | None:
    """Pick the video the loop settled on.

    An accepted run is unambiguous: take the attempt the critic accepted. When
    the loop exhausts its attempts without accepting, the final attempt is not
    the best one -- refinement often regresses, and the critic already scored
    every attempt -- so fall back to the highest-scoring attempt rather than the
    last.
    """
    attempt_index = result.get("attempt_index")
    if attempt_index:
        candidate = output_dir / f"attempt_{int(attempt_index):03d}" / "edited.mp4"
        if candidate.exists():
            return candidate

    scored: list[tuple[float, Path]] = []
    for video in sorted(output_dir.glob("attempt_*/edited.mp4")):
        evaluation = video.parent / "evaluation.response.json"
        score = -1.0
        if evaluation.exists():
            try:
                payload = json.loads(evaluation.read_text(encoding="utf-8"))
                score = float((payload.get("scores") or {}).get("overall_score") or 0.0)
            except (OSError, ValueError):
                score = -1.0
        scored.append((score, video))
    if not scored:
        return None
    # Ties keep the earliest attempt, which is the one the later passes failed to beat.
    best_score = max(score for score, _ in scored)
    return next(video for score, video in scored if score == best_score)


def _run_skill_sample(
    *,
    sample_id: str,
    instruction: str,
    low_video: Path,
    arm_dir: Path,
    variant: str,
    max_attempts: int,
    memory_path: Path,
    num_frames: int,
    fps: float,
) -> dict:
    from loop import run_planner_and_edit_loop

    work_dir = arm_dir / "loop" / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)
    # The loop owns prompt compilation and builds its own EditRequest, so the
    # bench protocol reaches the edit backend through the environment. Values of
    # 0 mean "follow the source video", which is what keeps the edit from
    # changing the clip's duration.
    if num_frames > 0:
        os.environ["LOW_HIGH_BENCH_NUM_FRAMES"] = str(num_frames)
    else:
        os.environ.pop("LOW_HIGH_BENCH_NUM_FRAMES", None)
    if fps > 0:
        os.environ["LOW_HIGH_BENCH_FPS"] = str(fps)
    else:
        os.environ.pop("LOW_HIGH_BENCH_FPS", None)
    result = run_planner_and_edit_loop(
        sample_id=sample_id,
        prompt=instruction,
        low_video=str(low_video),
        output_dir=work_dir,
        variant=variant,
        memory_path=str(memory_path),
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        max_attempts=max_attempts,
    )
    video = _accepted_video(work_dir, result)
    if video is None:
        raise RuntimeError("closed loop produced no video")
    return {"result": result, "video": video}


def _run_raw_sample(
    *,
    sample_id: str,
    instruction: str,
    low_video: Path,
    arm_dir: Path,
    variant: str,
    num_frames: int,
    fps: float,
    seed: int,
    reference_image: Path | None = None,
) -> dict:
    from edit import EditRequest, render_edit_video

    work_dir = arm_dir / "raw" / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_video = work_dir / "edited.mp4"
    payload = render_edit_video(
        EditRequest(
            low_video=low_video,
            output_video=output_video,
            positive_prompt=instruction,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            variant=variant,
            num_frames=num_frames if num_frames > 0 else None,
            fps=fps if fps > 0 else None,
            seed=seed,
            device="cuda",
            reference_image=reference_image,
        )
    )
    (work_dir / "edit.prompt.txt").write_text(instruction, encoding="utf-8")
    (work_dir / "edit.response.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"result": payload, "video": output_video}


def _run_translate_sample(
    *,
    sample_id: str,
    intent: str,
    reference_images: list[str],
    low_video: Path,
    arm_dir: Path,
    variant: str,
    num_frames: int,
    fps: float,
    seed: int,
    vace_reference: bool,
) -> dict:
    """T2 agent: translate the reference still(s) into an instruction, then render it.

    The translation cache lives at <output-root>/translations (shared with
    scripts/eval/bench_v3/precompute_translations.py), not per arm, so arms that
    differ only in ``vace_reference`` render the identical instruction and their
    comparison isolates the reference image.
    """
    from planner.reference_translator import load_or_translate

    translation = load_or_translate(cache_dir=arm_dir.parent / "translations", sample_id=sample_id, intent=intent,
                                    low_video=str(low_video), reference_images=reference_images)
    return _run_raw_sample(
        sample_id=sample_id,
        instruction=translation["instruction"],
        low_video=low_video,
        arm_dir=arm_dir,
        variant=variant,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        reference_image=Path(reference_images[0]) if vace_reference else None,
    )


def _run_bestof_sample(
    *,
    sample_id: str,
    instruction: str,
    low_video: Path,
    arm_dir: Path,
    variant: str,
    candidates: int,
    num_frames: int,
    fps: float,
    seed: int,
) -> dict:
    """Generate N independent candidates and keep the one the critic scores highest.

    Serial refinement spends its budget rewriting the prompt after each attempt,
    and measurement showed that regresses: on 73% of samples an earlier attempt
    outscored the one the loop ended on. Best-of-N spends the same number of
    diffusion passes on independent seeds against the unmodified instruction, so
    the critic ranks genuine alternatives instead of a drifting sequence. The
    candidates are also independent, so they can run concurrently rather than
    strictly one after another.
    """
    from critic import evaluate_edit_result
    from edit import EditRequest, render_edit_video
    from planner.core.video_planner import plan_video

    work_dir = arm_dir / "bestof" / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_video(
        sample_id=sample_id,
        prompt=instruction,
        low_video=str(low_video),
        memory_context={},
        memory_store=None,
        attempt_adaptation_state={},
        state_path=work_dir / "planner.state.json",
        variant=variant,
    )
    planner_payload = plan.model_dump()
    (work_dir / "planner.response.json").write_text(
        json.dumps(planner_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    edit_prompt = (
        planner_payload.get("compiled_edit_prompt")
        or planner_payload.get("normalized_prompt")
        or instruction
    )
    (work_dir / "edit.prompt.txt").write_text(edit_prompt, encoding="utf-8")

    scored: list[tuple[float, Path, dict]] = []
    for index in range(candidates):
        candidate_dir = work_dir / f"candidate_{index + 1:02d}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        output_video = candidate_dir / "edited.mp4"
        edit_payload = render_edit_video(
            EditRequest(
                low_video=low_video,
                output_video=output_video,
                positive_prompt=edit_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                variant=variant,
                num_frames=num_frames if num_frames > 0 else None,
                fps=fps if fps > 0 else None,
                # Decorrelated per candidate; a shared seed would make every
                # candidate a near-duplicate and the ranking meaningless.
                seed=seed + index * 1013,
                device="cuda",
            )
        )
        evaluation = evaluate_edit_result(
            sample_id=sample_id,
            prompt=edit_prompt,
            low_video=low_video,
            edited_video=edit_payload["output_video"],
            planner_payload=planner_payload,
            edit_payload=edit_payload,
            memory_mode="none",
            explicit_context={},
            implicit_state={},
        )
        payload = evaluation.to_dict()
        (candidate_dir / "evaluation.response.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        score = float((payload.get("scores") or {}).get("overall_score") or 0.0)
        scored.append((score, output_video, payload))
        _log(f"  {sample_id} candidate {index + 1}/{candidates} score={score:.3f}")

    if not scored:
        raise RuntimeError("best-of-N produced no candidates")
    best_score, best_video, _ = max(scored, key=lambda item: item[0])
    (work_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_video": str(best_video),
                "selected_score": best_score,
                "all_scores": [
                    {"candidate": str(video.parent.name), "overall_score": score}
                    for score, video, _ in scored
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"result": {"status": "bestof", "selected_score": best_score}, "video": best_video}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bench videos for one arm/shard.")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=None,
                        help="directory of <sample_id>.mp4 sources; omit to use each row's source_video")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=3,
                        help="best-of-N candidate count (mode=bestof only)")
    parser.add_argument("--num-frames", type=int, default=0,
                        help="0 = follow the source video (preserves clip duration)")
    parser.add_argument("--fps", type=float, default=0.0,
                        help="0 = follow the source video")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-sample-ids", type=Path, default=None,
                        help="restrict the run to sample ids listed in this file, one per line")
    parser.add_argument("--instruction-file", type=Path, default=None,
                        help="jsonl of {sample_id, intent}; replaces the bench instruction per sample. "
                        "The planning track uses this to feed a one-line user intent instead of the "
                        "expert instruction, which leaks the target.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="walk the shard back-to-front; use for catch-up workers added on freed GPUs "
        "so they meet the in-flight shard in the middle instead of redoing its queue",
    )
    return parser.parse_args()


def apply_instruction_file(rows: list[dict], path: Path) -> list[dict]:
    """Replace each row's instruction with the file's ``intent`` and attach its ``reference_images``.

    Every row must be covered: falling back to the expert instruction would
    silently mix tracks.
    """
    intents: dict[str, str] = {}
    references: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            intents[str(item["sample_id"])] = str(item["intent"]).strip()
            references[str(item["sample_id"])] = [str(p) for p in item.get("reference_images") or []]
    missing = [str(r.get("sample_id")) for r in rows if str(r.get("sample_id")) not in intents]
    if missing:
        raise SystemExit(f"--instruction-file has no intent for {len(missing)} samples, e.g. {missing[:3]}")
    return [
        {**r, "instruction": intents[str(r.get("sample_id"))], "reference_images": references[str(r.get("sample_id"))]}
        for r in rows
    ]


def main() -> int:
    args = parse_args()
    spec = ARMS[args.arm]
    rows = _read_index(args.sample_index)
    if args.only_sample_ids:
        keep = {
            line.strip()
            for line in args.only_sample_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        rows = [row for row in rows if str(row.get("sample_id")) in keep]
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.instruction_file:
        rows = apply_instruction_file(rows, args.instruction_file)
    needs_reference = spec["mode"] == "translate" or spec.get("vace_reference")
    if needs_reference:
        unreferenced = [str(r.get("sample_id")) for r in rows if not r.get("reference_images")]
        if unreferenced:
            raise SystemExit(f"arm {args.arm} needs reference_images; missing for {unreferenced[:3]}")
    shard = [row for i, row in enumerate(rows) if i % args.shard_count == args.shard_index]
    if args.reverse:
        shard.reverse()

    arm_dir = args.output_root / args.arm
    gen_dir = arm_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    memory_path = arm_dir / f"memory.shard{args.shard_index}.json"
    status_path = arm_dir / f"status.shard{args.shard_index}.jsonl"

    _log(
        f"arm={args.arm} variant={spec['variant']} mode={spec['mode']} "
        f"shard={args.shard_index}/{args.shard_count} samples={len(shard)} "
        f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"
    )

    failures = 0
    for position, row in enumerate(shard, start=1):
        sample_id = str(row.get("sample_id") or "").strip()
        instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
        final_video = gen_dir / f"{sample_id}.mp4"
        if final_video.exists() and not args.overwrite:
            _log(f"{args.arm} [{position}/{len(shard)}] {sample_id} skip (exists)")
            continue

        # --source-dir names sources by sample_id (top50). Rows that pair one source
        # with several edits (bench_v3 style track) carry their own source path
        # instead; sample_index.json's source_video points at the original host, so
        # it is only used when --source-dir is omitted.
        low_video = args.source_dir / f"{sample_id}.mp4" if args.source_dir else Path(str(row.get("source_video") or ""))
        started = time.time()
        record: dict = {"arm": args.arm, "sample_id": sample_id, "shard": args.shard_index}
        try:
            if not low_video.exists():
                raise FileNotFoundError(f"source video missing: {low_video}")
            if spec["mode"] == "bestof":
                outcome = _run_bestof_sample(
                    sample_id=sample_id,
                    instruction=instruction,
                    low_video=low_video,
                    arm_dir=arm_dir,
                    variant=spec["variant"],
                    candidates=args.candidates,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    seed=args.seed,
                )
                record["status"] = "bestof"
                record["selected_score"] = outcome["result"].get("selected_score")
            elif spec["mode"] == "skill":
                outcome = _run_skill_sample(
                    sample_id=sample_id,
                    instruction=instruction,
                    low_video=low_video,
                    arm_dir=arm_dir,
                    variant=spec["variant"],
                    max_attempts=args.max_attempts,
                    memory_path=memory_path,
                    num_frames=args.num_frames,
                    fps=args.fps,
                )
                loop_result = outcome["result"]
                record["status"] = loop_result.get("status", "")
                record["attempts"] = len(loop_result.get("attempts") or []) or loop_result.get("attempt_index")
            elif spec["mode"] == "translate":
                outcome = _run_translate_sample(
                    sample_id=sample_id,
                    intent=instruction,
                    reference_images=row["reference_images"],
                    low_video=low_video,
                    arm_dir=arm_dir,
                    variant=spec["variant"],
                    num_frames=args.num_frames,
                    fps=args.fps,
                    seed=args.seed,
                    vace_reference=bool(spec.get("vace_reference")),
                )
                record["status"] = "rendered"
            else:
                outcome = _run_raw_sample(
                    sample_id=sample_id,
                    instruction=instruction,
                    low_video=low_video,
                    arm_dir=arm_dir,
                    variant=spec["variant"],
                    num_frames=args.num_frames,
                    fps=args.fps,
                    seed=args.seed,
                    reference_image=Path(row["reference_images"][0]) if spec.get("vace_reference") else None,
                )
                record["status"] = "rendered"

            source = outcome["video"]
            final_video.parent.mkdir(parents=True, exist_ok=True)
            final_video.write_bytes(Path(source).read_bytes())
            record["video"] = str(final_video)
            record["error"] = ""
        except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the shard
            failures += 1
            record["status"] = "error"
            record["video"] = ""
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()[-1500:]

        record["elapsed_s"] = round(time.time() - started, 1)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        _log(
            f"{args.arm} [{position}/{len(shard)}] {sample_id} "
            f"status={record['status']} {record['elapsed_s']}s"
            + (f" ERROR={record['error'][:160]}" if record["error"] else "")
        )

    _log(f"arm={args.arm} shard={args.shard_index} done failures={failures}")
    return 1 if failures and failures == len(shard) else 0


if __name__ == "__main__":
    raise SystemExit(main())
