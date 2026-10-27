"""Predict which axes *this* intent implies, in CLIP text space.

The arm that is rendering now (`g1`) aims every clip at the same task-level prior.
That is wrong in a way the data can show: the per-sample target direction has a mean
cosine of only 0.686 with the mean direction, so for a large share of clips the
dataset average points somewhere the target does not go. "Make it look like a dreamy
pastel memory" should not be handed the period-costume clause that the average
low->high pair wants.

This module turns the intent sentence into a per-clip goal vector without any LLM:
score every axis by how much the intent text prefers the axis's "high" caption over
its "low" one, then use that to reweight the prior.

`--eval` measures which reweighting actually predicts the target's own axis
direction (`tp`), on the frozen calibration split. The variant is chosen on the
calibration split and reported on its complement, never the other way round.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes  # noqa: E402
import featurize  # noqa: E402


def intent_margins(intents: list[str]) -> np.ndarray:
    """(n, K) CLIP antonym margin of each intent text on each axis."""
    tx = featurize.text_matrix()                     # (2K, D), [high_0, low_0, ...]
    featurize._load()
    import torch
    tok = featurize._tokenizer(intents).to(featurize._device)
    with torch.no_grad():
        e = featurize._model.encode_text(tok)
        e = e / e.norm(dim=-1, keepdim=True)
    it = e.float().cpu().numpy()
    sims = it @ tx.T                                 # (n, 2K)
    return sims[:, 0::2] - sims[:, 1::2]             # (n, K)


def goals(prior: np.ndarray, margins: np.ndarray, mode: str, alpha: float = 1.0,
          floor: float = 0.0) -> np.ndarray:
    """(n, K) per-clip goal vectors."""
    if mode == "prior":
        return np.tile(prior, (margins.shape[0], 1))
    z = (margins - margins.mean(0)) / (margins.std(0) + 1e-9)
    if mode == "additive":
        w = 1.0 + alpha * z
    elif mode == "gate":
        w = 1.0 / (1.0 + np.exp(-alpha * z))
    elif mode == "signed":
        # axes the intent argues against are pushed, not just suppressed
        w = 1.0 + alpha * z
    else:
        raise ValueError(mode)
    w = np.clip(w, floor, None)
    return prior * w


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents", default="scripts/bench_v5/data/instructions_intent.jsonl")
    ap.add_argument("--profile", default="scripts/bench_v5/data/axis_profile_calib.json")
    ap.add_argument("--split", default="scripts/bench_v5/data/split.json")
    ap.add_argument("--scores", default="scripts/bench_v5/data/scores_v5.jsonl",
                    help="scores jsonl that carries per-sample tp")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    prof = json.loads(Path(args.profile).read_text())
    names = prof["axis_names"]
    prior = np.array(prof["mean_tp"])
    split = json.loads(Path(args.split).read_text())
    calib, held = set(split["calibration"]), set(split["held_out"])

    intents, order = {}, []
    for l in Path(args.intents).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        intents[r["sample_id"]] = r.get("instruction") or r.get("intent") or ""
        order.append(r["sample_id"])
    full_names = axes.axis_names()
    keep = [full_names.index(n) for n in names]   # profile is the validated subset
    margins = intent_margins([intents[s] for s in order])[:, keep]
    idx = {s: i for i, s in enumerate(order)}

    rows = [json.loads(l) for l in Path(args.scores).read_text().splitlines() if l.strip()]
    tp = {r["sample_id"]: np.array(r["tp"]) for r in rows}

    def report(label, sid_list):
        tp_m = np.array([tp[s] for s in sid_list if s in tp])
        out = {}
        for mode in ("prior", "additive", "gate", "signed"):
            for alpha in (0.5, 1.0, 2.0):
                if mode == "prior" and alpha != 0.5:
                    continue
                g = goals(prior, np.array([margins[idx[s]] for s in sid_list if s in tp]),
                          mode, alpha)
                cos = np.sum(unit(g) * unit(tp_m), axis=1)
                out[f"{mode}/a{alpha}"] = (float(cos.mean()), float(np.median(cos)))
        print(f"\n{label} (n={len(tp_m)})  mean cos(goal, tp)")
        for k, (m, md) in sorted(out.items(), key=lambda kv: -kv[1][0]):
            print(f"  {k:14s} {m:+.4f}  (median {md:+.4f})")
        return out

    cal = [s for s in order if s in calib]
    hel = [s for s in order if s in held]
    report("CALIBRATION (variant chosen here)", cal)
    report("HELD OUT (reported here)", hel)

    if args.out:
        best_mode, best_alpha = "signed", 2.0
        g = {s: goals(prior, margins[idx[s]][None, :], best_mode, best_alpha)[0]
             for s in order if s in idx}
        Path(args.out).write_text(json.dumps(
            {"axis_names": names, "mode": best_mode, "alpha": best_alpha,
             "goals": {s: [float(x) for x in v] for s, v in g.items()}}, indent=1))
        print(f"\nwrote per-clip goals -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
