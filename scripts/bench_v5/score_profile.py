"""Score arms on the axis-profile metrics.

The edit is a vector: one signed CLIP margin per validated axis, differenced
against the source. The target's own vector is the reference direction. Three
numbers come out of that pair of vectors and none of them can be gamed by doing
nothing:

* `delivery`  -- fraction of the target's change delivered along the target's
                 direction. noop 0, oracle 1. Editing less lowers it.
* `alignment` -- cos(edit, target change). Direction only; a tiny edit in exactly
                 the right direction scores 1, which is why delivery is the headline.
* `waste`     -- orthogonal part of the edit, in units of the target's change.
                 Over-editing in the wrong direction raises it.

`change` and `adherence` are kept from bench_v4 and reported separately: how much
moved, and whether it moved toward what was asked (LESSONS.md: never merge these).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axes  # noqa: E402


def load_cache_dir(cdir: Path) -> dict:
    out = {}
    for f in cdir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        out[d["key"]] = d
    return out


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def metrics(src_axis, tgt_axis, out_axis, src_emb, out_emb, intent_emb):
    tp = tgt_axis - src_axis
    cv = out_axis - src_axis
    tp_norm = float(np.linalg.norm(tp))
    if tp_norm < 1e-12:
        return None
    tph = tp / tp_norm
    proj = float(cv @ tph)
    cv_norm = float(np.linalg.norm(cv))
    resid = cv - proj * tph
    # `delivery` alone can be raised by overshooting along the target's direction
    # (cv = 3 * tp scores 3), so the profile error is reported next to it: the
    # distance from the output's axis profile to the target's, in units of the
    # target's change. cv - tp = (proj - |tp|) tp_hat + resid, so
    # error = sqrt((delivery - 1)^2 + waste^2): noop 1, oracle 0, and both
    # under- and over-editing raise it.
    error = math.sqrt((proj / tp_norm - 1.0) ** 2
                      + (float(np.linalg.norm(resid)) / tp_norm) ** 2)
    return dict(
        delivery=proj / tp_norm,
        profile_error=error,
        alignment=float(cv @ tp) / (cv_norm * tp_norm) if cv_norm > 1e-12 else 0.0,
        waste=float(np.linalg.norm(resid)) / tp_norm,
        magnitude=cv_norm / tp_norm,
        change=1.0 - float(src_emb @ out_emb),
        adherence=float(out_emb @ intent_emb) - float(src_emb @ intent_emb),
        tp_norm=tp_norm,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--arms", required=True, help="comma-separated name=prefix pairs")
    ap.add_argument("--sources", default="scripts/bench_v5/data/top50_sources.jsonl")
    ap.add_argument("--intents", default="scripts/bench_v5/data/instructions_intent.jsonl")
    ap.add_argument("--axis-validation", default="", help="json from validate_axes.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = load_cache_dir(Path(args.cache_dir))
    names = axes.axis_names()
    keep_idx = list(range(len(names)))
    if args.axis_validation and Path(args.axis_validation).exists():
        v = json.loads(Path(args.axis_validation).read_text())
        keep = set(v["keep"])
        keep_idx = [i for i, n in enumerate(names) if n in keep]
        print(f"using {len(keep_idx)} validated axes: {[names[i] for i in keep_idx]}")

    prov = {}
    for l in Path(args.sources).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            prov[r["sample_id"]] = r.get("source_dataset", "?")
    intents = {}
    for l in Path(args.intents).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            intents[r["sample_id"]] = r.get("intent") or r.get("instruction") or ""

    arms = []
    for part in args.arms.split(","):
        name, prefix = part.split("=", 1)
        arms.append((name.strip(), prefix.strip()))

    samples = sorted(k.split("/", 1)[1] for k in cache if k.startswith("src/"))
    samples = [s for s in samples if f"tgt/{s}" in cache]

    # Intent text embeddings, one pass.
    import featurize
    import torch
    featurize._load()
    tok = featurize._tokenizer([intents.get(s, "") or "a video" for s in samples])
    with torch.no_grad():
        e = featurize._model.encode_text(tok.to(featurize._device))
        e = e / e.norm(dim=-1, keepdim=True)
    intent_emb = e.float().cpu().numpy()

    rows = []
    for si, s in enumerate(samples):
        src = np.array(cache[f"src/{s}"]["axis"])[keep_idx]
        tgt = np.array(cache[f"tgt/{s}"]["axis"])[keep_idx]
        se = np.array(cache[f"src/{s}"]["emb"])
        ie = unit(intent_emb[si])
        tp = tgt - src
        rec = dict(sample_id=s, provenance=prov.get(s, "?"),
                   tp=[float(x) for x in tp], tp_norm=float(np.linalg.norm(tp)))
        for name, prefix in arms:
            key = f"{prefix}/{s}"
            if key not in cache:
                continue
            oa = np.array(cache[key]["axis"])[keep_idx]
            oe = np.array(cache[key]["emb"])
            m = metrics(src, tgt, oa, se, oe, ie)
            if m is None:
                continue
            rec[name] = m
            rec[name + "__cv"] = [float(x) for x in (oa - src)]
        rows.append(rec)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # Axis names travel with the scores: the profile is meaningless without them and
    # a re-run with a different validated set would otherwise be read silently.
    meta = dict(axis_names=[names[i] for i in keep_idx], arms=[a for a, _ in arms],
                n_samples=len(rows))
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(rows)} samples -> {args.out} (+meta)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
