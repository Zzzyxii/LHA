"""Pre-compute and cache CLIP axis vectors + image embeddings for every clip.

Writes one small json per clip into --cache-dir so progress is observable from the
filesystem and a crash never loses finished work. Re-running skips clips whose
(mtime, size) still match.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE_VERSION = "v2"   # v2 = v1 plus the axis_a/axis_b split halves


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)
    import featurize  # noqa: F401  -- warm the model once per worker
    featurize.text_matrix()


def _work(job):
    key, path = job
    import featurize
    try:
        frames = featurize.read_frames(path)
        # axis_a / axis_b are the same clip scored on alternate frames. They exist so
        # `delivery` can be computed twice from independent frames and the two
        # estimates correlated: if `delivery` were mostly sampling noise the arms
        # could not be separated at any n, and more frames would not help.
        halves = (featurize.axis_vector(frames[0::2]), featurize.axis_vector(frames[1::2])) \
            if len(frames) >= 4 else (None, None)
        return (key, path, featurize.axis_vector(frames), featurize.clip_embedding(frames),
                "", halves)
    except Exception as exc:  # noqa: BLE001
        return (key, path, None, None, f"{type(exc).__name__}: {exc}", (None, None))


def load_jobs(path: Path) -> list[tuple[str, str]]:
    raw = path.read_text(encoding="utf-8").strip()
    out = []
    if raw.startswith("["):
        rows = json.loads(raw)
    else:
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    for r in rows:
        out.append((r["key"], r["path"]))
    return out


def safe_name(key: str) -> str:
    return key.replace("/", "__") + ".json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    jobs = load_jobs(Path(args.jobs))
    cdir = Path(args.cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    todo = []
    missing = []
    for key, path in jobs:
        p = Path(path)
        if not p.exists():
            missing.append(key)
            continue
        cfile = cdir / safe_name(key)
        st = p.stat()
        sig = f"{CACHE_VERSION}|{st.st_mtime_ns}|{st.st_size}"
        if cfile.exists():
            try:
                if json.loads(cfile.read_text()).get("sig") == sig:
                    continue
            except Exception:  # noqa: BLE001 - unreadable cache entry is just recomputed
                pass
        todo.append((key, str(path)))

    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} cached, {len(todo)} to run, "
          f"{len(missing)} missing", flush=True)
    if missing:
        print("MISSING:", missing[:10], flush=True)

    done = 0
    failed = 0
    if todo:
        with Pool(processes=args.workers, initializer=_worker_init) as pool:
            for key, path, av, emb, err, halves in pool.imap_unordered(_work, todo, chunksize=1):
                done += 1
                if err:
                    failed += 1
                    print(f"FAIL {key}: {err}", flush=True)
                    continue
                st = Path(path).stat()
                (cdir / safe_name(key)).write_text(json.dumps({
                    "sig": f"{CACHE_VERSION}|{st.st_mtime_ns}|{st.st_size}",
                    "key": key, "path": str(path),
                    "axis": [float(x) for x in np.asarray(av)],
                    "emb": [float(x) for x in np.asarray(emb)],
                    "axis_a": None if halves[0] is None else [float(x) for x in halves[0]],
                    "axis_b": None if halves[1] is None else [float(x) for x in halves[1]],
                }))
                if done % 20 == 0:
                    print(f"  {done}/{len(todo)} (failed {failed})", flush=True)
    print(f"done {done}, failed {failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
