"""Job list for one rendered arm directory.

extract_cache.py takes a jsonl of {key, path}; a render directory is already that,
just with the arm name missing. One small script so every arm gets the same key
convention (`<arm>/<sample_id>`) instead of a hand-written file per arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.run_dir)
    jobs = [{"key": f"{args.arm}/{p.stem}", "path": str(p)} for p in sorted(d.glob("*.mp4"))]
    Path(args.out).write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
