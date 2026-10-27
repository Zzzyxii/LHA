"""Print the `--arms` argument for every arm present in a cache directory.

Hand-written arm lists go stale the moment an arm finishes rendering (or is
discovered to cover only part of the sample set), and a missing name silently
drops that arm from a report. This derives the list from what is actually cached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--exclude", default="src,tgt")
    args = ap.parse_args()

    skip = set(args.exclude.split(","))
    names = set()
    for f in Path(args.cache_dir).glob("*.json"):
        try:
            key = json.loads(f.read_text())["key"]
        except Exception:  # noqa: BLE001
            continue
        if "/" in key and key.split("/", 1)[0] not in skip:
            names.add(key.split("/", 1)[0])
    print(",".join(f"{n}={n}" for n in sorted(names)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
