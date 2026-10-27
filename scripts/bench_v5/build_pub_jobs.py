"""Job list for the archived published-method bundle.

`归档_extract/methods/<method>/<sample_id>.mp4` holds the released outputs of the
16 methods the previous project compared against (VACE, Ditto, Bernini, InsV2V,
...). Scoring them under bench_v5 costs no renders and puts our arms on the same
table as the published ones.

Caveat recorded in RESULTS: the released outputs do not all share the source's
frame count (most are 17 @20 fps, Ditto/VACE 73, InsV2V 49). The clips are
near-static (source frame-to-frame CLIP similarity 0.997, endpoints 0.957), so a
frame-averaged feature is not sensitive to which span was exported; the
`--self-check` mode of this script quantifies that by extracting the source twice,
once over all frames and once over the first 17.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARCHIVE = Path("归档_extract")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=str(ARCHIVE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--self-check", action="store_true",
                    help="also emit src_head17/<sid> jobs (first 17 frames of the source)")
    args = ap.parse_args()

    root = Path(args.archive)
    jobs = []
    for mdir in sorted((root / "methods").iterdir()):
        if not mdir.is_dir():
            continue
        method = mdir.name
        for mp4 in sorted(mdir.glob("*.mp4")):
            jobs.append({"key": f"pub_{method}/{mp4.stem}", "path": str(mp4)})
    if args.self_check:
        for mp4 in sorted((root / "source_low").glob("*.mp4")):
            jobs.append({"key": f"src_head17/{mp4.stem}", "path": str(mp4)})
    Path(args.out).write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + "\n")
    print(f"wrote {len(jobs)} jobs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
