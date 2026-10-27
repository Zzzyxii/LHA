#!/usr/bin/env python3
"""List sample ids in the instruction file that have no rendered video yet."""
import json, sys
from pathlib import Path
instr, out = Path(sys.argv[1]), Path(sys.argv[2])
for line in instr.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    sid = str(json.loads(line)["sample_id"])
    if not (out / f"{sid}.mp4").exists():
        print(sid)
