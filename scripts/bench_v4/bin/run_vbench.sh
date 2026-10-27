#!/bin/bash
# usage: run_vbench.sh <arm_dir> <instructions_jsonl> <out_json> <gpu> [dims...]
# Scores a directory of <sample_id>.mp4 with VBench in custom_input mode.
set -uo pipefail
# The pod is shared and VBench initialises torch.distributed on a fixed default
# port, so a stray process (ours or someone else's) on 29500 makes it fail with
# EADDRINUSE. Pick a distinct port per invocation.
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(( 30000 + RANDOM % 20000 ))
PROJ=/apdcephfs_szcf/share_304169095/hunyuan/juliayxzhu/dev/tmpp/outt
VB=/apdcephfs_szcf/share_304169095/junjhzhu/VBench
PY=/apdcephfs_szcf/share_304169095/junjhzhu/venvs/vbench-eval/bin/python
ARM=$1
INSTR=$2
OUT=$3
GPU=$4
shift 4
DIMS=${@:-"aesthetic_quality subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree"}
TMP=$PROJ/out/bench_v4/vbench_tmp
mkdir -p "$TMP" "$(dirname "$OUT")"
/opt/venv/bin/python - "$ARM" "$INSTR" "$TMP/prompts_$(basename "$ARM").json" <<'PY'
import json, os, sys
arm, instr, dest = sys.argv[1], sys.argv[2], sys.argv[3]
texts = {}
for line in open(instr):
    line = line.strip()
    if line:
        it = json.loads(line)
        texts[str(it["sample_id"])] = str(it.get("instruction") or it.get("intent") or "")
prompts = {}
for f in sorted(os.listdir(arm)):
    if not f.endswith(".mp4"):
        continue
    sid = os.path.splitext(f)[0]
    prompts[os.path.join(os.path.abspath(arm), f)] = texts.get(sid, sid)
json.dump(prompts, open(dest, "w"), ensure_ascii=False)
print("prompts:", len(prompts))
PY
cd "$VB" && CUDA_VISIBLE_DEVICES=$GPU "$PY" evaluate.py \
  --videos_path "$ARM" --dimension $DIMS --mode custom_input \
  --prompt_file "$TMP/prompts_$(basename "$ARM").json" \
  --load_ckpt_from_local True \
  --output_path "$TMP/$(basename "$ARM")_out" > "$TMP/$(basename "$ARM").log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "vbench failed rc=$rc"; tail -5 "$TMP/$(basename "$ARM").log"; exit $rc; fi
/opt/venv/bin/python - "$TMP/$(basename "$ARM")_out" "$OUT" <<'PY'
import json, glob, os, sys, re
src, dest = sys.argv[1], sys.argv[2]
best = None
for p in sorted(glob.glob(os.path.join(src, "*_eval_results.json"))):
    best = p
if best is None:
    raise SystemExit("no eval results in " + src)
data = json.load(open(best))
out = {}
for dim, payload in data.items():
    if isinstance(payload, list):
        vals = [v for v in payload if isinstance(v, (int, float))]
        out[dim] = round(sum(vals) / len(vals), 4) if vals else None
    elif isinstance(payload, dict):
        vals = [v for v in payload.values() if isinstance(v, (int, float))]
        out[dim] = round(sum(vals) / len(vals), 4) if vals else None
json.dump(out, open(dest, "w"), ensure_ascii=False, indent=2)
print(json.dumps(out, indent=2))
PY
