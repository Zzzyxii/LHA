#!/usr/bin/env bash
# Download one IVEBench short clip. Usage: dl.sh <name>
set -uo pipefail
N=$1
OUT=mp4_short/$N
[ -s "$OUT" ] && exit 0
timeout 300 curl -sSL -o "$OUT.tmp" "https://huggingface.co/datasets/Coraxor/IVEBench/resolve/main/ivebenchdb_mp4_short/$N" \
  && mv "$OUT.tmp" "$OUT" && echo "ok $N $(wc -c < "$OUT")"
