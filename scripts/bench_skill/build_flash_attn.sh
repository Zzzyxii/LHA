#!/usr/bin/env bash
# Build flash-attn from source: Bernini hard-requires FA2/FA3 and the mirror has
# no prebuilt wheel for torch 2.9.1+cu129 / py312. MAX_JOBS is capped so the
# compile does not starve the concurrently running GPU bench jobs.
set -eu
export MAX_JOBS=8
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="9.0"
pip install --no-deps --no-build-isolation \
  -i https://mirrors.tencent.com/pypi/simple/ flash-attn==2.8.3
