#!/usr/bin/env bash
# Pod bootstrap for the RunPod L40S (CUDA 12.4).
#
# RunPod pods cannot build Docker images in-pod (no Docker daemon), so the
# dependency layer is installed at pod start instead of baked into an image.
# Run on each fresh pod:  bash setup.sh
#
# Installs:
#   - uv (the package manager)
#   - the owned deps incl. torch (cu124) via `uv sync` from uv.lock
#   - TensorRT (cu12, CUDA-12.4-compatible) via uv pip, OUTSIDE the lock, so it
#     never drags a conflicting libnvinfer/CUDA stack into the project resolution.
#
# A Docker image is composed only at the very end, for reproducibility (off-pod).
#
# NOTE: a bare `uv sync` run later prunes TensorRT (it is not in the lock); re-run
# this script (or the step 3 install) to restore it.
set -euo pipefail

# Pin TensorRT so re-loading a pod reproduces the same engine toolchain. Must be a
# cu12 build compatible with CUDA 12.4; override if the L40S needs another.
TENSORRT_VERSION="${TENSORRT_VERSION:-10.7.0}"

# 1) uv (idempotent)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2) owned deps (torch cu124 + the rest), pinned by uv.lock
uv sync

# 3) TensorRT -- CUDA-12.x build, into the project venv but outside the lock
uv pip install --upgrade \
  --extra-index-url https://pypi.nvidia.com \
  "tensorrt-cu12==${TENSORRT_VERSION}"

# 4) sanity: versions + CUDA match
uv run python - <<'PY'
import torch, tensorrt
print("torch", torch.__version__, "| torch.cuda", torch.version.cuda)
print("tensorrt", tensorrt.__version__)
assert torch.version.cuda and torch.version.cuda.startswith("12."), torch.version.cuda
PY
