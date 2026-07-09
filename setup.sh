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
#   - the NVIDIA TensorRT Model Optimizer (nvidia-modelopt[onnx], the explicit-INT8 Q/DQ
#     tool) and its onnxruntime-gpu, the same way — via uv pip, outside the lock — but
#     pinned to CUDA-12 builds: onnxruntime-gpu's default PyPI wheel is now CUDA 13, which
#     pulls nvidia-*-cu13 and cannot init cuDNN against the pod's 12.x driver, so it is
#     installed from onnxruntime's dedicated CUDA-12 feed. The whole export stack thus stays
#     on CUDA major 12, matching torch cu124 + TensorRT + the pod driver.
#
# A Docker image is composed only at the very end, for reproducibility (off-pod).
#
# NOTE: a bare `uv sync` run later prunes TensorRT + onnxruntime-gpu + modelopt (not in the
# lock); re-run this script (or the step 3 installs) to restore them.
set -euo pipefail

# Pin TensorRT so re-loading a pod reproduces the same engine toolchain. Must be a
# cu12 build compatible with CUDA 12.4; override if the L40S needs another.
TENSORRT_VERSION="${TENSORRT_VERSION:-10.7.0}"

# The export toolchain must stay on CUDA major 12 (torch is locked to cu124 and the pod
# driver tops out at 12.x — a CUDA-13 build cannot run here). onnxruntime-gpu's default PyPI
# wheel is now CUDA 13, so it is pulled from onnxruntime's dedicated CUDA-12 feed instead;
# that feed hosts ONLY cu12 builds, so even the unpinned latest there is cu12. Override the
# feed if the URL moves. Leave the versions empty for the latest builds; pin both (export
# ONNXRUNTIME_GPU_VERSION=... MODELOPT_VERSION=...) once a known-good pair is confirmed here.
ONNXRUNTIME_CUDA12_INDEX="${ONNXRUNTIME_CUDA12_INDEX:-https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/}"
ONNXRUNTIME_GPU_VERSION="${ONNXRUNTIME_GPU_VERSION:-}"
MODELOPT_VERSION="${MODELOPT_VERSION:-}"

# 0) uv cache — force to ephemeral /tmp so the 15GB archive-v0 never lands on the
#    network volume (/workspace). Safe to lose on restart (just cached wheels).
export UV_CACHE_DIR=/tmp/uv-cache

# 1) uv (idempotent)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 2) owned deps (torch cu124 + the rest), pinned by uv.lock
uv sync

# 3) Export toolchain (TensorRT + onnxruntime-gpu + Model Optimizer) -- CUDA-12 builds, into
#    the project venv but outside the lock. Order matters: pin the cu12 onnxruntime-gpu BEFORE
#    modelopt so modelopt[onnx]'s unbounded onnxruntime-gpu dependency is already satisfied by
#    a cu12 wheel and is not re-resolved to the cu13 PyPI default.

#    TensorRT (cu12, pinned):
uv pip install --upgrade \
  --extra-index-url https://pypi.nvidia.com \
  "tensorrt-cu12==${TENSORRT_VERSION}"

#    onnxruntime-gpu -- from onnxruntime's CUDA-12 feed as the PRIMARY index (--index-url) so
#    the cu12 wheel wins over PyPI's cu13 default; PyPI is only the fallback for its pure-
#    Python deps (numpy, protobuf, ...). uv's default first-index strategy takes onnxruntime-
#    gpu from the primary feed, where only cu12 builds exist.
uv pip install \
  --index-url "${ONNXRUNTIME_CUDA12_INDEX}" \
  --extra-index-url https://pypi.org/simple/ \
  "onnxruntime-gpu${ONNXRUNTIME_GPU_VERSION:+==${ONNXRUNTIME_GPU_VERSION}}"

#    Model Optimizer -- NO --upgrade, so it keeps the cu12 onnxruntime-gpu just installed
#    instead of re-resolving it (which would pull the cu13 PyPI default back in).
uv pip install \
  --extra-index-url https://pypi.nvidia.com \
  "nvidia-modelopt[onnx]${MODELOPT_VERSION:+==${MODELOPT_VERSION}}"

# 4) sanity: versions + a REAL onnxruntime CUDA-EP session across the whole export toolchain.
#    The old check only imported modelopt and asserted torch's CUDA — it passed even with a
#    cu13 onnxruntime-gpu, which then failed cudnnCreate at INT8-export time. Opening a CUDA-EP
#    session here (mirroring modelopt's own preload_dlls) makes a CUDA-major mismatch fail
#    provisioning LOUDLY instead of silently later. Needs the pod GPU (this is a pod bootstrap).
uv run python - <<'PY'
import numpy as np, onnx, onnxruntime as ort
import torch, tensorrt, modelopt
from modelopt.onnx.quantization import quantize  # noqa: F401  (explicit-INT8 PTQ entrypoint)
from onnx import helper, TensorProto
print("torch", torch.__version__, "| torch.cuda", torch.version.cuda)
print("tensorrt", tensorrt.__version__)
print("modelopt", modelopt.__version__)
print("onnxruntime-gpu", ort.__version__, "| providers", ort.get_available_providers())
assert torch.version.cuda and torch.version.cuda.startswith("12."), torch.version.cuda
assert "CUDAExecutionProvider" in ort.get_available_providers(), "onnxruntime-gpu has no CUDA EP"
# modelopt loads the CUDA/cuDNN DLLs from the nvidia wheels this way; mirror it so the check
# sees the same libraries the real INT8 quantization will.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)]), "/tmp/_ort_cuda_check.onnx")
ort.InferenceSession("/tmp/_ort_cuda_check.onnx",
                     providers=[("CUDAExecutionProvider", {"device_id": 0})]).run(
    None, {"x": np.ones((1, 4), np.float32)})
print("onnxruntime CUDA EP: OK (cuDNN initialized on the CUDA-12 stack)")
PY

# 5) secrets: HF_TOKEN must be in the runtime env for gated DINOv3 downloads
#    (facebook/dinov3-*). Provisioning still succeeds without it (LeWM uses a
#    scratch ViT), so this warns loudly rather than aborting.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Gated DINOv3 downloads will 401." >&2
  echo "         export HF_TOKEN=hf_... (and accept the model license on HF)" >&2
  echo "         before running prejepa/introspection." >&2
else
  echo "HF_TOKEN is set."
fi

# 6) secrets: WANDB_API_KEY must be in the runtime env for training. The Phase-2
#    +experiment overlays set wandb.enabled=true, so the trainer inits WandbLogger at
#    startup and will stall on an interactive prompt (or fail) without it. Provisioning
#    still succeeds without it (smoke with wandb.enabled=false), so this warns rather
#    than aborting.
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WARNING: WANDB_API_KEY is not set. Training (+experiment overlays enable W&B)" >&2
  echo "         will stall on a login prompt. export WANDB_API_KEY=... before" >&2
  echo "         training, or pass wandb.enabled=false for a no-logging smoke." >&2
else
  echo "WANDB_API_KEY is set."
fi

# 7) persistent storage: STABLEWM_HOME is the platform's cache root — datasets land in
#    $STABLEWM_HOME/datasets and checkpoints in $STABLEWM_HOME/checkpoints/<run_name>/
#    (stable_worldmodel.wm.utils.save_pretrained). Its own default (~/.stable_worldmodel)
#    is the EPHEMERAL container fs on RunPod — a multi-hour run's checkpoints are lost on
#    pod restart. Default it to the persistent network volume (RunPod mounts it at
#    /workspace) unless already set in the env, and export so this script's own steps
#    (mkdir + dataset download below) target it.
#    NOTE: this export only reaches this script's subprocesses, not your other shells —
#    set STABLEWM_HOME as a RunPod env var too so training/eval terminals inherit it.
export STABLEWM_HOME="${STABLEWM_HOME:-/workspace/.stablewm}"
case "$STABLEWM_HOME" in
  "$HOME" | "$HOME"/*)
    echo "WARNING: STABLEWM_HOME=$STABLEWM_HOME is under \$HOME (ephemeral on RunPod)." >&2
    echo "         Point it at the network volume, e.g. /workspace/.stablewm." >&2
    ;;
esac
mkdir -p "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"
echo "STABLEWM_HOME=$STABLEWM_HOME (datasets + checkpoints persist here)."

# 8) Push-T expert dataset: the train configs request 'pusht_expert_train.lance' by
#    bare name, which the resolver does NOT auto-fetch from HF — it must exist under
#    $STABLEWM_HOME/datasets. Pull it once (idempotent: skipped if already present;
#    re-run resumes a partial download). Public dataset, no HF_TOKEN required.
ds="$STABLEWM_HOME/datasets/pusht_expert_train.lance"
if [ -d "$ds" ]; then
  echo "Push-T dataset present: $ds (skipping download)."
else
  echo "Fetching Push-T expert dataset (~14GB) into $STABLEWM_HOME/datasets ..."
  uv run hf download galilai-group/lewm-pusht \
    --repo-type dataset \
    --include "pusht_expert_train.lance/*" \
    --local-dir "$STABLEWM_HOME/datasets"
fi
