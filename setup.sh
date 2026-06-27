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
#    (stable_worldmodel.wm.utils.save_pretrained). It defaults to ~/.stable_worldmodel,
#    which on RunPod is the EPHEMERAL container fs — a multi-hour run's checkpoints are
#    lost on pod restart. Point it at the persistent network volume (RunPod mounts it at
#    /workspace). Set it in the pod's runtime env so every shell inherits it; this step
#    validates + creates the dirs, mirroring the secret checks above (warns, never aborts).
if [ -z "${STABLEWM_HOME:-}" ]; then
  echo "WARNING: STABLEWM_HOME is not set. Datasets + checkpoints default to" >&2
  echo "         ~/.stable_worldmodel on the ephemeral container fs and are LOST on pod" >&2
  echo "         restart. Point it at the network volume, e.g.:" >&2
  echo "             export STABLEWM_HOME=/workspace/.stablewm" >&2
  echo "         (set as a RunPod env var so training/eval shells inherit it.)" >&2
else
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
fi
