"""Phase-5 adapter-fidelity gate: the DINO-WM shim (encode + assemble + predict + rollout)
must reproduce the platform's `DINOv3PreJEPA.rollout`. Runs on a real `DINOv3PreJEPA` with a
tiny random backbone (no DINOv3 download), so the orchestration is exercised end to end on
CPU; the pod runs the same gate on the real checkpoint (`python -m src.fidelity`)."""

import torch
from torch import nn

from src.adapter import DINOWMAdapter
from src.interfaces import DINO_PREDICTOR_DIM, MODEL_ACTION_DIM, ExportConfig
from src.fidelity import (
    build_dummy_dino_model,
    dino_fidelity,
    lewm_action_encoder_per_frame,
)


def test_dino_shim_matches_platform_rollout():
    torch.manual_seed(0)
    model = build_dummy_dino_model()
    adapter = DINOWMAdapter(model)
    result = dino_fidelity(model, adapter, ExportConfig())
    # Same submodules on both sides -> a faithful shim matches bit-for-bit (float roundoff).
    assert result["passed"], result
    assert result["max_abs"] < 1e-4, result
    assert result["shape"][-1] == DINO_PREDICTOR_DIM  # trajectory carries the full 404


def test_lewm_per_frame_guard_passes_on_real_embedder():
    """The real LeWM Embedder (Conv1d kernel_size=1 + per-position MLP) is per-frame, so
    encoding a step within the full sequence == encoding it as the prefix's last step."""
    from stable_worldmodel.wm.lewm.module import Embedder

    torch.manual_seed(0)
    result = lewm_action_encoder_per_frame(Embedder(input_dim=MODEL_ACTION_DIM))
    assert result["passed"], result
    assert result["max_abs"] < 1e-4, result


def test_lewm_per_frame_guard_catches_temporal_kernel():
    """The guard must FAIL for a temporal (kernel_size>1) encoder — otherwise it would not
    protect the per-step predict-engine boundary it exists to guard."""

    class TemporalEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # kernel_size=3 with padding=1 -> output at t mixes actions t-1, t, t+1 (leaks future).
            self.conv = nn.Conv1d(MODEL_ACTION_DIM, MODEL_ACTION_DIM, kernel_size=3, padding=1)

        def forward(self, x):  # (B, T, D)
            return self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

    torch.manual_seed(0)
    result = lewm_action_encoder_per_frame(TemporalEncoder())
    assert not result["passed"], result
    assert result["max_abs"] > 1e-4, result
