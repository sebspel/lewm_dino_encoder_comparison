"""Phase-5 adapter-fidelity gate: the DINO-WM shim (encode + assemble + predict + rollout)
must reproduce the platform's `DINOv3PreJEPA.rollout`. Runs on a real `DINOv3PreJEPA` with a
tiny random backbone (no DINOv3 download), so the orchestration is exercised end to end on
CPU; the pod runs the same gate on the real checkpoint (`python -m src.fidelity`)."""

import torch

from src.adapter import DINOWMAdapter
from src.interfaces import DINO_PREDICTOR_DIM, ExportConfig
from src.fidelity import build_dummy_dino_model, dino_fidelity


def test_dino_shim_matches_platform_rollout():
    torch.manual_seed(0)
    model = build_dummy_dino_model()
    adapter = DINOWMAdapter(model)
    result = dino_fidelity(model, adapter, ExportConfig())
    # Same submodules on both sides -> a faithful shim matches bit-for-bit (float roundoff).
    assert result["passed"], result
    assert result["max_abs"] < 1e-4, result
    assert result["shape"][-1] == DINO_PREDICTOR_DIM  # trajectory carries the full 404
