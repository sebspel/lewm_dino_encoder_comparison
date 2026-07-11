"""Phase-5 owner-gated SR shim: the engine-backed ``get_cost`` used to produce the SR that
pairs with each precision's speed number.

The shim subclasses ``DINOv3PreJEPA`` and overrides only ``_encode_image`` / ``predict``, so
``get_cost`` / ``rollout`` / ``criterion`` / goal-encode are inherited unchanged. Routing the
two overrides through the adapter's pure-torch encode/predict (the exact functions the engines
reconstruct) must reproduce ``model.get_cost`` bit-for-bit on a real ``DINOv3PreJEPA`` (tiny
random backbone — no DINOv3 download); the pod runs the same check on the real checkpoint
(``python -m src.sr_shim``). Bit-for-bit here proves the subclass wiring introduces no silent
cost change; the engines' quantization drift is the only source of SR divergence on the pod.
"""

import torch

from stable_worldmodel.protocols import Actionable

from src.adapter import DINOWMAdapter
from src.fidelity import build_dummy_dino_model
from src.interfaces import ExportConfig
from src.sr_shim import DINOWMSRShim, sr_cost_parity


def test_sr_shim_get_cost_matches_platform():
    torch.manual_seed(0)
    model = build_dummy_dino_model()
    adapter = DINOWMAdapter(model)
    result = sr_cost_parity(model, adapter.encode, adapter.predict, ExportConfig())
    # Same submodules on both sides -> the inherited cost path is reproduced exactly.
    assert result["max_abs"] < 1e-4, result
    assert len(result["shape"]) == 2  # cost is (batch, candidates), the CEM contract


def test_sr_shim_is_non_actionable():
    # No get_action -> prepare_init_action zero-pads the warm-start exactly like the Phase-3
    # baseline (get_cost-only LeWM/DINO-WM); an Actionable shim would perturb the plan.
    torch.manual_seed(0)
    model = build_dummy_dino_model()
    adapter = DINOWMAdapter(model)
    shim = DINOWMSRShim.from_adapter(model, adapter)
    assert not isinstance(shim, Actionable)
    assert not hasattr(shim, "get_action")
    assert hasattr(shim, "get_cost")
