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

import pytest
import torch

from stable_worldmodel.protocols import Actionable

from src.adapter import DINOWMAdapter, LeWMAdapter
from src.fidelity import build_dummy_dino_model
from src.interfaces import ExportConfig
from src.sr_shim import (
    DINOWMSRShim,
    LeWMSRShim,
    _hist_adapt,
    _predict_hist_adapt,
    build_dummy_lewm_model,
    sr_cost_parity,
    sr_cost_parity_lewm,
)


def _per_frame_encode(pixels):
    # A temporally-independent stand-in for the encoder engine: each frame maps to an embedding
    # depending ONLY on that frame (folds (b t), per-frame op) — the exact property _hist_adapt's
    # pad/slice relies on. Returns (b, t, 8).
    b, t = pixels.shape[:2]
    return (pixels.reshape(b * t, -1)[:, :8] * 2.0).reshape(b, t, 8)


def _causal_predict(*inputs):
    # A CAUSAL stand-in for the predictor engine: output frame t depends ONLY on input frames
    # 0..t (a cumulative op over the frame axis) — the exact property _predict_hist_adapt's
    # pad/slice relies on (right-padding frames after T and slicing [:, :T] leaves 0..T-1
    # unchanged). Reduces any extra inputs to a per-frame scalar so it also covers the two-input
    # (LeWM emb + RAW action) form regardless of their feature widths.
    out = torch.cumsum(inputs[0], dim=1) * 2.0
    for extra in inputs[1:]:
        out = out + torch.cumsum(extra.mean(dim=-1, keepdim=True), dim=1)
    return out


def test_hist_adapt_short_hist_matches_native_encode():
    # The static-hist engine would raise on the goal encode (T=1). _hist_adapt repeat-pads T up
    # to the traced hist and slices back; because the encoder is temporally independent, the
    # result must equal a native T=1 encode bit-for-bit (the padded frames don't touch frame 0).
    torch.manual_seed(0)
    adapted = _hist_adapt(_per_frame_encode, enc_hist=3)
    px1 = torch.randn(2, 1, 3, 4, 4)  # T=1, goal-style
    assert torch.equal(adapted(px1), _per_frame_encode(px1))
    assert adapted(px1).shape[1] == 1


def test_hist_adapt_full_hist_passes_through():
    torch.manual_seed(0)
    adapted = _hist_adapt(_per_frame_encode, enc_hist=3)
    px3 = torch.randn(2, 3, 3, 4, 4)  # T == hist, init-style
    assert torch.equal(adapted(px3), _per_frame_encode(px3))


def test_hist_adapt_rejects_longer_hist():
    # T > enc_hist can't occur in the CEM cost path (init=hist, goal=1) and the static engine
    # cannot serve it -> loud error, never a silent wrong SR.
    adapted = _hist_adapt(_per_frame_encode, enc_hist=3)
    with pytest.raises(ValueError):
        adapted(torch.randn(1, 4, 3, 4, 4))


@pytest.mark.parametrize("t", [1, 2])
def test_predict_hist_adapt_short_hist_matches_native(t):
    # The static-HS predict engine can't bind the T<HS windows the rollout feeds (n_obs=1 ->
    # 1,2,3). _predict_hist_adapt right-pads the frame axis to HS and slices [:, :T] back;
    # because the predictor is causal with prefix positional embeddings, frames 0..T-1 never
    # attend the padded tail, so the result equals a native T-window predict bit-for-bit.
    adapted = _predict_hist_adapt(_causal_predict, pred_hist=3)
    x = torch.randn(2, t, 5, 8)  # (B, T, P, D)-style DINO embedding window
    assert torch.equal(adapted(x), _causal_predict(x))
    assert adapted(x).shape[1] == t


def test_predict_hist_adapt_full_hist_passes_through():
    adapted = _predict_hist_adapt(_causal_predict, pred_hist=3)
    x = torch.randn(2, 3, 5, 8)  # T == HS: no pad, straight through
    assert torch.equal(adapted(x), _causal_predict(x))


def test_predict_hist_adapt_two_inputs_lewm_shape():
    # LeWM's predict engine is two-input — (emb, RAW action) sharing the frame axis; the adapt
    # must pad BOTH and slice the single output back.
    adapted = _predict_hist_adapt(_causal_predict, pred_hist=3)
    emb = torch.randn(2, 1, 192)
    act = torch.randn(2, 1, 10)
    assert torch.equal(adapted(emb, act), _causal_predict(emb, act))
    assert adapted(emb, act).shape[1] == 1


def test_predict_hist_adapt_rejects_longer_hist():
    # T > HS cannot occur (the rollout caps the window at HS) and the static engine cannot serve
    # it -> loud error, never a silent wrong SR.
    adapted = _predict_hist_adapt(_causal_predict, pred_hist=3)
    with pytest.raises(ValueError):
        adapted(torch.randn(1, 4, 5, 8))


@pytest.mark.parametrize("n_obs", [None, 1])
def test_sr_shim_get_cost_matches_platform(n_obs):
    # n_obs=None -> history_size (traced-HS window); n_obs=1 -> the GROWING sub-HS windows the
    # eval feeds (1,2,3). Bit-for-bit either way (same submodules), so the shim + inherited
    # rollout are proven correct at the variable-window eval condition too.
    torch.manual_seed(0)
    model = build_dummy_dino_model()
    adapter = DINOWMAdapter(model)
    result = sr_cost_parity(model, adapter.encode, adapter.predict, ExportConfig(), n_obs=n_obs)
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


@pytest.mark.parametrize("n_obs", [None, 1])
def test_lewm_sr_shim_encode_override_matches_platform(n_obs):
    # LeWM.encode has no _encode_image seam, so LeWMSRShim RE-IMPLEMENTS encode's body, AND
    # predict routes through the engine boundary (raw action -> per-frame action_encoder inside).
    # Routing both through the adapter's torch encode/predict must reproduce LeWM.get_cost
    # bit-for-bit -> the override + predict boundary preserve the inherited cost path (no silent
    # SR corruption). The per-frame action_encoder is what makes the Identity-passthrough rollout
    # (raw actions windowed into predict) exactly equal the source's whole-sequence pre-encode.
    # n_obs=1 drives the GROWING sub-HS predict windows the eval feeds (the variable-window case).
    torch.manual_seed(0)
    model = build_dummy_lewm_model()
    adapter = LeWMAdapter(model)
    result = sr_cost_parity_lewm(
        model, adapter.encode, adapter.predict, ExportConfig(), n_obs=n_obs
    )
    assert result["max_abs"] < 1e-4, result
    assert result["shape"] == (1, 4)  # B=1 (batch_size=1 contract) x candidates


def test_lewm_sr_shim_is_non_actionable():
    torch.manual_seed(0)
    model = build_dummy_lewm_model()
    adapter = LeWMAdapter(model)
    shim = LeWMSRShim.from_adapter(model, adapter)
    assert not isinstance(shim, Actionable)
    assert not hasattr(shim, "get_action")
    assert hasattr(shim, "get_cost")
