"""INT8 calibration data-shaping tests (pure torch + onnx, off-pod).

The clip draw (`build_calibration_data`) needs the real dataset and the Model-Optimizer PTQ
call (`src.export.quantize_onnx`) needs `modelopt` (both pod-only); the batching /
adapter-streaming logic AND the numpy-dict producer keyed by ONNX input name are exercised
here on synthetic clips + the dummy adapters — the parts that could silently mis-shape a
calibration array or mis-key it and poison every INT8 scale.
"""

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from src.calibrate import CalibrationData, _sample_cem_actions, make_calibration_dict
from src.interfaces import (
    CEM_HORIZON,
    CEM_VAR_SCALE,
    HISTORY_SIZE,
    DINO_N_PATCHES,
    DINO_PREDICTOR_DIM,
    LATENT_DIM,
    MODEL_ACTION_DIM,
    DINO_PROPRIO_DIM,
)
from src.adapter import LeWMAdapter, DINOWMAdapter
from src.fidelity import build_dummy_dino_model
from src.smoke import build_dummy_lewm

N, BATCH, T = 20, 8, HISTORY_SIZE  # 20 clips, batch 8 -> trims to 16 (two full batches)
# The roll emits one window per predict call and keeps the `T == HS` ones. The roll's
# action-sequence length is CEM_HORIZON (matching the real CEMSolver candidates tensor, which
# is `horizon`-long, NOT `n_obs + horizon`); at eval n_obs=1 the windows are 1,2,3,3,3 over
# (CEM_HORIZON - 1) + 1 = CEM_HORIZON calls -> 3 kept per clip-chunk.
STEADY_PER_CHUNK = CEM_HORIZON - HISTORY_SIZE + 1


def _clips(n=N):
    obs = torch.randn(n, T, 3, 224, 224)
    proprio = torch.randn(n, T, DINO_PROPRIO_DIM)
    action = torch.randn(n, T, MODEL_ACTION_DIM)
    return CalibrationData(obs, proprio, action, BATCH)


def _tiny_onnx(tmp_path, input_names):
    """A minimal ONNX whose graph.input carries `input_names` (in order) — enough for
    `make_calibration_dict` to read the real names off the graph, as it will off the base
    export ONNX on the pod."""
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.FLOAT, [None, 1]) for n in input_names
    ]
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, 1])
    node = helper.make_node("Identity", [input_names[0]], ["out"])
    graph = helper.make_graph([node], "g", inputs, [out])
    path = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph), str(path))
    return path


def test_trims_to_whole_batches():
    data = _clips()
    assert len(data.obs) == (N // BATCH) * BATCH == 16  # partial tail dropped
    batches = data.encoder_batches()
    assert len(batches) == 2
    assert all(b[0].shape == (BATCH, T, 3, 224, 224) for b in batches)


def test_too_few_clips_raises():
    with pytest.raises(ValueError):  # 3 clips < batch 8 -> no full batch
        CalibrationData(
            torch.randn(3, T, 3, 224, 224),
            torch.randn(3, T, DINO_PROPRIO_DIM),
            torch.randn(3, T, MODEL_ACTION_DIM),
            BATCH,
        )


def test_lewm_predictor_stream_is_latent_action():
    data = _clips()
    adapter = LeWMAdapter(build_dummy_lewm())
    batches = data.predictor_batches(adapter)
    # 2 chunks x the roll's steady-state windows (no longer 1 sample per chunk)
    assert len(batches) == 2 * STEADY_PER_CHUNK
    latent, action = batches[0]  # LeWM predict is 2-arity
    assert latent.shape == (BATCH, T, LATENT_DIM)
    assert action.shape == (BATCH, T, MODEL_ACTION_DIM)


def test_dino_predictor_stream_is_404_embedding():
    data = _clips()
    adapter = DINOWMAdapter(build_dummy_dino_model())
    batches = data.predictor_batches(adapter)
    assert len(batches) == 2 * STEADY_PER_CHUNK
    (embedding,) = batches[0]  # DINO predict is a single assembled 404 embedding
    assert embedding.shape == (BATCH, T, DINO_N_PATCHES, DINO_PREDICTOR_DIM)


def test_every_captured_window_binds_the_static_hist_engine():
    """The predictor engine's frame axis is static at HS, so every emitted window must be
    exactly HS — a T<HS window would negative-dim-bind, the crash that killed the SR run."""
    for adapter in (LeWMAdapter(build_dummy_lewm()), DINOWMAdapter(build_dummy_dino_model())):
        for batch in _clips().predictor_batches(adapter):
            assert all(t.shape[1] == HISTORY_SIZE for t in batch)


def test_predictor_actions_are_cem_proposal_not_expert():
    """The whole fix: the stream's actions must come from the unclamped CEM proposal, not the
    clips' Box(-1,1) expert actions. Expert-scaled actions are the ~4x under-scale that
    saturated INT8, so a regression here silently restores the SR collapse."""
    data = _clips()
    data.action.clamp_(-1.0, 1.0)  # expert actions are bounded; the proposal is not
    batches = data.predictor_batches(LeWMAdapter(build_dummy_lewm()))
    assert max(a.abs().max().item() for _, a in batches) > 1.0


def test_cem_action_sample_is_deterministic_and_unclamped():
    a = _sample_cem_actions(64, 6, torch.Generator().manual_seed(0))
    b = _sample_cem_actions(64, 6, torch.Generator().manual_seed(0))
    assert torch.equal(a, b)  # seeded -> the calibration draw stays reproducible
    assert a.shape == (64, 6, MODEL_ACTION_DIM)  # CEM samples the 10-wide pack directly
    assert a.abs().max().item() > 1.0  # unclamped: no projection into Box(-1, 1)
    assert abs(a.std().item() - CEM_VAR_SCALE) < 0.1  # matches the solver's var_scale


def test_encoder_calib_dict_keyed_by_onnx_input(tmp_path):
    data = _clips()
    onnx_path = _tiny_onnx(tmp_path, ["obs"])  # encoder graph has one input
    d = make_calibration_dict(onnx_path, data.encoder_batches())
    assert list(d) == ["obs"]
    assert d["obs"].dtype == np.float32
    # batches concatenated over the whole (trimmed) clip set -> leading axis = 16
    assert d["obs"].shape == (16, T, 3, 224, 224)


def test_lewm_predictor_calib_dict_keyed_and_ordered(tmp_path):
    data = _clips()
    adapter = LeWMAdapter(build_dummy_lewm())
    onnx_path = _tiny_onnx(tmp_path, ["latent", "action"])  # LeWM predict is 2-arity
    d = make_calibration_dict(onnx_path, data.predictor_batches(adapter))
    assert list(d) == ["latent", "action"]  # positional zip preserves order
    # the roll emits STEADY_PER_CHUNK windows per chunk -> the stream is that much longer
    n = 16 * STEADY_PER_CHUNK
    assert d["latent"].shape == (n, T, LATENT_DIM)
    assert d["action"].shape == (n, T, MODEL_ACTION_DIM)


def test_dino_predictor_calib_dict_is_404(tmp_path):
    data = _clips()
    adapter = DINOWMAdapter(build_dummy_dino_model())
    onnx_path = _tiny_onnx(tmp_path, ["embedding"])
    d = make_calibration_dict(onnx_path, data.predictor_batches(adapter))
    assert list(d) == ["embedding"]
    assert d["embedding"].shape == (
        16 * STEADY_PER_CHUNK,
        T,
        DINO_N_PATCHES,
        DINO_PREDICTOR_DIM,
    )


def test_calib_dict_input_count_mismatch_raises(tmp_path):
    data = _clips()
    onnx_path = _tiny_onnx(tmp_path, ["latent", "action"])  # 2 inputs
    with pytest.raises(ValueError):  # encoder stream has 1 array -> mismatch, fails loud
        make_calibration_dict(onnx_path, data.encoder_batches())
