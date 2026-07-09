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

from src.calibrate import CalibrationData, make_calibration_dict
from src.interfaces import (
    HISTORY_SIZE,
    DINO_N_PATCHES,
    DINO_PREDICTOR_DIM,
    LATENT_DIM,
    MODEL_ACTION_DIM,
    DINO_PROPRIO_DIM,
)
from src.adapter import LeWMAdapter, DINOWMAdapter
from src.smoke import build_dummy_lewm, build_dummy_dino

N, BATCH, T = 20, 8, HISTORY_SIZE  # 20 clips, batch 8 -> trims to 16 (two full batches)


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
    assert len(batches) == 2
    latent, action = batches[0]  # LeWM predict is 2-arity
    assert latent.shape == (BATCH, T, LATENT_DIM)
    assert action.shape == (BATCH, T, MODEL_ACTION_DIM)


def test_dino_predictor_stream_is_404_embedding():
    data = _clips()
    adapter = DINOWMAdapter(build_dummy_dino())
    batches = data.predictor_batches(adapter)
    assert len(batches) == 2
    (embedding,) = batches[0]  # DINO predict is a single assembled 404 embedding
    assert embedding.shape == (BATCH, T, DINO_N_PATCHES, DINO_PREDICTOR_DIM)


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
    assert d["latent"].shape == (16, T, LATENT_DIM)
    assert d["action"].shape == (16, T, MODEL_ACTION_DIM)


def test_dino_predictor_calib_dict_is_404(tmp_path):
    data = _clips()
    adapter = DINOWMAdapter(build_dummy_dino())
    onnx_path = _tiny_onnx(tmp_path, ["embedding"])
    d = make_calibration_dict(onnx_path, data.predictor_batches(adapter))
    assert list(d) == ["embedding"]
    assert d["embedding"].shape == (16, T, DINO_N_PATCHES, DINO_PREDICTOR_DIM)


def test_calib_dict_input_count_mismatch_raises(tmp_path):
    data = _clips()
    onnx_path = _tiny_onnx(tmp_path, ["latent", "action"])  # 2 inputs
    with pytest.raises(ValueError):  # encoder stream has 1 array -> mismatch, fails loud
        make_calibration_dict(onnx_path, data.encoder_batches())
