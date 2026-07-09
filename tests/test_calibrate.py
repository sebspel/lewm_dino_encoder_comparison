"""INT8 calibration data-shaping tests (pure torch, off-pod).

The clip draw (`build_calibration_data`) and the `IInt8MinMaxCalibrator` subclass need the
real dataset + `tensorrt` (pod-only); the batching / adapter-streaming logic is exercised
here on synthetic clips + the dummy adapters — the part that could silently mis-shape a
calibration batch and poison every INT8 scale.
"""

import pytest
import torch

from src.calibrate import CalibrationData
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
