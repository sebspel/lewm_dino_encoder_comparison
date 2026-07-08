"""Phase-4 adapter boundary tests: latent shapes (both tracks, encode + predict) and
that a shape violation actually raises at both the encode and predict boundaries."""

import jaxtyping
import pytest
import torch
from beartype.roar import BeartypeCallHintViolation

# A shape/axis violation raises jaxtyping.TypeCheckError; a dtype/type violation raises
# beartype's. The boundary must reject either.
BOUNDARY_VIOLATION = (jaxtyping.TypeCheckError, BeartypeCallHintViolation)

from src.interfaces import (
    LATENT_DIM,
    DINO_N_PATCHES,
    DINO_LATENT_DIM,
    MODEL_ACTION_DIM,
    DINO_PROPRIO_DIM,
    HISTORY_SIZE,
)
from src.adapter import LeWMAdapter, DINOWMAdapter
from src.smoke import build_dummy_lewm, build_dummy_dino

B, T = 2, HISTORY_SIZE


def _obs():
    return torch.randn(B, T, 3, 224, 224)


def _action():
    return torch.randn(B, T, MODEL_ACTION_DIM)  # 10-wide frameskip pack


def _proprio():
    return torch.randn(B, T, DINO_PROPRIO_DIM)


def _predict_conditioning(adapter, action=None):
    """Per-track predict conditioning after the latent — DINO carries proprio + action,
    LeWM just action (the plumbing drives predict(latent, *conditioning))."""
    action = _action() if action is None else action
    if isinstance(adapter, DINOWMAdapter):
        return (_proprio(), action)
    return (action,)


def test_lewm_shapes():
    adapter = LeWMAdapter(build_dummy_lewm())
    latent = adapter.encode(_obs())
    assert latent.shape == (B, T, LATENT_DIM)  # single token
    nxt = adapter.predict(latent, *_predict_conditioning(adapter))
    assert nxt.shape == (B, T, LATENT_DIM)


def test_dino_shapes():
    adapter = DINOWMAdapter(build_dummy_dino())
    latent = adapter.encode(_obs())
    assert latent.shape == (B, T, DINO_N_PATCHES, DINO_LATENT_DIM)  # patch grid
    nxt = adapter.predict(latent, *_predict_conditioning(adapter))
    assert nxt.shape == (B, T, DINO_N_PATCHES, DINO_LATENT_DIM)


@pytest.mark.parametrize("build, cls", [
    (build_dummy_lewm, LeWMAdapter),
    (build_dummy_dino, DINOWMAdapter),
])
def test_encode_boundary_raises(build, cls):
    adapter = cls(build())
    bad = torch.randn(B, 3, 224, 224)  # missing the hist axis (4-D, not 5-D)
    with pytest.raises(BOUNDARY_VIOLATION):
        adapter.encode(bad)


@pytest.mark.parametrize("build, cls", [
    (build_dummy_lewm, LeWMAdapter),
    (build_dummy_dino, DINOWMAdapter),
])
def test_predict_boundary_raises(build, cls):
    adapter = cls(build())
    latent = adapter.encode(_obs())
    bad_action = torch.randn(B, T)  # missing the action_dim axis (2-D, not 3-D)
    # The @typed boundary validates args on entry, before the body runs, so this fires for
    # DINO too even though its predict body is not yet filled.
    conditioning = _predict_conditioning(adapter, action=bad_action)
    with pytest.raises(BOUNDARY_VIOLATION):
        adapter.predict(latent, *conditioning)
