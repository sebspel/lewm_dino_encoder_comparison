"""Phase-5 per-component profiling plumbing — validated on the dummy adapters (CPU).

The real drift numbers need trained weights on the pod, but the timing harness is dim-
driven, so the dummy stand-ins exercise every code path here, including the planner's
CEMSolver-mirroring micro-benchmark (`src.profile._planner_step`).
"""

import torch

from src.adapter import LeWMAdapter, DINOWMAdapter
from src.smoke import build_dummy_lewm, build_dummy_dino
from src.interfaces import HISTORY_SIZE, ACTION_DIM
from src.profile import profile


def _inputs(adapter, batch=2):
    obs = torch.randn(batch, HISTORY_SIZE, 3, 224, 224)
    latent = adapter.encode(obs)
    action = torch.randn(batch, HISTORY_SIZE, ACTION_DIM)
    return (obs,), (latent, action)


def test_profile_lewm_keys_and_positive():
    torch.manual_seed(0)
    adapter = LeWMAdapter(build_dummy_lewm())
    enc, pred = _inputs(adapter)
    p = profile(adapter, enc, pred, n_iters=3, warmup=1)
    assert set(p) == {"encoder_ms", "predictor_ms", "planner_ms"}
    assert p["encoder_ms"] > 0 and p["predictor_ms"] > 0
    assert p["planner_ms"] > 0


def test_profile_dino_runs():
    torch.manual_seed(0)
    adapter = DINOWMAdapter(build_dummy_dino())
    enc, pred = _inputs(adapter)
    p = profile(adapter, enc, pred, n_iters=2, warmup=1)
    assert p["encoder_ms"] > 0 and p["predictor_ms"] > 0
