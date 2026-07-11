"""Phase-5 per-component profiling plumbing — validated on the dummy adapters (CPU).

The real drift numbers need trained weights on the pod, but the timing harness is dim-
driven, so the dummy stand-ins exercise every code path here, including the planner's
CEMSolver-mirroring micro-benchmark (`src.profile._planner_step`).
"""

import torch

from src.adapter import LeWMAdapter, DINOWMAdapter
from src.smoke import build_dummy_lewm, build_dummy_dino
from src.interfaces import HISTORY_SIZE, MODEL_ACTION_DIM, DINO_PROPRIO_DIM
from src.profile import profile


def _inputs(adapter, batch=2):
    obs = torch.randn(batch, HISTORY_SIZE, 3, 224, 224)
    latent = adapter.encode(obs)
    action = torch.randn(batch, HISTORY_SIZE, MODEL_ACTION_DIM)
    if isinstance(adapter, DINOWMAdapter):
        proprio = torch.randn(batch, HISTORY_SIZE, DINO_PROPRIO_DIM)
        embedding = adapter.assemble_embedding(latent, proprio, action)
        return (obs,), (embedding,)
    return (obs,), (latent, action)


def test_profile_lewm_keys_and_positive():
    torch.manual_seed(0)
    adapter = LeWMAdapter(build_dummy_lewm())
    enc, pred = _inputs(adapter)
    p = profile(adapter, enc, pred, n_iters=3, warmup=1)
    assert set(p) == {
        "encoder_ms", "predictor_ms", "planner_ms",
        "encoder_calls", "predictor_calls", "planner_calls",
        "encoder_cycle_ms", "predictor_cycle_ms", "planner_cycle_ms",
        "optimizable_fraction", "amdahl_ceiling",
    }
    assert p["encoder_ms"] > 0 and p["predictor_ms"] > 0
    assert p["planner_ms"] > 0


def test_profile_weighting_and_amdahl_consistent():
    """The per-cycle shares are calls × per-call ms, and p / ceiling follow from them —
    the arithmetic the report reads off (issue 4: runtime-weighted, not per-call, shares)."""
    import math

    torch.manual_seed(0)
    adapter = LeWMAdapter(build_dummy_lewm())
    enc, pred = _inputs(adapter)
    p = profile(adapter, enc, pred, n_iters=3, warmup=1)

    assert math.isclose(p["encoder_cycle_ms"], p["encoder_calls"] * p["encoder_ms"])
    assert math.isclose(p["predictor_cycle_ms"], p["predictor_calls"] * p["predictor_ms"])
    assert math.isclose(p["planner_cycle_ms"], p["planner_calls"] * p["planner_ms"])
    # predict dominates the call count (180 vs 2 encode) — the whole point of weighting.
    assert p["predictor_calls"] > p["encoder_calls"]

    cycle = p["encoder_cycle_ms"] + p["predictor_cycle_ms"] + p["planner_cycle_ms"]
    frac = (p["encoder_cycle_ms"] + p["predictor_cycle_ms"]) / cycle
    assert math.isclose(p["optimizable_fraction"], frac)
    assert 0.0 < p["optimizable_fraction"] <= 1.0
    assert math.isclose(p["amdahl_ceiling"], 1.0 / (1.0 - frac))


def test_profile_dino_runs():
    torch.manual_seed(0)
    adapter = DINOWMAdapter(build_dummy_dino())
    enc, pred = _inputs(adapter)
    p = profile(adapter, enc, pred, n_iters=2, warmup=1)
    assert p["encoder_ms"] > 0 and p["predictor_ms"] > 0
