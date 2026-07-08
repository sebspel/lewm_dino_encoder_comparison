"""Per-component profiling (Phase 5) — encoder / predictor / planner timing.

Owned PLUMBING (fails LOUDLY; CLAUDE.md §8): attributes one CEM planning cycle to its three
components, per the Phase-1 decomposition (docs/platform_api.md §5), so the LeWM↔DINOv3 gap
can be pinned to the right place (SPEC §Parity):

  - ENCODER  : ``adapter.encode`` — runs ~once per cycle (cached across candidates × iters)
  - PREDICTOR: ``adapter.predict`` — autoregressive over the horizon for all candidates
               (dominates the call count; where the token-count asymmetry bites)
  - PLANNER  : the CEM Python/CUDA ops (candidate sampling + topk elites + mean/var update),
               with the model call AND the MSE-to-goal criterion EXCLUDED — the loop that
               stays in Python around the engine

Dims come from the example inputs, so this runs on CPU for the tracer bullet and on the
L40S for the real profile. Timing uses warmup + an optional ``torch.cuda.synchronize`` so a
CUDA number is an accurate device wall-clock, not an async-launch artifact.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable

import torch
from torch import Tensor

from src.interfaces import ComponentProfile, WMStepAdapter, ACTION_DIM

# CEM parity constants for the PLANNER micro-benchmark (docs/platform_api.md §3 — fixed,
# do not vary between tracks): 300 candidates, 30 elites, horizon 5.
_CEM_NUM_SAMPLES = 300
_CEM_TOPK = 30
_CEM_HORIZON = 5
_CEM_BATCH = 1


def _time_ms(
    fn: Callable[[], object], n_iters: int, warmup: int, device: torch.device
) -> float:
    """Mean wall-clock ms per call of `fn`, after `warmup` untimed calls. On CUDA a sync
    brackets the timed loop so the measurement covers real device work, not just launch.
    """
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(n_iters):
        fn()
    if is_cuda:
        torch.cuda.synchronize(device)
    return (perf_counter() - start) * 1000.0 / n_iters


def _planner_step(device: torch.device) -> Callable[[], object]:
    """Build the closure timed as PLANNER cost: one CEM iteration mirroring
    `stable_worldmodel.solver.cem.CEMSolver.solve` (sample -> force first candidate to the
    mean -> topk elites -> recompute mean/std) at `_CEM_BATCH=1` (the parity `batch_size=1`,
    docs/platform_api.md §3), with the model call excluded (docs/platform_api.md §5).
    `cost` stands in for `get_cost`'s entire output (model forward + MSE-to-goal criterion)
    since criterion cost scales with latent size and would otherwise leak the LeWM/DINO
    token-count asymmetry into `planner_ms` (SPEC §Parity attribution). The CEM optimizes the
    env action (ACTION_DIM), not the model-facing frameskip pack, so candidates are that
    width — and it barely affects timing (topk over 300 candidates dominates)."""
    action_dim = ACTION_DIM
    mean = torch.zeros(_CEM_BATCH, _CEM_HORIZON, action_dim, device=device)
    std = torch.ones(_CEM_BATCH, _CEM_HORIZON, action_dim, device=device)
    cost = torch.randn(_CEM_BATCH, _CEM_NUM_SAMPLES, device=device)

    def step() -> object:
        # (B,N,H,D) sampled from (mean, std), first candidate forced to the mean.
        candidates = torch.randn(
            _CEM_BATCH, _CEM_NUM_SAMPLES, _CEM_HORIZON, action_dim, device=device
        )
        candidates = mean.unsqueeze(1) + std.unsqueeze(1) * candidates
        candidates[:, 0] = mean

        # (B,K) elite indices by lowest cost, gathered via the batched two-index form
        # `candidates[batch_indices, topk_indices]` (matches CEMSolver.solve's compound
        # gather, not a plain single-axis index).
        _, topk_indices = torch.topk(cost, k=_CEM_TOPK, dim=1, largest=False)
        batch_indices = (
            torch.arange(_CEM_BATCH, device=device).unsqueeze(1).expand(-1, _CEM_TOPK)
        )
        elite_candidates = candidates[batch_indices, topk_indices]  # (B,K,H,D)

        new_mean = elite_candidates.mean(dim=1)
        new_std = elite_candidates.std(dim=1)
        return new_mean, new_std

    return step


def profile(
    adapter: WMStepAdapter,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
    n_iters: int,
    warmup: int,
) -> ComponentProfile:
    """Time the encoder, predictor, and planner components of one planning cycle."""
    adapter.eval()
    device = encode_inputs[0].device
    planner_step = _planner_step(device)
    with torch.no_grad():
        encoder_ms = _time_ms(
            lambda: adapter.encode(*encode_inputs), n_iters, warmup, device
        )
        predictor_ms = _time_ms(
            lambda: adapter.predict(*predict_inputs), n_iters, warmup, device
        )
        planner_ms = _time_ms(planner_step, n_iters, warmup, device)
    return ComponentProfile(
        encoder_ms=encoder_ms, predictor_ms=predictor_ms, planner_ms=planner_ms
    )
