"""Fixed-wall-clock-budget benchmark (Phase 5) — real timing harness.

Owned PLUMBING (fails LOUDLY; CLAUDE.md §8 / SPEC §Interface Contracts): drives the two
TensorRT engines built by `src.export` through a Python CEM-style rollout — encode ONCE,
then `predict` autoregressively over the horizon (the exact encoder-cached /
predictor-dominates call pattern, docs/platform_api.md §5) — for a fixed wall-clock budget,
and records per-step inference latency (p50/p95), rollouts completed, throughput, and peak
GPU memory. Only the model runs in the engine; the rollout loop stays in Python.

**SR is NOT produced here.** Every speed figure must carry its SR (SPEC §Parity), but the
SR path is the Phase-3 eval driver re-run on the optimized model through a get_cost /
get_action shim (OWNER-gated — needs the real checkpoint + adapter wiring). So `benchmark`
returns real speed metrics and `success_rate=NaN`; the headline runner (`src.report`) joins
in the SR per precision from that separate eval run.

Runs ONLY on the L40S: `EngineRunner` lazy-imports `tensorrt` and allocates CUDA buffers.
`peak_mem_mb` is `torch.cuda.max_memory_allocated` — torch-visible I/O + activation
buffers; TensorRT's own workspace arena is a separate cudaMalloc and is not counted here.
"""

from __future__ import annotations

import math
from time import perf_counter

import torch
from torch import Tensor

from src.interfaces import EnginePaths, BenchResult
from src.trt_runtime import EngineRunner

# One rollout replays PlanConfig.horizon predictor steps over the cached latent
# (docs/platform_api.md §3: horizon=5). Fixed across tracks — a parity condition.
_ROLLOUT_HORIZON = 5


def benchmark(
    engines: EnginePaths,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
    time_budget_s: float,
    warmup: int,
) -> BenchResult:
    for name, path in engines.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} engine missing: {path}")

    encoder = EngineRunner(engines["encoder"])
    predictor = EngineRunner(engines["predictor"])
    device = encoder.device
    # predict_inputs[0] is the dim-preserving predictor STATE (LeWM: latent 192; DINO: the
    # assembled 404 embedding) that `predict` returns unchanged in shape, so it re-feeds
    # autoregressively. predict_inputs[1:] is the per-track fixed conditioning (LeWM: action;
    # DINO: none) reused each step — this measures timing, not rollout correctness.
    state0 = predict_inputs[0]
    conditioning = predict_inputs[1:]

    def _rollout() -> list[float]:
        """One rollout: encode once (timed as the encoder cost), then HORIZON predictor
        steps feeding the state forward. Returns each predictor step's latency in ms
        (EngineRunner.run syncs its stream, so perf_counter around it is an accurate per-step
        GPU wall-clock number). DINO's 384->404 assembly is a Python step outside the engine,
        so the encoder output is not threaded directly into `predict` here."""
        encoder.run(encode_inputs)
        state = state0
        step_ms = []
        for _ in range(_ROLLOUT_HORIZON):
            t0 = perf_counter()
            state = predictor.run((state, *conditioning))
            step_ms.append((perf_counter() - t0) * 1000.0)
        return step_ms

    for _ in range(warmup):
        _rollout()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    step_latencies_ms: list[float] = []
    rollouts = 0
    start = perf_counter()
    while perf_counter() - start < time_budget_s:
        step_latencies_ms.extend(_rollout())
        rollouts += 1
    elapsed = perf_counter() - start

    lat = torch.tensor(step_latencies_ms)
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
    return BenchResult(
        latency_p50_ms=torch.quantile(lat, 0.50).item(),
        latency_p95_ms=torch.quantile(lat, 0.95).item(),
        rollouts_completed=rollouts,
        throughput=rollouts / elapsed,
        peak_mem_mb=peak_mem_mb,
        success_rate=math.nan,  # joined in by src.report from the gated eval-shim re-run
    )
