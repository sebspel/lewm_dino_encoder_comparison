"""Fixed-wall-clock-budget benchmark — real timing harness.

Owned PLUMBING (fails LOUDLY): drives the two
TensorRT engines built by `src.export` through a Python CEM-style rollout — encode ONCE,
then `predict` autoregressively over the horizon (the exact encoder-cached /
predictor-dominates call pattern, docs/platform_api.md §5) — for a fixed wall-clock budget.

**What the numbers are (and are NOT).** This harness runs the MODEL only — the CEM planner
is not in the loop (no candidate sampling / topk / elite update). So:
  - `rollouts_completed` / `throughput` are the **model-only** count (planner treated as
    free) — the *ceiling*, not the realized wall-clock. The realized rollouts-in-budget
    (planner in the loop) comes from the gated eval-shim re-run; their gap is the planner
    floor (SPEC §dilution disclosure, ≈ Amdahl from the profile shares).
  - `latency_p50/p95_ms` time the **predictor step only** — `encode` runs once per rollout
    and is deliberately untimed here (the encoder asymmetry surfaces in throughput and the
    profile). Each step syncs the stream, so for LeWM's tiny op this is a launch+sync floor,
    which compresses the LeWM↔DINO p95 ratio (LeWM is launch-latency-bound, SPEC §Parity).

**SR is NOT produced here.** Every speed figure must carry its SR, but the
SR path is the eval driver re-run on the optimized model through a get_cost /
get_action shim (owner-gated — needs the real checkpoint + adapter wiring). So `benchmark`
returns real speed metrics and `success_rate=NaN`; the headline runner (`src.report`) joins
in the SR per precision from that separate eval run and flags every still-unpaired row.

Runs ONLY on the L40S: `EngineRunner` lazy-imports `tensorrt` and allocates CUDA buffers.
`peak_mem_mb` is sampled from **cudaMemGetInfo** (`torch.cuda.mem_get_info`) — device-level
used memory, so it counts TensorRT's engine + execution-context arena, which a
`torch.cuda.max_memory_allocated` (torch-allocator) reading would silently miss on exactly
the optimized path (SPEC §Interface Contracts).
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

    def _used_mem_mb() -> float:
        # cudaMemGetInfo via torch: device-level (total - free) used bytes. Captures the TRT
        # engine + context arena (a separate cudaMalloc outside torch's allocator). Sampled
        # after warmup, so engines/contexts/I-O buffers are already resident.
        free, total = torch.cuda.mem_get_info(device)
        return (total - free) / 1e6

    peak_mem_mb = _used_mem_mb()
    step_latencies_ms: list[float] = []
    rollouts = 0
    start = perf_counter()
    while perf_counter() - start < time_budget_s:
        step_latencies_ms.extend(_rollout())
        rollouts += 1
        peak_mem_mb = max(peak_mem_mb, _used_mem_mb())  # once per rollout, not per step
    elapsed = perf_counter() - start

    lat = torch.tensor(step_latencies_ms)
    return BenchResult(
        latency_p50_ms=torch.quantile(lat, 0.50).item(),
        latency_p95_ms=torch.quantile(lat, 0.95).item(),
        rollouts_completed=rollouts,
        throughput=rollouts / elapsed,
        peak_mem_mb=peak_mem_mb,
        success_rate=math.nan,  # joined in by src.report from the gated eval-shim re-run
    )
