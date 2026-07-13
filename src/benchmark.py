"""Latency benchmark — real timing harness for the exported engines.

Owned PLUMBING (fails LOUDLY). **Latency is the headline** (SPEC §Interface Contracts):
this harness measures the two COMPONENT latency distributions in isolated, equal-n
fixed-iteration loops on the engines — **encode-step** (exposes the LeWM↔DINOv3 encoder
token-count asymmetry) and **predict-step** (quantization's kernel target) — each as p50/p95,
warm-up dropped. It also samples peak GPU memory.

There is **no fixed-wall-clock rollout-count run** (owner decision — redundant with the
per-cycle latency under serial planning). The HEADLINE **per-cycle** latency (full CEM solve)
and the **SR** are NOT produced here: both come from the gated eval-shim re-run (`src.sr_eval`,
via the observation-only CEM-solve-latency callback), so they share the same solves. This
harness therefore returns real component-latency + peak-mem numbers with `per_cycle_*` and
`success_rate` left NaN; `src.report` joins the per-cycle latency + SR back in per precision.

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


def _percentiles_ms(step_ms: list[float]) -> tuple[float, float]:
    """(p50, p95) over a list of per-call latencies in ms."""
    lat = torch.tensor(step_ms)
    return torch.quantile(lat, 0.50).item(), torch.quantile(lat, 0.95).item()


def _time_loop(runner: EngineRunner, inputs: tuple[Tensor, ...], n_iters: int) -> list[float]:
    """Time `n_iters` isolated calls of one engine. `EngineRunner.run` syncs its stream, so
    `perf_counter` around it is an accurate per-call GPU wall-clock (not an async-launch
    artifact). Warm-up is the caller's responsibility (run before the timed loop)."""
    step_ms: list[float] = []
    for _ in range(n_iters):
        t0 = perf_counter()
        runner.run(inputs)
        step_ms.append((perf_counter() - t0) * 1000.0)
    return step_ms


def benchmark(
    engines: EnginePaths,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
    n_iters: int,
    warmup: int,
) -> BenchResult:
    for name, path in engines.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} engine missing: {path}")

    encoder = EngineRunner(engines["encoder"])
    predictor = EngineRunner(engines["predictor"])
    device = encoder.device

    # encode_inputs is the single obs (batch 1 — the cached per-cycle encode); predict_inputs
    # is the dim-preserving predictor STATE (LeWM latent 192; DINO assembled 404) at the
    # candidate fan-out batch, plus per-track fixed conditioning (LeWM action; DINO none).
    # Values are held fixed across iters — this measures per-call timing, not rollout state.
    for _ in range(warmup):
        encoder.run(encode_inputs)
        predictor.run(predict_inputs)
    torch.cuda.synchronize(device)

    def _used_mem_mb() -> float:
        # cudaMemGetInfo via torch: device-level (total - free) used bytes. Captures the TRT
        # engine + context arena (a separate cudaMalloc outside torch's allocator). Sampled
        # after warm-up, so engines/contexts/I-O buffers are already resident.
        free, total = torch.cuda.mem_get_info(device)
        return (total - free) / 1e6

    peak_mem_mb = _used_mem_mb()
    encode_ms = _time_loop(encoder, encode_inputs, n_iters)
    peak_mem_mb = max(peak_mem_mb, _used_mem_mb())
    predict_ms = _time_loop(predictor, predict_inputs, n_iters)
    peak_mem_mb = max(peak_mem_mb, _used_mem_mb())

    encode_p50, encode_p95 = _percentiles_ms(encode_ms)
    predict_p50, predict_p95 = _percentiles_ms(predict_ms)
    return BenchResult(
        per_cycle_p50_ms=math.nan,  # joined by src.report from the gated eval-shim re-run
        per_cycle_p95_ms=math.nan,
        encode_p50_ms=encode_p50,
        encode_p95_ms=encode_p95,
        predict_p50_ms=predict_p50,
        predict_p95_ms=predict_p95,
        peak_mem_mb=peak_mem_mb,
        success_rate=math.nan,  # joined in by src.report from the gated eval-shim re-run
    )
