"""Latency benchmark — real timing harness for the exported engines.

Owned PLUMBING (fails LOUDLY). **Latency is the headline** (SPEC §Interface Contracts):
this harness measures the two COMPONENT latency distributions in isolated, equal-n
fixed-iteration loops on the engines — **encode-step** (exposes the LeWM↔DINOv3 encoder
token-count asymmetry) and **predict-step** (quantization's kernel target) — each as p50/p95,
warm-up dropped. Each step additionally carries its arithmetic
**mean**, which feeds `src.report`'s per-component decomposition ONLY (means compose additively,
percentiles do not) and is never reported as a headline.

Alongside those summaries it returns the loops' **raw per-call samples** (`ComponentSamples`), which
`src.study` persists to `latencies.<track>.json` — so the component p50's confidence interval and its
lag-1 independence test are re-derivable off-pod rather than taken on trust, and no added statistic
costs an L40S run (SPEC §Interface Contracts, docs/architecture.md §9).

There is **no fixed-wall-clock rollout-count run** (owner decision — redundant with the
per-cycle latency under serial planning). The HEADLINE **per-cycle** latency (one episode's full
decision) and the **SR** are NOT produced here: both come from the gated eval-shim re-run
(`src.sr_eval`, via the observation-only per-decision latency callback), so they share the same
solves. This
harness therefore returns real component-latency numbers with `per_cycle_*` and
`success_rate` left NaN; `src.report` joins the per-cycle latency + SR back in per precision.

Runs ONLY on the L40S: `EngineRunner` lazy-imports `tensorrt` and allocates CUDA buffers.
"""

from __future__ import annotations

import math
from statistics import fmean
from time import perf_counter

import torch
from torch import Tensor

from src.interfaces import EnginePaths, BenchResult, ComponentSamples
from src.trt_runtime import EngineRunner


def _percentiles_ms(step_ms: list[float]) -> tuple[float, float]:
    """(p50, p95) over a list of per-call latencies in ms.

    **float64, matching `src.report._percentile_ms` exactly.** The raw sample is persisted
    (`ComponentSamples`) and `src.stats` computes the p50's confidence interval from it, so the
    stored point estimate and the later interval must come from the SAME percentile definition —
    otherwise an interval could bracket a number the table does not print (docs/architecture.md §9).
    float32 would differ in the last bits."""
    lat = torch.tensor(step_ms, dtype=torch.float64)
    return torch.quantile(lat, 0.50).item(), torch.quantile(lat, 0.95).item()


def _mean_ms(step_ms: list[float]) -> float:
    """Arithmetic mean over a list of per-call latencies in ms — the DECOMPOSITION basis
    (the mean latency table), never a reported headline. Means are what make
    `cycle = enc·calls + pred·calls + overhead` exact (linearity of expectation); percentiles
    do not compose that way. Reported latency stays p50/p95 (SPEC §Interface Contracts)."""
    return fmean(step_ms)


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
) -> tuple[BenchResult, ComponentSamples]:
    """Time both engines' isolated step loops → the summary `BenchResult` **and** the raw per-call
    samples it was reduced from (`ComponentSamples`), which `src.study` persists to
    `latencies.<track>.json`. Retaining the samples changes nothing about the measurement — same
    loops, same `n_iters`, same `warmup`, same inputs — it only stops throwing them away, so the
    component p50's confidence interval and independence test can be computed off-pod later
    (SPEC §Interface Contracts, docs/architecture.md §9)."""
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

    encode_ms = _time_loop(encoder, encode_inputs, n_iters)
    predict_ms = _time_loop(predictor, predict_inputs, n_iters)

    encode_p50, encode_p95 = _percentiles_ms(encode_ms)
    predict_p50, predict_p95 = _percentiles_ms(predict_ms)
    result = BenchResult(
        per_cycle_p50_ms=math.nan,  # joined by src.report from the gated eval-shim re-run
        per_cycle_p95_ms=math.nan,
        per_cycle_mean_ms=math.nan,
        encode_p50_ms=encode_p50,
        encode_p95_ms=encode_p95,
        encode_mean_ms=_mean_ms(encode_ms),
        predict_p50_ms=predict_p50,
        predict_p95_ms=predict_p95,
        predict_mean_ms=_mean_ms(predict_ms),
        success_rate=math.nan,  # joined in by src.report from the gated eval-shim re-run
    )
    return result, ComponentSamples(encode_ms=encode_ms, predict_ms=predict_ms)
