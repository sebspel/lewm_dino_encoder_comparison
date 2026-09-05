"""Latency benchmark plumbing — the parts checkable without engines/CUDA (CPU).

The real timing needs the TensorRT engines on the L40S; here we cover the guard (missing
engine fails loud) and the pure percentile helper.
"""

import math
import pathlib

import pytest
import torch

from src import benchmark


def test_percentiles_ms_orders_p50_p95():
    p50, p95 = benchmark._percentiles_ms([1.0, 2.0, 3.0, 4.0, 100.0])
    assert p95 >= p50
    assert math.isclose(p50, 3.0)


def test_stored_p50_matches_the_percentile_the_interval_is_built_on():
    """ANTI-DRIFT (architecture.md §9): the p50 stored in `results.<track>.json` and the p50 that
    `src.stats` recomputes from the persisted sample must be the SAME number — otherwise the
    rendered interval brackets a value the table does not print. One shared percentile definition,
    float64 on both sides; a float32 reduction here would differ in the last bits."""
    from src import report

    sample = [0.1234567, 5.4321, 2.71828, 3.14159, 1.41421, 9.87654, 0.57721]
    p50, _ = benchmark._percentiles_ms(sample)
    assert p50 == report._percentile_ms(sample, 0.50)


def test_benchmark_returns_the_raw_samples_beside_the_summary(monkeypatch):
    """`benchmark` hands back the loops' raw per-call latencies (`ComponentSamples`) as well as the
    summary — one vector per component, `n_iters` long, with the untimed warm-up iters EXCLUDED, so
    the recorded vector is already the sample `src.stats` runs on (no truncation, no report-time
    drop). Engine-free: the two runners are stubbed, since real timing is pod-only."""

    class _FakeRunner:
        device = torch.device("cpu")

        def __init__(self, path):
            self.calls = 0

        def run(self, inputs):
            self.calls += 1

    monkeypatch.setattr(benchmark, "EngineRunner", _FakeRunner)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (1_000, 2_000))

    engines = {"encoder": __file__, "predictor": __file__}  # exist; content unused by the stub
    result, samples = benchmark.benchmark(
        {k: pathlib.Path(v) for k, v in engines.items()},
        (torch.zeros(1),),
        (torch.zeros(1),),
        n_iters=7,
        warmup=3,
    )

    assert set(samples) == {"encode_ms", "predict_ms"}
    assert all(len(v) == 7 for v in samples.values())  # timed iters only, warm-up excluded
    # The summary is a reduction OF that sample, not a second measurement.
    assert result["encode_p50_ms"] == benchmark._percentiles_ms(samples["encode_ms"])[0]


def test_benchmark_missing_engine_raises(tmp_path):
    engines = {"encoder": tmp_path / "e.plan", "predictor": tmp_path / "p.plan"}
    with pytest.raises(FileNotFoundError):
        benchmark.benchmark(
            engines,
            (torch.randn(1),),
            (torch.randn(1), torch.randn(1)),
            n_iters=1,
            warmup=0,
        )
