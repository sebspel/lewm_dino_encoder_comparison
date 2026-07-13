"""Latency benchmark plumbing — the parts checkable without engines/CUDA (CPU).

The real timing needs the TensorRT engines on the L40S; here we cover the guard (missing
engine fails loud) and the pure percentile helper.
"""

import math

import pytest
import torch

from src import benchmark


def test_percentiles_ms_orders_p50_p95():
    p50, p95 = benchmark._percentiles_ms([1.0, 2.0, 3.0, 4.0, 100.0])
    assert p95 >= p50
    assert math.isclose(p50, 3.0)


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
