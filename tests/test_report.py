"""Phase-5 headline runner plumbing — ratios, tables, plots, NaN-SR handling (CPU).

Synthetic bench/profile dicts stand in for the real pod results; the numbers are chosen so
the ratios reproduce the paper-flavoured ~48× rollouts / ~45× p95 headline for a sanity read.
"""

import math
from pathlib import Path

import pytest

from src import report


def _bench(rollouts, p95, sr):
    return dict(
        latency_p50_ms=p95 * 0.8,
        latency_p95_ms=p95,
        rollouts_completed=rollouts,
        throughput=rollouts / 10.0,
        peak_mem_mb=100.0,
        success_rate=sr,
    )


def _synthetic():
    bench = {
        "lewm": {"fp32": _bench(480, 2.0, 90.0), "fp16": _bench(720, 1.3, 89.0)},
        "dino": {"fp32": _bench(10, 90.0, 88.0), "fp16": _bench(16, 60.0, 87.0)},
    }
    prof = {
        "lewm": {"fp32": {"encoder_ms": 0.5, "predictor_ms": 0.3, "planner_ms": 0.2}},
        "dino": {"fp32": {"encoder_ms": 8.0, "predictor_ms": 6.0, "planner_ms": 0.2}},
    }
    return bench, prof


def test_rollouts_and_p95_ratio():
    bench, _ = _synthetic()
    assert report.rollouts_ratio(bench, "fp32") == 48.0
    assert report.p95_ratio(bench, "fp32") == 45.0


def test_fp32_relative_speed_and_sr():
    bench, _ = _synthetic()
    rel = report.fp32_relative(bench, "lewm")
    assert math.isclose(rel["fp16"]["p95_speedup_vs_fp32"], 2.0 / 1.3)
    assert math.isclose(rel["fp16"]["sr_delta_vs_fp32"], -1.0)


def test_tables_render_and_report_emits_plots(tmp_path):
    bench, prof = _synthetic()
    assert "lewm" in report.render_speed_table(bench)
    assert "predictor_ms" in report.render_component_table(prof)
    out = report.report(bench, prof, tmp_path)
    for path in out["plots"].values():
        assert Path(path).exists()
    assert out["ratios"]["fp32"]["rollouts_ratio"] == 48.0


def test_nan_sr_is_skipped_not_crashed(tmp_path):
    bench, prof = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan  # unfilled SR (benchmark default)
    out = report.report(bench, prof, tmp_path)  # must not raise
    assert Path(out["plots"]["speed_vs_sr"]).exists()


def test_benchmark_missing_engine_raises(tmp_path):
    import torch

    from src.benchmark import benchmark

    engines = {"encoder": tmp_path / "e.plan", "predictor": tmp_path / "p.plan"}
    with pytest.raises(FileNotFoundError):
        benchmark(
            engines,
            (torch.randn(1),),
            (torch.randn(1), torch.randn(1)),
            time_budget_s=1.0,
            warmup=0,
        )
