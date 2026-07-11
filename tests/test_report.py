"""Phase-5 headline runner plumbing — ratios, tables, plots, NaN-SR handling (CPU).

Synthetic bench/profile dicts stand in for the real pod results; the numbers are chosen so
the ratios reproduce the paper-flavoured ~48× rollouts / ~45× p95 headline for a sanity read.
"""

import math
from pathlib import Path

import pytest

from src import report


from src.profile import (
    _ENCODER_CALLS_PER_CYCLE,
    _PREDICTOR_CALLS_PER_CYCLE,
    _PLANNER_CALLS_PER_CYCLE,
)


def _bench(rollouts, p95, sr):
    return dict(
        latency_p50_ms=p95 * 0.8,
        latency_p95_ms=p95,
        rollouts_completed=rollouts,
        throughput=rollouts / 10.0,
        peak_mem_mb=100.0,
        success_rate=sr,
    )


def _prof(encoder_ms, predictor_ms, planner_ms):
    """Build a full ComponentProfile the way `src.profile.profile` does (runtime-weighted
    cycle shares + derived p/ceiling), so the report fixtures match the real shape."""
    enc_c = _ENCODER_CALLS_PER_CYCLE * encoder_ms
    pred_c = _PREDICTOR_CALLS_PER_CYCLE * predictor_ms
    plan_c = _PLANNER_CALLS_PER_CYCLE * planner_ms
    cycle = enc_c + pred_c + plan_c
    frac = (enc_c + pred_c) / cycle
    return dict(
        encoder_ms=encoder_ms, predictor_ms=predictor_ms, planner_ms=planner_ms,
        encoder_calls=_ENCODER_CALLS_PER_CYCLE,
        predictor_calls=_PREDICTOR_CALLS_PER_CYCLE,
        planner_calls=_PLANNER_CALLS_PER_CYCLE,
        encoder_cycle_ms=enc_c, predictor_cycle_ms=pred_c, planner_cycle_ms=plan_c,
        optimizable_fraction=frac, amdahl_ceiling=1.0 / (1.0 - frac),
    )


def _synthetic():
    bench = {
        "lewm": {"fp32": _bench(480, 2.0, 90.0), "fp16": _bench(720, 1.3, 89.0)},
        "dino": {"fp32": _bench(10, 90.0, 88.0), "fp16": _bench(16, 60.0, 87.0)},
    }
    prof = {
        "lewm": {"fp32": _prof(0.5, 0.3, 0.2)},
        "dino": {"fp32": _prof(8.0, 6.0, 0.2)},
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
    # component table now shows the runtime-WEIGHTED per-cycle shares (issue 4)
    assert "pred_cyc_ms" in report.render_component_table(prof)
    out = report.report(bench, prof, tmp_path)
    for path in out["plots"].values():
        assert Path(path).exists()
    assert out["ratios"]["fp32"]["rollouts_ratio"] == 48.0


def test_tables_persisted_to_disk(tmp_path):
    """Durability (SPEC §Headline-artifact durability): each table serialized to a .txt on
    disk, not stdout/W&B-HTML only, so a completed study survives pod teardown."""
    bench, prof = _synthetic()
    out = report.report(bench, prof, tmp_path)
    assert set(out["tables"]) == {"speed_table", "component_table", "dilution_table"}
    for path in out["tables"].values():
        assert Path(path).exists()
        assert Path(path).read_text().strip()  # non-empty
    # the serialized table matches what was rendered
    assert (
        Path(out["tables"]["speed_table"]).read_text().rstrip("\n")
        == report.render_speed_table(bench)
    )


def test_dilution_disclosure_model_only_and_predicted(tmp_path):
    """Issue 2/4: model-only speedup + Amdahl-predicted realized; measured realized is gated."""
    bench, prof = _synthetic()
    d = report.dilution_disclosure(bench, prof, "lewm")
    assert 0.0 < d["optimizable_fraction"] <= 1.0
    assert d["amdahl_ceiling"] > 1.0
    # fp16 model-only speedup = throughput ratio = 72/48 = 1.5
    fp16 = d["per_precision"]["fp16"]
    assert math.isclose(fp16["model_only_speedup"], 1.5)
    # predicted realized is diluted below the model-only speedup by the planner floor
    assert fp16["predicted_realized_speedup"] < fp16["model_only_speedup"]
    assert fp16["measured_realized_speedup"] is None  # gated eval-shim


def test_sr_pending_flagged_and_join(tmp_path):
    """Issue 5: unpaired SR must be surfaced (PEND), and injectable via sr_overrides."""
    bench, prof = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan
    out = report.report(bench, prof, tmp_path)
    assert "lewm-fp32" in out["sr_pending"]
    assert "PEND" in report.render_speed_table(bench)

    # the gated eval-shim join fills it -> no longer pending
    out2 = report.report(
        bench, prof, tmp_path, sr_overrides={"lewm": {"fp32": 91.5}}
    )
    assert "lewm-fp32" not in out2["sr_pending"]


def test_nan_sr_is_skipped_not_crashed(tmp_path):
    bench, prof = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan  # unfilled SR (benchmark default)
    out = report.report(bench, prof, tmp_path)  # must not raise
    assert Path(out["plots"]["speed_vs_sr"]).exists()


def test_single_track_render_skips_ratio_plots(tmp_path):
    """A single-track render must NOT emit the two cross-track ratio plots (they'd be empty),
    but still renders + persists the single-track tables (SPEC §Headline-artifact durability)."""
    bench, prof = _synthetic()
    del bench["dino"], prof["dino"]
    out = report.report(bench, prof, tmp_path)
    assert out["ratios"] == {}
    assert "rollouts_ratio" not in out["plots"] and "p95_ratio" not in out["plots"]
    assert not (tmp_path / "rollouts_ratio.png").exists()
    assert Path(out["tables"]["speed_table"]).exists()


def test_load_results_merges_per_track(tmp_path):
    """`load_results` merges whichever per-track `results.<track>.json` files exist back into
    the nested shape `report` consumes; NaN SRs round-trip via the json `NaN` token."""
    import json

    bench, prof = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan  # unfilled SR must survive the round-trip
    for track in ("lewm", "dino"):
        (tmp_path / f"results.{track}.json").write_text(
            json.dumps({"meta": {"track": track}, "bench": bench[track], "prof": prof[track]})
        )

    b, p = report.load_results(report._resolve_result_paths(tmp_path))
    assert set(b) == {"lewm", "dino"} and set(p) == {"lewm", "dino"}
    assert report.rollouts_ratio(b, "fp32") == 48.0  # cross-track headline reconstructs
    assert math.isnan(b["lewm"]["fp32"]["success_rate"])


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
