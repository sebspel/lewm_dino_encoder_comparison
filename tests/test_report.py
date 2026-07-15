"""Phase-5 headline runner plumbing — ratios, tables, plots, per-cycle join (CPU).

Synthetic bench dicts stand in for the real pod results. The per-component decomposition and
Amdahl dilution are derived in the report from the engine-step MEANS × CEM call counts minus the
measured mean per-cycle time (overhead by subtraction), so the fixtures carry per-cycle + step
latencies with positive overhead (a realistic full solve dominates the model step time).

Statistic split under test (SPEC §Interface Contracts): p50 = comparison basis, p95 = reported
tail, mean = decomposition basis only. The fixture defaults each mean to its p50 so the
arithmetic stays readable; `test_decompose_uses_mean_not_p50` breaks that tie deliberately to
pin which one `decompose` actually reads.
"""

import math
from pathlib import Path

from src import report


def _bench(
    cyc_p50, cyc_p95, enc_p50, pred_p50, sr, mem=100.0,
    cyc_mean=None, enc_mean=None, pred_mean=None,
):
    """A BenchResult with per-cycle latency already filled (as if joined). Means default to the
    corresponding p50 — pass them explicitly to distinguish the two bases."""
    return dict(
        per_cycle_p50_ms=cyc_p50,
        per_cycle_p95_ms=cyc_p95,
        per_cycle_mean_ms=cyc_p50 if cyc_mean is None else cyc_mean,
        encode_p50_ms=enc_p50,
        encode_p95_ms=enc_p50 * 1.1,
        encode_mean_ms=enc_p50 if enc_mean is None else enc_mean,
        predict_p50_ms=pred_p50,
        predict_p95_ms=pred_p50 * 1.1,
        predict_mean_ms=pred_p50 if pred_mean is None else pred_mean,
        peak_mem_mb=mem,
        success_rate=sr,
    )


def _synthetic():
    bench = {
        "lewm": {
            "fp32": _bench(100.0, 100.0, 1.0, 0.25, 90.0),
            "fp16": _bench(60.0, 60.0, 0.6, 0.15, 89.0),
        },
        "dino": {
            "fp32": _bench(1000.0, 2000.0, 10.0, 5.0, 88.0),
            "fp16": _bench(600.0, 1200.0, 6.0, 3.0, 87.0),
        },
    }
    return bench


def test_per_cycle_ratio():
    bench = _synthetic()
    assert report.per_cycle_ratio(bench, "fp32", "p95") == 20.0  # 2000 / 100
    assert report.per_cycle_ratio(bench, "fp32", "p50") == 10.0  # 1000 / 100


def test_per_cycle_ratio_nan_until_joined():
    bench = _synthetic()
    bench["lewm"]["fp32"]["per_cycle_p95_ms"] = math.nan
    assert math.isnan(report.per_cycle_ratio(bench, "fp32", "p95"))


def test_fp32_relative_speed_and_sr():
    bench = _synthetic()
    rel = report.fp32_relative(bench, "lewm")
    # p50 basis — agrees with the headline ratio rather than adding a second, tail-based speedup
    assert math.isclose(rel["fp16"]["per_cycle_p50_speedup_vs_fp32"], 100.0 / 60.0)
    assert math.isclose(rel["fp16"]["sr_delta_vs_fp32"], -1.0)


def test_fp32_relative_survives_single_track_render():
    """`report` iterates both tracks unconditionally, so a single-track render (separate pod
    sessions, SPEC §Headline-artifact durability) reaches a track with no rows. Must not KeyError."""
    bench = {"lewm": _synthetic()["lewm"]}
    assert report.fp32_relative(bench, "dino") == {}
    assert "lewm" in report.render_fp32_relative_table(bench)


def test_fp32_relative_table_shows_speed_and_sr_together(tmp_path):
    """SPEC §Parity: a precision that is faster but degrades task quality must be VISIBLE —
    which means the speedup and the SR delta land in the same row, not two tables."""
    bench = _synthetic()
    out = report.report(bench, tmp_path)
    text = Path(out["tables"]["fp32_relative_table"]).read_text()
    assert "ΔSR_vs_fp32" in text and "cyc_p50_speedup" in text
    fp16_row = [ln for ln in text.splitlines() if ln.split()[:2] == ["lewm", "fp16"]][0]
    assert "1.667" in fp16_row and "-1.0" in fp16_row


def test_decompose_overhead_by_subtraction():
    """overhead = cycle − enc·2 − pred·150; p = (enc+pred)/cycle."""
    bench = _synthetic()
    d = report.decompose(bench["lewm"]["fp32"])
    # enc_cyc = 1.0*2 = 2; pred_cyc = 0.25*150 = 37.5; model = 39.5; cycle = 100
    assert math.isclose(d["enc_cyc_ms"], 2.0)
    assert math.isclose(d["pred_cyc_ms"], 37.5)
    assert math.isclose(d["overhead_ms"], 60.5)  # 100 - 39.5
    assert math.isclose(d["optimizable_fraction"], 0.395)


def test_decompose_cycle_none_until_joined():
    bench = _synthetic()
    bench["lewm"]["fp32"]["per_cycle_mean_ms"] = math.nan
    d = report.decompose(bench["lewm"]["fp32"])
    assert d["overhead_ms"] is None and d["optimizable_fraction"] is None
    assert math.isclose(d["model_cyc_ms"], 39.5)  # enc/pred model shares still stand


def test_decompose_uses_mean_not_p50():
    """The decomposition basis is the MEAN, not p50: `cycle = enc·calls + pred·calls + overhead`
    is exact for means (linearity of expectation) and only approximate for percentiles. Means are
    set well away from the p50s here, so reading the wrong field fails loudly."""
    row = _bench(
        cyc_p50=100.0, cyc_p95=100.0, enc_p50=1.0, pred_p50=0.25, sr=90.0,
        cyc_mean=200.0, enc_mean=2.0, pred_mean=0.5,
    )
    d = report.decompose(row)
    assert math.isclose(d["enc_cyc_ms"], 4.0)  # 2.0 mean × 2 calls (p50 would give 2.0)
    assert math.isclose(d["pred_cyc_ms"], 75.0)  # 0.5 mean × 150 calls (p50 would give 37.5)
    assert math.isclose(d["overhead_ms"], 121.0)  # 200 mean cycle − 79 (p50 would give 60.5)
    assert math.isclose(d["optimizable_fraction"], 79.0 / 200.0)


def test_dilution_disclosure_model_only_and_realized():
    bench = _synthetic()
    d = report.dilution_disclosure(bench, "lewm")
    assert math.isclose(d["optimizable_fraction"], 0.395)
    assert d["amdahl_ceiling"] > 1.0
    fp16 = d["per_precision"]["fp16"]
    # model-only speedup = 39.5 / (0.6*2 + 0.15*150) = 39.5 / 23.7
    assert math.isclose(fp16["model_only_speedup"], 39.5 / 23.7)
    # measured realized = per-cycle MEAN ratio = 100 / 60 (mean-based so it reconciles against
    # predicted_realized, which is derived from the mean-based `p`)
    assert math.isclose(fp16["measured_realized_speedup"], 100.0 / 60.0)
    # predicted realized diluted below the model-only speedup by the overhead floor
    assert fp16["predicted_realized_speedup"] < fp16["model_only_speedup"]


def test_tables_render_and_report_emits_plots(tmp_path):
    bench = _synthetic()
    assert "lewm" in report.render_speed_table(bench)
    assert "pred_cyc_ms" in report.render_component_table(bench)
    out = report.report(bench, tmp_path)
    for path in out["plots"].values():
        assert Path(path).exists()
    assert out["ratios"]["fp32"]["per_cycle_p50_ratio"] == 10.0  # headline basis
    assert out["ratios"]["fp32"]["per_cycle_p95_ratio"] == 20.0  # tail, reported alongside


def test_speed_table_reports_all_three_distributions_at_p50_p95():
    """SPEC §Interface Contracts: three latency distributions, ALL at p50/p95. The encode/predict
    p95s were computed and persisted but rendered nowhere before."""
    text = report.render_speed_table(_synthetic())
    for col in ("cyc_p50", "cyc_p95", "enc_p50", "enc_p95", "pred_p50", "pred_p95"):
        assert col in text


def test_tables_persisted_to_disk(tmp_path):
    """Durability (SPEC §Headline-artifact durability): each table serialized to a .txt."""
    bench = _synthetic()
    out = report.report(bench, tmp_path)
    assert set(out["tables"]) == {
        "speed_table", "fp32_relative_table", "component_table", "dilution_table"
    }
    for path in out["tables"].values():
        assert Path(path).exists()
        assert Path(path).read_text().strip()
    assert (
        Path(out["tables"]["speed_table"]).read_text().rstrip("\n")
        == report.render_speed_table(bench)
    )


def test_sr_pending_flagged_and_join(tmp_path):
    bench = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan
    out = report.report(bench, tmp_path)
    assert "lewm-fp32" in out["sr_pending"]
    assert "PEND" in report.render_speed_table(bench)

    # a plain-number override fills SR (manual-override back-compat) -> no longer pending
    out2 = report.report(bench, tmp_path, sr_overrides={"lewm": {"fp32": 91.5}})
    assert "lewm-fp32" not in out2["sr_pending"]


def test_join_eval_fills_sr_and_equal_n_per_cycle(tmp_path):
    """The gated eval-shim join fills SR + per-cycle latency; per-cycle p50/p95 are taken after
    truncating each track to the common min-n across tracks (equal-n, SPEC §Interface Contracts)."""
    bench = {
        "lewm": {"fp32": _bench(math.nan, math.nan, 1.0, 0.25, math.nan)},
        "dino": {"fp32": _bench(math.nan, math.nan, 10.0, 5.0, math.nan)},
    }
    overrides = {
        "lewm": {"fp32": {"success_rate": 90.0, "per_cycle_latencies_ms": [10, 11, 12, 13]}},
        "dino": {"fp32": {"success_rate": 88.0, "per_cycle_latencies_ms": [100, 110, 120]}},
    }
    report.report(bench, tmp_path, sr_overrides=overrides)
    assert bench["lewm"]["fp32"]["success_rate"] == 90.0
    # equal-n: both truncated to min n=3 -> lewm uses [10,11,12], p50 = 11
    assert math.isclose(bench["lewm"]["fp32"]["per_cycle_p50_ms"], 11.0)
    assert not math.isnan(bench["dino"]["fp32"]["per_cycle_p95_ms"])
    # the decomposition mean comes off the SAME truncated sample as the reported percentiles
    assert math.isclose(bench["lewm"]["fp32"]["per_cycle_mean_ms"], 11.0)  # mean(10,11,12)
    assert math.isclose(bench["dino"]["fp32"]["per_cycle_mean_ms"], 110.0)


def test_equal_n_truncation_is_temporal_not_smallest(tmp_path):
    """Equal-n truncation keeps the first n in TEMPORAL order (a representative subset), NOT the
    n smallest — otherwise the upper tail is censored and p95 deflated (SPEC §Interface Contracts)."""
    bench = {
        "lewm": {"fp32": _bench(math.nan, math.nan, 1.0, 0.25, math.nan)},
        "dino": {"fp32": _bench(math.nan, math.nan, 10.0, 5.0, math.nan)},
    }
    overrides = {
        # lewm: a large latency arrives FIRST (temporal), then small ones; min-n across tracks = 3
        "lewm": {"fp32": {"success_rate": 90.0, "per_cycle_latencies_ms": [100.0, 1.0, 2.0, 3.0, 4.0]}},
        "dino": {"fp32": {"success_rate": 88.0, "per_cycle_latencies_ms": [10.0, 11.0, 12.0]}},
    }
    report.report(bench, tmp_path, sr_overrides=overrides)
    # temporal first-3 = [100,1,2] -> p95 near 100; sorted()[:3] = [1,2,3] would give ~3
    assert bench["lewm"]["fp32"]["per_cycle_p95_ms"] > 50.0


def test_nan_sr_is_skipped_not_crashed(tmp_path):
    bench = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan
    out = report.report(bench, tmp_path)  # must not raise
    assert Path(out["plots"]["speed_vs_sr"]).exists()


def test_single_track_render_skips_ratio_plot(tmp_path):
    bench = _synthetic()
    del bench["dino"]
    out = report.report(bench, tmp_path)
    assert out["ratios"] == {}
    assert "per_cycle_ratio" not in out["plots"]
    assert not (tmp_path / "per_cycle_ratio.png").exists()
    assert Path(out["tables"]["speed_table"]).exists()


def test_load_results_merges_per_track(tmp_path):
    """`load_results` merges whichever per-track `results.<track>.json` files exist back into
    the nested `bench[track][precision]` shape `report` consumes."""
    import json

    bench = _synthetic()
    for track in ("lewm", "dino"):
        (tmp_path / f"results.{track}.json").write_text(
            json.dumps({"meta": {"track": track}, "bench": bench[track]})
        )

    b = report.load_results(report._resolve_result_paths(tmp_path))
    assert set(b) == {"lewm", "dino"}
    assert report.per_cycle_ratio(b, "fp32", "p95") == 20.0


def test_negative_overhead_surfaced_not_clamped(capsys):
    """A cycle smaller than the enc+pred model time means the weighting/timing is off; the
    report must surface it loudly (SPEC §Interface Contracts), not clamp it to 0."""
    bench = {"lewm": {"fp32": _bench(10.0, 10.0, 1.0, 0.25, 90.0)}}  # cycle 10 < model 47
    d = report.decompose(bench["lewm"]["fp32"])
    assert d["overhead_ms"] < 0
    assert "negative overhead" in capsys.readouterr().out
