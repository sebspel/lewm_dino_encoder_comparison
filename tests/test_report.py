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
    """Durability (SPEC §Headline-artifact durability): each table serialized to a .txt. The two
    data-dependent tables (calibration, isolation) are absent here — no sr.json overrides."""
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


def test_headline_tables_are_method_scoped_and_labelled(tmp_path):
    """SPEC §Parity / ADR-0002 3rd amendment: the method must survive into the PERSISTED artefact,
    and rendering the other method must not clobber the first. Both are load-bearing — the SR and
    the per-cycle sample it was measured on are method-sourced."""
    overrides = {"lewm": {"int8": {"max": {"success_rate": 76.0},
                                   "entropy": {"success_rate": 71.0}}}}

    def _b():
        return {"lewm": {"fp32": _bench(100.0, 100.0, 1.0, 0.25, 90.0),
                         "int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}

    out_max = report.report(_b(), tmp_path, sr_overrides=overrides, method="max")
    out_ent = report.report(_b(), tmp_path, sr_overrides=overrides, method="entropy")

    # distinct files -> the entropy render did not overwrite the max one
    assert Path(out_max["tables"]["speed_table"]).name == "speed_table.max.txt"
    assert Path(out_ent["tables"]["speed_table"]).name == "speed_table.entropy.txt"
    for out, method in ((out_max, "max"), (out_ent, "entropy")):
        for key in ("speed_table", "fp32_relative_table", "component_table", "dilution_table"):
            text = Path(out["tables"][key]).read_text()
            assert f"calibration_method = {method}" in text
    # both SRs still on disk, each under its own label
    assert "76.0" in Path(out_max["tables"]["speed_table"]).read_text()
    assert "71.0" in Path(out_ent["tables"]["speed_table"]).read_text()


def test_speed_table_reports_equal_n(tmp_path):
    """ADR-0003 amendment: the n each percentile was computed from is ON the artefact, so the
    equal-n truncation is verifiable rather than asserted. n is what SURVIVES both reductions —
    the warm-up drop then the common MINIMUM across tracks."""
    bench = {
        "lewm": {"fp32": _bench(math.nan, math.nan, 1.0, 0.25, math.nan)},
        "dino": {"fp32": _bench(math.nan, math.nan, 10.0, 5.0, math.nan)},
    }
    overrides = {
        "lewm": {"fp32": {"success_rate": 90.0, "per_cycle_latencies_ms": [10, 11, 12, 13, 14]}},
        "dino": {"fp32": {"success_rate": 88.0, "per_cycle_latencies_ms": [100, 110, 120]}},
    }
    report.report(bench, tmp_path, sr_overrides=overrides)
    # k=1 drop -> lewm 4, dino 2; equal-n min = 2 (not lewm's own 4)
    assert bench["lewm"]["fp32"]["_per_cycle_n"] == 2
    text = report.render_speed_table(bench)
    assert "cyc_n" in text
    assert [ln for ln in text.splitlines() if ln.split()[:2] == ["lewm", "fp32"]][0].split()[4] == "2"


def test_method_invariant_precisions_join_across_methods(tmp_path):
    """FP32/FP16 build data-free, so their SR cannot depend on a PTQ method — but `src.sr_eval`
    stamps every precision in a run with that run's label. An entropy render must still join an
    fp32 SR that only a `max` run recorded, or every FP32-relative ΔSR goes NaN from a label
    alone. Quantized precisions must NOT fall back."""
    overrides = {"lewm": {"fp32": {"max": {"success_rate": 90.0}},
                          "int8": {"max": {"success_rate": 76.0}}}}
    bench = {"lewm": {"fp32": _bench(100.0, 100.0, 1.0, 0.25, math.nan),
                      "int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}
    out = report.report(bench, tmp_path, sr_overrides=overrides, method="entropy")
    assert bench["lewm"]["fp32"]["success_rate"] == 90.0  # method-invariant -> falls back
    assert "lewm-int8" in out["sr_pending"]  # quantized -> no fallback, stays pending


def test_sr_pending_flagged_and_join(tmp_path):
    bench = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan
    out = report.report(bench, tmp_path)
    assert "lewm-fp32" in out["sr_pending"]
    assert "PEND" in report.render_speed_table(bench)

    # a plain-number override fills SR (manual-override back-compat) -> no longer pending
    out2 = report.report(bench, tmp_path, sr_overrides={"lewm": {"fp32": 91.5}})
    assert "lewm-fp32" not in out2["sr_pending"]


def test_method_labelled_sr_join_selects_and_coexists(tmp_path):
    """The gated eval-shim sr.json holds int8 SR under BOTH methods
    (`{track:{precision:{method:SR}}}`); `report(method=…)` selects one for a like-for-like render,
    and the other stays available (SPEC §Parity, ADR-0002). fp32 stays method-invariant."""
    overrides = {
        "lewm": {
            "fp32": {"max": {"success_rate": 90.0}},
            "int8": {"max": {"success_rate": 76.0}, "entropy": {"success_rate": 71.0}},
        }
    }

    b_max = {"lewm": {"fp32": _bench(100.0, 100.0, 1.0, 0.25, math.nan),
                      "int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}
    report.report(b_max, tmp_path, sr_overrides=overrides, method="max")
    assert b_max["lewm"]["int8"]["success_rate"] == 76.0
    assert b_max["lewm"]["fp32"]["success_rate"] == 90.0  # method-invariant fp32 joined

    # The SAME overrides re-rendered under entropy picks the entropy point — no rebuild, no clobber.
    b_ent = {"lewm": {"fp32": _bench(100.0, 100.0, 1.0, 0.25, math.nan),
                      "int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}
    report.report(b_ent, tmp_path, sr_overrides=overrides, method="entropy")
    assert b_ent["lewm"]["int8"]["success_rate"] == 71.0


def test_legacy_flat_sr_joins_only_under_max(tmp_path):
    """A legacy flat sr.json entry (pre-labelling) is `max`-calibrated: it joins for method=max and
    is absent (SR-PENDING) for method=entropy — so an entropy render never mislabels a max point."""
    overrides = {"lewm": {"int8": {"success_rate": 48.0}}}  # flat == max

    b = {"lewm": {"int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}
    report.report(b, tmp_path, sr_overrides=overrides, method="max")
    assert b["lewm"]["int8"]["success_rate"] == 48.0

    b2 = {"lewm": {"int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}
    out = report.report(b2, tmp_path, sr_overrides=overrides, method="entropy")
    assert "lewm-int8" in out["sr_pending"]  # no entropy point -> stays pending, not mislabelled


def test_join_eval_fills_sr_and_equal_n_per_cycle(tmp_path):
    """The gated eval-shim join fills SR + per-cycle latency; per-cycle p50/p95 are taken after
    dropping the warm-up head and truncating each track to the common min-n across tracks
    (equal-n, SPEC §Interface Contracts)."""
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
    # warm-up drop k=1 -> lewm [11,12,13], dino [110,120]; equal-n min = 2 -> lewm [11,12]
    assert math.isclose(bench["lewm"]["fp32"]["per_cycle_p50_ms"], 11.5)
    assert not math.isnan(bench["dino"]["fp32"]["per_cycle_p95_ms"])
    # the decomposition mean comes off the SAME reduced sample as the reported percentiles
    assert math.isclose(bench["lewm"]["fp32"]["per_cycle_mean_ms"], 11.5)
    assert math.isclose(bench["dino"]["fp32"]["per_cycle_mean_ms"], 115.0)


def test_equal_n_truncation_is_temporal_not_smallest(tmp_path):
    """Equal-n truncation keeps the first n in TEMPORAL order (a representative subset), NOT the
    n smallest — otherwise the upper tail is censored and p95 deflated (SPEC §Interface Contracts).
    The large sample sits at index 1 so it survives the k=1 warm-up drop: this pins the truncation
    rule, not the warm-up rule."""
    bench = {
        "lewm": {"fp32": _bench(math.nan, math.nan, 1.0, 0.25, math.nan)},
        "dino": {"fp32": _bench(math.nan, math.nan, 10.0, 5.0, math.nan)},
    }
    overrides = {
        "lewm": {"fp32": {"success_rate": 90.0,
                          "per_cycle_latencies_ms": [5.0, 100.0, 1.0, 2.0, 3.0, 4.0]}},
        "dino": {"fp32": {"success_rate": 88.0,
                          "per_cycle_latencies_ms": [9.0, 10.0, 11.0, 12.0]}},
    }
    report.report(bench, tmp_path, sr_overrides=overrides)
    # post-drop lewm [100,1,2,3,4], dino [10,11,12]; n=3 -> temporal first-3 = [100,1,2] -> p95 ~100
    # (sorted()[:3] = [1,2,3] would give ~3)
    assert bench["lewm"]["fp32"]["per_cycle_p95_ms"] > 50.0


def test_per_cycle_warmup_drops_cold_decision_and_discloses_it(tmp_path):
    """ADR-0003 amendment: the cold first decision is dropped BEFORE truncation, so it cannot bias
    the mean the decomposition subtracts from — and the exclusion is disclosed (`drop×`), not
    hidden. `warmup_drop=0` reproduces the old, biased view."""
    def _b():
        return {"lewm": {"fp32": _bench(math.nan, math.nan, 1.0, 0.25, math.nan)}}

    # a 10x cold first decision, then a steady 10ms
    lat = [100.0] + [10.0] * 9
    overrides = {"lewm": {"fp32": {"success_rate": 90.0, "per_cycle_latencies_ms": lat}}}

    dropped = _b()
    report.report(dropped, tmp_path / "drop", sr_overrides=overrides)
    row = dropped["lewm"]["fp32"]
    assert math.isclose(row["per_cycle_mean_ms"], 10.0)  # cold sample gone from the mean
    assert row["_per_cycle_n"] == 9
    assert row["_per_cycle_dropped_ms"] == [100.0]  # stashed, not discarded

    kept = _b()
    report.report(kept, tmp_path / "keep", sr_overrides=overrides, warmup_drop=0)
    assert math.isclose(kept["lewm"]["fp32"]["per_cycle_mean_ms"], 19.0)  # 10x sample inflates it
    assert kept["lewm"]["fp32"]["_per_cycle_n"] == 10

    # the exclusion is ON the artefact: drop× = 100 / retained p50 (10) = 10.00
    text = report.render_speed_table(dropped)
    assert "drop×" in text
    assert [ln for ln in text.splitlines() if ln.split()[:2] == ["lewm", "fp32"]][0].split()[5] == "10.00"


def test_warmup_drop_does_not_move_the_p50_headline(tmp_path):
    """The drop exists for the MEAN-based decomposition. p50 is robust to one cold sample in n,
    so the headline ratio is unmoved either way — which is why this is a defensible correction
    rather than a result-changing one (ADR-0003 amendment)."""
    lat = {"lewm": [900.0] + [10.0 + (i % 5) for i in range(60)],
           "dino": [9000.0] + [100.0 + (i % 5) for i in range(60)]}
    overrides = {
        t: {"fp32": {"success_rate": 90.0, "per_cycle_latencies_ms": v}} for t, v in lat.items()
    }

    def _ratio(k):
        bench = {
            "lewm": {"fp32": _bench(math.nan, math.nan, 1.0, 0.25, math.nan)},
            "dino": {"fp32": _bench(math.nan, math.nan, 10.0, 5.0, math.nan)},
        }
        report.report(bench, tmp_path / f"k{k}", sr_overrides=overrides, warmup_drop=k)
        return report.per_cycle_ratio(bench, "fp32", "p50")

    assert math.isclose(_ratio(0), _ratio(1))


def test_report_never_rewrites_canonical_results(tmp_path):
    """`src.report` renders VIEWS. The canonical artefacts — `results.<track>.json` (src.study) and
    `sr.json` (src.sr_eval) — are read-only to it and must survive a render byte-for-byte, even when
    out_dir IS the directory holding them (CLAUDE §8, SPEC §Headline-artifact durability)."""
    import json

    bench = _synthetic()
    canonical = {}
    for track in ("lewm", "dino"):
        p = tmp_path / f"results.{track}.json"
        p.write_text(json.dumps({"meta": {"track": track}, "bench": bench[track]}))
        canonical[p] = p.read_bytes()
    sr_path = tmp_path / "sr.json"
    sr_path.write_text(json.dumps({"lewm": {"int8": {"max": {"success_rate": 76.0}}}}))
    canonical[sr_path] = sr_path.read_bytes()

    loaded = report.load_results(report._resolve_result_paths(tmp_path))
    report.report(loaded, tmp_path, sr_overrides=json.loads(sr_path.read_text()), method="max")

    for path, before in canonical.items():
        assert path.read_bytes() == before, f"{path.name} was rewritten by a render"


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


def test_calibration_table_shows_both_methods(tmp_path):
    """ADR-0002 3rd amendment: the ONE table spanning methods. int8/fp8 only (fp32/fp16 are
    method-invariant), PEND where a method was never built, and a `headline` column naming which
    method the single-method tables were rendered at."""
    overrides = {
        "lewm": {
            "fp32": {"max": {"success_rate": 98.0}},  # method-invariant -> excluded
            "int8": {"max": {"success_rate": 76.0}},  # entropy not yet built -> PEND
        },
        "dino": {"int8": {"max": {"success_rate": 20.0}, "entropy": {"success_rate": 16.0}}},
    }
    text = report.render_calibration_table(overrides, headline_method="entropy")
    assert "SR@max" in text and "SR@entropy" in text
    # method-invariant precisions get no ROW here (a max-vs-entropy comparison of them is vacuous)
    assert not [ln for ln in text.splitlines() if ln.split()[:2] == ["lewm", "fp32"]]
    lewm = [ln for ln in text.splitlines() if ln.split()[:2] == ["lewm", "int8"]][0]
    assert "76.0" in lewm and "PEND" in lewm
    dino = [ln for ln in text.splitlines() if ln.split()[:2] == ["dino", "int8"]][0]
    assert "20.0" in dino and "16.0" in dino and "-4.0" in dino  # Δ(entropy − max)
    assert dino.split()[-1] == "entropy"  # headline marker

    bench = {"lewm": {"int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}
    out = report.report(bench, tmp_path, sr_overrides=overrides, method="max")
    assert Path(out["tables"]["calibration_table"]).name == "calibration_table.txt"


def test_calibration_table_absent_without_quantized_sr(tmp_path):
    """No quantized SR -> no artefact, rather than an empty one."""
    assert report.render_calibration_table(None) == ""
    out = report.report(_synthetic(), tmp_path)
    assert "calibration_table" not in out["tables"]


def test_isolation_table_attributes_component(tmp_path):
    """ADR-0005: the mixed-precision diagnostic attributes a measured SR drop to encoder or
    predictor. ΔSR is quoted vs the track's FP16 row (the held component's precision), NOT FP32."""
    bench = {
        "dino": {
            "fp16": _bench(600.0, 1200.0, 6.0, 3.0, 70.0),
            "int8": _bench(400.0, 800.0, 4.0, 2.0, math.nan),
        }
    }
    overrides = {
        "dino": {
            "enc-int8+pred-fp16": {"entropy": {"success_rate": 16.0}},
            "enc-fp16+pred-int8": {"entropy": {"success_rate": 42.0}},
        }
    }
    text = report.render_isolation_table(bench, overrides, method="entropy")
    enc = [ln for ln in text.splitlines() if ln.split()[:3] == ["dino", "int8", "encoder"]][0]
    pred = [ln for ln in text.splitlines() if ln.split()[:3] == ["dino", "int8", "predictor"]][0]
    assert "16.0" in enc and "-54.0" in enc  # 16 − 70, vs FP16 not FP32
    assert "42.0" in pred and "-28.0" in pred
    # cyc_share = that component's per-cycle time × calls ÷ the joined cycle, at that precision
    assert enc.split()[-1] == format(4.0 * 2 / 400.0, ".3f")
    assert pred.split()[-1] == format(2.0 * 150 / 400.0, ".3f")


def test_isolation_keys_never_reach_the_headline(tmp_path):
    """The composite keys are diagnostics: they must leave every headline artefact byte-identical
    (ADR-0005 — a mixed pairing is never a fifth precision)."""
    overrides = {"dino": {"fp16": {"entropy": {"success_rate": 70.0}}}}
    with_iso = {
        "dino": {**overrides["dino"], "enc-fp16+pred-int8": {"entropy": {"success_rate": 42.0}}}
    }

    def _render(ov, out_dir):
        bench = {"dino": {"fp16": _bench(600.0, 1200.0, 6.0, 3.0, math.nan),
                          "int8": _bench(400.0, 800.0, 4.0, 2.0, math.nan)}}
        out = report.report(bench, out_dir, sr_overrides=ov, method="entropy")
        return {k: Path(v).read_text() for k, v in out["tables"].items()}

    plain = _render(overrides, tmp_path / "a")
    isolated = _render(with_iso, tmp_path / "b")
    assert "isolation_table" in isolated and "isolation_table" not in plain
    for key in plain:
        assert plain[key] == isolated[key], f"{key} changed when isolation keys were present"


def test_negative_overhead_surfaced_not_clamped(capsys):
    """A cycle smaller than the enc+pred model time means the weighting/timing is off; the
    report must surface it loudly (SPEC §Interface Contracts), not clamp it to 0."""
    bench = {"lewm": {"fp32": _bench(10.0, 10.0, 1.0, 0.25, 90.0)}}  # cycle 10 < model 47
    d = report.decompose(bench["lewm"]["fp32"])
    assert d["overhead_ms"] < 0
    assert "negative overhead" in capsys.readouterr().out
