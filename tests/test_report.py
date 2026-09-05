"""Phase-5 headline runner plumbing — ratios, tables, plots, per-cycle join (CPU).

Synthetic bench dicts stand in for the real pod results. The per-component decomposition is
derived from the engine-step MEANS × CEM call counts minus the measured mean per-cycle time
(residual overhead by subtraction), so the fixtures carry per-cycle + step latencies with a
positive residual (a realistic full solve dominates the model step time).

Statistic split under test (SPEC §Interface Contracts): p50 = comparison basis, p95 = reported
tail, mean = decomposition basis only. The fixture defaults each mean to its p50 so the
arithmetic stays readable.
"""

import math
from pathlib import Path

import pytest

from src import report
from src.interfaces import ENCODER_CALLS_PER_CYCLE, PREDICTOR_CALLS_PER_CYCLE


def _bench(
    cyc_p50, cyc_p95, enc_p50, pred_p50, sr,
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


def test_tables_render_and_report_emits_plots(tmp_path):
    bench = _synthetic()
    assert "lewm" in report.render_speed_table(bench)
    out = report.report(bench, tmp_path)
    for path in out["plots"].values():
        assert Path(path).exists()


def test_speed_table_reports_the_per_cycle_distribution_only():
    """The reported speed surface is the per-cycle distribution at p50/p95 plus SR. The engine
    steps are reported as MEANS on the latency-means table, never as percentiles here, and peak
    memory is not a reported quantity at all."""
    text = report.render_speed_table(_synthetic())
    for col in ("cyc_p50", "cyc_p95", "cyc_n", "SR"):
        assert col in text
    for col in ("enc_p50", "enc_p95", "pred_p50", "pred_p95", "mem_MB"):
        assert col not in text


def test_tables_persisted_to_disk(tmp_path):
    """Durability (SPEC §Headline-artifact durability): each table serialized to a .txt. The two
    data-dependent tables (isolation, latency means) are absent here — no sr.json overrides."""
    bench = _synthetic()
    out = report.report(bench, tmp_path)
    assert set(out["tables"]) == {"speed_table"}
    for path in out["tables"].values():
        assert Path(path).exists()
        assert Path(path).read_text().strip()
    assert (
        Path(out["tables"]["speed_table"]).read_text().rstrip("\n")
        == report.render_speed_table(bench)
    )


def test_headline_tables_are_method_scoped_and_labelled(tmp_path):
    """SPEC §Parity / architecture.md §6: the method must survive into the PERSISTED artefact,
    and rendering the other method must not clobber the first. Both are load-bearing — the SR and
    the per-cycle sample it was measured on are method-sourced."""
    overrides = {"lewm": {"int8": {"max": {"success_rate": 76.0},
                                   "entropy": {"success_rate": 72.0}}}}

    def _b():
        return {"lewm": {"fp32": _bench(100.0, 100.0, 1.0, 0.25, 90.0),
                         "int8": _bench(40.0, 40.0, 0.4, 0.1, math.nan)}}

    out_max = report.report(_b(), tmp_path, sr_overrides=overrides, method="max")
    out_ent = report.report(_b(), tmp_path, sr_overrides=overrides, method="entropy")

    # distinct files -> the entropy render did not overwrite the max one
    assert Path(out_max["tables"]["speed_table"]).name == "speed_table.max.txt"
    assert Path(out_ent["tables"]["speed_table"]).name == "speed_table.entropy.txt"
    for out, method in ((out_max, "max"), (out_ent, "entropy")):
        text = Path(out["tables"]["speed_table"]).read_text()
        assert f"calibration_method = {method}" in text
    # both SRs still on disk, each under its own label
    assert "76.0" in Path(out_max["tables"]["speed_table"]).read_text()
    assert "72.0" in Path(out_ent["tables"]["speed_table"]).read_text()


def test_speed_table_reports_equal_n(tmp_path):
    """architecture.md §7: the n each percentile was computed from is ON the artefact, so the
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
    assert [ln for ln in text.splitlines() if ln.split()[:2] == ["lewm", "fp32"]][0].split()[6] == "2"


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
    out2 = report.report(bench, tmp_path, sr_overrides={"lewm": {"fp32": 92.0}})
    assert "lewm-fp32" not in out2["sr_pending"]


def test_method_labelled_sr_join_selects_and_coexists(tmp_path):
    """The gated eval-shim sr.json holds int8 SR under BOTH methods
    (`{track:{precision:{method:SR}}}`); `report(method=…)` selects one for a like-for-like render,
    and the other stays available (SPEC §Parity, architecture.md §6). fp32 stays method-invariant."""
    overrides = {
        "lewm": {
            "fp32": {"max": {"success_rate": 90.0}},
            "int8": {"max": {"success_rate": 76.0}, "entropy": {"success_rate": 72.0}},
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
    assert b_ent["lewm"]["int8"]["success_rate"] == 72.0


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
    """architecture.md §7: the cold first decision is dropped BEFORE truncation, so it cannot bias
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
    assert [ln for ln in text.splitlines() if ln.split()[:2] == ["lewm", "fp32"]][0].split()[7] == "10.00"


def test_warmup_drop_does_not_move_the_p50_headline(tmp_path):
    """The drop exists for the MEAN-based decomposition. p50 is robust to one cold sample in n,
    so the headline ratio is unmoved either way — which is why this is a defensible correction
    rather than a result-changing one (architecture.md §7)."""
    lat = {"lewm": [900.0] + [10.0 + (i % 5) for i in range(60)],
           "dino": [9000.0] + [100.0 + (i % 5) for i in range(60)]}
    overrides = {
        t: {"fp32": {"success_rate": 90.0, "per_cycle_latencies_ms": v}} for t, v in lat.items()
    }

    def _p50s(k):
        bench = {
            "lewm": {"fp32": _bench(math.nan, math.nan, 1.0, 0.25, math.nan)},
            "dino": {"fp32": _bench(math.nan, math.nan, 10.0, 5.0, math.nan)},
        }
        report.report(bench, tmp_path / f"k{k}", sr_overrides=overrides, warmup_drop=k)
        return {t: bench[t]["fp32"]["per_cycle_p50_ms"] for t in ("lewm", "dino")}

    dropped, undropped = _p50s(1), _p50s(0)
    for track in ("lewm", "dino"):
        assert math.isclose(dropped[track], undropped[track])


def test_report_never_rewrites_canonical_results(tmp_path):
    """`src.report` renders VIEWS. The canonical artefacts — `results.<track>.json` +
    `latencies.<track>.json` (src.study) and `sr.json` (src.sr_eval) — are read-only to it and must
    survive a render byte-for-byte, even when out_dir IS the directory holding them (CLAUDE §8,
    SPEC §Headline-artifact durability)."""
    import json

    bench = _synthetic()
    canonical = {}
    for track in ("lewm", "dino"):
        p = tmp_path / f"results.{track}.json"
        p.write_text(json.dumps({"meta": {"track": track}, "bench": bench[track]}))
        canonical[p] = p.read_bytes()
        # The engine-step samples are the ONLY copy of that measurement — a render that rewrote
        # them would put an L40S run at risk.
        lat = tmp_path / f"latencies.{track}.json"
        lat.write_text(
            json.dumps({"meta": {"track": track}, "latencies": _component_samples(3)})
        )
        canonical[lat] = lat.read_bytes()
    sr_path = tmp_path / "sr.json"
    sr_path.write_text(json.dumps({"lewm": {"int8": {"max": {"success_rate": 76.0}}}}))
    canonical[sr_path] = sr_path.read_bytes()

    loaded = report.load_results(report._resolve_result_paths(tmp_path))
    report.report(
        loaded,
        tmp_path,
        sr_overrides=json.loads(sr_path.read_text()),
        method="max",
        component_latencies={"lewm": _component_samples(3)},
    )

    for path, before in canonical.items():
        assert path.read_bytes() == before, f"{path.name} was rewritten by a render"


# --- component p50 intervals on the speed table (Phase 9) --------------------------------
def _component_samples(seed: int, n: int = 40, method: str = "max") -> dict:
    """One track's stored engine-step samples, in `latencies.<track>.json`'s `latencies` shape —
    keyed by (precision, calibration method), since the quantized engines are per-method builds."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return {
        p: {
            method: {
                "encode_ms": list(rng.normal(1.0, 0.05, size=n)),
                "predict_ms": list(rng.normal(0.25, 0.01, size=n)),
            }
        }
        for p in ("fp32", "fp16")
    }


def test_speed_table_stays_parseable(tmp_path):
    """Every cell must remain ONE whitespace-delimited token — the artefacts are read with
    `split()`, so a stray space in an interval would shift every column after it."""
    bench = _synthetic()
    overrides = {
        t: {"fp32": {"success_rate": 90.0, "per_cycle_latencies_ms": [10.0 + i for i in range(20)]}}
        for t in ("lewm", "dino")
    }
    out = report.report(bench, tmp_path, sr_overrides=overrides)
    text = Path(out["tables"]["speed_table"]).read_text()

    header = [ln for ln in text.splitlines() if ln.split()[:1] == ["track"]][0].split()
    assert header == [
        "track", "prec", "cyc_p50", "cyc_p50_CI95", "ac", "cyc_p95", "cyc_n", "drop×",
        "SR", "SR_CI95",
    ]

    rows = [ln.split() for ln in text.splitlines() if ln.split()[:1] in (["lewm"], ["dino"])]
    assert {len(r) for r in rows} == {len(header)}  # fixed token count, every row
    lewm_fp32 = [r for r in rows if r[:2] == ["lewm", "fp32"]][0]
    assert lewm_fp32[3].startswith("[") and "," in lewm_fp32[3]  # p50 interval, unspaced
    assert lewm_fp32[4] in {"*", "-"}  # independence flag, never empty
    # fp16 has no stored per-cycle sample here -> blank cells, not a crash
    assert [r for r in rows if r[:2] == ["lewm", "fp16"]][0][3] == "—"


def test_component_samples_do_not_touch_the_speed_surface(tmp_path):
    """The component samples feed the MEAN latency table and nothing else: the speed table and the
    speed-vs-SR figure are per-cycle/SR surfaces, so rendering with and without the stored engine
    steps must leave them byte-identical."""
    without = report.report(_synthetic(), tmp_path / "a")
    with_ = report.report(
        _synthetic(), tmp_path / "b", component_latencies={"lewm": _component_samples(22)}
    )

    assert (
        Path(without["tables"]["speed_table"]).read_bytes()
        == Path(with_["tables"]["speed_table"]).read_bytes()
    )
    assert (
        Path(without["plots"]["speed_vs_sr"]).read_bytes()
        == Path(with_["plots"]["speed_vs_sr"]).read_bytes()
    )


# --- mean latencies + overhead, with bootstrap intervals ---------------------------------
def _mean_fixture(seed=31):
    """Both surfaces the mean decomposition needs, wired as the pod writes them: the engine-step
    samples (`latencies.*.json`) and a bench whose `*_mean_ms` are the means OF THOSE SAMPLES — the
    same loop produces both on a real run — plus per-cycle vectors under two method labels."""
    import numpy as np
    from statistics import fmean

    rng = np.random.default_rng(seed)
    samples = {t: _component_samples(seed + i) for i, t in enumerate(("lewm", "dino"))}
    bench = _synthetic()
    for track, by_prec in samples.items():
        for prec, by_method in by_prec.items():
            vectors = by_method["max"]
            bench[track][prec]["encode_mean_ms"] = fmean(vectors["encode_ms"])
            bench[track][prec]["predict_mean_ms"] = fmean(vectors["predict_ms"])
        # int8 exercises the QUANTIZED case: a component sample PER METHOD (each method is its own
        # engine build) beside two per-cycle samples, hence two rows — `INT8 (max)`/`INT8 (entropy)`.
        by_prec["int8"] = {
            m: _component_samples(seed + 7 + i, method=m)["fp32"][m]
            for i, m in enumerate(("max", "entropy"))
        }
    overrides = {
        track: {
            prec: {
                m: {
                    "success_rate": 90.0,
                    "per_cycle_latencies_ms": list(rng.normal(120.0, 4.0, size=40)),
                }
                for m in (("max", "entropy") if prec == "int8" else ("max",))
            }
            for prec in ("fp32", "fp16", "int8")
        }
        for track in ("lewm", "dino")
    }
    return bench, overrides, samples


def _mean_render(out_dir, method="max", seed=31):
    bench, overrides, samples = _mean_fixture(seed)
    out = report.report(
        bench, out_dir, sr_overrides=overrides, method=method, component_latencies=samples
    )
    return out, bench


def test_latency_means_decompose_the_measured_cycle(tmp_path):
    """The anti-drift guard on the decomposition itself: the two per-call components are the
    engine-step means as timed, the composite is those weighted by the CEM call counts, and the
    residual is what is left of the MEASURED cycle after subtracting it (SPEC §Interface
    Contracts). `cycle` must be the mean of the same truncated sample the p50 is read off."""
    import json

    out, bench = _mean_render(tmp_path)  # `report` finalizes `bench` in place
    payload = json.loads(Path(out["stats"]).read_text())

    for track in ("lewm", "dino"):
        for prec in ("fp32", "fp16"):
            e = payload["points_means"][track][prec]["max"]
            assert e["t_comp_mean_ms"] == (
                ENCODER_CALLS_PER_CYCLE * e["enc_mean_ms"]
                + PREDICTOR_CALLS_PER_CYCLE * e["pred_mean_ms"]
            )
            assert e["cycle_mean_ms"] == bench[track][prec]["per_cycle_mean_ms"]
            assert e["overhead_mean_ms"] == e["cycle_mean_ms"] - e["t_comp_mean_ms"]


def test_latency_means_table_is_parseable_and_carries_its_markers(tmp_path):
    """Each of the five VALUE cells is ONE whitespace-delimited token — point, interval and (for the
    three quantities that ARE a sample) its independence marker together — so the artefact stays
    `split()`-parseable off the last five fields. `t_comp`/`ovh` end in a bracket: a flag describes a
    sample, and they are functions of two and three (architecture.md §9). The config column is the
    one that may split (`INT8 (max)`), which is why the values are read from the END."""
    import json

    out, _ = _mean_render(tmp_path)
    text = Path(out["tables"]["latency_means_table"]).read_text()
    payload = json.loads(Path(out["stats"]).read_text())

    header = [ln for ln in text.splitlines() if ln.split()[:1] == ["track"]][0].split()
    assert header == ["track", "config", "enc_call_ms", "pred_call_ms", "t_comp_ms", "cycle_ms",
                      "ovh_ms"]
    rows = [ln.split() for ln in text.splitlines() if ln.split()[:1] in (["lewm"], ["dino"])]
    assert len(rows) == sum(len(m) for t in payload["points_means"].values() for m in t.values())
    configs = [" ".join(r[1:-5]) for r in rows]
    assert set(configs) == {"FP32", "FP16", "INT8 (max)", "INT8 (entropy)"}  # method only where it applies
    for row in rows:
        enc, pred, t_comp, cycle, ovh = row[-5:]
        for cell in (enc, pred, cycle):
            assert cell.endswith("*") or cell.endswith("-")
        assert t_comp.endswith("]") and ovh.endswith("]")  # no marker of their own
        assert all("[" in c and "," in c for c in row[-5:])


def test_latency_means_table_is_unscoped_and_identical_across_methods(tmp_path):
    """The config column names the method, so ONE file spans both and either render writes the same
    bytes — unlike the method-scoped speed and isolation tables (SPEC §Parity)."""
    at_max, _ = _mean_render(tmp_path / "max", method="max")
    at_entropy, _ = _mean_render(tmp_path / "entropy", method="entropy")

    assert Path(at_max["tables"]["latency_means_table"]).name == "latency_means_table.txt"
    assert (
        Path(at_max["tables"]["latency_means_table"]).read_bytes()
        == Path(at_entropy["tables"]["latency_means_table"]).read_bytes()
    )


def test_render_writes_exactly_the_reported_artifacts(tmp_path):
    """The reported surface is closed: ONE figure and three tables (the speed table, the isolation
    table where isolation runs exist, and the method-unscoped mean latency table) plus the
    stats.json they read their intervals from. Nothing else is written, and there is no
    subdirectory of further figures."""
    out, _ = _mean_render(tmp_path)

    assert {p.name for p in tmp_path.glob("*.png")} == {"speed_vs_sr.png"}
    assert not [p for p in tmp_path.iterdir() if p.is_dir()]
    assert set(out["tables"]) == {"speed_table", "latency_means_table"}
    assert {p.name for p in tmp_path.glob("*.txt")} == {
        "speed_table.max.txt", "latency_means_table.txt"
    }
    assert Path(out["stats"]).name == "stats.json"


def test_mean_table_does_not_touch_the_other_artifacts(tmp_path):
    """The mean surface is additive: it adds a file and changes none. The speed table and the
    figure must be byte-identical with and without the component samples that unlock it."""
    import numpy as np

    rng = np.random.default_rng(41)
    overrides = {
        t: {"fp32": {"max": {"success_rate": 90.0,
                             "per_cycle_latencies_ms": list(rng.normal(120.0, 4.0, size=40))}}}
        for t in ("lewm", "dino")
    }
    without = report.report(_synthetic(), tmp_path / "a", sr_overrides=overrides, method="max")
    with_ = report.report(
        _synthetic(), tmp_path / "b", sr_overrides=overrides, method="max",
        component_latencies={t: _component_samples(42) for t in ("lewm", "dino")},
    )

    assert "latency_means_table" not in without["tables"]
    assert "latency_means_table" in with_["tables"]
    assert (
        Path(without["tables"]["speed_table"]).read_bytes()
        == Path(with_["tables"]["speed_table"]).read_bytes()
    )
    assert (
        Path(without["plots"]["speed_vs_sr"]).read_bytes()
        == Path(with_["plots"]["speed_vs_sr"]).read_bytes()
    )


def test_nan_sr_is_skipped_not_crashed(tmp_path):
    bench = _synthetic()
    bench["lewm"]["fp32"]["success_rate"] = math.nan
    out = report.report(bench, tmp_path)  # must not raise
    assert Path(out["plots"]["speed_vs_sr"]).exists()


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
    assert b["dino"]["fp32"]["per_cycle_p95_ms"] == 2000.0


def test_isolation_table_attributes_component(tmp_path):
    """architecture.md §8: the mixed-precision diagnostic attributes a measured SR drop to encoder
    or predictor. Each row is read against that track's FP16 row on the speed table — the precision
    the held component is running at, NOT FP32."""
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
    # One row per diagnostic run, naming the precision EACH component ran at — the quantized side
    # and the FP16-held side — so which component the SR belongs to is read off the row itself.
    enc = [ln for ln in text.splitlines() if ln.split()[:3] == ["dino", "int8", "fp16"]][0]
    pred = [ln for ln in text.splitlines() if ln.split()[:3] == ["dino", "fp16", "int8"]][0]
    assert enc.split()[3] == "16.0"
    assert pred.split()[3] == "42.0"


def test_isolation_keys_never_reach_the_headline(tmp_path):
    """The composite keys are diagnostics: they must leave every headline artefact byte-identical
    (architecture.md §8 — a mixed pairing is never a fifth precision)."""
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
