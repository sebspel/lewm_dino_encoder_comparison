"""Phase-7 clock-confound render plumbing (CPU, off-pod).

Synthetic `dmon` logs + synthetic canonical results stand in for the pod artifacts. What is pinned
here is the OWNER-SET construction (SPEC §Implementation Boundaries) and the additivity contract
(SPEC §Parity): the util-conditioned statistic, the unmeasured-run refusal to invent a clock, the
`R′ ≤ R` bound direction, and that no canonical artefact is rewritten by a render.
"""

import json
from pathlib import Path

from src import clock_norm, report
from src.interfaces import CLOCK_F_REF_MHZ

_HEADER = (
    "#Date        Time         gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg"
    "    ofa   mclk   pclk     fb   bar1   ccpm \n"
    "#YYYYMMDD    HH:MM:SS     Idx      W      C      C      %      %      %      %      %"
    "      %    MHz    MHz     MB     MB     MB \n"
)


def _dmon(path: Path, samples) -> Path:
    """Write a dmon log. `samples` is `(sm_util_pct, pclk_mhz, power_w)` per line."""
    rows = "".join(
        f" 20260721    17:07:39       0   {pwr:4.0f}     31      -   {sm:4.0f}      1"
        f"      0      0      0      0   9001   {pclk:4.0f}   3689      4      0 \n"
        for sm, pclk, pwr in samples
    )
    path.write_text(_HEADER + rows)
    return path


def _logs(tmp_path: Path, *, lewm_bench_samples=None) -> Path:
    """A `gpu_logs/` dir mirroring the real shape: DINO throttled below the ceiling under load,
    LeWM pinned at the ceiling, and (by default) LeWM's engine loops too short to sample."""
    d = tmp_path / "gpu_logs"
    d.mkdir(exist_ok=True)
    for prec, dino_clk in (("fp32", 2000.0), ("fp16", 2400.0)):
        _dmon(d / f"dino.{prec}.sr_eval.dmon.log", [(100, dino_clk, 340)] * 20)
        _dmon(d / f"dino.{prec}.benchmark.dmon.log", [(100, dino_clk, 340)] * 20)
        _dmon(
            d / f"lewm.{prec}.sr_eval.dmon.log",
            [(0, CLOCK_F_REF_MHZ, 100)] * 10 + [(90, CLOCK_F_REF_MHZ, 230)] * 5,
        )
        _dmon(
            d / f"lewm.{prec}.benchmark.dmon.log",
            lewm_bench_samples if lewm_bench_samples is not None else [(7, 1260.0, 56)],
        )
    return d


def _bench_and_sr(tmp_path: Path):
    """Canonical per-track results + sr.json on disk, in the real file shapes."""
    bench = {
        "lewm": {
            "fp32": _row(1.0, 0.25, 100.0),
            "fp16": _row(0.6, 0.15, 60.0),
        },
        "dino": {
            "fp32": _row(10.0, 5.0, 1000.0),
            "fp16": _row(6.0, 3.0, 600.0),
        },
    }
    for track, rows in bench.items():
        (tmp_path / f"results.{track}.json").write_text(
            json.dumps({"meta": {"track": track}, "bench": rows})
        )
    sr = {
        track: {
            prec: {"max": {"success_rate": 90.0, "per_cycle_latencies_ms": r["_cyc"]}}
            for prec, r in rows.items()
        }
        for track, rows in bench.items()
    }
    (tmp_path / "sr.json").write_text(json.dumps(sr))
    return sr


def _row(enc_ms, pred_ms, cycle_ms):
    """A BenchResult with a flat per-cycle vector (so p50 == mean == `cycle_ms`), plus one warm-up
    head decision for `report.PER_CYCLE_WARMUP_DROP` to drop."""
    return {
        "per_cycle_p50_ms": float("nan"),
        "per_cycle_p95_ms": float("nan"),
        "per_cycle_mean_ms": float("nan"),
        "encode_p50_ms": enc_ms,
        "encode_p95_ms": enc_ms,
        "encode_mean_ms": enc_ms,
        "predict_p50_ms": pred_ms,
        "predict_p95_ms": pred_ms,
        "predict_mean_ms": pred_ms,
        "peak_mem_mb": 100.0,
        "success_rate": float("nan"),
        "_cyc": [cycle_ms * 3] + [cycle_ms] * 10,
    }


# --- the owner-set clock statistic ----------------------------------------------------
def test_util_conditioned_median_ignores_idle_samples(tmp_path):
    """The conditioning is the whole point: an unconditioned median over a mostly-idle log lands
    at a clock nothing was running at."""
    rows = clock_norm.parse_dmon(
        _dmon(tmp_path / "x.log", [(0, 500.0, 60)] * 9 + [(100, 2400.0, 340)] * 5)
    )
    s = clock_norm.summarize_run(rows)
    assert s["f_measured_mhz"] == 2400.0  # busy samples only
    assert s["sm_clock_median_mhz"] == 500.0  # the naive statistic, carried for contrast only
    assert s["power_w_busy_median"] == 340.0
    assert s["n_samples"] == 14 and s["n_busy"] == 5


def test_undersampled_run_is_unmeasured_never_assumed(tmp_path):
    """A run too short for 1 Hz dmon gets NO clock — inventing one from an idle/ramp sample is the
    silent wrong-number failure this gate exists to prevent."""
    rows = clock_norm.parse_dmon(_dmon(tmp_path / "x.log", [(7, 1260.0, 56)]))
    s = clock_norm.summarize_run(rows)
    assert s["f_measured_mhz"] is None
    assert "need" in s["unmeasured_reason"]
    assert s["sm_clock_median_mhz"] == 1260.0  # the value it refused to normalize with


def test_harvest_keys_by_track_precision_runtype(tmp_path):
    clocks = clock_norm.harvest(_logs(tmp_path))
    assert clocks["dino"]["fp32"]["sr_eval"]["f_measured_mhz"] == 2000.0
    assert clocks["lewm"]["fp32"]["sr_eval"]["f_measured_mhz"] == CLOCK_F_REF_MHZ
    assert clocks["lewm"]["fp32"]["benchmark"]["f_measured_mhz"] is None


def test_harvest_fails_loud_on_an_unreadable_tag(tmp_path):
    d = tmp_path / "gpu_logs"
    d.mkdir()
    _dmon(d / "mystery.dmon.log", [(100, 2400.0, 340)] * 5)
    try:
        clock_norm.harvest(d)
    except SystemExit as e:
        assert "mystery" in str(e)
    else:
        raise AssertionError("a run with an unreadable tag was silently skipped")


# --- the three surfaces ---------------------------------------------------------------
def _normalized(tmp_path):
    sr = _bench_and_sr(tmp_path)
    bench = report.load_results(report._resolve_result_paths(tmp_path))
    report._join_eval(bench, sr, "max")
    report._finalize_per_cycle(bench)
    return bench, clock_norm.normalize(bench, clock_norm.harvest(_logs(tmp_path)))


def test_ratio_bound_brackets_the_measured_ratio(tmp_path):
    """R′ ≤ R when the heavier track is the throttled one — the measured headline is the upper end
    of the bracket, which is the claim the disclosure makes."""
    _, norm = _normalized(tmp_path)
    rows = {r["precision"]: r for r in norm["ratio"]}
    fp32 = rows["fp32"]
    assert fp32["r_p50"] == 10.0  # 1000 / 100 measured
    assert fp32["r_p50_norm"] < fp32["r_p50"]
    # f_ref cancels: R' = R * f_dino / f_lewm
    assert abs(fp32["r_p50_norm"] - 10.0 * 2000.0 / CLOCK_F_REF_MHZ) < 1e-9
    for r in norm["ratio"]:
        assert r["r_p50_norm"] <= r["r_p50"]


def test_precision_delta_normalizes_absolute_ms_and_speedup(tmp_path):
    _, norm = _normalized(tmp_path)
    rows = {r["precision"]: r for r in norm["precision_delta"]["dino"]}
    # T_ref = T * f/f_ref, applied to the joined p50
    assert abs(rows["fp32"]["cyc_p50_ms_norm"] - 1000.0 * 2000.0 / CLOCK_F_REF_MHZ) < 1e-9
    # DINO's FP32 baseline was the MORE throttled run, so its FP16 speedup shrinks once both are
    # taken to a common clock (the within-model direction is not fixed — SPEC surface (b) caveat).
    assert rows["fp16"]["speedup_norm"] < rows["fp16"]["speedup"]
    # LeWM held the ceiling, so its normalization is a no-op
    lewm = {r["precision"]: r for r in norm["precision_delta"]["lewm"]}
    assert abs(lewm["fp16"]["speedup_norm"] - lewm["fp16"]["speedup"]) < 1e-9


def test_overhead_needs_both_clocks_and_leaves_unmeasured_rows_blank(tmp_path):
    """Surface (c) subtracts terms from two different runs. LeWM's engine loops are unmeasured, so
    its derived columns stay blank rather than half-corrected."""
    _, norm = _normalized(tmp_path)
    dino = {r["precision"]: r for r in norm["overhead"]["dino"]}
    lewm = {r["precision"]: r for r in norm["overhead"]["lewm"]}
    assert dino["fp32"]["ovh_ms_norm"] is not None and dino["fp32"]["p_norm"] is not None
    assert lewm["fp32"]["ovh_ms"] is not None  # measured side still reported
    assert lewm["fp32"]["ovh_ms_norm"] is None and lewm["fp32"]["p_norm"] is None


def test_unresolvable_overhead_is_flagged_not_clamped(tmp_path, capsys):
    """When the cycle-vs-component clock mismatch exceeds the overhead's share of the cycle, the
    derived overhead flips negative. It must be reported with the two numbers that explain it —
    never clamped, never silently dropped (SPEC §Interface Contracts)."""
    d = tmp_path / "gpu_logs"
    d.mkdir()
    _dmon(d / "dino.fp32.sr_eval.dmon.log", [(100, 2000.0, 340)] * 20)
    _dmon(d / "dino.fp32.benchmark.dmon.log", [(100, 2400.0, 340)] * 20)  # +16% of f_ref
    # overhead is 1% of the cycle — far below the clock mismatch
    row = _row(1.0, 1.0, 100.0)
    model = 1.0 * 2 + 1.0 * 150
    row["_per_cycle_latencies_ms"] = [300.0] + [model / 0.99] * 10
    bench = {"dino": {"fp32": row}}
    report._finalize_per_cycle(bench)
    norm = clock_norm.normalize(bench, clock_norm.harvest(d))
    r = norm["overhead"]["dino"][0]
    assert r["ovh_ms"] > 0 and r["ovh_ms_norm"] < 0  # measured positive, derived flips
    assert r["ovh_share"] < r["clock_mismatch"]  # ...and this is why
    # surfaced loudly at compute time...
    alarm = capsys.readouterr().out
    assert "negative DERIVED overhead" in alarm and "not resolvable" in alarm.lower()
    # ...and the table carries both numbers, so the resolvability test is checkable off the artifact
    table = clock_norm.render_overhead_table(norm)
    assert "RESOLVABLE only where" in table
    assert "1-p" in table and "Δf/f_ref" in table


# --- artefacts ------------------------------------------------------------------------
def _run(tmp_path, method="max"):
    """`repo_root` is redirected into tmp_path so a test never writes the real repo's display copy."""
    sr = _bench_and_sr(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return clock_norm.run(tmp_path, _logs(tmp_path), tmp_path, sr, method=method, repo_root=repo)


def test_derived_clocks_json_round_trips(tmp_path):
    out = _run(tmp_path)
    data = json.loads(Path(out["paths"]["derived_clocks"]).read_text())
    assert data["meta"]["f_ref_mhz"] == CLOCK_F_REF_MHZ
    assert data["meta"]["derived"] is True
    assert data["runs"]["dino"]["fp32"]["sr_eval"]["f_measured_mhz"] == 2000.0
    # one entry per (track, precision, run-type), matching the diagnostic
    assert set(data["runs"]["lewm"]["fp32"]) == {"sr_eval", "benchmark"}
    assert data["runs"] == out["clocks"]


def test_every_derived_table_names_itself_derived_and_its_construction(tmp_path):
    out = _run(tmp_path)
    for key in ("ratio", "precision_delta", "overhead"):
        path = Path(out["paths"][key])
        assert path.name == f"{key}_normalized.derived.max.txt"  # method-scoped, additive
        text = path.read_text()
        assert "DERIVED" in text
        assert f"f_ref = {CLOCK_F_REF_MHZ} MHz" in text
        assert "util-conditioned median SM clock" in text
        assert "OVER-corrects" in text


def test_a_second_method_render_is_additive(tmp_path):
    first = _run(tmp_path, method="max")
    before = {k: Path(p).read_bytes() for k, p in first["paths"].items() if str(p).endswith(".txt")}
    _run(tmp_path, method="entropy")
    for path, data in before.items():
        assert Path(first["paths"][path]).read_bytes() == data, "a re-render clobbered `max`"
    assert (tmp_path / "ratio_normalized.derived.entropy.txt").exists()


def test_throttle_plot_renders_per_run_type(tmp_path):
    out = _run(tmp_path)
    for run_type in ("sr_eval", "benchmark"):
        path = Path(out["plots"][run_type])
        assert path.name == f"{run_type}_clock_diag.png"
        assert path.parent.name == "gpu_logs"  # beside the telemetry it summarizes
        assert path.stat().st_size > 0


def test_throttle_plot_is_copied_to_the_repo_display_dir(tmp_path):
    """`reports/figs/` is the committed display view — regenerable, never the canonical copy, which
    stays on the volume beside the telemetry."""
    out = _run(tmp_path)
    repo_fig = Path(out["paths"]["throttle_fig_repo"])
    assert repo_fig.parent.name == "figs"
    assert repo_fig.read_bytes() == Path(out["plots"]["sr_eval"]).read_bytes()


def test_clock_norm_never_rewrites_canonical_results(tmp_path):
    """The same guard as `test_report.py::test_report_never_rewrites_canonical_results`, extended
    to the Phase-7 render and to the measured `.txt` tables: normalized numbers are ADDITIVE and
    never overwrite `results.*.json`, `sr.json`, or a measured table (SPEC §Parity, CLAUDE §8)."""
    sr = _bench_and_sr(tmp_path)
    measured = tmp_path / "speed_table.max.txt"
    measured.write_text("measured, must survive\n")
    canonical = {
        p: p.read_bytes()
        for p in (
            tmp_path / "results.lewm.json",
            tmp_path / "results.dino.json",
            tmp_path / "sr.json",
            measured,
        )
    }
    clock_norm.run(tmp_path, _logs(tmp_path), tmp_path, sr, method="max", repo_root=tmp_path)
    for path, before in canonical.items():
        assert path.read_bytes() == before, f"{path.name} was rewritten by a derived render"
