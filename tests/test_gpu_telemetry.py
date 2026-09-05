"""GPU-telemetry render plumbing (CPU, off-pod).

Synthetic `dmon` logs stand in for the pod artifacts. What is pinned here is how a run's log is
reduced to ONE clock — the util-conditioned statistic and the refusal to invent one for an
undersampled run — plus the tag parsing that keeps two calibration methods' telemetry from
colliding.
"""

from pathlib import Path

from src import gpu_telemetry

_BOOST_CEILING_MHZ = gpu_telemetry._BOOST_CEILING_MHZ

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
            [(0, _BOOST_CEILING_MHZ, 100)] * 10 + [(90, _BOOST_CEILING_MHZ, 230)] * 5,
        )
        _dmon(
            d / f"lewm.{prec}.benchmark.dmon.log",
            lewm_bench_samples if lewm_bench_samples is not None else [(7, 1260.0, 56)],
        )
    return d


# --- the owner-set clock statistic ----------------------------------------------------
def test_util_conditioned_median_ignores_idle_samples(tmp_path):
    """The conditioning is the whole point: an unconditioned median over a mostly-idle log lands
    at a clock nothing was running at."""
    rows = gpu_telemetry.parse_dmon(
        _dmon(tmp_path / "x.log", [(0, 500.0, 60)] * 9 + [(100, 2400.0, 340)] * 5)
    )
    s = gpu_telemetry.summarize_run(rows)
    assert s["f_measured_mhz"] == 2400.0  # busy samples only
    assert s["sm_clock_median_mhz"] == 500.0  # the naive statistic, carried for contrast only
    assert s["power_w_busy_median"] == 340.0
    assert s["n_samples"] == 14 and s["n_busy"] == 5


def test_undersampled_run_is_unmeasured_never_assumed(tmp_path):
    """A run too short for 1 Hz dmon gets NO clock — inventing one from an idle/ramp sample is the
    silent wrong-number failure this gate exists to prevent."""
    rows = gpu_telemetry.parse_dmon(_dmon(tmp_path / "x.log", [(7, 1260.0, 56)]))
    s = gpu_telemetry.summarize_run(rows)
    assert s["f_measured_mhz"] is None
    assert "need" in s["unmeasured_reason"]
    assert s["sm_clock_median_mhz"] == 1260.0  # the value it refused to normalize with


def test_harvest_keys_by_track_precision_method_runtype(tmp_path):
    """Legacy 3-part logs land under the reserved `unscoped` method rather than being dropped —
    they are durable artifacts already on the volume (CLAUDE §8)."""
    clocks = gpu_telemetry.harvest(_logs(tmp_path))
    assert clocks["dino"]["fp32"]["unscoped"]["sr_eval"]["f_measured_mhz"] == 2000.0
    assert clocks["lewm"]["fp32"]["unscoped"]["sr_eval"]["f_measured_mhz"] == _BOOST_CEILING_MHZ
    assert clocks["lewm"]["fp32"]["unscoped"]["benchmark"]["f_measured_mhz"] is None


def test_harvest_fails_loud_on_an_unreadable_tag(tmp_path):
    d = tmp_path / "gpu_logs"
    d.mkdir()
    _dmon(d / "mystery.dmon.log", [(100, 2400.0, 340)] * 5)
    try:
        gpu_telemetry.harvest(d)
    except SystemExit as e:
        assert "mystery" in str(e)
    else:
        raise AssertionError("a run with an unreadable tag was silently skipped")


def test_method_scoped_logs_do_not_collide_and_are_selected_by_method(tmp_path):
    """The bug this guards: the pre-fix tag omitted the method and `log_gpu` opens with `"w"`, so
    an `entropy` re-run overwrote the `max` run's telemetry and the render paired the survivor with
    whichever method it happened to be rendering. Scoped logs coexist and each render picks its
    own."""
    d = tmp_path / "gpu_logs"
    d.mkdir()
    _dmon(d / "dino.int8.max.sr_eval.dmon.log", [(100, 2000.0, 340)] * 20)
    _dmon(d / "dino.int8.entropy.sr_eval.dmon.log", [(100, 2300.0, 340)] * 20)
    clocks = gpu_telemetry.harvest(d)
    assert set(clocks["dino"]["int8"]) == {"max", "entropy"}  # neither clobbered the other
    assert gpu_telemetry._summary(clocks, "dino", "int8", "sr_eval", "max")["f_measured_mhz"] == 2000.0
    assert gpu_telemetry._summary(clocks, "dino", "int8", "sr_eval", "entropy")["f_measured_mhz"] == 2300.0


def test_unscoped_log_is_the_fallback_and_falls_back_per_run_type(tmp_path):
    """A partially re-run volume holds a method-scoped `sr_eval` beside a legacy `benchmark`; the
    fallback is per run type, so the legacy component clock is still found."""
    d = tmp_path / "gpu_logs"
    d.mkdir()
    _dmon(d / "dino.int8.entropy.sr_eval.dmon.log", [(100, 2300.0, 340)] * 20)
    _dmon(d / "dino.int8.benchmark.dmon.log", [(100, 2400.0, 340)] * 20)
    clocks = gpu_telemetry.harvest(d)
    assert gpu_telemetry._summary(clocks, "dino", "int8", "sr_eval", "entropy")["f_measured_mhz"] == 2300.0
    assert gpu_telemetry._summary(clocks, "dino", "int8", "benchmark", "entropy")["f_measured_mhz"] == 2400.0



# --- the throttle diagnostic -----------------------------------------------------------
def test_throttle_plot_renders_per_run_type_beside_the_telemetry(tmp_path):
    """The PNG lands in `gpu_logs/`, beside the logs it summarizes — on the volume, never
    committed to the repo."""
    out = gpu_telemetry.run(_logs(tmp_path))
    for run_type in ("sr_eval", "benchmark"):
        path = Path(out["plots"][run_type])
        assert path.name == f"{run_type}_clock_diag.png"
        assert path.parent.name == "gpu_logs"
        assert path.stat().st_size > 0


def test_telemetry_render_writes_nothing_but_its_plots(tmp_path):
    """It reads telemetry and renders a diagnostic — it derives no latency and touches no
    canonical artefact (SPEC §Parity, CLAUDE §8)."""
    logs = _logs(tmp_path)
    (tmp_path / "results.lewm.json").write_text("{}")
    (tmp_path / "speed_table.max.txt").write_text("measured, must survive\n")
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

    gpu_telemetry.run(logs)

    for path, content in before.items():
        assert path.read_bytes() == content, f"{path.name} was rewritten by the telemetry render"
    assert {p.name for p in logs.iterdir() if p.suffix == ".png"} == {
        "sr_eval_clock_diag.png",
        "benchmark_clock_diag.png",
    }
