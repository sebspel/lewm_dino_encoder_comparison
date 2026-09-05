"""GPU-telemetry render — owned PLUMBING (fails LOUDLY), runs OFF-POD.

GPU clocks cannot be locked on the benchmark platform (`nvidia-smi -lgc` denied by the RunPod
virtualization layer, confirmed as root with persistence mode on — SPEC §Execution Environment),
so the per-run clock/thermal state is RECORDED rather than controlled: `src.gpu_clocks` runs a
passive `nvidia-smi dmon` observer beside every timed run. This module reads those logs back and
renders the throttle diagnostic they support.

It reads `gpu_logs/*.dmon.log` and writes one PNG per run type BESIDE them, on the volume. Nothing
else is touched: `results.*.json`, `sr.json`, `stats.json` and the rendered tables are never read
or written here.

**It derives no corrected latency.** Rescaling a measured time to a reference clock would
over-correct (memory-bound and host/Python time do not scale with SM clock) and would put a
derived number beside a measured one where the two could be confused. The measured numbers are the
only latency numbers; this render states the conditions they were measured under.

    uv run python -m src.gpu_telemetry
    uv run python -m src.gpu_telemetry from=$STABLEWM_HOME/reports/phase5 calibration_method=entropy
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window (off-pod + CI)
import matplotlib.pyplot as plt  # noqa: E402

from src import report  # noqa: E402  — the shared track/precision vocabulary + figure chrome
from src.interfaces import (  # noqa: E402
    CLOCK_BUSY_UTIL_PCT,
    CLOCK_MIN_BUSY_SAMPLES,
    DEFAULT_CALIBRATION_METHOD,
    check_calibration_method,
)

# Which timed run each telemetry log belongs to. The per-cycle vector is recorded by the latency
# callback over the SR eval-shim run; the encode/predict step distributions come from the isolated
# per-precision engine loops in `src.benchmark`.
_CYCLE_RUN = "sr_eval"
_COMPONENT_RUN = "benchmark"

# Reserved method key for legacy logs tagged before `gpu_clocks.run_tag` carried the calibration
# method. Not a method any run was made at — a marker that the log's method is unrecorded.
_UNSCOPED = "unscoped"

# Display-only reference lines: the L40S boost ceiling and its board power limit. They mark where
# "at the ceiling" and "pinned at the cap" sit on the axes; neither scales any number.
_BOOST_CEILING_MHZ = 2520
_POWER_LIMIT_W = 350

_MUTED = "#898781"


# --- dmon telemetry -> per-run summary -------------------------------------------------
def _num(tok: str):
    """One dmon field -> float, or None for a not-supported field (`-`) / a non-numeric column."""
    try:
        return float(tok)
    except ValueError:
        return None



def parse_dmon(path) -> list[dict]:
    """One `nvidia-smi dmon -s pucm -o DT` log -> its per-sample rows.

    Column names are read from the log's **own header** (`#Date Time gpu pwr gtemp mtemp sm mem
    … mclk pclk …`) rather than fixed positionally, so a different `nvidia-smi` field order fails
    as a missing key instead of silently reading power out of the clock column."""
    names: list[str] | None = None
    rows: list[dict] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if names is None:  # first header line carries the field names; the second, the units
                names = line.lstrip("#").split()
            continue
        parts = line.split()
        if names is None or len(parts) != len(names):
            continue  # truncated final sample (the observer is SIGTERMed mid-write)
        rows.append({k: _num(v) for k, v in zip(names, parts)})
    return rows


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None



def summarize_run(rows: list[dict]) -> dict:
    """Per-sample dmon rows -> the one run's telemetry summary, including the **owner-set clock
    statistic** `f_measured_mhz` that feeds the normalization.

    `f_measured_mhz` is the median `pclk` over the **busy** samples only (SM util ≥
    `CLOCK_BUSY_UTIL_PCT`). The conditioning is load-bearing: dmon samples at ~1 Hz, so a short
    run's log is mostly idle/ramp samples, and an unconditioned median can land at a clock nothing
    was ever running at. A run with fewer than `CLOCK_MIN_BUSY_SAMPLES` busy samples is reported
    **unmeasured** (`f_measured_mhz = None`) rather than summarised — never a fabricated clock
    (SPEC §Implementation Boundaries: a wrong choice here is a plausible wrong number).

    `sm_clock_median_mhz` is the *unconditioned* median, carried for contrast only — it shows how
    far the naive statistic would have been off."""
    busy = [r for r in rows if (r.get("sm") or 0.0) >= CLOCK_BUSY_UTIL_PCT]
    measured = len(busy) >= CLOCK_MIN_BUSY_SAMPLES
    return {
        "n_samples": len(rows),
        "n_busy": len(busy),
        # THE normalization input (owner-set statistic)
        "f_measured_mhz": _median(r.get("pclk") for r in busy) if measured else None,
        "unmeasured_reason": (
            None
            if measured
            else f"{len(busy)} sample(s) at >= {CLOCK_BUSY_UTIL_PCT}% SM util "
            f"(need {CLOCK_MIN_BUSY_SAMPLES}) — run too short for 1 Hz dmon"
        ),
        # descriptive, unconditioned (context for the diagnostic, not normalization inputs)
        "sm_clock_median_mhz": _median(r.get("pclk") for r in rows),
        "sm_util_median_pct": _median(r.get("sm") for r in rows),
        "mem_clock_median_mhz": _median(r.get("mclk") for r in rows),
        "power_w_median": _median(r.get("pwr") for r in rows),
        "temp_c_median": _median(r.get("gtemp") for r in rows),
        # power over the busy samples — the throttle's CAUSE, so worth conditioning too
        "power_w_busy_median": _median(r.get("pwr") for r in busy) if measured else None,
    }



def harvest(gpu_log_dir) -> dict:
    """`gpu_logs/*.dmon.log` -> `{track: {precision: {method: {run_type: summary}}}}`.

    Log basenames are the tags `src.gpu_clocks.run_tag` builds:
    `<track>.<precision>.<method>.<run_type>.dmon.log` (`src.study` / `src.sr_eval`). `precision`
    may be a composite component-isolation key (`enc-fp16+pred-int8`); those are harvested like any
    other run but reach no reported surface, exactly as they reach no headline table
    (architecture.md §8).

    **Legacy 3-part `<track>.<precision>.<run_type>` logs are still read**, under the reserved
    method key `unscoped` — they are durable artifacts already on the volume and predate the
    method-scoped tag (CLAUDE §8).

    An unexpected basename raises rather than being skipped — a silently-dropped run would look
    like a run that was never made."""
    clocks: dict = {}
    for path in sorted(Path(gpu_log_dir).glob("*.dmon.log")):
        parts = path.name[: -len(".dmon.log")].split(".")
        if len(parts) == 4:
            track, precision, method, run_type = parts
        elif len(parts) == 3:  # legacy, pre method-scoped tag
            (track, precision, run_type), method = parts, _UNSCOPED
        else:
            raise SystemExit(
                f"[gpu_telemetry] cannot read a (track, precision, method, run_type) tag off "
                f"{path.name!r} — expected <track>.<precision>.<method>.<run_type>.dmon.log "
                f"(or the legacy <track>.<precision>.<run_type>.dmon.log)"
            )
        by_method = clocks.setdefault(track, {}).setdefault(precision, {})
        by_method.setdefault(method, {})[run_type] = summarize_run(parse_dmon(path))
    return clocks



def _summary(clocks: dict, track: str, precision: str, run_type: str, method: str) -> dict:
    """The one run's telemetry summary, selected by calibration method.

    Prefers the method-scoped log and falls back to a legacy `unscoped` one, which is the only
    telemetry that exists for runs made before the tag carried the method. The fallback is **per
    run type**, not per method: after a partial re-run a precision can hold a method-scoped
    `sr_eval` log beside a legacy `benchmark` one. `{}` when absent."""
    by_method = clocks.get(track, {}).get(precision, {})
    scoped = by_method.get(method, {}).get(run_type)
    return scoped or by_method.get(_UNSCOPED, {}).get(run_type) or {}



def plot_throttle(clocks: dict, run_type: str, out_dir: Path, method: str) -> Path:
    """The differential-throttle diagnostic for one run type: LeWM vs DINOv3-WM, per precision.

    Two panels — **SM clock** over **power**, its driver — so a track that saturates the board
    power limit and trades clock for it reads differently from one that never approaches it. The `f_ref` boost ceiling and the board power limit
    are each drawn as a reference line, so "at the ceiling" vs "throttled below it" and "pinned at
    the cap" vs "never near it" are readable without arithmetic or knowing the L40S TDP. Bars use
    the measured (util-conditioned) statistic; an UNMEASURED run is labelled as such rather than
    plotted at some fallback value. Each measured clock bar carries its busy-sample count `n=…` —
    the lighter track's eval-shim medians rest on an order of magnitude fewer samples than the
    heavier track's, and the label keeps that visible instead of rendering both with equal
    authority.

    Composite component-isolation runs are excluded, like everywhere else outside the isolation
    table (architecture.md §8)."""
    precs = [
        p
        for p in report._PRECISIONS
        if any(_summary(clocks, t, p, run_type, method) for t in report._TRACKS)
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    width = 0.38
    for ax, (field, label) in zip(
        axes, (("f_measured_mhz", "SM clock (MHz)"), ("power_w_busy_median", "Board power (W)"))
    ):
        for i, track in enumerate(report._TRACKS):
            xs = [j + (i - 0.5) * width for j in range(len(precs))]
            runs = [_summary(clocks, track, p, run_type, method) for p in precs]
            vals = [r.get(field) for r in runs]
            ax.bar(
                xs,
                [v or 0.0 for v in vals],
                width=width,
                color=report._TRACK_COLOR[track],
                label=report._TRACK_DISPLAY[track],
                zorder=2,
            )
            for x, v, run in zip(xs, vals, runs):
                # Value-label every bar: the throttle is a low-tens-of-percent effect, so a
                # zero-based axis (never truncated for bars) reads it as a small notch. The exact
                # number belongs on the mark rather than in a truncated axis.
                ax.annotate(
                    "unmeasured" if v is None else f"{v:.0f}",
                    (x, 0 if v is None else v),
                    rotation=90 if v is None else 0,
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=_MUTED if v is None else report._INK,
                )
                if field == "f_measured_mhz" and v is not None:
                    # The busy-sample count behind the median, inside the bar: a 6-sample median
                    # and a 4000-sample one must not read with equal authority.
                    ax.annotate(
                        f"n={run['n_busy']}",
                        (x, 0),
                        rotation=90,
                        textcoords="offset points",
                        xytext=(0, 4),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="white",
                        zorder=3,
                    )
        ax.set_ylabel(label)
        report._style(ax)
    # The reference lines (boost ceiling, board power limit). Deliberately unexplained in-figure —
    # the PNG carries no caption text (owner ruling, 2026-07-26); what the dashes and the n=…
    # labels mean is documented here, not on the figure.
    axes[0].axhline(_BOOST_CEILING_MHZ, color=_MUTED, ls="--", lw=0.9, zorder=1)
    axes[1].axhline(_POWER_LIMIT_W, color=_MUTED, ls="--", lw=0.9, zorder=1)
    # Headroom above the bars for the legend / the value labels of bars at the cap. The bar axes
    # keep their zero baseline — truncating one to dramatize the gap would overstate the spread.
    axes[0].set_ylim(0, _BOOST_CEILING_MHZ * 1.32)
    axes[1].set_ylim(0, _POWER_LIMIT_W * 1.2)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xticks(range(len(precs)), [report._PREC_DISPLAY[p] for p in precs])
    axes[-1].set_xlabel(f"Precision  ({run_type} runs)", fontsize=8)
    fig.tight_layout()
    path = Path(out_dir) / f"{run_type}_clock_diag.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --- driver ---------------------------------------------------------------------------
def run(gpu_log_dir, method: str = DEFAULT_CALIBRATION_METHOD) -> dict:
    """Harvest the telemetry and render the throttle diagnostic per run type, beside the logs.

    `method` selects which method's logs are read: the quantized rows are timed on that method's
    engines, so its runs carry their own telemetry. Legacy unscoped logs are read as a fallback."""
    gpu_log_dir = Path(gpu_log_dir)
    clocks = harvest(gpu_log_dir)
    plots = {
        rt: plot_throttle(clocks, rt, gpu_log_dir, method)
        for rt in (_CYCLE_RUN, _COMPONENT_RUN)
    }
    return {"plots": plots, "clocks": clocks}


def main() -> None:
    """Off-pod telemetry render — reads the saved dmon logs, writes only the diagnostic PNGs.

        uv run python -m src.gpu_telemetry
        uv run python -m src.gpu_telemetry from=<dir> gpu_logs=<dir> calibration_method=entropy

    `from` defaults to `$STABLEWM_HOME/reports/phase5` (the same default `src.study`/`src.report`
    use) and `gpu_logs` to `<from>/gpu_logs`."""
    src_dir = gpu_log_dir = None
    method = DEFAULT_CALIBRATION_METHOD
    for a in sys.argv[1:]:
        if a.startswith("from="):
            src_dir = a.split("=", 1)[1]
        elif a.startswith("gpu_logs="):
            gpu_log_dir = Path(a.split("=", 1)[1])
        elif a.startswith("calibration_method="):
            method = check_calibration_method(a.split("=", 1)[1])
    if src_dir is None:
        from src.study import default_out_dir  # shared default; lazy to avoid an import cycle

        src_dir = default_out_dir()
    base = Path(src_dir) if Path(src_dir).is_dir() else Path(src_dir).parent
    gpu_log_dir = gpu_log_dir or base / "gpu_logs"
    if not Path(gpu_log_dir).is_dir():
        raise SystemExit(f"[gpu_telemetry] no telemetry directory at {gpu_log_dir}")

    out = run(gpu_log_dir, method=method)
    print(
        f"[gpu_telemetry] throttle diagnostics (method={method}) -> \n  "
        + "\n  ".join(str(p) for p in out["plots"].values())
    )


if __name__ == "__main__":
    main()
