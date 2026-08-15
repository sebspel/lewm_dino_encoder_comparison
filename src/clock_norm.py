"""Phase-7 clock-confound render — owned PLUMBING (fails LOUDLY), runs OFF-POD.

GPU clocks cannot be locked on the benchmark platform (`nvidia-smi -lgc` denied by the RunPod
virtualization layer, confirmed as root with persistence mode on — SPEC §Execution Environment),
and the observed throttle is **differential**: the heavier track power-throttles to a lower SM
clock while the lighter track holds the boost ceiling, so it does **not** cancel in the
cross-model ratio. This module **quantifies** that confound from the telemetry `src.gpu_clocks`
already logged beside every timed run.

It reads only saved artifacts — `results.{lewm,dino}.json` (via `report.load_results`), `sr.json`,
and `gpu_logs/*.dmon.log` — so it needs no L40S, exactly like the decoupled `src.report` render.

**Everything it writes is ADDITIVE and labelled `derived`.** The measured numbers stay canonical
and remain the headline (owner framing, 2026-07-25); `results.*.json`, `sr.json`, and the measured
`.txt`/`.png` tables are **read-only** here (SPEC §Parity, CLAUDE §8).

Output: `derived_clocks.json`, three `*_normalized.derived.<method>.txt` tables, and the throttle
diagnostic plot. **The limitations write-up is deliberately not among them** — deciding how the
confound is framed, what the bound licenses, and which caveats are load-bearing is an interpretation
of the measured result, so the disclosure doc is OWNER-ONLY (SPEC §Implementation Boundaries).

Unlike the Phase-5 render, the derived artifacts are **volume-only** — not mirrored to W&B, so there
is no second place a derived number could be read as a measured one (SPEC §Requirements).

The normalization construction is **owner-set**, not chosen here (SPEC §Implementation Boundaries;
the constants live in `src.interfaces`):

  - scaling model `T_ref = T × f_measured / f_ref` (time ∝ 1/f_sm);
  - `f_ref = CLOCK_F_REF_MHZ` (the L40S boost ceiling the lighter track actually held);
  - `f_measured` = the **util-conditioned median SM clock** per run, `None` (→ no derived value)
    when the run has fewer than `CLOCK_MIN_BUSY_SAMPLES` busy samples;
  - **all** measured latency is treated as clock-bound — per-cycle, encode-step, predict-step —
    so the overhead decomposition subtracts terms taken at a matched clock.

The rescaling **over-corrects** (memory-bound and host/Python time do not scale with SM clock), and
that is the point: the normalized figure is the **maximum plausible correction**, a bound beside
the measured value, never a point estimate and never a replacement.

On the recorded Phase-5 data the overhead surface is unresolvable by construction for DINO — its
overhead share of the cycle (1−p ≈ 0.01–0.03) sits below the cycle-vs-component clock mismatch
(Δf/f_cmp ≈ 0.04–0.07) — so its derived overhead rows flip negative, and LeWM's are blank (component
runs too short for the 1 Hz sampler). Expected output, not a defect: architecture.md §11.

Which run's clock normalizes which latency follows where the latency was measured: per-cycle rides
the SR eval-shim run (`*.sr_eval.dmon.log`), the isolated encode/predict engine loops ride the
benchmark run (`*.benchmark.dmon.log`), each selected at the render's calibration method
(`gpu_clocks.run_tag`). Logs written before that tag carried the method are read as `unscoped` and
their provenance is recorded by `attribute_unscoped_runs`, never assumed.

    uv run python -m src.clock_norm
    uv run python -m src.clock_norm from=$STABLEWM_HOME/reports/phase5 calibration_method=entropy
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window (off-pod + CI)
import matplotlib.pyplot as plt  # noqa: E402

from src import report  # noqa: E402  — canonical results loader + the measured statistics
from src.interfaces import (  # noqa: E402
    CLOCK_BUSY_UTIL_PCT,
    CLOCK_F_REF_MHZ,
    CLOCK_MIN_BUSY_SAMPLES,
    DEFAULT_CALIBRATION_METHOD,
    ENCODER_CALLS_PER_CYCLE as _ENCODER_CALLS,
    PER_CYCLE_WARMUP_DROP,
    PREDICTOR_CALLS_PER_CYCLE as _PREDICTOR_CALLS,
    check_calibration_method,
)

# Which timed run each latency was measured on — hence whose clock normalizes it. The per-cycle
# vector is recorded by the latency callback over the SR eval-shim run; the encode/predict step
# distributions come from the isolated per-precision engine loops in `src.benchmark`.
_CYCLE_RUN = "sr_eval"
_COMPONENT_RUN = "benchmark"

# Reserved method key for legacy logs tagged before `gpu_clocks.run_tag` carried the calibration
# method. Not a method any run was made at — a marker that the log's method is unrecorded.
_UNSCOPED = "unscoped"

_STATISTIC = (
    f"util-conditioned median SM clock (dmon samples with SM util >= {CLOCK_BUSY_UTIL_PCT}%)"
)
_SCALING_MODEL = "T_ref = T * f_measured / f_ref"

# L40S board power limit (W) — the cap the heavier track saturates and trades clock against.
# Display-only reference line in the throttle diagnostic; it normalizes nothing.
_POWER_LIMIT_W = 350

# Where the committed display copy of the throttle plot lands (`reports/figs/`), resolved relative
# to this file so the render does not depend on the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent


# --- dmon telemetry -> per-run clock statistic ----------------------------------------
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
    **unmeasured** (`f_measured_mhz = None`) and gets no derived counterpart — never a fabricated
    clock (SPEC §Implementation Boundaries: a wrong choice here is a plausible wrong number).

    `sm_clock_median_mhz` is the *unconditioned* median, carried for contrast only — it shows how
    far the naive statistic would have been off. It normalizes nothing."""
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
    other run but reach no derived surface, exactly as they reach no headline table
    (architecture.md §9).

    **Legacy 3-part `<track>.<precision>.<run_type>` logs are still read**, under the reserved
    method key `unscoped` — they are durable artifacts already on the volume and predate the
    method-scoped tag (CLAUDE §8). Because the pre-fix tag was overwritten in place by a re-run,
    an `unscoped` sr_eval log for a quantized precision cannot be attributed to a method from its
    name alone; `attribute_unscoped_runs` records the evidence for which run wrote it.

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
                f"[clock_norm] cannot read a (track, precision, method, run_type) tag off "
                f"{path.name!r} — expected <track>.<precision>.<method>.<run_type>.dmon.log "
                f"(or the legacy <track>.<precision>.<run_type>.dmon.log)"
            )
        by_method = clocks.setdefault(track, {}).setdefault(precision, {})
        by_method.setdefault(method, {})[run_type] = summarize_run(parse_dmon(path))
    return clocks


def attribute_unscoped_runs(clocks: dict, sr: dict | None) -> dict:
    """Which calibration method's run wrote each LEGACY unscoped `sr_eval` log?

    The pre-fix tag omitted the method and `log_gpu` opens with `"w"`, so where a precision was
    evaluated under both methods only the last run's telemetry survives. This does **not** repair
    that — one log is one log — it records the evidence so the pairing is auditable rather than
    fortunate.

    The test: a dmon log samples at ~1 Hz for the whole run, so `n_samples ≈ run seconds`, and the
    run must contain at least its own planning time. `residual_s = n_samples − Σ per-cycle
    latencies` is therefore **≥ 0 for the method that actually wrote the log** and the leftover is
    that run's non-planning time (setup, env stepping, teardown). A candidate whose planning alone
    exceeds the log is physically **excluded**.

    Verdicts: `single-method` (only one method ran — unambiguous by construction), `decisive`
    (exactly one candidate survives the ≥ 0 test), `ambiguous` (several do; `best_fit` is the
    smallest residual but the log does not settle it). Only `sr_eval` is checked — the benchmark
    logs have no per-run duration recorded anywhere to check against."""
    out: dict = {}
    for track, precisions in sorted(clocks.items()):
        for precision, by_method in sorted(precisions.items()):
            if _CYCLE_RUN not in by_method.get(_UNSCOPED, {}):
                continue
            n_samples = by_method[_UNSCOPED][_CYCLE_RUN]["n_samples"]
            cands = {
                m: n_samples - sum(v["per_cycle_latencies_ms"]) / 1000.0
                for m, v in sorted((sr or {}).get(track, {}).get(precision, {}).items())
                if "per_cycle_latencies_ms" in v
            }
            feasible = [m for m, resid in cands.items() if resid >= 0]
            out[f"{track}.{precision}.{_CYCLE_RUN}"] = {
                "n_samples": n_samples,
                "residual_s": {m: round(r, 1) for m, r in cands.items()},
                "excluded": [m for m in cands if m not in feasible],
                "best_fit": min(feasible, key=lambda m: cands[m]) if feasible else None,
                "verdict": (
                    "single-method"
                    if len(cands) == 1
                    else "decisive"
                    if len(feasible) == 1
                    else "ambiguous"
                ),
            }
    return out


def write_derived_clocks(clocks: dict, attribution: dict, out_dir: Path) -> Path:
    """Persist the harvest to `derived_clocks.json` — the durable, re-readable record of the clock
    statistic every derived number was computed from (so a reader can recheck the correction
    without re-parsing 30 MB of dmon logs). Read-only over `gpu_logs/`; `results.*.json` and
    `sr.json` are never touched."""
    path = Path(out_dir) / "derived_clocks.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "statistic": _STATISTIC,
                    "scaling_model": _SCALING_MODEL,
                    "f_ref_mhz": CLOCK_F_REF_MHZ,
                    "busy_util_pct": CLOCK_BUSY_UTIL_PCT,
                    "min_busy_samples": CLOCK_MIN_BUSY_SAMPLES,
                    "clock_bound_time": (
                        "all measured latency (per-cycle, encode-step, predict-step)"
                    ),
                    "cycle_run": _CYCLE_RUN,
                    "component_run": _COMPONENT_RUN,
                    "log_tag": (
                        "<track>.<precision>.<method>.<run_type>.dmon.log; legacy 3-part logs "
                        f"are recorded under method {_UNSCOPED!r} (see unscoped_attribution)"
                    ),
                    "derived": True,
                    "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                "unscoped_attribution": attribution,
                "runs": clocks,
            },
            indent=1,
        )
        + "\n"
    )
    return path


# --- normalization (the owner-set formula, applied to the canonical latencies) ---------
def _summary(clocks: dict, track: str, precision: str, run_type: str, method: str) -> dict:
    """The one run's telemetry summary, selected by calibration method.

    Prefers the method-scoped log and falls back to a legacy `unscoped` one, which is the only
    telemetry that exists for runs made before the tag carried the method. The fallback is **per
    run type**, not per method: after a partial re-run a precision can hold a method-scoped
    `sr_eval` log beside a legacy `benchmark` one. `{}` when absent."""
    by_method = clocks.get(track, {}).get(precision, {})
    scoped = by_method.get(method, {}).get(run_type)
    return scoped or by_method.get(_UNSCOPED, {}).get(run_type) or {}


def _f(clocks: dict, track: str, precision: str, run_type: str, method: str):
    """The run's `f_measured` (MHz), or None when that run is unmeasured / absent."""
    run = _summary(clocks, track, precision, run_type, method)
    return run["f_measured_mhz"] if run else None


def _at_ref(ms, f_measured):
    """`T_ref = T × f_measured / f_ref` — one latency expressed at the reference clock. None when
    the latency is missing or the run's clock is unmeasured (no derived counterpart)."""
    if f_measured is None or report._missing(ms):
        return None
    return ms * f_measured / CLOCK_F_REF_MHZ


def normalize(bench: dict, clocks: dict, method: str) -> dict:
    """Apply the owner-set normalization to the canonical latencies, on the **three surfaces the
    confound touches** (SPEC §Requirements "Clock-state confound disclosure"):

    (a) `ratio` — the cross-model per-cycle ratio `R` and its normalized bound `R′`. Because
        `R′ = R × f_dino / f_lewm` the reference clock cancels; with the heavier track throttled
        below the lighter one's ceiling, `R′ ≤ R` — the measured ratio is the upper end.
    (b) `precision_delta` — the within-model FP32→FP16→INT8→FP8 per-cycle speedups re-expressed at
        a common clock. `f_ref` cancels in the speedup but not in the absolute ms, which is why
        both are carried. SR is clock-independent and therefore absent here.
    (c) `overhead` — the decomposition recomputed with the cycle **and** the component loops at a
        matched clock: `ovh′ = cycle′ − enc′·calls − pred′·calls`, `p′ = (enc′+pred′)/cycle′`.
        Needs BOTH the cycle run's and the component run's clock; either unmeasured leaves the row
        without a derived value. MEAN-based, like the measured decomposition it mirrors
        (architecture.md §8).

    `bench` must already carry the joined per-cycle statistics (`report._join_eval` +
    `report._finalize_per_cycle`), so the derived numbers describe exactly the same truncated,
    warm-up-dropped sample as the measured tables.

    `method` selects each run's telemetry, so an `entropy` render normalizes with the `entropy`
    run's clock — the per-cycle sample and the engine-step latencies being normalized both came from
    that method's runs (`report.load_results` selects them by the same label). Legacy unscoped logs
    are the documented fallback (`_summary`)."""
    out: dict = {"ratio": [], "precision_delta": {}, "overhead": {}}

    for prec in report._PRECISIONS:
        if prec not in bench.get("lewm", {}) or prec not in bench.get("dino", {}):
            continue
        f_l = _f(clocks, "lewm", prec, _CYCLE_RUN, method)
        f_d = _f(clocks, "dino", prec, _CYCLE_RUN, method)
        row = {"precision": prec, "f_lewm_mhz": f_l, "f_dino_mhz": f_d}
        for pct in ("p50", "p95"):
            r = report.per_cycle_ratio(bench, prec, pct)
            row[f"r_{pct}"] = r
            row[f"r_{pct}_norm"] = (
                None if f_l is None or f_d is None or report._missing(r) else r * f_d / f_l
            )
        out["ratio"].append(row)

    for track in report._TRACKS:
        base = bench.get(track, {}).get("fp32")
        deltas, overheads = [], []
        f_base = _f(clocks, track, "fp32", _CYCLE_RUN, method)
        for prec in report._PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is None:
                continue
            f_cyc = _f(clocks, track, prec, _CYCLE_RUN, method)
            f_cmp = _f(clocks, track, prec, _COMPONENT_RUN, method)

            # (b) within-model, at a common clock
            p50, p50_ref = r["per_cycle_p50_ms"], _at_ref(r["per_cycle_p50_ms"], f_cyc)
            base_p50 = None if base is None else base["per_cycle_p50_ms"]
            base_ref = None if base is None else _at_ref(base_p50, f_base)
            deltas.append(
                {
                    "track": track,
                    "precision": prec,
                    "f_mhz": f_cyc,
                    "cyc_p50_ms": p50,
                    "cyc_p50_ms_norm": p50_ref,
                    "speedup": (
                        None
                        if base_p50 is None or report._missing(base_p50) or report._missing(p50)
                        else base_p50 / p50
                    ),
                    "speedup_norm": (
                        None if base_ref is None or p50_ref is None else base_ref / p50_ref
                    ),
                }
            )

            # (c) overhead at a matched clock — MEAN basis, mirroring `report.decompose`
            measured = report.decompose(r)
            cyc_ref = _at_ref(r["per_cycle_mean_ms"], f_cyc)
            enc_ref = _at_ref(r["encode_mean_ms"], f_cmp)
            pred_ref = _at_ref(r["predict_mean_ms"], f_cmp)
            ovh_ref = p_ref = None
            # The two sides were measured on different runs, so they carry different clocks. The
            # overhead term is only RESOLVABLE where the measured overhead share (1−p) exceeds
            # Δf/f_cmp — otherwise the correction is larger than the quantity being corrected and
            # the derived value flips sign. The denominator is f_cmp, NOT f_ref: substituting
            # model = p·cyc gives ovh′ = (cyc/f_ref)·(f_cyc − p·f_cmp), so f_ref factors out of the
            # bracket and cannot enter the sign condition — the threshold is set by the two
            # measurement clocks alone. Carried per row so the table shows why, not just that.
            mismatch = None if f_cyc is None or f_cmp is None else (f_cmp - f_cyc) / f_cmp
            if None not in (cyc_ref, enc_ref, pred_ref):
                model_ref = enc_ref * _ENCODER_CALLS + pred_ref * _PREDICTOR_CALLS
                ovh_ref = cyc_ref - model_ref
                p_ref = model_ref / cyc_ref
                if ovh_ref < 0:
                    print(
                        f"⚠ negative DERIVED overhead ({ovh_ref:.4f} ms) for {track} {prec}: "
                        f"the measured overhead share (1−p = {1 - measured['optimizable_fraction']:.3f}) "
                        f"is smaller than the cycle-vs-component clock mismatch ({mismatch:+.3f} of "
                        "f_cmp), so this row's overhead is NOT resolvable under unlocked clocks. "
                        "Surfaced, never clamped (SPEC §Interface Contracts)."
                    )
            overheads.append(
                {
                    "track": track,
                    "precision": prec,
                    "f_cycle_mhz": f_cyc,
                    "f_component_mhz": f_cmp,
                    "ovh_ms": measured["overhead_ms"],
                    "ovh_ms_norm": ovh_ref,
                    "p": measured["optimizable_fraction"],
                    "p_norm": p_ref,
                    "ovh_share": (
                        None
                        if measured["optimizable_fraction"] is None
                        else 1.0 - measured["optimizable_fraction"]
                    ),
                    "clock_mismatch": mismatch,
                }
            )
        out["precision_delta"][track] = deltas
        out["overhead"][track] = overheads
    return out


# --- derived tables -------------------------------------------------------------------
def _derived_preamble() -> list[str]:
    """The provenance block every derived table carries: that it is `derived` (not measured), the
    scaling model, `f_ref`, the clock statistic, and why it is a bound. SPEC §Parity — a normalized
    number presented without this framing would read as a measurement."""
    return [
        "  DERIVED — clock-normalized BOUND, not a measurement. The measured numbers are "
        "canonical and remain the headline (SPEC §Parity).",
        f"  scaling model: {_SCALING_MODEL}   f_ref = {CLOCK_F_REF_MHZ} MHz (L40S boost ceiling, "
        "the clock the lighter track held)",
        f"  f_measured    = {_STATISTIC}",
        f"                  a run with < {CLOCK_MIN_BUSY_SAMPLES} busy samples is UNMEASURED and "
        "gets NO derived value ('—'), never an assumed clock",
        "  This 1/f_sm rescaling OVER-corrects — memory-bound and host/Python time do not scale "
        "with SM clock — so it is the MAXIMUM plausible correction.",
    ]


_fmt = report._fmt  # shared "—" rendering for a missing/unmeasured cell
_sig = report._sig  # shared significant-figure rendering (5 for a latency), "—" when unmeasured


def render_ratio_table(norm: dict) -> str:
    """Surface (a): the cross-model per-cycle ratio and its clock-normalized bound. `f_ref` cancels
    in `R′ = R × f_dino/f_lewm`, so this table is a pure differential-throttle correction."""
    hdr = (
        f"{'prec':>5} {'f_lewm':>8} {'f_dino':>8} {'R_p50':>9} {'R_p50′':>9} "
        f"{'R_p95':>9} {'R_p95′':>9}"
    )
    lines = _derived_preamble() + [
        "  R = DINOv3-WM ÷ LeWM per-cycle latency (the HEADLINE ratio, compared at p50); "
        "R′ = R × f_dino/f_lewm, so f_ref cancels.",
        "  R′ ≤ R whenever the heavier track ran at the lower clock — the measured ratio is then "
        "the UPPER end of the bracket [R′, R].",
        hdr,
        "-" * len(hdr),
    ]
    for row in norm["ratio"]:
        lines.append(
            f"{row['precision']:>5} {_fmt(row['f_lewm_mhz'], '.0f'):>8} "
            f"{_fmt(row['f_dino_mhz'], '.0f'):>8} "
            f"{_fmt(row['r_p50'], '.1f'):>9} {_fmt(row['r_p50_norm'], '.1f'):>9} "
            f"{_fmt(row['r_p95'], '.1f'):>9} {_fmt(row['r_p95_norm'], '.1f'):>9}"
        )
    return "\n".join(lines)


def render_precision_delta_table(norm: dict) -> str:
    """Surface (b): the within-model FP32→FP16→INT8→FP8 per-cycle deltas at a common clock.

    The measured speedup can move in EITHER direction here, unlike surface (a): it shrinks when
    the FP32 baseline was the more throttled run (part of its apparent slowness was clock, not
    precision) and grows when the quantized run was. SR is clock-independent, so it is not
    re-expressed — read it off the measured `fp32_relative_table`."""
    hdr = (
        f"{'track':>6} {'prec':>5} {'f_meas':>8} {'cyc_p50':>10} {'cyc_p50′':>10} "
        f"{'speedup':>9} {'speedup′':>9}"
    )
    lines = _derived_preamble() + [
        "  Per-cycle p50 vs that track's FP32, measured and at a common clock. f_ref cancels in "
        "the speedup but NOT in the ms columns.",
        "  ΔSR is clock-independent and deliberately absent — it is unchanged from the measured "
        "fp32_relative_table.",
        hdr,
        "-" * len(hdr),
    ]
    for track in report._TRACKS:
        for row in norm["precision_delta"].get(track, []):
            lines.append(
                f"{row['track']:>6} {row['precision']:>5} {_fmt(row['f_mhz'], '.0f'):>8} "
                f"{_sig(row['cyc_p50_ms'], 5):>10} {_sig(row['cyc_p50_ms_norm'], 5):>10} "
                f"{_fmt(row['speedup'], '.3f'):>9} {_fmt(row['speedup_norm'], '.3f'):>9}"
            )
    return "\n".join(lines)


def render_overhead_table(norm: dict) -> str:
    """Surface (c): `overhead = cycle − enc·calls − pred·calls` and the optimizable fraction `p`,
    recomputed with the cycle and the component loops taken to the same clock.

    The two sides are measured on DIFFERENT runs — the cycle on the SR eval-shim run, the
    components on the isolated engine loops — so they carry different clocks, and the measured
    subtraction silently mixes them. This is the surface that exposes it. A row needs both clocks;
    an unmeasured component run (a track whose engine loops finish inside a dmon sample) leaves
    the derived columns blank rather than half-corrected."""
    hdr = (
        f"{'track':>6} {'prec':>5} {'f_cyc':>7} {'f_cmp':>7} {'ovh_ms':>11} {'ovh_ms′':>11} "
        f"{'p':>7} {'p′':>7} {'1-p':>7} {'Δf/f_cmp':>9}"
    )
    lines = _derived_preamble() + [
        "  MEAN basis (means compose additively; percentiles do not — architecture.md §8). "
        "f_cyc normalizes the cycle, f_cmp the encode/predict loops.",
        "  Δf/f_cmp = (f_cmp − f_cyc)/f_cmp — the clock mismatch between the two runs the "
        "subtraction differences. The overhead term is",
        "  RESOLVABLE only where 1-p (its measured share of the cycle) EXCEEDS |Δf/f_cmp|; below "
        "that the correction is bigger than the quantity",
        "  and ovh_ms′ flips negative. A negative derived overhead is SURFACED, never clamped "
        "(SPEC §Interface Contracts) — read it as",
        "  'not resolvable under unlocked clocks', not as a measurement. Blank derived columns = "
        "one of the two runs was UNMEASURED.",
        hdr,
        "-" * len(hdr),
    ]
    for track in report._TRACKS:
        for row in norm["overhead"].get(track, []):
            lines.append(
                f"{row['track']:>6} {row['precision']:>5} "
                f"{_fmt(row['f_cycle_mhz'], '.0f'):>7} {_fmt(row['f_component_mhz'], '.0f'):>7} "
                f"{_sig(row['ovh_ms'], 5):>11} {_sig(row['ovh_ms_norm'], 5):>11} "
                f"{_fmt(row['p'], '.3f'):>7} {_fmt(row['p_norm'], '.3f'):>7} "
                f"{_fmt(row['ovh_share'], '.3f'):>7} {_fmt(row['clock_mismatch'], '+.3f'):>9}"
            )
    return "\n".join(lines)


# --- throttle diagnostic plot ---------------------------------------------------------
def plot_throttle(clocks: dict, run_type: str, out_dir: Path, method: str) -> Path:
    """The differential-throttle diagnostic for one run type: LeWM vs DINOv3-WM, per precision.

    Two panels — **SM clock** (the confound) over **power** (its cause) — because the story is
    that the heavier track saturates the board power limit and trades clock for it while the
    lighter track never approaches the limit. The `f_ref` boost ceiling and the board power limit
    are each drawn as a reference line, so "at the ceiling" vs "throttled below it" and "pinned at
    the cap" vs "never near it" are readable without arithmetic or knowing the L40S TDP. Bars use
    the measured (util-conditioned) statistic; an UNMEASURED run is labelled as such rather than
    plotted at some fallback value. Each measured clock bar carries its busy-sample count `n=…` —
    the lighter track's eval-shim medians rest on an order of magnitude fewer samples than the
    heavier track's, and the label keeps that visible instead of rendering both with equal
    authority.

    Composite component-isolation runs are excluded, like everywhere else outside the isolation
    table (architecture.md §9)."""
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
                    color=report._MUTED if v is None else report._INK,
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
        report._style(ax, grid_axis="y")
    # The reference lines (boost ceiling, board power limit). Deliberately unexplained in-figure —
    # the PNG carries no caption text (owner ruling, 2026-07-26); what the dashes and the n=…
    # labels mean is documented here and in the disclosure prose, not on the figure.
    axes[0].axhline(CLOCK_F_REF_MHZ, color=report._MUTED, ls="--", lw=0.9, zorder=1)
    axes[1].axhline(_POWER_LIMIT_W, color=report._MUTED, ls="--", lw=0.9, zorder=1)
    # Headroom above the bars for the legend / the value labels of bars at the cap. The bar axes
    # keep their zero baseline — truncating one to dramatize the gap would overstate the confound.
    axes[0].set_ylim(0, CLOCK_F_REF_MHZ * 1.32)
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
def run(
    src,
    gpu_log_dir,
    out_dir,
    sr_overrides: dict | None,
    method: str = DEFAULT_CALIBRATION_METHOD,
    warmup_drop: int = PER_CYCLE_WARMUP_DROP,
    repo_root: Path = _REPO_ROOT,
) -> dict:
    """Harvest → normalize → render, all additive. Returns the derived artifact paths.

    Writes `derived_clocks.json` + the three derived tables under `out_dir` (the volume) and copies
    the throttle plot to `repo_root/reports/figs/` as the committed display view.

    It does **not** write the limitations doc. Framing the confound is an interpretation of the
    measured result, so the write-up is OWNER-ONLY (SPEC §Implementation Boundaries); this render
    produces the numbers it cites and nothing more."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `method` selects the measured numbers as well as the telemetry: the quantized rows are timed
    # on that method's engines, so pairing them with the other method's clocks would normalize one
    # measurement with another's conditions.
    bench = report.load_results(report._resolve_result_paths(src), method)
    # The SAME join + reduction `src.report` performs, so the derived numbers describe exactly the
    # truncated, warm-up-dropped sample the measured tables were rendered from. Neither call
    # touches a file — `bench` is an in-memory copy of the canonical results.
    report._join_eval(bench, sr_overrides, method)
    report._finalize_per_cycle(bench, warmup_drop)

    clocks = harvest(gpu_log_dir)
    attribution = attribute_unscoped_runs(clocks, sr_overrides)
    norm = normalize(bench, clocks, method)

    # Surface every legacy log this render had to fall back on. Silence here is what let a
    # method-ambiguous log be paired with a render and read as if it had been scoped.
    for tag, ev in attribution.items():
        if ev["verdict"] != "single-method":
            print(
                f"⚠ {tag}.dmon.log is UNSCOPED (legacy tag) and {ev['verdict']}: "
                f"best fit {ev['best_fit']!r}, residual_s {ev['residual_s']}"
                + (f", excluded {ev['excluded']}" if ev["excluded"] else "")
                + f" — this {method} render normalized with it. Evidence in derived_clocks.json."
            )

    texts = {
        "ratio": render_ratio_table(norm),
        "precision_delta": render_precision_delta_table(norm),
        "overhead": render_overhead_table(norm),
    }
    # Method-scoped like the measured tables: the per-cycle sample these numbers reduce was
    # recorded by a method-labelled `src.sr_eval` run, so an unscoped name would let an `entropy`
    # render overwrite the `max` artefact (SPEC §Parity, CLAUDE §8).
    paths = {"derived_clocks": write_derived_clocks(clocks, attribution, out_dir)}
    for key, text in texts.items():
        path = out_dir / f"{key}_normalized.derived.{method}.txt"
        path.write_text(text + "\n")
        paths[key] = path

    plots = {
        rt: plot_throttle(clocks, rt, Path(gpu_log_dir), method)
        for rt in (_CYCLE_RUN, _COMPONENT_RUN)
    }

    # Committed display copy in the repo — the same `reports/figs/` exception the headline plots
    # use (SPEC §Headline-artifact durability): regenerable, display-only, never the canonical copy,
    # which stays beside the telemetry it summarizes.
    figs = Path(repo_root) / "reports" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    paths["throttle_fig_repo"] = figs / plots[_CYCLE_RUN].name
    paths["throttle_fig_repo"].write_bytes(plots[_CYCLE_RUN].read_bytes())

    for key in ("ratio", "precision_delta", "overhead"):
        print(texts[key])
        print()

    return {"paths": paths, "plots": plots, "norm": norm, "clocks": clocks}


def main() -> None:
    """Off-pod Phase-7 render — reads the saved results + telemetry, writes only derived artifacts.

        uv run python -m src.clock_norm
        uv run python -m src.clock_norm from=<dir> gpu_logs=<dir> out=<dir> sr=<sr.json>
        uv run python -m src.clock_norm calibration_method=entropy

    `from` defaults to `$STABLEWM_HOME/reports/phase5` (the same default `src.study`/`src.report`
    use), `gpu_logs` to `<from>/gpu_logs`, `out` to `<from>`, and `sr` to `<from>/sr.json` when it
    exists — every surface needs the per-cycle latency that only `sr.json` carries, so there is
    nothing to render without it."""
    src = out_dir = gpu_log_dir = sr_path = None
    method = DEFAULT_CALIBRATION_METHOD
    warmup_drop = PER_CYCLE_WARMUP_DROP
    for a in sys.argv[1:]:
        if a.startswith("from="):
            src = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("gpu_logs="):
            gpu_log_dir = Path(a.split("=", 1)[1])
        elif a.startswith("sr="):
            sr_path = Path(a.split("=", 1)[1])
        elif a.startswith("calibration_method="):
            method = check_calibration_method(a.split("=", 1)[1])
        elif a.startswith("per_cycle_warmup="):
            warmup_drop = int(a.split("=", 1)[1])
    if src is None:
        from src.study import default_out_dir  # shared default; lazy to avoid an import cycle

        src = default_out_dir()
    base = Path(src) if Path(src).is_dir() else Path(src).parent
    out_dir = out_dir or base
    gpu_log_dir = gpu_log_dir or base / "gpu_logs"
    if sr_path is None and (base / "sr.json").exists():
        sr_path = base / "sr.json"
    if sr_path is None:
        raise SystemExit(
            f"[clock_norm] no sr.json found under {base} — every derived surface needs the "
            "per-cycle latency it carries; pass sr=<file>"
        )
    if not Path(gpu_log_dir).is_dir():
        raise SystemExit(f"[clock_norm] no telemetry directory at {gpu_log_dir}")

    out = run(
        src,
        gpu_log_dir,
        out_dir,
        json.loads(Path(sr_path).read_text()),
        method=method,
        warmup_drop=warmup_drop,
    )
    print(
        f"[clock_norm] derived artifacts (method={method}, f_ref={CLOCK_F_REF_MHZ} MHz) -> "
        f"{out_dir}\n  " + "\n  ".join(str(p) for p in out["paths"].values())
    )
    print("  " + "\n  ".join(str(p) for p in out["plots"].values()))


if __name__ == "__main__":
    main()
