"""Headline table + plot runner — owned PLUMBING (fails LOUDLY).

Consumes the benchmark results (per track × precision) and emits the headline outputs:
  - LeWM-vs-DINOv3 **per-cycle latency ratio** (the headline speed measure, per precision) —
    DINOv3 ÷ LeWM full planning-cycle latency, compared at **p50**
  - **Amdahl dilution**: optimizable fraction `p`, ceiling `1/(1-p)`, and per-precision
    model-only vs realized speedup
  - per-model **FP32→FP16→INT8→FP8 delta** in both **speed and SR**, degradation quoted vs FP32
  - **speed-vs-SR** scatter
  - per-component **encoder / predictor / overhead** bottleneck breakdown, derived from the
    engine-step times × CEM call counts minus the measured per-cycle time (overhead by
    subtraction, SPEC §Interface Contracts)
  - **component analysis** — the five mean latency quantities of `latency_means_table.txt` as one
    dot plot each, per model and configuration, in a `component_analysis/` subdirectory
  - **calibration-method comparison** (`max` vs `entropy` SR side by side) — the ONE table that
    spans methods, so the two labelled points coexist on the page (architecture.md §7)
  - **component-precision isolation** — which component's quantization caused a measured SR drop,
    read off the mixed-precision diagnostic runs (architecture.md §9)

**Which statistic goes where** (SPEC §Interface Contracts — do not mix these):
  - **p50** — the COMPARISON basis: the LeWM-vs-DINOv3 headline ratio and the FP32-relative
    speedup. The headline is a mechanistic claim about encoder compute, and p50 is what this
    sample size supports.
  - **p95** — reported as the descriptive tail, never the basis of a comparison.
  - **mean** — the DECOMPOSITION basis ONLY (`decompose`, `dilution_disclosure`), never a
    headline. `cycle = enc·calls + pred·calls + overhead` is exact for means (linearity of
    expectation) and merely approximate for percentiles; Amdahl is an expectation model too.

Pure data → tables/plots; runs anywhere (matplotlib Agg, no CUDA, no `stable_worldmodel`).
The HEADLINE **per-cycle** latency and the **SR** are NOT in the benchmark output (they come
from the gated `src.sr_eval` run, same solves); both are joined here per precision via
`sr_overrides` — a `{track: {precision: {success_rate, per_cycle_latencies_ms}}}` dict (a plain
number value is accepted as SR-only, for manual overrides). Any still-unpaired row is flagged
SR-PENDING (not a validated win) and plots skip it. Per-cycle percentiles are taken AFTER
truncating each track to the common min-n across tracks (equal-n, SPEC §Interface Contracts).

All headline artifacts are persisted to `out_dir` on disk — each table serialized to a `.txt`,
each plot to a `.png` — so a completed study survives pod teardown (SPEC §Headline-artifact
durability). Logging to an open W&B run is optional and **additive, never the sole copy**.

The four single-method tables are **method-scoped by filename** (`<name>.<method>.txt`) and state
their calibration method in the table body, so rendering the other method neither clobbers them nor
leaves an unlabelled artefact behind (SPEC §Parity, architecture.md §7). The calibration table
is NOT method-scoped — it spans both methods by construction; only its `headline` marker moves.

Input shape: ``bench[track][precision] -> BenchResult`` (missing entries are skipped).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from statistics import fmean
from typing import NamedTuple

import matplotlib
import torch

matplotlib.use("Agg")  # headless: save figures, never open a window (pod + CI)
import matplotlib.pyplot as plt  # noqa: E402

from src.interfaces import (  # noqa: E402  — CEM per-cycle call counts (the decomposition weights)
    ENCODER_CALLS_PER_CYCLE as _ENCODER_CALLS,
    PREDICTOR_CALLS_PER_CYCLE as _PREDICTOR_CALLS,
    CALIBRATION_METHODS,
    DEFAULT_CALIBRATION_METHOD,
    EVAL_NUM_EPISODES,
    PER_CYCLE_WARMUP_DROP,
    QUANTIZED_PRECISIONS,
    check_calibration_method,
)

_TRACKS = ("lewm", "dino")
_PRECISIONS = ("fp32", "fp16", "int8", "fp8")
# The precisions whose SR does not depend on the PTQ calibration method — they build data-free off
# the base graph, so whichever run recorded them, the point is the same (SPEC §Parity). Used to let
# an `entropy` render join an fp32/fp16 SR that only a `max` run happened to record.
_METHOD_INVARIANT_PRECISIONS = tuple(p for p in _PRECISIONS if p not in QUANTIZED_PRECISIONS)
# Composite precision label written by the mixed-precision component-isolation runs
# (`src.sr_eval encoder_precision=/predictor_precision=`): `enc-<A>+pred-<B>`. Deliberately NOT in
# `_PRECISIONS`, so these diagnostic points reach no headline table, plot, or ratio — they are read
# ONLY by the isolation table (architecture.md §9).
_ISOLATION_KEY = re.compile(r"^enc-([a-z0-9]+)\+pred-([a-z0-9]+)$")
# The component held at FP16 while the other is quantized — the "undamaged" reference an isolation
# run's ΔSR is quoted against (architecture.md §7).
_ISOLATION_HELD = "fp16"


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt(x, spec: str = ".3g") -> str:
    return "—" if _missing(x) else format(x, spec)


def _sig(x, n: int) -> str:
    """`x` at exactly `n` significant figures, TRAILING ZEROS KEPT — `76.0` at n=3, not `76`: the
    trailing zero is a significant digit and reports the precision the number is quoted to, which
    plain `g` would strip. `#` forces them; the bare trailing `.` it leaves on a value with no
    fractional digits left to show (`100.`) comes back off.

    Every latency in every table is rendered at n=5 and every success rate at n=3 — significant
    figures, never a fixed decimal count, so one column serves a sub-ms component step and a
    multi-second cycle alike."""
    if _missing(x):
        return "—"
    s = format(x, f"#.{n}g")
    return s[:-1] if s.endswith(".") else s


def _percentile_ms(values, q: float) -> float:
    return torch.quantile(torch.tensor(values, dtype=torch.float64), q).item()


# --- calibration-method selection (shared by src.study / src.stats / this module) ------
def as_method_map(entry: dict, recorded_method: str = DEFAULT_CALIBRATION_METHOD) -> dict:
    """One stored per-precision entry -> `{method: payload}`.

    Every measured artefact keyed by precision — the benchmark numbers in `results.<track>.json`
    and the engine-step samples in `latencies.<track>.json` — is keyed by (precision, METHOD), so a
    quantized precision's two builds are two coexisting points rather than one overwriting the other
    (SPEC §Parity). A method map's values are themselves dicts; a flat entry's are floats or lists,
    which is how the two are told apart without a schema flag.

    `recorded_method` is the label a flat entry belongs to — the writing run's own
    `meta.calibration_method`, never a guess: folding an `entropy`-recorded measurement under `max`
    would mislabel a real measurement, the failure this keying exists to prevent."""
    if entry and all(isinstance(v, dict) for v in entry.values()):
        return entry
    return {recorded_method: entry}


def method_key(by_method: dict, precision: str, method: str) -> str | None:
    """WHICH label a `method` render reads out of a `{method: payload}` map, or None if none.

    **FP32/FP16 fall back across labels; quantized precisions never do.** The unquantized engines
    build data-free off the base graph — one build, no scales — so whichever run recorded them, the
    number describes the same engine and a label is only the stamp of the run that recorded it.
    An INT8/FP8 engine is a per-method BUILD, so falling back there would report one method's engine
    under the other's name — silently, and in the exact column a reader compares the methods in.

    Returned rather than only the payload so a consumer can RECORD which measurement it read
    (`src.stats` stamps it on the mean rows), instead of leaving the fallback implicit."""
    if by_method.get(method) is not None:
        return method
    if precision not in _METHOD_INVARIANT_PRECISIONS:
        return None
    for m in (DEFAULT_CALIBRATION_METHOD, *sorted(by_method)):
        if by_method.get(m) is not None:
            return m
    return None


def select_by_method(by_method: dict, precision: str, method: str):
    """The payload `method_key` selects, or None."""
    key = method_key(by_method, precision, method)
    return None if key is None else by_method[key]


# --- per-component decomposition (overhead by subtraction from the measured cycle) ----
def decompose(r: dict) -> dict:
    """One BenchResult → the per-cycle time decomposition (SPEC §Interface Contracts).

    Runs entirely on **MEANS**, not percentiles. The decomposition asserts
    `cycle = enc·calls + pred·calls + overhead`; linearity of expectation makes that identity
    exact for means under any distribution, whereas `p50(a+b) ≠ p50(a)+p50(b)` — a percentile
    decomposition would book its own non-additivity error as planner overhead. Amdahl is itself
    an expectation model, so `p` and the ceiling are mean-derived too. Reported/compared latency
    stays p50/p95 elsewhere; the mean appears in no headline.

    The engine-step means are weighted by the CEM per-cycle call counts into `enc_cyc`/`pred_cyc`;
    the **measured** mean per-cycle time (joined from the eval-shim run) is the cycle, and
    `overhead = cycle − enc_cyc − pred_cyc` (the un-optimizable floor: CEM planner + criterion
    + assembly + glue). A NEGATIVE overhead is surfaced loudly — never clamped — as a sign the
    call-count weighting or the isolated timing is off. `p = (enc+pred)/cycle` is the optimizable
    fraction; the Amdahl ceiling is `1/(1-p)`. When the cycle is not yet joined, the cycle-derived
    fields are None (the enc/pred model shares still stand).

    The warm-up asymmetry that used to bias this — enc/pred loops dropping `warmup` iters while the
    per-cycle callback recorded from the first decision of the first solve, so cold-start cost landed
    in the cycle mean and was booked entirely as overhead — is closed by
    `_finalize_per_cycle(warmup_drop=…)` (architecture.md §8). The excluded decisions are disclosed
    as the speed table's `drop×`, and `warmup_drop=0` reproduces the old, biased view.
    """
    enc_cyc = r["encode_mean_ms"] * _ENCODER_CALLS
    pred_cyc = r["predict_mean_ms"] * _PREDICTOR_CALLS
    model_cyc = enc_cyc + pred_cyc
    cycle = r["per_cycle_mean_ms"]
    out = {
        "enc_cyc_ms": enc_cyc,
        "pred_cyc_ms": pred_cyc,
        "model_cyc_ms": model_cyc,
        "cycle_ms": None,
        "overhead_ms": None,
        "optimizable_fraction": None,
        "amdahl_ceiling": None,
    }
    if _missing(cycle):
        return out
    overhead = cycle - model_cyc
    if overhead < 0:
        print(
            f"⚠ negative overhead ({overhead:.4f} ms): enc+pred model time exceeds the measured "
            "per-cycle time — the call-count weighting or the isolated engine-step timing is off "
            "(SPEC §Interface Contracts). Reporting as-is; do not trust the decomposition."
        )
    p = model_cyc / cycle
    out.update(
        cycle_ms=cycle,
        overhead_ms=overhead,
        optimizable_fraction=p,
        amdahl_ceiling=float("inf") if p >= 1.0 else 1.0 / (1.0 - p),
    )
    return out


# --- ratios (the LeWM-vs-DINOv3 headline) ---------------------------------------------
def per_cycle_ratio(bench: dict, precision: str, pct: str = "p50") -> float:
    """DINOv3 ÷ LeWM full **per-cycle** planning latency at `pct` — the headline speed ratio
    (how many× slower a DINO planning cycle is). Defaults to **p50**: the headline is a
    mechanistic claim about encoder compute, and p50 is the statistic this sample size supports
    (SPEC §Parity). p95 stays available as the descriptive tail. NaN until the per-cycle latency
    is joined from the gated eval-shim run."""
    key = f"per_cycle_{pct}_ms"
    d = bench["dino"][precision][key]
    l = bench["lewm"][precision][key]
    if _missing(d) or _missing(l):
        return math.nan
    return d / l


def _missing_sr_rows(bench: dict) -> list[str]:
    """`track-precision` labels whose SR is still unpaired (NaN) — the gated eval-shim join
    has not filled them. Used to flag speed numbers that are NOT yet validated wins."""
    rows = []
    for track in _TRACKS:
        for prec in _PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is not None and _missing(r["success_rate"]):
                rows.append(f"{track}-{prec}")
    return rows


def fp32_relative(bench: dict, track: str) -> dict[str, dict]:
    """Per precision: per-cycle **p50** speedup and SR delta **relative to FP32** for one track —
    a precision that is faster but degrades task quality must be visible (SPEC §Parity). Quoted
    at p50, the comparison basis, so it agrees with the headline ratio rather than introducing a
    second, tail-based notion of "speedup".

    Distinct from `dilution_disclosure`'s `measured_realized_speedup`, which is **mean**-based
    because it must reconcile against an Amdahl prediction. Same shape, different question — do
    not conflate them.

    `base` is guarded: `report` iterates both tracks unconditionally, so a single-track render
    (SPEC §Headline-artifact durability) reaches a track with no rows."""
    base = bench.get(track, {}).get("fp32")
    if base is None:
        return {}
    out: dict[str, dict] = {}
    for prec in _PRECISIONS:
        r = bench.get(track, {}).get(prec)
        if r is None:
            continue
        base_p50, r_p50 = base["per_cycle_p50_ms"], r["per_cycle_p50_ms"]
        out[prec] = {
            "per_cycle_p50_speedup_vs_fp32": (
                math.nan if _missing(base_p50) or _missing(r_p50) else base_p50 / r_p50
            ),
            "sr_delta_vs_fp32": (
                math.nan
                if _missing(r["success_rate"]) or _missing(base["success_rate"])
                else r["success_rate"] - base["success_rate"]
            ),
        }
    return out


def dilution_disclosure(bench: dict, track: str) -> dict:
    """Per track: the Amdahl dilution picture (SPEC §dilution disclosure). From the FP32
    decomposition comes the optimizable fraction `p` and the ceiling `1/(1-p)`. Per precision:
    the **model-only** speedup `s` (FP32 ÷ precision encode+predict model time), the **measured
    realized** speedup (FP32 ÷ precision per-cycle latency), and the Amdahl-**predicted** realized
    `1/((1-p)+p/s)`. The gap between model-only `s` and the realized is the overhead floor.
    Cycle-derived fields are None until the per-cycle latency is joined."""
    base = bench.get(track, {}).get("fp32")
    out: dict = {"optimizable_fraction": None, "amdahl_ceiling": None, "per_precision": {}}
    if base is None:
        return out
    base_dec = decompose(base)
    out["optimizable_fraction"] = base_dec["optimizable_fraction"]
    out["amdahl_ceiling"] = base_dec["amdahl_ceiling"]
    frac = out["optimizable_fraction"]
    # MEAN-based, like the rest of this block: `predicted_realized` below is derived from `p`,
    # which is mean-derived, so the measured counterpart must be too or the reconciliation
    # compares two different statistics and a units mismatch reads as an Amdahl-model failure.
    # The p50 FP32-relative speedup (the reported comparison) lives in `fp32_relative`.
    base_cycle = base["per_cycle_mean_ms"]
    for prec in _PRECISIONS:
        r = bench.get(track, {}).get(prec)
        if r is None:
            continue
        dec = decompose(r)
        s = base_dec["model_cyc_ms"] / dec["model_cyc_ms"]  # model-only speedup vs FP32
        measured_realized = (
            math.nan
            if _missing(base_cycle) or _missing(r["per_cycle_mean_ms"])
            else base_cycle / r["per_cycle_mean_ms"]
        )
        predicted_realized = None if frac is None else 1.0 / ((1.0 - frac) + frac / s)
        out["per_precision"][prec] = {
            "model_only_speedup": s,
            "predicted_realized_speedup": predicted_realized,
            "measured_realized_speedup": measured_realized,
        }
    return out


# --- tables ---------------------------------------------------------------------------
def _method_line(method: str) -> str:
    """The calibration-method label, carried INSIDE every single-method table body.

    SPEC §Parity: the label must survive into the persisted artefact, never stdout-only — a
    rendered table that does not name its method is not a valid artefact (architecture.md §7).
    Both the SR and (via the joined eval-shim run) the per-cycle percentiles are method-sourced,
    so this line qualifies the whole table, not just the SR column."""
    return (
        f"  calibration_method = {method}  (every int8/fp8 number in this table — SR, the per-cycle "
        "sample it was measured on, and the engine-step latencies — comes from that method's own "
        "runs; fp32/fp16 build data-free and are read across labels)"
    )


def _stats_lookup(payload: dict | None, track: str, precision: str, method: str) -> dict:
    """One point out of a `src.stats.compute` payload, or `{}` so a missing point renders as a blank
    cell rather than raising. A plain dict walk, deliberately NOT an import of `src.stats` — that
    module imports this one for the shared per-cycle sample rule, and the intervals must stay
    optional here (a render without them is still a valid artefact).

    **Falls back across methods for fp32/fp16 exactly as `_select_method` does.** `src.sr_eval`
    stamps every precision in a run with that run's method label, so a method-invariant fp32/fp16
    point recorded by a `max` run is filed under `max`; without the fallback an `entropy` render
    shows those rows an SR but no interval, purely from a label."""
    if not payload:
        return {}
    # Empty points are dropped before selection: `src.stats` files an entry for every (label, method)
    # it sees, including ones with neither an SR nor a sample, and a fallback that stopped at one of
    # those would report "no interval" while a usable point sat under the next label.
    by_method = {
        m: e for m, e in payload.get("points", {}).get(track, {}).get(precision, {}).items() if e
    }
    return select_by_method(by_method, precision, method) or {}


def _component_stats_lookup(
    payload: dict | None, track: str, precision: str, method: str, component: str
) -> dict:
    """One component point out of a `src.stats` payload's `points_components` section, or `{}` so a
    missing point renders as a blank cell.

    Selected by calibration method with `select_by_method`'s rule — an int8/fp8 engine is a
    per-method build, so its timing is that method's and never stands in for the other's, while
    fp32/fp16 fall back across labels because there is only one such engine to have timed."""
    if not payload:
        return {}
    by_method = payload.get("points_components", {}).get(track, {}).get(precision, {})
    return (select_by_method(by_method, precision, method) or {}).get(component, {})


def _ci(bounds, n: int = 3) -> str:
    """A `[lo,hi]` interval rendered for a fixed-width table, or "—" when the sample could not
    support one (`src.stats` returns None rather than inventing an interval).

    Both bounds carry the same significant figures as the point they bracket — `n` defaults to the
    SR's 3; latency bounds pass 5.

    NO space after the comma, deliberately: every cell in these tables is one whitespace-delimited
    token, which is what lets the artefacts be parsed with `split()`."""
    if not bounds:
        return "—"
    return f"[{_sig(bounds[0], n)},{_sig(bounds[1], n)}]"


def _ac_flag(point: dict) -> str:
    """Independence marker for one row: `*` when the Dwass lag-1 permutation test rejects at the
    UNADJUSTED p-value (the decision — SPEC §Interface Contracts), `-` when it does not, `—` when
    no sample was available to test. The interval beside a `*` is anti-conservative (too NARROW),
    so the flag is what a reader acts on. Holm values live in `stats.json` as secondary reporting
    and deliberately drive nothing here.

    Always emits a token — an empty cell would shift every column after it under `split()`."""
    if not point or point.get("lag1_p_permutation") is None:
        return "—"
    return "*" if point.get("lag1_reject") else "-"


def _mean_cell(value, bounds, flag: str | None = None, n: int = 5) -> str:
    """One mean quantity as ONE whitespace-delimited token: `18.334[18.112,18.601]*` — the point, its
    bootstrap interval, and (for the three quantities that ARE a sample) that sample's independence
    marker, in the same cell.

    `flag=None` is the deliberate case, not a missing one: `t_comp` and `overhead` are functions of
    two and three samples, and a flag describes a sample (SPEC §Interface Contracts), so they carry
    the constituent flags on the same row rather than a composite of their own."""
    if _missing(value):
        return "—"
    return f"{_sig(value, n)}{_ci(bounds, n)}{'' if flag is None else flag}"


def _mean_flag(entry: dict, prefix: str) -> str:
    """`*`/`-`/`—` for one constituent sample of a mean row, off the flag `src.stats` carried over
    from that sample's already-run lag-1 test (no test is re-run — architecture.md §12)."""
    if entry.get(f"{prefix}_lag1_p_permutation") is None:
        return "—"
    return "*" if entry.get(f"{prefix}_lag1_reject") else "-"


def render_latency_means_table(stats_payload: dict | None) -> str:
    """The five MEAN latency quantities per configuration, with bootstrap intervals and the
    inherited independence markers (SPEC §Interface Contracts, architecture.md §12).

    `enc`/`pred` are per ENGINE CALL — the scale the loops time them on; `t_comp`, `cycle` and `ovh`
    are per cycle, where the CEM call counts belong.

    A pure walk of `stats.json`'s `points_means` — deliberately NOT a recomputation off `bench`, so
    the rendered numbers and the persisted ones cannot drift; the three per-cycle quantities are the
    same numbers `decompose` reports, now with intervals.

    **Method-unscoped**: the config column names the method (`INT8 (max)`), so one table spans both
    and a render at either method writes the same file — like `calibration_table.txt`. Returns ""
    when the payload has no mean section, and the table is then simply not written."""
    points = (stats_payload or {}).get("points_means") or {}
    if not points:
        return ""
    hdr = (
        f"{'track':>6} {'config':>15} {'enc_call_ms':>30} {'pred_call_ms':>30} "
        f"{'t_comp_ms':>30} {'cycle_ms':>30} {'ovh_ms':>30}"
    )
    lines = [
        "  (MEAN basis. enc_call/pred_call = one engine call, UNWEIGHTED — the scale the "
        "fixed-iteration loops time them on)",
        f"  (per-cycle scale: t_comp = {_ENCODER_CALLS} × enc_call + {_PREDICTOR_CALLS} × "
        "pred_call; ovh = cycle − t_comp — those three POINTS add up, the intervals do not)",
        "  (each cell = point[lo,hi]: a 95% non-parametric percentile BOOTSTRAP interval, "
        "paired=False, over the same stored samples the p50 intervals use — construction + seed in "
        "stats.json)",
        "  (enc/pred sample = the fixed-iteration engine-step loop as recorded, timed on THAT "
        "method's engines; cycle sample = the warm-up-dropped, equal-n-truncated per-cycle vector "
        "from the same method's eval run)",
        "  (trailing * = that sample's Dwass lag-1 test REJECTS independence at the unadjusted p "
        "(interval too NARROW), - = does not, — = untested; t_comp and ovh carry no marker — a flag "
        "describes a sample, and they are functions of two and three)",
        "  (config = <precision> (<calibration method>), the method shown only where it applies — "
        "fp32/fp16 build data-free. It is the ONE column that may split into two tokens; the five "
        "value cells are always the LAST five)",
        hdr,
        "-" * len(hdr),
    ]
    for track in _TRACKS:
        by_label = points.get(track, {})
        for prec in _PRECISIONS:
            by_method = by_label.get(prec, {})
            for method in sorted(by_method, key=lambda m: (m != DEFAULT_CALIBRATION_METHOD, m)):
                e = by_method[method]
                lines.append(
                    f"{track:>6} {e['label']:>15} "
                    f"{_mean_cell(e['enc_mean_ms'], e['enc_ci95_ms'], _mean_flag(e, 'enc')):>30} "
                    f"{_mean_cell(e['pred_mean_ms'], e['pred_ci95_ms'], _mean_flag(e, 'pred')):>30} "
                    f"{_mean_cell(e['t_comp_mean_ms'], e['t_comp_ci95_ms']):>30} "
                    f"{_mean_cell(e['cycle_mean_ms'], e['cycle_ci95_ms'], _mean_flag(e, 'cycle')):>30} "
                    f"{_mean_cell(e['overhead_mean_ms'], e['overhead_ci95_ms']):>30}"
                )
    return "\n".join(lines)


def render_speed_table(
    bench: dict,
    method: str = DEFAULT_CALIBRATION_METHOD,
    warmup_drop: int = PER_CYCLE_WARMUP_DROP,
    stats_payload: dict | None = None,
) -> str:
    # All three latency distributions at p50/p95 (SPEC §Interface Contracts): per-cycle is the
    # HEADLINE (joined from the eval-shim; PEND until then) with **p50 the comparison basis** and
    # p95 the descriptive tail; enc/pred are the isolated engine-step components. SR is PEND
    # until the gated eval-shim pairs it. `cyc_n` is the post-truncation per-cycle sample count
    # those percentiles + the decomposition mean were computed from (architecture.md §8).
    hdr = (
        f"{'track':>6} {'prec':>5} {'cyc_p50':>8} {'cyc_p50_CI95':>22} {'ac':>3} "
        f"{'cyc_p95':>8} {'cyc_n':>6} {'drop×':>7} "
        f"{'enc_p50':>9} {'enc_p50_CI95':>21} {'enc_ac':>6} {'enc_p95':>9} "
        f"{'pred_p50':>9} {'pred_p50_CI95':>21} {'pred_ac':>7} {'pred_p95':>9} "
        f"{'mem_MB':>9} {'SR':>7} {'SR_CI95':>16}"
    )
    lines = [
        "  (cyc = per-cycle HEADLINE, joined from eval-shim; p50 = comparison basis, "
        "p95 = tail; enc/pred = engine step; PEND = gated eval-shim)",
        "  (cyc_n = equal-n truncated sample size — the common min across tracks at that "
        "precision; SR-dependent, hence why p50 carries the comparison)",
        f"  (drop× = the {warmup_drop} dropped warm-up decision(s) ÷ the retained cyc_p50 — the "
        "EXCLUSION disclosed, not hidden; ≈1 means the cold decision was unremarkable)",
        "  (CI95 = 95% interval on the ABSOLUTE value: exact binomial order-statistic for every p50 "
        "(cyc over cyc_n cycles; enc/pred over their fixed-iteration loop sample), Clopper-Pearson "
        f"for SR over {EVAL_NUM_EPISODES} episodes. Full construction + p-values in stats.json — "
        "architecture.md §12)",
        "  (the enc/pred loop sample needs no truncation and no warm-up drop — fixed-iteration, "
        "warm-up dropped at record time — so it is the vector as recorded)",
        "  (ac = * where the Dwass lag-1 permutation test REJECTS independence at the unadjusted p, "
        "- where it does not, — untested; a * interval is anti-conservative — too NARROW, not wide)",
        "  (no interval on a difference or a ratio — see the fp32-relative table — nor on any p95 "
        "or mean; the component/dilution tables are mean-based and carry none)",
        _method_line(method),
        hdr,
        "-" * len(hdr),
    ]
    for track in _TRACKS:
        for prec in _PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is None:
                continue
            sr = "PEND" if _missing(r["success_rate"]) else _sig(r["success_rate"], 3)
            n = r.get("_per_cycle_n")
            # Worst dropped decision relative to the retained p50 — with the default k=1 that IS
            # the cold decision; for k>1 the max is the conservative disclosure.
            dropped = r.get("_per_cycle_dropped_ms") or []
            p50 = r["per_cycle_p50_ms"]
            drop_x = max(dropped) / p50 if dropped and not _missing(p50) and p50 else None
            point = _stats_lookup(stats_payload, track, prec, method)
            # Component intervals are keyed by (track, precision, method): a quantized engine is a
            # per-method build, so its step loop is that method's measurement (SPEC §Parity).
            enc_pt = _component_stats_lookup(stats_payload, track, prec, method, "encode")
            pred_pt = _component_stats_lookup(stats_payload, track, prec, method, "predict")
            lines.append(
                f"{track:>6} {prec:>5} "
                f"{_sig(r['per_cycle_p50_ms'], 5):>8} "
                f"{_ci(point.get('p50_ci95_ms'), 5):>22} {_ac_flag(point):>3} "
                f"{_sig(r['per_cycle_p95_ms'], 5):>8} "
                f"{('—' if n is None else str(n)):>6} {_fmt(drop_x, '.2f'):>7} "
                f"{_sig(r['encode_p50_ms'], 5):>9} {_ci(enc_pt.get('p50_ci95_ms'), 5):>21} "
                f"{_ac_flag(enc_pt):>6} {_sig(r['encode_p95_ms'], 5):>9} "
                f"{_sig(r['predict_p50_ms'], 5):>9} {_ci(pred_pt.get('p50_ci95_ms'), 5):>21} "
                f"{_ac_flag(pred_pt):>7} {_sig(r['predict_p95_ms'], 5):>9} "
                f"{r['peak_mem_mb']:>9.1f} {sr:>7} {_ci(point.get('sr_ci95_pct')):>16}"
            )
    return "\n".join(lines)


def render_fp32_relative_table(bench: dict, method: str = DEFAULT_CALIBRATION_METHOD) -> str:
    """FP32-relative degradation per track × precision: per-cycle p50 speedup **and** SR delta,
    side by side — SPEC §Parity requires a precision that is faster but degrades task quality to
    be visible, which means both numbers in one row."""
    hdr = f"{'track':>6} {'prec':>5} {'cyc_p50_speedup':>16} {'ΔSR_vs_fp32':>12}"
    lines = [
        "  (vs that track's FP32; speedup = FP32 p50 ÷ this p50, >1 = faster; "
        "ΔSR in percentage points, <0 = task quality lost)",
        _method_line(method),
        hdr,
        "-" * len(hdr),
    ]
    for track in _TRACKS:
        for prec, v in fp32_relative(bench, track).items():
            lines.append(
                f"{track:>6} {prec:>5} "
                f"{_fmt(v['per_cycle_p50_speedup_vs_fp32'], '.3f'):>16} "
                f"{_fmt(v['sr_delta_vs_fp32'], '+.1f'):>12}"
            )
    return "\n".join(lines)


def render_component_table(bench: dict, method: str = DEFAULT_CALIBRATION_METHOD) -> str:
    # Runtime-WEIGHTED per-cycle shares (step MEAN × CEM call counts); overhead by subtraction
    # from the measured mean cycle. `p` = optimizable fraction, ceiling = 1/(1-p).
    hdr = (
        f"{'track':>6} {'prec':>5} {'enc_cyc_ms':>11} {'pred_cyc_ms':>12} "
        f"{'ovh_cyc_ms':>11} {'p':>7} {'ceil×':>7}"
    )
    lines = [
        "  (MEAN basis — means compose additively, percentiles do not; reported latency is "
        "p50/p95 above)",
        "  (cyc_ms = per-cycle = step mean × calls; predict called "
        f"{_PREDICTOR_CALLS} × / cycle, encode {_ENCODER_CALLS} ×; ovh = cycle − enc − pred)",
        _method_line(method),
        hdr,
        "-" * len(hdr),
    ]
    for track in _TRACKS:
        for prec in _PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is None:
                continue
            d = decompose(r)
            lines.append(
                f"{track:>6} {prec:>5} "
                f"{_sig(d['enc_cyc_ms'], 5):>11} {_sig(d['pred_cyc_ms'], 5):>12} "
                f"{_sig(d['overhead_ms'], 5):>11} {_fmt(d['optimizable_fraction'], '.3f'):>7} "
                f"{_fmt(d['amdahl_ceiling'], '.2f'):>7}"
            )
    return "\n".join(lines)


def render_dilution_table(bench: dict, method: str = DEFAULT_CALIBRATION_METHOD) -> str:
    """Amdahl dilution table: p, ceiling, and per-precision model-only vs realized speedup —
    makes the overhead floor that dilutes the model-only ratio visible."""
    lines = [
        "  (MEAN basis throughout — Amdahl is an expectation model; the reported p50 "
        "FP32-relative speedup is in the fp32-relative table)",
        _method_line(method),
    ]
    for track in _TRACKS:
        d = dilution_disclosure(bench, track)
        if d["optimizable_fraction"] is None and not d["per_precision"]:
            continue
        lines.append(
            f"{track}: optimizable fraction p={_fmt(d['optimizable_fraction'], '.3f')}  "
            f"Amdahl ceiling 1/(1-p)={_fmt(d['amdahl_ceiling'], '.2f')}×"
        )
        hdr = f"    {'prec':>5} {'model_only_s':>13} {'predicted_realized':>19} {'measured_realized':>18}"
        lines.append(hdr)
        for prec, v in d["per_precision"].items():
            lines.append(
                f"    {prec:>5} {_fmt(v['model_only_speedup'], '.2f'):>13} "
                f"{_fmt(v['predicted_realized_speedup'], '.2f'):>19} "
                f"{_fmt(v['measured_realized_speedup'], '.2f'):>18}"
            )
    return "\n".join(lines)


def _sr_value(point):
    """The success rate out of one sr.json point — a `{success_rate, ...}` dict, a bare number
    (manual override), or None when the point is absent."""
    if point is None:
        return None
    return point.get("success_rate") if isinstance(point, dict) else point


def render_calibration_table(
    overrides: dict | None,
    headline_method: str = DEFAULT_CALIBRATION_METHOD,
    stats_payload: dict | None = None,
) -> str:
    """The ONE table that spans calibration methods: int8/fp8 SR under `max` AND `entropy`, side
    by side (architecture.md §7).

    The four single-method tables answer "what does the study report"; this one answers "what did
    the calibration method buy". Keeping them separate preserves the guard that a single-method
    render can never place `dino int8@max` beside `lewm int8@entropy` — the cross-track comparison
    SPEC §Parity forbids — while still putting both measured points on the page.

    Reads `sr.json` directly (`{track: {precision: {method: SR}}}`) rather than `bench`, because
    `bench` holds only the ONE method the render selected. No `results.<track>.json` schema change.
    The `headline` column names the method the single-method tables were rendered at, linking the
    two artefacts. FP32/FP16 are excluded: they are method-invariant, so a comparison is vacuous.

    Returns "" when no quantized SR exists at all — the caller then writes no artefact rather than
    an empty one."""
    delta_label = f"Δ({CALIBRATION_METHODS[-1][:3]}−{CALIBRATION_METHODS[0][:3]})"
    hdr = (
        f"{'track':>6} {'prec':>5} "
        + " ".join(f"{'SR@' + m:>11} {'CI95@' + m:>16}" for m in CALIBRATION_METHODS)
        + f" {delta_label:>12} {'headline':>9}"
    )
    rows = []
    for track in _TRACKS:
        for prec in QUANTIZED_PRECISIONS:
            raw = (overrides or {}).get(track, {}).get(prec)
            if raw is None:
                continue
            srs = {m: _sr_value(_select_method(raw, m, prec)) for m in CALIBRATION_METHODS}
            if all(v is None for v in srs.values()):
                continue
            first, last = srs[CALIBRATION_METHODS[0]], srs[CALIBRATION_METHODS[-1]]
            delta = None if first is None or last is None else last - first
            rows.append(
                f"{track:>6} {prec:>5} "
                + " ".join(
                    f"{('PEND' if srs[m] is None else _sig(srs[m], 3)):>11} "
                    f"{_ci(_stats_lookup(stats_payload, track, prec, m).get('sr_ci95_pct')):>16}"
                    for m in CALIBRATION_METHODS
                )
                + f" {_fmt(delta, '+.1f'):>12} {headline_method:>9}"
            )
    if not rows:
        return ""
    return "\n".join(
        [
            "  (int8/fp8 only — fp32/fp16 are method-invariant. PEND = that method's engines were "
            "not built/evaluated for this cell)",
            "  (`headline` = the method the single-method tables were rendered at; this table "
            "compares SR alone — each method's latencies are in its own method-scoped speed table)",
            f"  (CI95 = Clopper-Pearson 95% on each ABSOLUTE SR over {EVAL_NUM_EPISODES} episodes; "
            f"{delta_label} carries none — no interval on a difference. Overlapping intervals do "
            "NOT by themselves settle whether the methods differ.)",
            hdr,
            "-" * len(hdr),
        ]
        + rows
    )


def _parse_isolation_key(key: str):
    """`enc-<A>+pred-<B>` -> `(encoder_precision, predictor_precision)`, or None for a normal
    precision key. Written by the mixed-precision component-isolation runs (`src.sr_eval`)."""
    m = _ISOLATION_KEY.match(key)
    return m.groups() if m else None


def render_isolation_table(
    bench: dict,
    overrides: dict | None,
    method: str = DEFAULT_CALIBRATION_METHOD,
    stats_payload: dict | None = None,
) -> str:
    """Component-precision isolation (architecture.md §9): which component's quantization caused a measured
    SR drop. Placed immediately after the FP32-relative table — it answers the question that table
    provokes.

    Each row is one diagnostic run with ONE component quantized and the other held at FP16 (the
    undamaged reference — architecture.md §7). ΔSR is quoted against that track's **FP16**
    row, not FP32, because FP16 is what the held component is running at; that makes it a different
    number from the FP32-relative table's ΔSR, deliberately.

    `cyc_share` is the isolated component's share of the measured per-cycle time AT THAT PRECISION
    (its step mean × CEM call count ÷ the joined cycle) — diagnostic context for how much latency
    the damage was buying. It is "—" until the pure-precision per-cycle latency is joined.

    These rows come from composite `enc-<A>+pred-<B>` sr.json keys that are NOT in `_PRECISIONS`,
    so they reach no headline table, plot, or ratio — the mixed pairing is a diagnostic, never a
    fifth precision (architecture.md §9). Returns "" when no isolation runs exist.

    ONE table per calibration method (`isolation_table.<method>.txt`). The diagnostic is run at
    BOTH `max` and `entropy` (SPEC §Requirements) and a row only explains a headline row rendered
    at the SAME method, so a method with no isolation runs of its own renders no table at all
    rather than borrowing the other method's rows."""
    order = {p: i for i, p in enumerate(_PRECISIONS)}
    rows = []
    for track in _TRACKS:
        for key, raw in (overrides or {}).get(track, {}).items():
            parsed = _parse_isolation_key(key)
            if parsed is None:
                continue
            enc_p, pred_p = parsed
            # Isolation points are method-DEPENDENT (they are quantized runs), so `_select_method`
            # must not fall back across methods — the composite key is not method-invariant.
            sr = _sr_value(_select_method(raw, method, key))
            if sr is None:
                continue
            quantized = [(c, p) for c, p in (("encoder", enc_p), ("predictor", pred_p))
                         if p != _ISOLATION_HELD]
            if not quantized:
                continue  # both held at FP16 == the FP16 baseline; already the speed table's row
            if len(quantized) == 1:
                component, prec = quantized[0]
            else:  # both sides quantized — not a single-component isolation; no share attributable
                component, prec = "enc+pred", f"{enc_p}+{pred_p}"
            base = bench.get(track, {}).get(_ISOLATION_HELD, {}).get("success_rate")
            delta = None if base is None or _missing(base) else sr - base
            share = None
            if len(quantized) == 1 and prec in bench.get(track, {}):
                d = decompose(bench[track][prec])
                cycle = d["cycle_ms"]
                if cycle:
                    key_ms = "enc_cyc_ms" if component == "encoder" else "pred_cyc_ms"
                    share = d[key_ms] / cycle
            point = _stats_lookup(stats_payload, track, key, method)
            rows.append(
                (
                    _TRACKS.index(track), order.get(prec, len(order)), component,
                    f"{track:>6} {prec:>9} {component:>10} {_sig(sr, 3):>7} "
                    f"{_ci(point.get('sr_ci95_pct')):>16} "
                    f"{_fmt(delta, '+.1f'):>12} {_fmt(share, '.3f'):>10}",
                )
            )
    if not rows:
        return ""
    hdr = (
        f"{'track':>6} {'prec':>9} {'quantized':>10} {'SR':>7} {'SR_CI95':>16} "
        f"{'ΔSR_vs_fp16':>12} {'cyc_share':>10}"
    )
    return "\n".join(
        [
            "  (DIAGNOSTIC — mixed-precision component isolation; the OTHER component is held at "
            "fp16. Never a reported configuration, never in the headline sweep.)",
            "  (ΔSR is vs that track's FP16 row — the held component's precision — NOT vs FP32. "
            "cyc_share = that component's share of the measured cycle at that precision.)",
            f"  (SR_CI95 = Clopper-Pearson 95% interval on the ABSOLUTE SR over "
            f"{EVAL_NUM_EPISODES} episodes; ΔSR deliberately carries none — no interval on a "
            "difference, architecture.md §12)",
            f"  calibration_method = {method}  (isolation runs are method-dependent; a row only "
            "explains a headline row rendered at the SAME method)",
            hdr,
            "-" * len(hdr),
        ]
        + [r[-1] for r in sorted(rows, key=lambda r: r[:3])]
    )


# --- plots ----------------------------------------------------------------------------
# Static report/print PNGs on the light surface (matplotlib Agg). Validated categorical hues in
# FIXED order so the figure set reads as one system (dataviz skill): colour follows the ENTITY
# (track / component), never rank. Sentence-case titles, UPPERCASE precisions, black axes + axis
# values; grey is reserved for grid/leader lines on the bar charts.
# DINO is RED (not orange) so the track colours stay clear of the component palette's orange
_TRACK_COLOR = {"lewm": "#2a78d6", "dino": "#e34948"}  # blue / red (slots 1, 8)
_TRACK_DISPLAY = {"lewm": "LeWM", "dino": "DINOv3-WM"}
_PREC_DISPLAY = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8", "fp8": "FP8"}
_PRECISION_MARKER = {"fp32": "o", "fp16": "s", "int8": "^", "fp8": "D"}
# The quantized precisions are plotted ONCE PER CALIBRATION METHOD in the same panel, so a shape per
# precision is not enough — each (precision, method) pair needs its own. Kept in one family per
# precision (triangles = INT8, diamond/plus = FP8) with the `entropy` shapes unchanged, so the
# earlier single-method figures still read the same.
_QUANTIZED_MARKER = {
    ("int8", "max"): "v", ("int8", "entropy"): "^",
    ("fp8", "max"): "P", ("fp8", "entropy"): "D",
}
_RATIO_HUE = "#81c784"  # light green — distinct from the component encoder green and the track hues
# encoder green / predictor purple / overhead orange (validated slots 3, 7, 2)
_COMPONENT_COLOR = {"encoder": "#1baf7a", "predictor": "#4a3aa7", "overhead": "#eb6834"}
_GRID = "#e1e0d9"
_MUTED = "#898781"
_INK = "#0b0b0b"

_SERIF_PREFERRED = "Nimbus Roman"  # URW Times clone; apt `fonts-urw-base35`, installed by setup.sh
_SERIF_BUNDLED = "STIXGeneral"  # Times-metric, SHIPS WITH matplotlib — always resolvable


def _serif_rc() -> dict:
    """rcParams for the speed-vs-SR figure's serif typography (owner request) — the rest of the
    figure set keeps the default sans, so this is scoped to that one render via `rc_context`.

    Resolved at render time rather than fixed as a constant, because the two rc groups fall back
    DIFFERENTLY: `font.serif` walks its list, but `mathtext.rm` takes a single font NAME with no
    list and resolves a missing one to **DejaVu Sans**. A host without Nimbus Roman would therefore
    set the label serif and the `$p_{50}$` in it sans — a silently inconsistent figure, logged by
    `findfont` but raising nothing. Picking the family first and mapping mathtext to whatever won
    keeps text and math in one face on any host: the real Nimbus where setup.sh has installed it,
    matplotlib's bundled STIXGeneral (also Times-metric) where it has not."""
    from matplotlib import font_manager

    if _SERIF_PREFERRED in {f.name for f in font_manager.fontManager.ttflist}:
        return {
            "font.family": "serif",
            "font.serif": [_SERIF_PREFERRED],
            "mathtext.fontset": "custom",
            "mathtext.rm": _SERIF_PREFERRED,
            "mathtext.it": f"{_SERIF_PREFERRED}:italic",
            "mathtext.bf": f"{_SERIF_PREFERRED}:bold",
            # Unused by this figure, but the `custom` fontset resolves EVERY math family up front
            # and the default `cursive` matches nothing here — a findfont log line per render.
            "mathtext.cal": f"{_SERIF_PREFERRED}:italic",
        }
    return {"font.family": "serif", "font.serif": [_SERIF_BUNDLED], "mathtext.fontset": "stix"}


def _prec_label(prec: str, method: str) -> str:
    """Display label for a precision. Quantized precisions carry the calibration method in
    brackets (their SR depends on it); FP32/FP16 do not (method-invariant)."""
    return f"{_PREC_DISPLAY[prec]} ({method})" if prec in QUANTIZED_PRECISIONS else _PREC_DISPLAY[prec]


def _prec_marker(prec: str, method: str) -> str:
    """Marker for one plotted point. Quantized precisions are shape-keyed by (precision, method) —
    both methods share a panel — everything else by precision alone."""
    if prec not in QUANTIZED_PRECISIONS:
        return _PRECISION_MARKER[prec]
    return _QUANTIZED_MARKER[(prec, method)]


def _fmt_time_ms(ms) -> str:
    """A duration in ms, rendered ms below 1 s and seconds above, so a short component time and a
    long cycle both read cleanly."""
    if _missing(ms):
        return "—"
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.3g} ms"


def _style(ax, *, grid_axis: str = "y") -> None:
    """Recessive grey grid, BLACK spines + tick values — the shared chrome across the figure set."""
    ax.grid(axis=grid_axis, color=_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK)
    ax.tick_params(colors=_INK)
    for item in (ax.xaxis.label, ax.yaxis.label, ax.title):
        item.set_color(_INK)


def _asym_err(value, bounds):
    """A `[lo, hi]` interval → matplotlib's asymmetric `(2, 1)` error array around `value`, or None.
    Clamped at zero: the order-statistic and Clopper-Pearson intervals are NOT symmetric about the
    point (architecture.md §12), and an interpolated p50 can even fall marginally outside its own
    order-statistic bracket — a negative bar length would raise rather than render."""
    if not bounds or _missing(value):
        return None
    return [[max(0.0, value - bounds[0])], [max(0.0, bounds[1] - value)]]


class _Point(NamedTuple):
    """One plotted speed-vs-SR marker."""

    x_ms: float
    sr: float
    xerr: list | None
    yerr: list | None
    marker: str
    label: str
    zorder: float


def _speed_vs_sr_points(bench: dict, track: str, method: str, stats_payload: dict | None):
    """One panel's plotted points, in precision order: FP32/FP16 once (method-invariant), and each
    quantized precision ONCE PER CALIBRATION METHOD so `max` and `entropy` coexist in the panel —
    the same both-methods-on-the-page treatment `calibration_table.txt` gives the SR (architecture.md §7).
    Being a within-track view, it does not place a `max` point beside an `entropy` one ACROSS tracks;
    every method appears in both panels, so the cross-track reading stays like-for-like (SPEC §Parity).

    The render's own `method` is read off the joined `bench` row; the other method's quantized points
    come from `stats.json`, whose `per_cycle_p50_ms` is built from `report.per_cycle_samples` — the
    same warm-up-dropped, equal-n-truncated sample `_finalize_per_cycle` reduces — so the two sources
    cannot disagree, and no result is recomputed here. Without a stats payload only the joined
    method's points are available.

    Yields `_Point`s in LEGEND order. Draw order is carried separately as `zorder`, because the two
    methods of a quantized precision differ only in their PTQ scales and so land nearly on top of
    each other: whichever draws second hides the other. Yield order puts `max` first to match
    `CALIBRATION_METHODS` and the calibration table's columns, while `zorder` puts it on top, where
    the shape it covers is the one that still shows around the edges."""
    for prec in _PRECISIONS:
        methods = CALIBRATION_METHODS if prec in QUANTIZED_PRECISIONS else (method,)
        for i, m in enumerate(methods):
            point = _stats_lookup(stats_payload, track, prec, m)
            r = bench.get(track, {}).get(prec) or {} if m == method else {}
            x, y = r.get("per_cycle_p50_ms"), r.get("success_rate")
            if _missing(x) or _missing(y):
                # The off-method quantized point, and any config whose engines this render has no
                # benchmark row for: both axes are eval-shim quantities, so the stats payload
                # carries them whether or not the component benchmark has been run at this method.
                x, y = point.get("per_cycle_p50_ms"), point.get("success_rate")
            if _missing(x) or _missing(y):
                continue
            # 95% intervals on BOTH absolute axes: x = exact binomial order-statistic on the
            # per-cycle p50, y = Clopper-Pearson on the SR (architecture.md §12). Absent intervals
            # simply draw no bar.
            yield _Point(x, y, _asym_err(x, point.get("p50_ci95_ms")),
                         _asym_err(y, point.get("sr_ci95_pct")),
                         _prec_marker(prec, m), _prec_label(prec, m), 4 - i)


def _render_speed_vs_sr(
    bench: dict, path: Path, method: str, title: str | None, stats_payload: dict | None = None
) -> Path:
    """One speed-vs-SR figure. Two panels (LeWM | DINOv3-WM) with a SHARED y-axis but SEPARATE
    linear x-axes — the cross-track latency gap that a single linear axis would collapse is handled
    by faceting, so no log scale is needed. Marker = precision (and, for the quantized ones, the
    calibration method), with NO connecting line (the points are discrete, not a continuum). Each
    panel carries its OWN legend, coloured in that panel's track hue. `title` None omits the figure
    title (RESULTS.md); a title is passed for the README headline copy.

    Set in a serif face (`_serif_rc`) via `rc_context`, so the typography is scoped to this figure
    and the rest of the set is untouched. Every text artist — including `tight_layout`'s metrics —
    must be created inside the context, hence the whole body sits in the `with`."""
    from matplotlib.lines import Line2D

    with plt.rc_context(_serif_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        legend_loc = {"lewm": "lower left", "dino": "upper left"}  # each clears its panel's points
        for ax, track in zip(axes, _TRACKS):
            handles = []
            hue = _TRACK_COLOR[track]
            for p in _speed_vs_sr_points(bench, track, method, stats_payload):
                # Error bars UNDER the markers in a recessive grey, so the points still read first.
                if p.xerr or p.yerr:
                    ax.errorbar(p.x_ms, p.sr, xerr=p.xerr, yerr=p.yerr, fmt="none", ecolor=_MUTED,
                                elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=2)
                ax.scatter(p.x_ms, p.sr, marker=p.marker, s=90, color=hue,
                           edgecolor="white", linewidth=0.8, zorder=p.zorder)
                handles.append(Line2D([], [], marker=p.marker, ls="", color=hue, label=p.label))
            ax.set_title(_TRACK_DISPLAY[track])
            ax.set_xlabel("Per-cycle latency $p_{50}$ (ms)")
            _style(ax, grid_axis="both")
            if handles:
                ax.legend(handles=handles, title="Precision", fontsize=7.5, loc=legend_loc[track],
                          borderpad=0.6, labelspacing=0.35, handletextpad=0.4)
        axes[0].set_ylabel("Success rate (%)")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return path


def plot_speed_vs_sr(
    bench: dict,
    out_dir: Path,
    method: str = DEFAULT_CALIBRATION_METHOD,
    stats_payload: dict | None = None,
) -> Path:
    """Speed vs SR, rendered twice: untitled `speed_vs_sr.png` (for RESULTS.md, which titles it in
    prose) and titled `speed_vs_sr.titled.png` (the README headline copy). Returns the untitled
    path (the canonical RESULTS artefact).

    Unlike the four single-method tables these two filenames are NOT method-scoped, and do not need
    to be: the panels carry BOTH calibration methods' quantized points (`_speed_vs_sr_points`) and
    FP32/FP16 are method-invariant, so re-rendering at the other `method` reproduces the same figure
    rather than clobbering a different one (SPEC §Parity, CLAUDE §8)."""
    _render_speed_vs_sr(bench, out_dir / "speed_vs_sr.titled.png", method,
                        title="Per-cycle planning latency vs success rate",
                        stats_payload=stats_payload)
    return _render_speed_vs_sr(bench, out_dir / "speed_vs_sr.png", method, title=None,
                               stats_payload=stats_payload)


def plot_per_cycle_ratio(bench: dict, out_dir: Path) -> Path:
    """DINOv3-WM ÷ LeWM per-cycle p50 latency ratio per precision. RESULTS.md only, so NO figure
    title; one series → orange bars, no legend; zero baseline KEPT (truncating a ratio axis
    misleads); each bar value-labelled so the spread is exact rather than muted.

    Measured values only, and no caveat text on the PNG (owner ruling, 2026-07-26): the bar-to-bar
    (cross-precision) spread partly reflects the differential throttle (architecture.md §11), but
    that caveat lives in the surrounding prose and the derived ratio table, never on the figure."""
    vals = {
        p: per_cycle_ratio(bench, p, "p50")
        for p in _PRECISIONS
        if p in bench.get("lewm", {}) and p in bench.get("dino", {})
    }
    precs = [p for p in _PRECISIONS if p in vals and not _missing(vals[p])]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([_PREC_DISPLAY[p] for p in precs], [vals[p] for p in precs],
                  color=_RATIO_HUE, width=0.6, zorder=2)
    for b, p in zip(bars, precs):
        ax.annotate(f"{vals[p]:.0f}×", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=_INK)
    ax.set_ylabel("p50 latency ratio (DINOv3-WM relative to LeWM)")
    _style(ax, grid_axis="y")
    fig.tight_layout()
    path = out_dir / "per_cycle_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_component_breakdown(bench: dict, out_dir: Path, precision: str = "fp32") -> Path:
    """Per-track encoder/predictor/overhead split at one precision, as TWO panels each NORMALISED
    to 100% of that track's own cycle. The absolute cross-track gap otherwise collapses the faster
    track to a flat line and hides the slower track's encoder; normalising makes both compositions
    readable, and the absolute cycle time is kept in each panel title so the gap is not erased.
    Runtime-WEIGHTED (step mean × CEM call counts) with overhead by subtraction. Each component is
    a LEADER-LINE callout (so a component that renders as a sliver is still labelled), the callouts
    fanning outward per panel to avoid collisions; callout text is black, leader lines grey.

    No caveat text on the PNG (owner ruling, 2026-07-26): the overhead slice's absolute ms is
    measured on unlocked clocks and an overhead share below the run-to-run clock mismatch is
    bounded small, not resolved (architecture.md §11) — that caveat lives in the surrounding
    prose and the derived overhead table, never on the figure."""
    tracks = [t for t in _TRACKS if precision in bench.get(t, {})]
    decs = {t: decompose(bench[t][precision]) for t in tracks}
    segments = (("Encoder", "enc_cyc_ms"), ("Predictor", "pred_cyc_ms"), ("Overhead", "overhead_ms"))
    anchor_y = {"Encoder": 12, "Predictor": 50, "Overhead": 88}  # spread callout heights, arrows to true centre
    n = max(len(tracks), 1)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.5), squeeze=False)
    for i, (ax, t) in enumerate(zip(axes[0], tracks)):
        d = decs[t]
        total = d["cycle_ms"] or d["model_cyc_ms"]  # cycle if joined, else enc+pred only
        outward_right = len(tracks) == 1 or i == len(tracks) - 1  # last panel fans right, first fans left
        tip_x, text_x, ha = (0.35, 0.62, "left") if outward_right else (-0.35, -0.62, "right")
        ax.set_xlim((-0.5, 2.0) if outward_right else (-2.0, 0.5))
        bottom = 0.0
        for label, key in segments:
            ms = d[key]
            if ms is None:
                continue
            pct = 100.0 * ms / total
            ax.bar(0, pct, bottom=bottom, width=0.7, color=_COMPONENT_COLOR[label.lower()],
                   edgecolor="white", linewidth=1.2, zorder=2)
            ax.annotate(f"{label} ({_fmt_time_ms(ms)}, {pct:.0f}%)",
                        xy=(tip_x, bottom + pct / 2), xytext=(text_x, anchor_y[label]),
                        ha=ha, va="center", fontsize=8, color=_INK,
                        arrowprops=dict(arrowstyle="-", color=_MUTED, lw=0.8))
            bottom += pct
        cyc = d["cycle_ms"]
        ax.set_title(f"{_TRACK_DISPLAY[t]}\n{_fmt_time_ms(cyc) if cyc else 'cycle pending'} / cycle",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_ylim(0, 100)
        _style(ax, grid_axis="y")
    axes[0][0].set_ylabel("Share of per-cycle time (%)")
    fig.tight_layout()
    path = out_dir / f"component_breakdown_{precision}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- component analysis (the five mean latency quantities, one figure each) ------------
# The columns of `latency_means_table.txt`, as (file stem = that column's name, `points_means` key
# prefix, figure title). One figure per quantity — five in all — so each is read on its own scale.
_MEAN_QUANTITIES = (
    ("enc_call_ms", "enc", "Mean encode-step latency per engine call"),
    ("pred_call_ms", "pred", "Mean predictor-step latency per engine call"),
    ("t_comp_ms", "t_comp", "Mean component latency per cycle"),
    ("cycle_ms", "cycle", "Mean per-cycle latency"),
    ("ovh_ms", "overhead", "Mean overhead per cycle"),
)
# x-axis order for those figures (SPEC §Uncertainty quantification): FP32, FP16, then the quantized
# precisions with FP8 BEFORE INT8, each once per calibration method — so `INT8 (entropy)` is the
# rightmost category. Deliberately not `_PRECISIONS`, whose order is the sweep's, not this axis's.
_MEAN_PLOT_PRECISIONS = ("fp32", "fp16", "fp8", "int8")


def _mean_configs() -> list[tuple[str, str]]:
    """Every x category as `(label, marker)`, in the fixed axis order above. `_prec_label` renders
    the same string `stats.config_label` writes into `points_means`, which is what the panel lookup
    keys on; the method-invariant precisions carry the default method only to pick that label and
    marker, never to select a point (they have one row whichever run recorded it)."""
    return [
        (_prec_label(prec, m), _prec_marker(prec, m))
        for prec in _MEAN_PLOT_PRECISIONS
        for m in (
            CALIBRATION_METHODS
            if prec in QUANTIZED_PRECISIONS
            else (DEFAULT_CALIBRATION_METHOD,)
        )
    ]


def _mean_panel(points: dict, track: str) -> dict:
    """One panel's points as `{config label: mean entry}` — a pure read of `points_means`, keyed by
    the same label the mean table prints, so figure and table cannot name a configuration
    differently."""
    return {
        e["label"]: e
        for prec in _MEAN_PLOT_PRECISIONS
        for e in (points.get(track, {}).get(prec) or {}).values()
    }


def plot_component_analysis(stats_payload: dict | None, out_dir: Path) -> dict[str, Path]:
    """The five mean latency quantities as five dot plots, written to `out_dir/component_analysis/`
    — a subdirectory, so this figure set is ADDITIVE and overwrites no existing plot, table or
    result (CLAUDE §8). Returns `{stem: path}`, empty when the payload carries no mean section.

    Two panels per figure (LeWM | DINOv3-WM) on SEPARATE y-axes: the cross-track latency gap that
    one shared scale would collapse is handled by faceting, the same treatment (and the same serif
    typography, chrome and grey error bars) as the speed-vs-SR figure. x = the configuration, in
    `_mean_configs` order; y = that quantity's mean with its 95% bootstrap interval as the error
    bar. No legend: the x tick labels already name every configuration, so one would only restate
    the axis. Marker still follows (precision, method), keeping the shapes consistent across the
    figure set.

    A pure walk of `stats.json`'s `points_means`, exactly like `render_latency_means_table` — the
    plotted numbers are the persisted ones, never a recomputation off `bench` — and likewise
    method-unscoped: the x labels name the method, so either render writes the same five figures."""
    points = (stats_payload or {}).get("points_means") or {}
    if not points:
        return {}
    sub_dir = Path(out_dir) / "component_analysis"
    sub_dir.mkdir(parents=True, exist_ok=True)
    panels = {t: _mean_panel(points, t) for t in _TRACKS}
    # One shared category list across both panels, so a configuration sits at the same x in each.
    configs = [(lbl, mk) for lbl, mk in _mean_configs() if any(lbl in p for p in panels.values())]
    paths = {}
    with plt.rc_context(_serif_rc()):
        for stem, key, title in _MEAN_QUANTITIES:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            for ax, track in zip(axes, _TRACKS):
                hue = _TRACK_COLOR[track]
                for x, (label, marker) in enumerate(configs):
                    entry = panels[track].get(label)
                    value = None if entry is None else entry[f"{key}_mean_ms"]
                    if _missing(value):
                        continue
                    err = _asym_err(value, entry[f"{key}_ci95_ms"])
                    if err:
                        ax.errorbar(x, value, yerr=err, fmt="none", ecolor=_MUTED, elinewidth=0.9,
                                    capsize=2.5, capthick=0.9, zorder=2)
                    ax.scatter(x, value, marker=marker, s=90, color=hue,
                               edgecolor="white", linewidth=0.8, zorder=3)
                ax.set_title(_TRACK_DISPLAY[track])
                ax.set_xlim(-0.5, len(configs) - 0.5)
                ax.set_xticks(range(len(configs)))
                ax.set_xticklabels([lbl for lbl, _ in configs], rotation=30, ha="right",
                                   fontsize=8)
                ax.set_ylabel("Mean latency (ms)")
                _style(ax, grid_axis="y")
            fig.suptitle(title)
            fig.tight_layout()
            path = sub_dir / f"{stem}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths[stem] = path
    return paths


# --- eval-shim join (SR + per-cycle latency, equal-n) ---------------------------------
def _select_method(raw, method: str, precision: str | None = None):
    """From one sr.json precision entry, return the point for `method` (or None if this precision
    has no point for it). The entry is one of:
      - a plain number — a manual SR-only override, method-agnostic (returned for any method);
      - a legacy flat `{success_rate, ...}` (pre-labelling, always `max`-calibrated) — returned
        when `method` is the default (`max`), else None (no such method here);
      - a labelled `{method: {success_rate, ...}}` map — `raw.get(method)`.
    So `int8` @ `max` and `int8` @ `entropy` coexist under one precision and a render selects one,
    like-for-like across tracks (SPEC §Parity).

    **FP32/FP16 fall back across methods** when `precision` says they are method-invariant. Those
    engines build data-free off the base graph, so their SR cannot depend on a PTQ method — but
    `src.sr_eval` stamps EVERY precision in a run with that run's method label, so an fp32/fp16 SR
    recorded by a `max` run is filed under `max`. Without the fallback an `entropy` render leaves
    fp32/fp16 SR-PENDING and NaNs every FP32-relative ΔSR, purely from a label. Quantized precisions
    (and the composite isolation keys, which are not method-invariant) never fall back — an
    `entropy` render must never silently show a `max` point."""
    if not isinstance(raw, dict):
        return raw  # plain number: method-agnostic manual SR override
    if "success_rate" in raw:  # legacy flat entry == max-calibrated (pre-labelling)
        return raw if method == DEFAULT_CALIBRATION_METHOD else None
    return select_by_method(raw, precision, method)  # labelled {method: SR} map


def _join_eval(bench: dict, overrides: dict | None, method: str) -> None:
    """Merge the gated eval-shim results into `bench` in place, selecting the `method`-calibrated
    point for each quantized precision (`_select_method`). `overrides` is
    `{track: {precision: entry}}`; see `_select_method` for the accepted entry shapes. Raw
    latencies are stashed on the row for `_finalize_per_cycle` to reduce to equal-n percentiles."""
    if not overrides:
        return
    for track, by_prec in overrides.items():
        for prec, raw in by_prec.items():
            if track not in bench or prec not in bench[track]:
                continue
            val = _select_method(raw, method, prec)
            if val is None:  # this precision has no point for the selected method -> leave pending
                continue
            row = bench[track][prec]
            if isinstance(val, dict):
                if "success_rate" in val:
                    row["success_rate"] = val["success_rate"]
                row["_per_cycle_latencies_ms"] = list(val.get("per_cycle_latencies_ms", []))
            else:
                row["success_rate"] = val


def per_cycle_samples(raw_by_track: dict, warmup_drop: int = PER_CYCLE_WARMUP_DROP) -> dict:
    """`{track: raw per-decision latency vector}` → `{track: THE sample}`, for one precision label.

    The single definition of what "the per-cycle sample" is: drop the warm-up head, then truncate
    every track to the common min-n across the tracks present. `_finalize_per_cycle` reduces it to
    the reported p50/p95/mean; `src.stats` builds the confidence interval and runs the independence
    test on it. Shared rather than reimplemented so the interval can never end up describing a
    different sample than the point estimate it brackets (architecture.md §12) — and so the same rule
    reaches the composite `enc-<A>+pred-<B>` isolation labels, which `_finalize_per_cycle` never
    iterates.

    A vector too short to survive the warm-up drop is left OUT rather than reduced to nothing (only
    reachable with synthetic/degenerate data — a recorded vector is far longer)."""
    lat_by_track = {t: v[warmup_drop:] for t, v in raw_by_track.items() if len(v) > warmup_drop}
    if not lat_by_track:
        return {}
    n = min(len(v) for v in lat_by_track.values())
    # First n in TEMPORAL order — a representative chronological subset; NOT sorted()[:n] (the n
    # smallest), which would censor the upper tail. Order is preserved because the independence
    # test reads it: a permutation test on lag-1 autocorrelation is meaningless on a reordered
    # sample. (`_percentile_ms` sorts internally, so order does not affect the percentile itself.)
    return {t: lat[:n] for t, lat in lat_by_track.items()}


def _finalize_per_cycle(bench: dict, warmup_drop: int = PER_CYCLE_WARMUP_DROP) -> None:
    """Compute per-cycle p50/p95 **and the mean** on each row from its joined raw per-DECISION
    latencies (one per alive episode per solve — `src.eval_latency`), after dropping a warm-up head
    and truncating every track to the common min-n across tracks per precision (equal-n, SPEC
    §Interface Contracts). A single-track render truncates to that track's own n.

    p50/p95 are reported (p50 the comparison basis); the mean feeds `decompose` only. All three
    come off the SAME truncated sample, so the decomposition and the headline describe the same
    decisions.

    **Warm-up (`warmup_drop`, default 1 decision — architecture.md §8).** The engine-step loops drop
    `ExportConfig.warmup` iters, but the per-cycle callback records from the first decision of the
    first solve. Keeping the cold decision would put first-`execute_v2` / kernel-autotune / clock-ramp
    cost in the cycle mean and NOT in the component means, so `overhead = cycle − enc − pred` would
    book all of it as planner overhead — a one-sided bias the negative-overhead alarm structurally
    cannot catch (it makes overhead *more* positive). Dropping restores the symmetry the subtraction
    assumes. It is applied HERE, at report time, so `sr.json`'s raw vector stays complete (CLAUDE §8),
    the architecture.md §8 span-sum reconciliation still holds, and `warmup_drop=0` re-renders the undropped
    view off-pod.

    **Order matters: drop BEFORE truncating.** Truncation keeps the temporal head (`lat[:n]`), so
    truncating first would preserve the cold decision by construction while discarding clean tail
    samples.

    The dropped values are STASHED (`_per_cycle_dropped_ms`) rather than discarded, so the speed
    table can disclose how anomalous the excluded decision actually was (`drop×`) — the exclusion is
    reported, not hidden. The truncated `n` is stashed too (`_per_cycle_n`): the whole statistic
    ruling rests on n being small and SR-dependent (SPEC §Interface Contracts), so a reader must be
    able to verify the equal-n truncation off the artefact. Truncation takes the common MINIMUM, so the highest-SR track sets n
    for every row at that precision — one more reason p95 carries no claim."""
    for prec in _PRECISIONS:
        raw_by_track = {
            t: bench[t][prec]["_per_cycle_latencies_ms"]
            for t in _TRACKS
            if prec in bench.get(t, {}) and bench[t][prec].get("_per_cycle_latencies_ms")
        }
        for t, sample in per_cycle_samples(raw_by_track, warmup_drop).items():
            bench[t][prec]["per_cycle_p50_ms"] = _percentile_ms(sample, 0.50)
            bench[t][prec]["per_cycle_p95_ms"] = _percentile_ms(sample, 0.95)
            bench[t][prec]["per_cycle_mean_ms"] = fmean(sample)
            bench[t][prec]["_per_cycle_n"] = len(sample)
            bench[t][prec]["_per_cycle_dropped_ms"] = raw_by_track[t][:warmup_drop]


# --- durable results I/O (canonical per-track JSON <-> render) ------------------------
def load_results(paths, method: str = DEFAULT_CALIBRATION_METHOD) -> dict:
    """Load + merge the per-track `results.<track>.json` files (written by `src.study`) back
    into the nested `bench[track][precision]` shape `report` consumes — so the headline
    re-renders OFF-POD from the canonical numbers, no L40S benchmark re-run (to join the gated
    per-cycle latency + SR, or tweak a plot). Whichever track files exist are merged; NaN
    latencies/SRs round-trip via the `NaN` json token.

    On disk each precision holds a `{method: BenchResult}` map — the quantized engines are per-method
    builds, so their timings coexist (SPEC §Parity) — and `method` selects which one this render
    reads, by `select_by_method`'s rule: fp32/fp16 fall back across labels, int8/fp8 never do. A
    quantized precision this method never benchmarked is simply ABSENT from `bench`, which is what
    keeps its row out of every table rather than borrowing the other method's number.

    A legacy flat entry (written before the method keying) belongs to the label its file's
    `meta.calibration_method` records — that run's own method, not an assumed default."""
    bench: dict = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        recorded = data["meta"].get("calibration_method", DEFAULT_CALIBRATION_METHOD)
        rows = {}
        for precision, entry in data["bench"].items():
            r = select_by_method(as_method_map(entry, recorded), precision, method)
            if r is not None:
                rows[precision] = r
        bench[data["meta"]["track"]] = rows
    return bench


def _resolve_result_paths(src) -> list[Path]:
    """`src` is a directory (glob its `results.*.json`) or a single result file."""
    src = Path(src)
    return sorted(src.glob("results.*.json")) if src.is_dir() else [src]


def report(
    bench: dict,
    out_dir: Path,
    wandb_run=None,
    sr_overrides: dict | None = None,
    method: str = DEFAULT_CALIBRATION_METHOD,
    warmup_drop: int = PER_CYCLE_WARMUP_DROP,
    component_latencies: dict | None = None,
) -> dict:
    """Emit all headline tables + plots to `out_dir`; optionally log to an open W&B run.
    Returns the artifact paths and the computed ratios for programmatic use.

    `sr_overrides` ({track: {precision: entry}}) joins in the gated eval-shim SR + per-cycle
    latency; any still-unpaired row is flagged loudly (a speed number without its SR is NOT a
    validated win — SPEC "no speed number without its task-quality counterpart").

    `method` (`max` | `entropy`) selects which calibration method's quantized points to render —
    the SR and per-cycle sample joined from sr.json AND the engine-step latencies the caller loaded
    for `bench`/`component_latencies` — so a render is like-for-like across tracks (SPEC §Parity).
    Both methods' points sit on the volume, so switching `method` re-renders the other without
    rebuilding or re-measuring. FP32/FP16 carry one data-free build and are read across labels.

    `warmup_drop` (default 1) drops that many cold decisions from the head of each per-cycle vector
    before the equal-n truncation, matching the engine loops' warm-up drop so the decomposition
    subtracts like from like (architecture.md §8). The exclusion is disclosed as the speed table's
    `drop×`; `warmup_drop=0` re-renders the undropped view.

    The four single-method tables are written METHOD-SCOPED (`<name>.<method>.txt`) and name their
    method in the body, so the two methods' artefacts coexist on disk. Three further tables render
    only when their data exists: `calibration_table.txt` (both methods' SR side by side),
    `isolation_table.<method>.txt` (component-precision isolation, architecture.md §9), and
    `latency_means_table.txt` (the five mean latency quantities with their bootstrap intervals —
    unscoped, since its config column names the method).

    When `sr_overrides` is present the render also computes the 95% confidence intervals on every
    absolute SR and absolute per-cycle p50 plus the lag-1 independence test (`src.stats`), surfaces
    them as table columns and plot error bars, and persists them to **`stats.json`**. The mean
    section additionally renders as five dot plots under `out_dir/component_analysis/`. It is pure
    re-analysis of the same stored samples — no run, no GPU — so it rides this cheap render rather
    than requiring a `src.study` pass.

    `component_latencies` ({track: {precision: {method: {encode_ms, predict_ms}}}}, from
    `latencies.<track>.json`) adds the component p50 intervals + independence flags to the speed
    table's enc/pred columns. Independent of `sr_overrides`: the component samples are their own
    surface, so they render with or without a joined SR. With BOTH present it also yields
    `latency_means_table.txt` — the mean decomposition needs a component sample to weight and a
    cycle sample to subtract it from.

    **Writes only `.txt`, `.png` and `stats.json` into `out_dir`** (plus the `component_analysis/`
    subdirectory of `.png`s). The canonical inputs —
    `results.<track>.json` + `latencies.<track>.json` (`src.study`) and `sr.json` (`src.sr_eval`) —
    are read-only here and are
    never rewritten, even when `out_dir` is the directory holding them (CLAUDE §8; pinned by
    `tests/test_report.py::test_report_never_rewrites_canonical_results`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _join_eval(bench, sr_overrides, method)
    _finalize_per_cycle(bench, warmup_drop)

    # Confidence intervals on the ABSOLUTE SR and per-cycle p50, plus the independence test the p50
    # interval rests on (SPEC §Requirements "Uncertainty quantification"). Re-analysis of the same
    # stored samples — no run, no GPU — so it rides the cheap render rather than needing `src.study`.
    # Imported HERE, not at module scope: `src.stats` imports this module for the shared per-cycle
    # sample rule, and the intervals stay optional (a render without sr.json is still valid).
    stats_payload = None
    if sr_overrides or component_latencies:
        from src import stats as _stats

        stats_payload = _stats.compute(
            sr_overrides or {}, warmup_drop, component_latencies=component_latencies
        )

    missing_sr = _missing_sr_rows(bench)
    if missing_sr:
        print(
            "⚠ SR PENDING — speed numbers below are NOT validated wins until the gated "
            "eval-shim re-run pairs an SR per precision (SPEC: no speed number without its "
            "task-quality counterpart).\n  unpaired: " + ", ".join(missing_sr) + "\n"
        )

    print(
        f"Calibration method for the int8/fp8 rows: {method} "
        "(SR + latency, both taken from that method's own runs; fp32/fp16 build data-free and are "
        "read across labels — SPEC §Parity)\n"
    )
    speed_table = render_speed_table(bench, method, warmup_drop, stats_payload)
    fp32_table = render_fp32_relative_table(bench, method)
    component_table = render_component_table(bench, method)
    dilution_table = render_dilution_table(bench, method)
    calibration_table = render_calibration_table(sr_overrides, method, stats_payload)
    isolation_table = render_isolation_table(bench, sr_overrides, method, stats_payload)
    latency_means_table = render_latency_means_table(stats_payload)
    print(speed_table)
    print()
    print("FP32-relative degradation (speed AND task quality):")
    print(fp32_table)
    if isolation_table:
        print()
        print("Component-precision isolation (which component caused the drop above):")
        print(isolation_table)
    print()
    print(component_table)
    if latency_means_table:
        print()
        print(
            "Mean latencies with bootstrap intervals "
            "(components per engine call; the decomposition per cycle):"
        )
        print(latency_means_table)
    print()
    print("Amdahl dilution (model-only vs realized per-cycle speedup):")
    print(dilution_table)
    if calibration_table:
        print()
        print("Calibration method comparison (SR side by side, both methods):")
        print(calibration_table)

    # Durability: serialize each table to a .txt on disk (not stdout/W&B-HTML only), so a
    # completed study survives pod teardown — same contract as the plots + checkpoints
    # (SPEC §Headline-artifact durability; W&B logging below stays additive).
    #
    # The single-method tables are METHOD-SCOPED by filename: their SR (and the per-cycle sample
    # it was measured on) is method-sourced, so a fixed name would let an `entropy` render
    # overwrite the `max` artefacts in place — the artefact-preservation rule broken at the last
    # step (SPEC §Parity, architecture.md §7, CLAUDE §8). The calibration table spans both
    # methods by construction, so it stays unscoped; only its `headline` marker moves on re-render.
    tables = {
        "speed_table": (out_dir / f"speed_table.{method}.txt", speed_table),
        "fp32_relative_table": (out_dir / f"fp32_relative_table.{method}.txt", fp32_table),
        "component_table": (out_dir / f"component_table.{method}.txt", component_table),
        "dilution_table": (out_dir / f"dilution_table.{method}.txt", dilution_table),
    }
    if isolation_table:
        tables["isolation_table"] = (out_dir / f"isolation_table.{method}.txt", isolation_table)
    if calibration_table:
        tables["calibration_table"] = (out_dir / "calibration_table.txt", calibration_table)
    if latency_means_table:
        # Unscoped like the calibration table: its config column names the method, so it spans both
        # and either method's render writes the same bytes.
        tables["latency_means_table"] = (out_dir / "latency_means_table.txt", latency_means_table)
    table_paths = {}
    for key, (path, text) in tables.items():
        path.write_text(text + "\n")
        table_paths[key] = path

    # The intervals' own durable artefact, beside the tables it feeds. Method-UNSCOPED (it covers
    # every point in sr.json — both methods and the isolation composites), so unlike the
    # single-method tables it has one fixed name and a re-render at the other method does not
    # clobber a different set of numbers.
    stats_path = None
    if stats_payload is not None:
        from src import stats as _stats

        stats_path = _stats.write_stats_json(stats_payload, out_dir)

    ratios = {
        p: {
            "per_cycle_p50_ratio": per_cycle_ratio(bench, p, "p50"),
            "per_cycle_p95_ratio": per_cycle_ratio(bench, p, "p95"),
        }
        for p in _PRECISIONS
        if p in bench.get("lewm", {}) and p in bench.get("dino", {})
    }

    plots = {
        # untitled (RESULTS.md); error bars = the 95% intervals on both absolute axes
        "speed_vs_sr": plot_speed_vs_sr(bench, out_dir, method, stats_payload),
        "component_breakdown": plot_component_breakdown(bench, out_dir),
    }
    plots["speed_vs_sr_titled"] = plots["speed_vs_sr"].with_name("speed_vs_sr.titled.png")  # README headline
    # The five mean latency quantities, one figure each, in their own `component_analysis/`
    # subdirectory — additive to every artefact above (SPEC §Uncertainty quantification).
    plots.update(
        {f"mean_{stem}": path
         for stem, path in plot_component_analysis(stats_payload, out_dir).items()}
    )
    # The cross-track (LeWM-vs-DINOv3) ratio plot needs BOTH tracks at a shared precision; a
    # single-track render would emit misleading empty bars, so skip it unless a ratio exists
    # (SPEC §Headline-artifact durability).
    if ratios:
        plots["per_cycle_ratio"] = plot_per_cycle_ratio(bench, out_dir)

    dilution = {t: dilution_disclosure(bench, t) for t in _TRACKS}

    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "headline/speed_table": wandb.Html(f"<pre>{speed_table}</pre>"),
                "headline/fp32_relative_table": wandb.Html(f"<pre>{fp32_table}</pre>"),
                "headline/component_table": wandb.Html(f"<pre>{component_table}</pre>"),
                "headline/dilution_table": wandb.Html(f"<pre>{dilution_table}</pre>"),
                **(
                    {"headline/calibration_table": wandb.Html(f"<pre>{calibration_table}</pre>")}
                    if calibration_table
                    else {}
                ),
                **(
                    {"headline/isolation_table": wandb.Html(f"<pre>{isolation_table}</pre>")}
                    if isolation_table
                    else {}
                ),
                **(
                    {
                        "headline/latency_means_table": wandb.Html(
                            f"<pre>{latency_means_table}</pre>"
                        )
                    }
                    if latency_means_table
                    else {}
                ),
                "headline/sr_pending": len(missing_sr),
                "headline/calibration_method": method,  # which method's int8/fp8 SR is rendered
                **{f"headline/{k}": wandb.Image(str(v)) for k, v in plots.items()},
                # p50 is the headline comparison basis; p95 logged alongside as the tail.
                **{
                    f"headline/per_cycle_p50_ratio_{p}": r["per_cycle_p50_ratio"]
                    for p, r in ratios.items()
                },
                **{
                    f"headline/per_cycle_p95_ratio_{p}": r["per_cycle_p95_ratio"]
                    for p, r in ratios.items()
                },
            }
        )

    return {
        "tables": table_paths,  # durable .txt on disk (SPEC §Headline-artifact durability)
        "stats": stats_path,  # stats.json — the intervals + independence test (None without sr)
        "plots": plots,
        "ratios": ratios,  # DINOv3 ÷ LeWM per-cycle latency (headline speed ratio)
        "dilution": dilution,
        "sr_pending": missing_sr,
        "calibration_method": method,  # which method's int8/fp8 SR these artifacts reflect
    }


def main() -> None:
    """Off-pod re-render entrypoint: load the canonical per-track results JSON that `src.study`
    wrote and rebuild the headline tables/plots — no L40S, no benchmark re-run. This is how the
    later, separately-gated per-cycle latency + SR are joined in, and how a plot is tweaked.

        uv run python -m src.report                              # default $STABLEWM_HOME/reports/phase5
        uv run python -m src.report from=<dir|results.json>      # explicit source
        uv run python -m src.report from=<dir> sr=<sr.json> wandb=<eval overlay> out=<dir>
        uv run python -m src.report from=<dir> sr=<sr.json> calibration_method=entropy
        uv run python -m src.report from=<dir> sr=<sr.json> per_cycle_warmup=0

    `calibration_method` (default `max`) selects which method's int8/fp8 points to render — the SR
    from sr.json and the benchmark numbers from `results.<track>.json`, both of which hold every
    method; re-run with `=entropy` for the entropy view — same files, no rebuild and no re-measure.
    `per_cycle_warmup` (default 1) is the cold decisions dropped before the equal-n truncation;
    `=0` reproduces the undropped view. Neither rewrites any canonical results file.

    `latencies.<track>.json` is picked up automatically from the source dir when present, adding the
    component p50 intervals for the rendered method to the speed table's enc/pred columns.
    """
    src = None
    out_dir = None
    sr_overrides = None
    wandb_experiment = None
    method = DEFAULT_CALIBRATION_METHOD
    warmup_drop = PER_CYCLE_WARMUP_DROP
    for a in sys.argv[1:]:
        if a.startswith("from="):
            src = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("sr="):
            sr_overrides = json.loads(Path(a.split("=", 1)[1]).read_text())
        elif a.startswith("calibration_method="):
            method = check_calibration_method(a.split("=", 1)[1])
        elif a.startswith("per_cycle_warmup="):
            warmup_drop = int(a.split("=", 1)[1])
        elif a.startswith("wandb="):
            wandb_experiment = a.split("=", 1)[1]
    if src is None:
        from src.study import default_out_dir  # shared default; lazy to avoid an import cycle

        src = default_out_dir()
    paths = _resolve_result_paths(src)
    if not paths:
        raise SystemExit(f"[report] no results.*.json under {src} — run `src.study` first")
    bench = load_results(paths, method)
    src_dir = Path(src) if Path(src).is_dir() else Path(src).parent
    if out_dir is None:
        out_dir = src_dir

    # The engine-step loops' raw samples, if `src.study` has persisted them beside the results —
    # they carry the component p50 intervals. Absent (a pre-Phase-9 results dir), the enc/pred CI
    # columns simply render blank; the rest of the report is unaffected.
    from src import stats as _stats  # lazy: src.stats imports this module

    component_latencies = (
        _stats.load_component_latencies(_stats.component_latency_paths(src_dir)) or None
    )

    run = None
    if wandb_experiment is not None:
        from src import wandb_log

        run = wandb_log.init(
            wandb_experiment, name="phase5-report", config={"phase": "phase5-report"}
        )
    try:
        report(
            bench,
            out_dir,
            wandb_run=run,
            sr_overrides=sr_overrides,
            method=method,
            warmup_drop=warmup_drop,
            component_latencies=component_latencies,
        )
    finally:
        if run is not None:
            run.finish()
    print(
        f"[report] headline artifacts (method={method}, per_cycle_warmup={warmup_drop}) "
        f"-> {out_dir}  (from {len(paths)} track file(s))"
    )


if __name__ == "__main__":
    main()
