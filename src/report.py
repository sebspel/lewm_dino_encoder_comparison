"""Headline table + plot runner — owned PLUMBING (fails LOUDLY).

Consumes the benchmark results (per track × precision) and emits the headline outputs:
  - LeWM-vs-DINOv3 **per-cycle p50/p95 latency ratio** (the headline speed measure, per
    precision) — DINOv3 ÷ LeWM full planning-cycle latency
  - **Amdahl dilution**: optimizable fraction `p`, ceiling `1/(1-p)`, and per-precision
    model-only vs realized speedup (realized = measured per-cycle latency ratio)
  - per-model **FP32→FP16→INT8 delta** in both **speed and SR**, degradation quoted vs FP32
  - **speed-vs-SR** scatter
  - per-component **encoder / predictor / overhead** bottleneck breakdown, derived from the
    engine-step latencies × CEM call counts minus the measured per-cycle time (overhead by
    subtraction, SPEC §Interface Contracts)

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

Input shape: ``bench[track][precision] -> BenchResult`` (missing entries are skipped).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")  # headless: save figures, never open a window (pod + CI)
import matplotlib.pyplot as plt  # noqa: E402

from src.interfaces import (  # noqa: E402  — CEM per-cycle call counts (the decomposition weights)
    ENCODER_CALLS_PER_CYCLE as _ENCODER_CALLS,
    PREDICTOR_CALLS_PER_CYCLE as _PREDICTOR_CALLS,
)

_TRACKS = ("lewm", "dino")
_PRECISIONS = ("fp32", "fp16", "int8")


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt(x, spec: str = ".3g") -> str:
    return "—" if _missing(x) else format(x, spec)


def _percentile_ms(values, q: float) -> float:
    return torch.quantile(torch.tensor(values, dtype=torch.float64), q).item()


# --- per-component decomposition (overhead by subtraction from the measured cycle) ----
def decompose(r: dict) -> dict:
    """One BenchResult → the per-cycle time decomposition (SPEC §Interface Contracts). The
    engine-step p50s are weighted by the CEM per-cycle call counts into `enc_cyc`/`pred_cyc`;
    the **measured** per-cycle p50 (joined from the eval-shim run) is the cycle, and
    `overhead = cycle − enc_cyc − pred_cyc` (the un-optimizable floor: CEM planner + criterion
    + assembly + glue). A NEGATIVE overhead is surfaced loudly — never clamped — as a sign the
    call-count weighting or the isolated timing is off. `p = (enc+pred)/cycle` is the optimizable
    fraction; the Amdahl ceiling is `1/(1-p)`. When the cycle is not yet joined, the cycle-derived
    fields are None (the enc/pred model shares still stand)."""
    enc_cyc = r["encode_p50_ms"] * _ENCODER_CALLS
    pred_cyc = r["predict_p50_ms"] * _PREDICTOR_CALLS
    model_cyc = enc_cyc + pred_cyc
    cycle = r["per_cycle_p50_ms"]
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
def per_cycle_ratio(bench: dict, precision: str, pct: str = "p95") -> float:
    """DINOv3 ÷ LeWM full **per-cycle** planning latency at `pct` — the headline speed ratio
    (how many× slower a DINO planning cycle is). NaN until the per-cycle latency is joined from
    the gated eval-shim run."""
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
    """Per precision: per-cycle p95 speedup and SR delta **relative to FP32** for one track — a
    precision that is faster but degrades task quality must be visible."""
    base = bench[track]["fp32"]
    out: dict[str, dict] = {}
    for prec in _PRECISIONS:
        r = bench.get(track, {}).get(prec)
        if r is None:
            continue
        base_p95, r_p95 = base["per_cycle_p95_ms"], r["per_cycle_p95_ms"]
        out[prec] = {
            "per_cycle_p95_speedup_vs_fp32": (
                math.nan if _missing(base_p95) or _missing(r_p95) else base_p95 / r_p95
            ),
            "sr_delta_vs_fp32": r["success_rate"] - base["success_rate"],
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
    base_cycle = base["per_cycle_p50_ms"]
    for prec in _PRECISIONS:
        r = bench.get(track, {}).get(prec)
        if r is None:
            continue
        dec = decompose(r)
        s = base_dec["model_cyc_ms"] / dec["model_cyc_ms"]  # model-only speedup vs FP32
        measured_realized = (
            math.nan
            if _missing(base_cycle) or _missing(r["per_cycle_p50_ms"])
            else base_cycle / r["per_cycle_p50_ms"]
        )
        predicted_realized = None if frac is None else 1.0 / ((1.0 - frac) + frac / s)
        out["per_precision"][prec] = {
            "model_only_speedup": s,
            "predicted_realized_speedup": predicted_realized,
            "measured_realized_speedup": measured_realized,
        }
    return out


# --- tables ---------------------------------------------------------------------------
def render_speed_table(bench: dict) -> str:
    # per_cycle p50/p95 are the HEADLINE (joined, PEND until then); enc/pred p50 are the
    # component steps; SR is PEND until the gated eval-shim pairs it.
    hdr = (
        f"{'track':>6} {'prec':>5} {'cyc_p50':>8} {'cyc_p95':>8} "
        f"{'enc_p50':>8} {'pred_p50':>9} {'mem_MB':>9} {'SR':>7}"
    )
    lines = [
        "  (cyc = per-cycle HEADLINE, joined from eval-shim; enc/pred = engine step p50; "
        "PEND = gated eval-shim)",
        hdr,
        "-" * len(hdr),
    ]
    for track in _TRACKS:
        for prec in _PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is None:
                continue
            sr = "PEND" if _missing(r["success_rate"]) else format(r["success_rate"], ".1f")
            lines.append(
                f"{track:>6} {prec:>5} "
                f"{_fmt(r['per_cycle_p50_ms'], '.3f'):>8} {_fmt(r['per_cycle_p95_ms'], '.3f'):>8} "
                f"{r['encode_p50_ms']:>8.3f} {r['predict_p50_ms']:>9.3f} "
                f"{r['peak_mem_mb']:>9.1f} {sr:>7}"
            )
    return "\n".join(lines)


def render_component_table(bench: dict) -> str:
    # Runtime-WEIGHTED per-cycle shares (step p50 × CEM call counts); overhead by subtraction
    # from the measured cycle. `p` = optimizable fraction, ceiling = 1/(1-p).
    hdr = (
        f"{'track':>6} {'prec':>5} {'enc_cyc_ms':>11} {'pred_cyc_ms':>12} "
        f"{'ovh_cyc_ms':>11} {'p':>7} {'ceil×':>7}"
    )
    lines = [
        "  (cyc_ms = per-cycle = step p50 × calls; predict called "
        f"{_PREDICTOR_CALLS} × / cycle, encode {_ENCODER_CALLS} ×; ovh = cycle − enc − pred)",
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
                f"{d['enc_cyc_ms']:>11.4f} {d['pred_cyc_ms']:>12.4f} "
                f"{_fmt(d['overhead_ms'], '.4f'):>11} {_fmt(d['optimizable_fraction'], '.3f'):>7} "
                f"{_fmt(d['amdahl_ceiling'], '.2f'):>7}"
            )
    return "\n".join(lines)


def render_dilution_table(bench: dict) -> str:
    """Amdahl dilution table: p, ceiling, and per-precision model-only vs realized speedup —
    makes the overhead floor that dilutes the model-only ratio visible."""
    lines = []
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


# --- plots ----------------------------------------------------------------------------
def plot_speed_vs_sr(bench: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"fp32": "o", "fp16": "s", "int8": "^"}
    for track in _TRACKS:
        for prec in _PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is None or _missing(r["success_rate"]) or _missing(r["per_cycle_p50_ms"]):
                continue
            ax.scatter(
                r["per_cycle_p50_ms"], r["success_rate"],
                marker=markers[prec], s=80, label=f"{track}-{prec}",
            )
    ax.set_xlabel("per-cycle latency p50 (ms) — lower = faster")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Speed vs task quality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "speed_vs_sr.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _bar_over_precisions(values: dict[str, float], title: str, ylabel: str, out_dir: Path, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    precs = [p for p in _PRECISIONS if p in values and not _missing(values[p])]
    ax.bar(precs, [values[p] for p in precs])
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)  # parity line
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_per_cycle_ratio(bench: dict, out_dir: Path) -> Path:
    vals = {
        p: per_cycle_ratio(bench, p, "p95")
        for p in _PRECISIONS
        if p in bench.get("lewm", {}) and p in bench.get("dino", {})
    }
    return _bar_over_precisions(
        vals, "DINOv3 ÷ LeWM per-cycle p95 latency", "ratio (×)", out_dir, "per_cycle_ratio.png"
    )


def plot_component_breakdown(bench: dict, out_dir: Path, precision: str = "fp32") -> Path:
    """Stacked encoder/predictor/overhead bar per track at one precision, from the runtime-
    WEIGHTED per-cycle shares (step p50 × calls) with overhead by subtraction. Attributes the
    LeWM↔DINO gap to the right component. `overhead` is 0 until the per-cycle latency is joined."""
    fig, ax = plt.subplots(figsize=(5, 4))
    tracks = [t for t in _TRACKS if precision in bench.get(t, {})]
    decs = {t: decompose(bench[t][precision]) for t in tracks}
    bottoms = [0.0] * len(tracks)
    segments = (("encoder", "enc_cyc_ms"), ("predictor", "pred_cyc_ms"), ("overhead", "overhead_ms"))
    for label, key in segments:
        heights = [(decs[t][key] or 0.0) for t in tracks]
        ax.bar(tracks, heights, bottom=bottoms, label=label)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_ylabel("per-cycle time (ms), runtime-weighted")
    ax.set_title(f"Component breakdown ({precision})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"component_breakdown_{precision}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --- eval-shim join (SR + per-cycle latency, equal-n) ---------------------------------
def _join_eval(bench: dict, overrides: dict | None) -> None:
    """Merge the gated eval-shim results into `bench` in place. `overrides` is
    `{track: {precision: value}}` where value is either a plain number (SR only, for manual
    overrides) or `{success_rate, per_cycle_latencies_ms}`. Raw latencies are stashed on the
    row for `_finalize_per_cycle` to reduce to equal-n percentiles."""
    if not overrides:
        return
    for track, by_prec in overrides.items():
        for prec, val in by_prec.items():
            if track not in bench or prec not in bench[track]:
                continue
            row = bench[track][prec]
            if isinstance(val, dict):
                if "success_rate" in val:
                    row["success_rate"] = val["success_rate"]
                row["_per_cycle_latencies_ms"] = list(val.get("per_cycle_latencies_ms", []))
            else:
                row["success_rate"] = val


def _finalize_per_cycle(bench: dict) -> None:
    """Compute per-cycle p50/p95 on each row from its joined raw per-DECISION latencies (one per
    alive episode per solve — `src.eval_latency`), AFTER truncating every track to the common
    min-n across tracks per precision (equal-n, SPEC §Interface Contracts). A single-track render
    truncates to that track's own n."""
    for prec in _PRECISIONS:
        lat_by_track = {
            t: bench[t][prec]["_per_cycle_latencies_ms"]
            for t in _TRACKS
            if prec in bench.get(t, {}) and bench[t][prec].get("_per_cycle_latencies_ms")
        }
        if not lat_by_track:
            continue
        n = min(len(v) for v in lat_by_track.values())
        for t, lat in lat_by_track.items():
            # First n in TEMPORAL order — a representative chronological subset; NOT sorted()[:n]
            # (the n smallest), which would censor the upper tail (`_percentile_ms` sorts
            # internally via torch.quantile, so the input order does not matter for the value).
            sample = lat[:n]
            bench[t][prec]["per_cycle_p50_ms"] = _percentile_ms(sample, 0.50)
            bench[t][prec]["per_cycle_p95_ms"] = _percentile_ms(sample, 0.95)


# --- durable results I/O (canonical per-track JSON <-> render) ------------------------
def load_results(paths) -> dict:
    """Load + merge the per-track `results.<track>.json` files (written by `src.study`) back
    into the nested `bench[track][precision]` shape `report` consumes — so the headline
    re-renders OFF-POD from the canonical numbers, no L40S benchmark re-run (to join the gated
    per-cycle latency + SR, or tweak a plot). Whichever track files exist are merged; NaN
    latencies/SRs round-trip via the `NaN` json token."""
    bench: dict = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        bench[data["meta"]["track"]] = data["bench"]
    return bench


def _resolve_result_paths(src) -> list[Path]:
    """`src` is a directory (glob its `results.*.json`) or a single result file."""
    src = Path(src)
    return sorted(src.glob("results.*.json")) if src.is_dir() else [src]


def report(
    bench: dict, out_dir: Path, wandb_run=None, sr_overrides: dict | None = None
) -> dict:
    """Emit all headline tables + plots to `out_dir`; optionally log to an open W&B run.
    Returns the artifact paths and the computed ratios for programmatic use.

    `sr_overrides` ({track: {precision: {success_rate, per_cycle_latencies_ms}}}) joins in the
    gated eval-shim SR + per-cycle latency; any still-unpaired row is flagged loudly (a speed
    number without its SR is NOT a validated win — SPEC "no speed number without its task-quality
    counterpart")."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _join_eval(bench, sr_overrides)
    _finalize_per_cycle(bench)

    missing_sr = _missing_sr_rows(bench)
    if missing_sr:
        print(
            "⚠ SR PENDING — speed numbers below are NOT validated wins until the gated "
            "eval-shim re-run pairs an SR per precision (SPEC: no speed number without its "
            "task-quality counterpart).\n  unpaired: " + ", ".join(missing_sr) + "\n"
        )

    speed_table = render_speed_table(bench)
    component_table = render_component_table(bench)
    dilution_table = render_dilution_table(bench)
    print(speed_table)
    print()
    print(component_table)
    print()
    print("Amdahl dilution (model-only vs realized per-cycle speedup):")
    print(dilution_table)

    # Durability: serialize each table to a .txt on disk (not stdout/W&B-HTML only), so a
    # completed study survives pod teardown — same contract as the plots + checkpoints
    # (SPEC §Headline-artifact durability; W&B logging below stays additive).
    tables = {
        "speed_table": (out_dir / "speed_table.txt", speed_table),
        "component_table": (out_dir / "component_table.txt", component_table),
        "dilution_table": (out_dir / "dilution_table.txt", dilution_table),
    }
    table_paths = {}
    for key, (path, text) in tables.items():
        path.write_text(text + "\n")
        table_paths[key] = path

    ratios = {
        p: {
            "per_cycle_p50_ratio": per_cycle_ratio(bench, p, "p50"),
            "per_cycle_p95_ratio": per_cycle_ratio(bench, p, "p95"),
        }
        for p in _PRECISIONS
        if p in bench.get("lewm", {}) and p in bench.get("dino", {})
    }

    plots = {
        "speed_vs_sr": plot_speed_vs_sr(bench, out_dir),
        "component_breakdown": plot_component_breakdown(bench, out_dir),
    }
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
                "headline/component_table": wandb.Html(f"<pre>{component_table}</pre>"),
                "headline/dilution_table": wandb.Html(f"<pre>{dilution_table}</pre>"),
                "headline/sr_pending": len(missing_sr),
                **{f"headline/{k}": wandb.Image(str(v)) for k, v in plots.items()},
                **{
                    f"headline/per_cycle_p95_ratio_{p}": r["per_cycle_p95_ratio"]
                    for p, r in ratios.items()
                },
            }
        )

    return {
        "tables": table_paths,  # durable .txt on disk (SPEC §Headline-artifact durability)
        "plots": plots,
        "ratios": ratios,  # DINOv3 ÷ LeWM per-cycle latency (headline speed ratio)
        "dilution": dilution,
        "sr_pending": missing_sr,
    }


def main() -> None:
    """Off-pod re-render entrypoint: load the canonical per-track results JSON that `src.study`
    wrote and rebuild the headline tables/plots — no L40S, no benchmark re-run. This is how the
    later, separately-gated per-cycle latency + SR are joined in, and how a plot is tweaked.

        uv run python -m src.report                              # default $STABLEWM_HOME/reports/phase5
        uv run python -m src.report from=<dir|results.json>      # explicit source
        uv run python -m src.report from=<dir> sr=<sr.json> wandb=<eval overlay> out=<dir>
    """
    src = None
    out_dir = None
    sr_overrides = None
    wandb_experiment = None
    for a in sys.argv[1:]:
        if a.startswith("from="):
            src = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("sr="):
            sr_overrides = json.loads(Path(a.split("=", 1)[1]).read_text())
        elif a.startswith("wandb="):
            wandb_experiment = a.split("=", 1)[1]
    if src is None:
        from src.study import default_out_dir  # shared default; lazy to avoid an import cycle

        src = default_out_dir()
    paths = _resolve_result_paths(src)
    if not paths:
        raise SystemExit(f"[report] no results.*.json under {src} — run `src.study` first")
    bench = load_results(paths)
    if out_dir is None:
        s = Path(src)
        out_dir = s if s.is_dir() else s.parent

    run = None
    if wandb_experiment is not None:
        from src import wandb_log

        run = wandb_log.init(
            wandb_experiment, name="phase5-report", config={"phase": "phase5-report"}
        )
    try:
        report(bench, out_dir, wandb_run=run, sr_overrides=sr_overrides)
    finally:
        if run is not None:
            run.finish()
    print(f"[report] headline artifacts -> {out_dir}  (from {len(paths)} track file(s))")


if __name__ == "__main__":
    main()
