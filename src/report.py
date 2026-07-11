"""Headline table + plot runner — owned PLUMBING (fails LOUDLY).

Consumes the benchmark + profile results (per track × precision) and emits the
headline outputs:
  - LeWM-vs-DINOv3 **model-only rollouts-in-budget ratio** + **predictor-step p95-latency
    ratio** (per precision) — both labelled for what they measure (the planner floor and
    the untimed encoder are NOT in them)
  - **Amdahl dilution**: optimizable fraction `p`, ceiling `1/(1-p)`, and per-precision
    model-only vs Amdahl-predicted-realized speedup (measured-realized is gated)
  - per-model **FP32→FP16→INT8 delta** in both **speed and SR**, degradation quoted vs FP32
  - **speed-vs-SR** scatter
  - runtime-**weighted** per-component **encoder / predictor / planner** bottleneck breakdown

Pure data → tables/plots; runs anywhere (matplotlib Agg, no CUDA). SR is NaN wherever the
gated eval-shim re-run has not paired it — every such row is flagged SR-PENDING (not a
validated win) and plots skip those points; feed SR back in via `sr_overrides`. Optionally
logs the tables + figures to an open W&B run (shared project).

Input shape: ``bench[track][precision] -> BenchResult`` and
``prof[track][precision] -> ComponentProfile`` (missing entries are skipped).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window (pod + CI)
import matplotlib.pyplot as plt  # noqa: E402

from src.profile import (  # noqa: E402  — CEM per-cycle call counts (for the weighted table header)
    _ENCODER_CALLS_PER_CYCLE as _ENCODER_CALLS,
    _PREDICTOR_CALLS_PER_CYCLE as _PREDICTOR_CALLS,
)

_TRACKS = ("lewm", "dino")
_PRECISIONS = ("fp32", "fp16", "int8")


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt(x, spec: str = ".3g") -> str:
    return "—" if _missing(x) else format(x, spec)


# --- ratios (the LeWM-vs-DINOv3 headlines) --------------------------------------------
def rollouts_ratio(bench: dict, precision: str) -> float:
    """LeWM ÷ DINOv3 **model-only** rollouts in the fixed budget (planner treated as free —
    the benchmark runs encode+predict without the CEM loop). This is the model-portion
    speedup ratio; it OVERSTATES the realized LeWM advantage, because the Python planner
    floor dilutes it (the Amdahl point). The realized ratio comes from the gated eval-shim
    re-run (planner in the loop)."""
    return (
        bench["lewm"][precision]["rollouts_completed"]
        / bench["dino"][precision]["rollouts_completed"]
    )


def p95_ratio(bench: dict, precision: str) -> float:
    """DINOv3 ÷ LeWM p95 **predictor-step** latency — how many× slower a DINO predictor step
    is. This captures the 196-vs-1 token PREDICTOR gap only; the DINOv3-vs-ViT-Tiny ENCODER
    gap is amortized (untimed) and surfaces in rollouts + the profile, not here. And each
    step syncs the stream, so LeWM sits on a launch+sync floor — under-representing the true
    compute asymmetry (LeWM is launch-latency-bound, SPEC §Parity)."""
    return (
        bench["dino"][precision]["latency_p95_ms"]
        / bench["lewm"][precision]["latency_p95_ms"]
    )


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
    """Per precision: p95 speedup and SR delta **relative to FP32** for one track — a
    precision that is faster but degrades task quality must be visible."""
    base = bench[track]["fp32"]
    out: dict[str, dict] = {}
    for prec in _PRECISIONS:
        r = bench.get(track, {}).get(prec)
        if r is None:
            continue
        out[prec] = {
            "p95_speedup_vs_fp32": base["latency_p95_ms"] / r["latency_p95_ms"],
            "sr_delta_vs_fp32": r["success_rate"] - base["success_rate"],
        }
    return out


def dilution_disclosure(bench: dict, prof: dict, track: str) -> dict:
    """Per track: the Amdahl dilution picture (SPEC §dilution disclosure). From the FP32
    profile shares comes the optimizable fraction `p` and the speedup ceiling `1/(1-p)`.
    Per precision: the **model-only** speedup `s` (from the planner-free benchmark throughput
    vs FP32) and the Amdahl-**predicted** realized wall-clock speedup `1/((1-p)+p/s)`. The
    *measured* realized speedup needs the gated eval-shim re-run (planner in the loop) — left
    as `None` until that lands. The gap between model-only `s` and the predicted realized is
    the planner floor."""
    p = prof.get(track, {}).get("fp32")
    base = bench.get(track, {}).get("fp32")
    out: dict = {
        "optimizable_fraction": p["optimizable_fraction"] if p else None,
        "amdahl_ceiling": p["amdahl_ceiling"] if p else None,
        "per_precision": {},
    }
    if base is None:
        return out
    frac = out["optimizable_fraction"]
    for prec in _PRECISIONS:
        r = bench.get(track, {}).get(prec)
        if r is None:
            continue
        s = r["throughput"] / base["throughput"]  # model-only speedup vs FP32
        predicted_realized = (
            None if frac is None else 1.0 / ((1.0 - frac) + frac / s)
        )
        out["per_precision"][prec] = {
            "model_only_speedup": s,
            "predicted_realized_speedup": predicted_realized,
            "measured_realized_speedup": None,  # gated eval-shim re-run (planner in the loop)
        }
    return out


def render_dilution_table(bench: dict, prof: dict) -> str:
    """Amdahl dilution table: p, ceiling, and per-precision model-only vs predicted-realized
    speedup — makes the planner floor that dilutes the model-only ratio visible."""
    lines = []
    for track in _TRACKS:
        d = dilution_disclosure(bench, prof, track)
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
                f"{_fmt(v['measured_realized_speedup'], '.2f') + ' (gated)':>18}"
            )
    return "\n".join(lines)


# --- tables ---------------------------------------------------------------------------
def render_speed_table(bench: dict) -> str:
    # p50/p95 are PREDICTOR-STEP latency; rollouts/thrpt are MODEL-ONLY (planner-free); SR
    # shows PEND where the gated eval-shim has not paired it yet.
    hdr = (
        f"{'track':>6} {'prec':>5} {'p50_step':>9} {'p95_step':>9} "
        f"{'rollouts*':>9} {'thrpt*':>8} {'mem_MB':>9} {'SR':>7}"
    )
    lines = [
        "  (* = model-only, planner-free; p50/p95 = predictor step only; SR PEND = gated eval-shim)",
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
                f"{r['latency_p50_ms']:>9.3f} {r['latency_p95_ms']:>9.3f} "
                f"{r['rollouts_completed']:>9d} {r['throughput']:>8.2f} "
                f"{r['peak_mem_mb']:>9.1f} {sr:>7}"
            )
    return "\n".join(lines)


def render_component_table(prof: dict) -> str:
    # Runtime-WEIGHTED cycle shares (calls × per-call ms) — the honest decomposition, which
    # the raw per-call means are NOT. `p` = optimizable fraction, ceiling = 1/(1-p).
    hdr = (
        f"{'track':>6} {'prec':>5} {'enc_cyc_ms':>11} {'pred_cyc_ms':>12} "
        f"{'plan_cyc_ms':>12} {'p':>7} {'ceil×':>7}"
    )
    lines = [
        "  (cyc_ms = per-cycle = calls × per-call ms; predict called "
        f"{_PREDICTOR_CALLS} × / cycle, encode {_ENCODER_CALLS} ×)",
        hdr,
        "-" * len(hdr),
    ]
    for track in _TRACKS:
        for prec in _PRECISIONS:
            p = prof.get(track, {}).get(prec)
            if p is None:
                continue
            lines.append(
                f"{track:>6} {prec:>5} "
                f"{p['encoder_cycle_ms']:>11.4f} {p['predictor_cycle_ms']:>12.4f} "
                f"{p['planner_cycle_ms']:>12.4f} {p['optimizable_fraction']:>7.3f} "
                f"{_fmt(p['amdahl_ceiling'], '.2f'):>7}"
            )
    return "\n".join(lines)


# --- plots ----------------------------------------------------------------------------
def plot_speed_vs_sr(bench: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"fp32": "o", "fp16": "s", "int8": "^"}
    for track in _TRACKS:
        for prec in _PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is None or _missing(r["success_rate"]):
                continue
            ax.scatter(
                r["throughput"], r["success_rate"],
                marker=markers[prec], s=80, label=f"{track}-{prec}",
            )
    ax.set_xlabel("throughput (rollouts/s)")
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
    precs = [p for p in _PRECISIONS if p in values]
    ax.bar(precs, [values[p] for p in precs])
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)  # parity line
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_rollouts_ratio(bench: dict, out_dir: Path) -> Path:
    vals = {p: rollouts_ratio(bench, p) for p in _PRECISIONS if p in bench.get("lewm", {}) and p in bench.get("dino", {})}
    return _bar_over_precisions(vals, "LeWM ÷ DINOv3 rollouts in budget", "ratio (×)", out_dir, "rollouts_ratio.png")


def plot_p95_ratio(bench: dict, out_dir: Path) -> Path:
    vals = {p: p95_ratio(bench, p) for p in _PRECISIONS if p in bench.get("lewm", {}) and p in bench.get("dino", {})}
    return _bar_over_precisions(vals, "DINOv3 ÷ LeWM p95 step latency", "ratio (×)", out_dir, "p95_ratio.png")


def plot_component_breakdown(prof: dict, out_dir: Path, precision: str = "fp32") -> Path:
    """Stacked encoder/predictor/planner bar per track at one precision, using the
    runtime-WEIGHTED per-cycle shares (calls × per-call ms) — NOT the raw per-call means,
    which under-weight the predictor and don't sum to the cycle. Attributes the LeWM↔DINO gap
    to the right component (docs/platform_api.md §5)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    tracks = [t for t in _TRACKS if precision in prof.get(t, {})]
    bottoms = [0.0] * len(tracks)
    for comp in ("encoder_cycle_ms", "predictor_cycle_ms", "planner_cycle_ms"):
        heights = [prof[t][precision][comp] for t in tracks]
        ax.bar(tracks, heights, bottom=bottoms, label=comp.replace("_cycle_ms", ""))
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_ylabel("per-cycle time (ms), runtime-weighted")
    ax.set_title(f"Component breakdown ({precision})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"component_breakdown_{precision}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _join_sr(bench: dict, sr_overrides: dict | None) -> None:
    """Merge SR from the gated eval-shim re-run into `bench` in place: `sr_overrides` is
    `{track: {precision: success_rate}}`. Lets the owner pair SR per precision without editing
    any benchmark output (the benchmark leaves `success_rate=NaN`)."""
    if not sr_overrides:
        return
    for track, by_prec in sr_overrides.items():
        for prec, sr in by_prec.items():
            if track in bench and prec in bench[track]:
                bench[track][prec]["success_rate"] = sr


def report(
    bench: dict, prof: dict, out_dir: Path, wandb_run=None, sr_overrides: dict | None = None
) -> dict:
    """Emit all headline tables + plots to `out_dir`; optionally log to an open W&B run.
    Returns the artifact paths and the computed ratios for programmatic use.

    `sr_overrides` ({track: {precision: SR}}) joins in the SR from the gated eval-shim re-run;
    any still-unpaired row is flagged loudly (a speed number without its SR is NOT a validated
    win — SPEC "no speed number without its task-quality counterpart")."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _join_sr(bench, sr_overrides)

    missing_sr = _missing_sr_rows(bench)
    if missing_sr:
        print(
            "⚠ SR PENDING — speed numbers below are NOT validated wins until the gated "
            "eval-shim re-run pairs an SR per precision (SPEC: no speed number without its "
            "task-quality counterpart).\n  unpaired: " + ", ".join(missing_sr) + "\n"
        )

    speed_table = render_speed_table(bench)
    component_table = render_component_table(prof)
    dilution_table = render_dilution_table(bench, prof)
    print(speed_table)
    print()
    print(component_table)
    print()
    print("Amdahl dilution (model-only vs realized wall-clock speedup):")
    print(dilution_table)

    plots = {
        "speed_vs_sr": plot_speed_vs_sr(bench, out_dir),
        "rollouts_ratio": plot_rollouts_ratio(bench, out_dir),
        "p95_ratio": plot_p95_ratio(bench, out_dir),
        "component_breakdown": plot_component_breakdown(prof, out_dir),
    }
    ratios = {
        p: {"rollouts_ratio": rollouts_ratio(bench, p), "p95_ratio": p95_ratio(bench, p)}
        for p in _PRECISIONS
        if p in bench.get("lewm", {}) and p in bench.get("dino", {})
    }

    dilution = {t: dilution_disclosure(bench, prof, t) for t in _TRACKS}

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
                    f"headline/rollouts_ratio_model_only_{p}": r["rollouts_ratio"]
                    for p, r in ratios.items()
                },
                **{f"headline/p95_ratio_{p}": r["p95_ratio"] for p, r in ratios.items()},
            }
        )

    return {
        "plots": plots,
        "ratios": ratios,  # rollouts_ratio is MODEL-ONLY (planner-free) — see dilution
        "dilution": dilution,
        "sr_pending": missing_sr,
    }
