"""Headline table + plot runner (Phase 5) — owned PLUMBING (fails LOUDLY).

Consumes the benchmark + profile results (per track × precision) and emits the SPEC
§Requirements headline outputs:
  - LeWM-vs-DINOv3 **rollouts-in-budget ratio** + **p95-latency ratio** (per precision)
  - per-model **FP32→FP16→INT8 delta** in both **speed and SR**, degradation quoted vs FP32
  - **speed-vs-SR** scatter
  - per-component **encoder / predictor / planner** bottleneck breakdown

Pure data → tables/plots; runs anywhere (matplotlib Agg, no CUDA). SR may be NaN where the
gated eval-shim re-run has not filled it yet — tables show "—" and plots skip those points.
Optionally logs the tables + figures to an open W&B run (shared project, SPEC §W&B).

Input shape: ``bench[track][precision] -> BenchResult`` and
``prof[track][precision] -> ComponentProfile`` (missing entries are skipped).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window (pod + CI)
import matplotlib.pyplot as plt  # noqa: E402

_TRACKS = ("lewm", "dino")
_PRECISIONS = ("fp32", "fp16", "int8")


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt(x, spec: str = ".3g") -> str:
    return "—" if _missing(x) else format(x, spec)


# --- ratios (the LeWM-vs-DINOv3 headlines) --------------------------------------------
def rollouts_ratio(bench: dict, precision: str) -> float:
    """LeWM ÷ DINOv3 rollouts completed in the fixed budget — the headline speedup measure
    (how many more rollouts LeWM fits, SPEC §Parity 'mechanistic')."""
    return (
        bench["lewm"][precision]["rollouts_completed"]
        / bench["dino"][precision]["rollouts_completed"]
    )


def p95_ratio(bench: dict, precision: str) -> float:
    """DINOv3 ÷ LeWM p95 per-step latency — how many× slower a DINO predictor step is."""
    return (
        bench["dino"][precision]["latency_p95_ms"]
        / bench["lewm"][precision]["latency_p95_ms"]
    )


def fp32_relative(bench: dict, track: str) -> dict[str, dict]:
    """Per precision: p95 speedup and SR delta **relative to FP32** for one track (SPEC
    §Parity — a precision that is faster but degrades task quality must be visible)."""
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


# --- tables ---------------------------------------------------------------------------
def render_speed_table(bench: dict) -> str:
    hdr = (
        f"{'track':>6} {'prec':>5} {'p50_ms':>9} {'p95_ms':>9} "
        f"{'rollouts':>9} {'thrpt':>8} {'mem_MB':>9} {'SR':>7}"
    )
    lines = [hdr, "-" * len(hdr)]
    for track in _TRACKS:
        for prec in _PRECISIONS:
            r = bench.get(track, {}).get(prec)
            if r is None:
                continue
            lines.append(
                f"{track:>6} {prec:>5} "
                f"{r['latency_p50_ms']:>9.3f} {r['latency_p95_ms']:>9.3f} "
                f"{r['rollouts_completed']:>9d} {r['throughput']:>8.2f} "
                f"{r['peak_mem_mb']:>9.1f} {_fmt(r['success_rate'], '.1f'):>7}"
            )
    return "\n".join(lines)


def render_component_table(prof: dict) -> str:
    hdr = f"{'track':>6} {'prec':>5} {'encoder_ms':>11} {'predictor_ms':>13} {'planner_ms':>11}"
    lines = [hdr, "-" * len(hdr)]
    for track in _TRACKS:
        for prec in _PRECISIONS:
            p = prof.get(track, {}).get(prec)
            if p is None:
                continue
            lines.append(
                f"{track:>6} {prec:>5} "
                f"{p['encoder_ms']:>11.4f} {p['predictor_ms']:>13.4f} {p['planner_ms']:>11.4f}"
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
    """Stacked encoder/predictor/planner bar per track at one precision — attributes the
    LeWM↔DINO gap to the right component (docs/platform_api.md §5)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    tracks = [t for t in _TRACKS if precision in prof.get(t, {})]
    bottoms = [0.0] * len(tracks)
    for comp in ("encoder_ms", "predictor_ms", "planner_ms"):
        heights = [prof[t][precision][comp] for t in tracks]
        ax.bar(tracks, heights, bottom=bottoms, label=comp.replace("_ms", ""))
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_ylabel("per-cycle time (ms)")
    ax.set_title(f"Component breakdown ({precision})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / f"component_breakdown_{precision}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def report(bench: dict, prof: dict, out_dir: Path, wandb_run=None) -> dict:
    """Emit all headline tables + plots to `out_dir`; optionally log to an open W&B run.
    Returns the artifact paths and the computed ratios for programmatic use."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speed_table = render_speed_table(bench)
    component_table = render_component_table(prof)
    print(speed_table)
    print()
    print(component_table)

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

    if wandb_run is not None:
        import wandb

        wandb_run.log(
            {
                "headline/speed_table": wandb.Html(f"<pre>{speed_table}</pre>"),
                "headline/component_table": wandb.Html(f"<pre>{component_table}</pre>"),
                **{f"headline/{k}": wandb.Image(str(v)) for k, v in plots.items()},
                **{
                    f"headline/rollouts_ratio_{p}": r["rollouts_ratio"]
                    for p, r in ratios.items()
                },
                **{f"headline/p95_ratio_{p}": r["p95_ratio"] for p, r in ratios.items()},
            }
        )

    return {"plots": plots, "ratios": ratios}
