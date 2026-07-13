"""Phase-5 speedup-study driver — owned PLUMBING (fails LOUDLY).

Ties the finished Phase-5 workers into one L40S command: for each track it benchmarks every
precision whose engines `src.export` has already built (component-latency + peak memory), and
hands the assembled bench dict to `src.report` for the headline tables + plots.

    uv run python -m src.study                  # both tracks, all built precisions
    uv run python -m src.study track=lewm       # one track
    uv run python -m src.study wandb=eval_lewm  # also log the headline artifacts to that
                                                # overlay's (shared) W&B project
    uv run python -m src.study out=/some/dir    # override the output dir

**Latency is the headline** (SPEC §Interface Contracts). This driver produces the two COMPONENT
latency distributions (encode-step + predict-step p50/p95, engine step loops) + peak memory; the
HEADLINE **per-cycle** latency and the **SR** come from the separate, gated `src.sr_eval` run
(same solves) and are joined off-pod by `src.report`. There is no fixed-wall-clock rollout-count
run. GPU clocks are locked around the run so the numbers aren't thermal artifacts.

The headline tables (`.txt`) + plots (`.png`) are persisted to `$STABLEWM_HOME/reports/phase5/`
by default — the persistent network volume, so a completed study survives pod teardown (SPEC
§Headline-artifact durability); off-pod (no `STABLEWM_HOME`) it falls back to repo-local
`reports/phase5`. W&B logging stays additive.

Engines are NOT built here — run `uv run python -m src.export model=<t> precision=<p>` first
(after the precision-match gate). A precision whose
`engines/<track>/{encoder,predictor}.<p>.plan` is missing is skipped with a note, which is
exactly the FP16-only fallback (SPEC §Caps / PLAN §Phase-5 cap). Runs on the L40S (benchmark
needs CUDA / TensorRT).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.interfaces import EnginePaths, ExportConfig, CEM_NUM_SAMPLES
from src.export import _ENGINE_ROOT
from src.benchmark import benchmark
from src.precision_match import _build_adapter, example_inputs
from src.report import report

_TRACKS = ("lewm", "dino")


def default_out_dir() -> Path:
    """Where the headline artifacts land by default: `$STABLEWM_HOME/reports/phase5/` — the
    persistent network volume, same durability contract as checkpoints + engines, so a
    completed study survives pod teardown (SPEC §Headline-artifact durability). Falls back to
    the repo-local `reports/phase5` off-pod where `STABLEWM_HOME` is unset."""
    home = os.environ.get("STABLEWM_HOME")
    return Path(home) / "reports" / "phase5" if home else Path("reports/phase5")


def engine_paths(
    track: str, precision: str, engine_root: Path = _ENGINE_ROOT
) -> EnginePaths:
    """Where `src.export` writes a track's two engines for one precision
    (`engines/<track>/{encoder,predictor}.<precision>.plan`)."""
    d = Path(engine_root) / track
    return EnginePaths(
        encoder=d / f"encoder.{precision}.plan",
        predictor=d / f"predictor.{precision}.plan",
    )


def lock_gpu_clocks() -> int | None:
    """Best-effort GPU clock lock so latency + memory numbers aren't thermal artifacts
    (SPEC §Parity — a fairness condition). Enables persistence mode and pins the graphics
    clock to the max supported. Returns the locked clock (MHz), or None if the pod denies
    clock control (a common unprivileged-container case) — then the caller records None and
    the run proceeds unlocked (throttling would still show in `nvidia-smi dmon`). The owner may
    prefer a lower *sustainable* clock; this pins the max as a reproducible default."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        subprocess.run(["nvidia-smi", "-pm", "1"], check=True, capture_output=True)
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.graphics", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        clock = int(out.stdout.strip().splitlines()[0])
        subprocess.run(
            ["nvidia-smi", "-lgc", f"{clock},{clock}"], check=True, capture_output=True
        )
        print(f"[study] locked GPU graphics clock to {clock} MHz")
        return clock
    except Exception as e:  # denied / unsupported — proceed unlocked, record None
        print(
            f"[study] GPU clock lock unavailable ({e}); proceeding unlocked "
            "(throttling not prevented — check nvidia-smi dmon)"
        )
        return None


def reset_gpu_clocks() -> None:
    """Undo `lock_gpu_clocks` (`nvidia-smi -rgc`). Best-effort; safe if the lock never took."""
    if shutil.which("nvidia-smi") is None:
        return
    try:
        subprocess.run(["nvidia-smi", "-rgc"], check=True, capture_output=True)
    except Exception:
        pass


def dump_track_results(
    name: str, bench: dict, cfg: ExportConfig, out_dir: Path, locked_clock: int | None
) -> Path:
    """Persist one track's canonical raw results to `results.<name>.json` — the machine-
    readable benchmark numbers plus this run's fairness conditions (locked clock, batch, seed),
    from which the tables/plots are regenerable views (`src.report from=<dir>`). Written PER
    TRACK so lewm/dino can be benchmarked in separate pod sessions without one clobbering the
    other (CLAUDE.md §8, SPEC §Headline-artifact durability). NaN latencies/SRs serialize as the
    `NaN` token (Python's json round-trips them; `src.report.load_results` reads them back)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results.{name}.json"
    payload = {
        "meta": {
            "track": name,
            "precisions_built": list(bench),
            "locked_clock_mhz": locked_clock,
            "n_latency_iters": cfg.n_latency_iters,
            "warmup": cfg.warmup,
            "num_samples": CEM_NUM_SAMPLES,
            "seed": cfg.seed,
            "obs_shape": list(cfg.obs_shape),
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "bench": bench,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_track(
    track: str,
    cfg: ExportConfig,
    device: torch.device,
    engine_root: Path = _ENGINE_ROOT,
) -> tuple[str, dict]:
    """Benchmark one track's every built precision. Returns ``(name, bench_by_precision)`` in
    the shape `src.report` consumes.

    The engine step loops are timed at the CYCLE's real batches so the report's runtime-weighted
    decomposition is honest: **encode** once at batch 1 (single obs), **predict** at the candidate
    fan-out `CEM_NUM_SAMPLES` (the batch the CEM evaluates all candidates in per horizon step).
    """
    adapter, name = _build_adapter(track)
    adapter.to(device)
    encode_inputs, _ = example_inputs(adapter, cfg, batch=1, device=device)
    _, predict_inputs = example_inputs(adapter, cfg, batch=CEM_NUM_SAMPLES, device=device)

    bench: dict = {}
    for precision in cfg.precisions:
        engines = engine_paths(track, precision, engine_root)
        if not (engines["encoder"].exists() and engines["predictor"].exists()):
            print(
                f"[study:{name}] {precision}: engines missing under "
                f"{Path(engine_root) / name} — skipped "
                f"(build with `src.export model={name} precision={precision}`)"
            )
            continue
        bench[precision] = benchmark(
            engines, encode_inputs, predict_inputs, cfg.n_latency_iters, cfg.warmup
        )
    return name, bench


def main() -> None:
    tracks = _TRACKS
    wandb_experiment = None
    out_dir = default_out_dir()  # $STABLEWM_HOME/reports/phase5 (repo-local fallback); out= overrides
    sr_overrides = None
    for a in sys.argv[1:]:
        if a.startswith("track="):
            tracks = (a.split("=", 1)[1],)
        elif a.startswith("wandb="):
            wandb_experiment = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("sr="):
            # Optional: join SR + per-cycle latency from the gated eval-shim re-run — a JSON
            # file {track: {precision: {success_rate, per_cycle_latencies_ms}}}. Absent -> every
            # row stays SR-PENDING / per-cycle NaN.
            sr_overrides = json.loads(Path(a.split("=", 1)[1]).read_text())

    cfg = ExportConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bench_all: dict = {}
    locked_clock = lock_gpu_clocks()
    try:
        for track in tracks:
            name, bench = run_track(track, cfg, device)
            bench_all[name] = bench
            # Dump the canonical per-track results BEFORE rendering, so the raw numbers persist
            # even if the (cheap) render step later changes — and so `src.report from=<out_dir>`
            # can re-render/join per-cycle latency + SR off-pod without re-running this benchmark.
            dump_track_results(name, bench, cfg, out_dir, locked_clock)
    finally:
        reset_gpu_clocks()

    run = None
    if wandb_experiment is not None:
        from src import wandb_log

        run = wandb_log.init(
            wandb_experiment, name="phase5-study", config={"phase": "phase5-study"}
        )
    try:
        report(bench_all, out_dir, wandb_run=run, sr_overrides=sr_overrides)
    finally:
        if run is not None:
            run.finish()
    print(f"[study] headline artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
