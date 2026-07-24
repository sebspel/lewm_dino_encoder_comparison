"""Phase-5 speedup-study driver — owned PLUMBING (fails LOUDLY).

Ties the finished Phase-5 workers into one L40S command: for each track it benchmarks every
precision whose engines `src.export` has already built (component-latency + peak memory), and
hands the assembled bench dict to `src.report` for the headline tables + plots.

    uv run python -m src.study                  # both tracks, all built precisions
    uv run python -m src.study track=lewm       # one track
    uv run python -m src.study wandb=eval_lewm  # also log the headline artifacts to that
                                                # overlay's (shared) W&B project
    uv run python -m src.study out=/some/dir    # override the output dir
    uv run python -m src.study calibration_method=entropy  # label the run's int8/fp8 engines +
                                                # render the entropy SR (latency is method-invariant)

**Latency is the headline** (SPEC §Interface Contracts). This driver produces the two COMPONENT
latency distributions (encode-step + predict-step p50/p95, engine step loops) + peak memory; the
HEADLINE **per-cycle** latency and the **SR** come from the separate, gated `src.sr_eval` run
(same solves) and are joined off-pod by `src.report`. There is no fixed-wall-clock rollout-count
run.

The headline tables (`.txt`) + plots (`.png`) are persisted to `$STABLEWM_HOME/reports/phase5/`
by default — the persistent network volume, so a completed study survives pod teardown (SPEC
§Headline-artifact durability); off-pod (no `STABLEWM_HOME`) it falls back to repo-local
`reports/phase5`. W&B logging stays additive.

Engines are NOT built here — run `uv run python -m src.export model=<t> precision=<p>` first
(after the precision-match gate). A precision whose
`$STABLEWM_HOME/engines/<track>/{encoder,predictor}.<p>.plan` (int8/fp8 method-tagged
`…<p>.<method>.plan`; repo-local `engines/` fallback off-pod) is missing is skipped with a note
(that precision is reported as absent, not a run
failure). Runs on the L40S (benchmark needs CUDA / TensorRT).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.interfaces import (
    EnginePaths,
    ExportConfig,
    CEM_NUM_SAMPLES,
    QUANTIZED_PRECISIONS,
    DEFAULT_CALIBRATION_METHOD,
    check_calibration_method,
)
from src.export import engine_root as default_engine_root, engine_filename
from src.benchmark import benchmark
from src.gpu_clocks import log_gpu
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
    track: str,
    precision: str,
    engine_root: Path | None = None,
    method: str = DEFAULT_CALIBRATION_METHOD,
) -> EnginePaths:
    """Where `src.export` writes a track's two engines for one precision
    (`$STABLEWM_HOME/engines/<track>/{encoder,predictor}.<precision>.plan` by default;
    repo-local `engines/` fallback off-pod). For int8/fp8 the plan is METHOD-TAGGED
    (`…<precision>.<method>.plan`, `export.engine_filename`) so `max`/`entropy` engines coexist;
    `method` selects which to load (fp32/fp16 ignore it — method-invariant).

    Back-compat: engines built before method-tagging are untagged and were `max`-calibrated, so a
    `method=max` request falls back to the legacy `…<precision>.plan` when the tagged file is absent
    (mirrors the sr.json legacy fold). A non-default method never falls back — its engine must be
    tagged, or it is correctly reported missing."""
    root = engine_root if engine_root is not None else default_engine_root()
    d = Path(root) / track

    def _resolve(component: str) -> Path:
        p = d / engine_filename(component, precision, method)
        if (
            precision in QUANTIZED_PRECISIONS
            and method == DEFAULT_CALIBRATION_METHOD
            and not p.exists()
        ):
            legacy = d / f"{component}.{precision}.plan"
            if legacy.exists():
                return legacy
        return p

    return EnginePaths(encoder=_resolve("encoder"), predictor=_resolve("predictor"))


def dump_track_results(
    name: str,
    bench: dict,
    cfg: ExportConfig,
    out_dir: Path,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
) -> Path:
    """Persist one track's canonical raw results to `results.<name>.json` — the machine-
    readable benchmark numbers plus this run's fairness conditions (batch, seed), from which
    the tables/plots are regenerable views (`src.report from=<dir>`). Written PER TRACK so
    lewm/dino can be benchmarked in separate pod sessions without one clobbering the other
    (CLAUDE.md §8, SPEC §Headline-artifact durability). NaN latencies/SRs serialize as the
    `NaN` token (Python's json round-trips them; `src.report.load_results` reads them back).

    The per-precision benchmark numbers here are LATENCY + peak-mem, which are calibration-method-
    INVARIANT (SPEC §Parity), so ONE file per track serves every method; `calibration_method` is
    recorded as provenance for which engines this run benchmarked. This run's precisions are MERGED
    into any existing file per precision (not a whole-file replace), so benchmarking a precision
    subset later (e.g. adding fp8) leaves the track's other precisions on disk intact (CLAUDE §8).
    The method-DEPENDENT quantity — quantized SR — is labelled and kept coexisting in sr.json
    (`src.sr_eval`), not here."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results.{name}.json"
    existing_bench: dict = {}
    if path.exists():
        existing_bench = json.loads(path.read_text()).get("bench", {})
    # New precisions overwrite their own key (fresh measurement); precisions absent from this run
    # are preserved from disk — additive, no silent loss of a prior precision's numbers.
    merged_bench = {**existing_bench, **bench}
    payload = {
        "meta": {
            "track": name,
            "precisions_built": sorted(merged_bench),
            "n_latency_iters": cfg.n_latency_iters,
            "warmup": cfg.warmup,
            "num_samples": CEM_NUM_SAMPLES,
            "seed": cfg.seed,
            "obs_shape": list(cfg.obs_shape),
            # The int8/fp8 PTQ method this run's engines were built with (a build option for both
            # tracks — architecture.md §7). Latency is method-invariant, so this is provenance, not a
            # cross-track parity condition; the method-dependent SR is labelled in sr.json.
            "calibration_method": calibration_method,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "bench": merged_bench,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_track(
    track: str,
    cfg: ExportConfig,
    device: torch.device,
    engine_root: Path | None = None,
    gpu_log_dir: Path | None = None,
    method: str = DEFAULT_CALIBRATION_METHOD,
) -> tuple[str, dict]:
    """Benchmark one track's every built precision. Returns ``(name, bench_by_precision)`` in
    the shape `src.report` consumes.

    The engine step loops are timed at the CYCLE's real batches so the report's runtime-weighted
    decomposition is honest: **encode** once at batch 1 (single obs), **predict** at the candidate
    fan-out `CEM_NUM_SAMPLES` (the batch the CEM evaluates all candidates in per horizon step).

    Latency is calibration-method-invariant, but int8/fp8 engine plans are method-TAGGED, so
    `method` selects which quantized engines to locate (fp32/fp16 ignore it). The numbers are the
    same either way — the label just records which built engines were timed.

    Each precision's benchmark run is bracketed by an `nvidia-smi dmon` telemetry observer
    (`gpu_log_dir/<track>.<precision>.benchmark.dmon.log`) so the unlocked GPU clock/power/temp
    state during the timed loops is recorded, not merely assumed (SPEC §Parity).
    """
    root = engine_root if engine_root is not None else default_engine_root()
    adapter, name = _build_adapter(track)
    adapter.to(device)
    encode_inputs, _ = example_inputs(adapter, cfg, batch=1, device=device)
    _, predict_inputs = example_inputs(adapter, cfg, batch=CEM_NUM_SAMPLES, device=device)

    bench: dict = {}
    for precision in cfg.precisions:
        engines = engine_paths(track, precision, root, method)
        if not (engines["encoder"].exists() and engines["predictor"].exists()):
            print(
                f"[study:{name}] {precision}: engines missing under "
                f"{Path(root) / name} — skipped "
                f"(build with `src.export model={name} precision={precision}`)"
            )
            continue
        with log_gpu(f"{name}.{precision}.benchmark", gpu_log_dir):
            bench[precision] = benchmark(
                engines, encode_inputs, predict_inputs, cfg.n_latency_iters, cfg.warmup
            )
    return name, bench


def main() -> None:
    tracks = _TRACKS
    wandb_experiment = None
    out_dir = default_out_dir()  # $STABLEWM_HOME/reports/phase5 (repo-local fallback); out= overrides
    sr_overrides = None
    method = DEFAULT_CALIBRATION_METHOD
    for a in sys.argv[1:]:
        if a.startswith("track="):
            tracks = (a.split("=", 1)[1],)
        elif a.startswith("wandb="):
            wandb_experiment = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("calibration_method="):
            # Provenance label for the int8/fp8 engines this study benchmarked, and which method's
            # SR to render from sr.json (latency is method-invariant, so this drives no numbers).
            method = check_calibration_method(a.split("=", 1)[1])
        elif a.startswith("sr="):
            # Optional: join SR + per-cycle latency from the gated eval-shim re-run — a JSON
            # file {track: {precision: {method: {success_rate, per_cycle_latencies_ms}}}}. Absent
            # -> every row stays SR-PENDING / per-cycle NaN.
            sr_overrides = json.loads(Path(a.split("=", 1)[1]).read_text())

    cfg = ExportConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bench_all: dict = {}
    for track in tracks:
        name, bench = run_track(
            track, cfg, device, gpu_log_dir=out_dir / "gpu_logs", method=method
        )
        bench_all[name] = bench
        # Dump the canonical per-track results BEFORE rendering, so the raw numbers persist
        # even if the (cheap) render step later changes — and so `src.report from=<out_dir>`
        # can re-render/join per-cycle latency + SR off-pod without re-running this benchmark.
        dump_track_results(name, bench, cfg, out_dir, method)

    run = None
    if wandb_experiment is not None:
        from src import wandb_log

        run = wandb_log.init(
            wandb_experiment, name="phase5-study", config={"phase": "phase5-study"}
        )
    try:
        report(bench_all, out_dir, wandb_run=run, sr_overrides=sr_overrides, method=method)
    finally:
        if run is not None:
            run.finish()
    print(f"[study] headline artifacts (method={method}) -> {out_dir}")


if __name__ == "__main__":
    main()
