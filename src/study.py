"""Phase-5 speedup-study driver — owned PLUMBING (fails LOUDLY).

Ties the finished Phase-5 workers into one L40S command: for each track it benchmarks every
precision whose engines `src.export` has already built (component-latency + peak memory), and
hands the assembled bench dict to `src.report` for the headline tables + plots.

    uv run python -m src.study                  # both tracks, all built precisions
    uv run python -m src.study track=lewm       # one track
    uv run python -m src.study wandb=eval_lewm  # also log the headline artifacts to that
                                                # overlay's (shared) W&B project
    uv run python -m src.study out=/some/dir    # override the output dir
    uv run python -m src.study calibration_method=entropy  # time THAT method's int8/fp8 engines and
                                                # record + render under its label
    uv run python -m src.study precision=int8,fp8  # benchmark a subset (the method-invariant
                                                # fp32/fp16 need timing once, not once per method)

**Latency is the headline** (SPEC §Interface Contracts). This driver produces the two COMPONENT
latency distributions (encode-step + predict-step p50/p95, engine step loops) + peak memory; the
HEADLINE **per-cycle** latency and the **SR** come from the separate, gated `src.sr_eval` run
(same solves) and are joined off-pod by `src.report`. There is no fixed-wall-clock rollout-count
run.

The canonical per-track numbers (`results.<track>.json`), the engine-step loops' **raw per-call
samples** (`latencies.<track>.json` — what `src.stats` builds the component p50 intervals from), and
the headline tables (`.txt`) + plots (`.png`) are persisted to `$STABLEWM_HOME/reports/phase5/`
by default — the persistent network volume, so a completed study survives pod teardown (SPEC
§Headline-artifact durability); off-pod (no `STABLEWM_HOME`) it falls back to repo-local
`reports/phase5`. W&B logging stays additive.

Both measured files are keyed by **(precision, calibration method)** and merged per cell, so timing
the `max` engines never overwrites what the `entropy` pass measured (CLAUDE §8, SPEC §Parity); the
render selects one method's points with `src.report calibration_method=`.

Engines are NOT built here — run `uv run python -m src.export model=<t> precision=<p>` first
(after the precision-match gate). A precision whose
`$STABLEWM_HOME/engines/<track>/{encoder,predictor}.<p>.plan` (int8/fp8 method-tagged
`…<p>.<method>.plan`; repo-local `engines/` fallback off-pod) is missing is skipped with a note
(that precision is reported as absent, not a run
failure). Runs on the L40S (benchmark needs CUDA / TensorRT).
"""

from __future__ import annotations

import dataclasses
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
from src.gpu_clocks import log_gpu, run_tag
from src.precision_match import _build_adapter, example_inputs
from src.report import as_method_map, report

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


def _merge_by_method(path: Path, section: str, new: dict, method: str) -> dict:
    """Merge this run's `{precision: payload}` into whatever the file already holds, keyed
    `{precision: {method: payload}}`.

    The merge is per **(precision, method)**: a run replaces only the cells it just measured, so a
    second calibration method's timings land BESIDE the first's and a precision-subset run leaves
    the other precisions untouched (CLAUDE §8 — never silently discard a completed measurement).
    Existing flat entries are folded under the label the file's own `meta.calibration_method`
    records — the method that run actually built and timed, never an assumed default."""
    existing, recorded = {}, DEFAULT_CALIBRATION_METHOD
    if path.exists():
        data = json.loads(path.read_text())
        existing = data.get(section, {})
        recorded = data.get("meta", {}).get("calibration_method", DEFAULT_CALIBRATION_METHOD)
    merged = {p: dict(as_method_map(e, recorded)) for p, e in existing.items()}
    for precision, payload in new.items():
        merged.setdefault(precision, {})[method] = payload
    return merged


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

    Keyed by **(precision, calibration method)**: an int8/fp8 engine is a per-method BUILD, so the
    latency + peak-mem measured off it belong to that method and coexist with the other's as
    separately-labelled points (SPEC §Parity), which `src.report calibration_method=` then selects
    like-for-like across tracks. FP32/FP16 carry one data-free build, so whichever run recorded them
    the number describes the same engine and a render reads it across labels
    (`report.select_by_method`). `_merge_by_method` keeps every cell this run did not measure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results.{name}.json"
    merged_bench = _merge_by_method(path, "bench", bench, calibration_method)
    payload = {
        "meta": {
            "track": name,
            "precisions_built": sorted(merged_bench),
            "methods": sorted({m for by in merged_bench.values() for m in by}),
            "n_latency_iters": cfg.n_latency_iters,
            "warmup": cfg.warmup,
            "num_samples": CEM_NUM_SAMPLES,
            "seed": cfg.seed,
            "obs_shape": list(cfg.obs_shape),
            # The int8/fp8 PTQ method THIS run's engines were built with (a build option for both
            # tracks — architecture.md §7) and therefore the key its rows landed under. The per-cell
            # labels in `bench` are the authority; this records the latest writer, and is what a
            # legacy flat entry is folded under.
            "calibration_method": calibration_method,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "bench": merged_bench,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def dump_track_latencies(
    name: str,
    samples: dict,
    cfg: ExportConfig,
    out_dir: Path,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
) -> Path:
    """Persist one track's RAW engine-step samples to `latencies.<name>.json` — the per-call
    latencies `src.benchmark` reduced to the p50/p95/mean in `results.<name>.json`.

    Written **beside** the results file, never inside it: `results.<track>.json` is the
    summary-shaped canonical artifact every table, plot and derived-clock render parses, and folding
    ~800 floats per track into it would make that schema heavier for one consumer. This file is what
    `src.stats` computes the component p50 confidence intervals + lag-1 independence tests from, so a
    later statistic over the component distributions is an off-pod re-analysis rather than an L40S
    booking (SPEC §Interface Contracts, docs/architecture.md §12).

    Keyed by **(precision, calibration method)**, exactly as `dump_track_results` is: these are the
    per-call latencies of a specific pair of engine plans, and the quantized plans are per-method
    builds (SPEC §Parity), so a `max` timing run records beside the `entropy` one rather than over
    it. Merged **per (precision, method)** with the same no-clobber discipline, so benchmarking a
    subset later leaves the track's other cells on disk intact (CLAUDE §8) — this file is the only
    copy of these samples, and re-measuring one costs an L40S booking."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"latencies.{name}.json"
    merged = _merge_by_method(path, "latencies", samples, calibration_method)
    payload = {
        "meta": {
            "track": name,
            "precisions": sorted(merged),
            "methods": sorted({m for by in merged.values() for m in by}),
            # The loop conditions these samples were recorded under. n is equal across tracks by
            # construction (fixed-iteration), and `warmup` iters ran UNTIMED before the first
            # recorded call — so each vector is already the sample, needing no truncation and no
            # report-time warm-up drop (docs/architecture.md §12).
            "n_latency_iters": cfg.n_latency_iters,
            "warmup": cfg.warmup,
            "calibration_method": calibration_method,
            "seed": cfg.seed,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "latencies": merged,
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def run_track(
    track: str,
    cfg: ExportConfig,
    device: torch.device,
    engine_root: Path | None = None,
    gpu_log_dir: Path | None = None,
    method: str = DEFAULT_CALIBRATION_METHOD,
) -> tuple[str, dict, dict]:
    """Benchmark one track's every built precision. Returns
    ``(name, bench_by_precision, samples_by_precision)`` — the summary numbers in the shape
    `src.report` consumes, plus the raw engine-step samples `dump_track_latencies` persists.

    The engine step loops are timed at the CYCLE's real batches so the report's runtime-weighted
    decomposition is honest: **encode** once at batch 1 (single obs), **predict** at the candidate
    fan-out `CEM_NUM_SAMPLES` (the batch the CEM evaluates all candidates in per horizon step).

    `method` selects which engines to locate and therefore which are timed: int8/fp8 plans are
    method-TAGGED (`export.engine_filename`), so a `max` pass times the `max` build and an `entropy`
    pass the `entropy` one, and the two results are recorded under their own labels rather than one
    standing in for the other. FP32/FP16 are untagged — one build, timed under whichever label the
    run carries.

    Each precision's benchmark run is bracketed by an `nvidia-smi dmon` telemetry observer
    (`gpu_log_dir/<track>.<precision>.<method>.benchmark.dmon.log` — `gpu_clocks.run_tag`) so the
    unlocked GPU clock/power/temp state during the timed loops is recorded, not merely assumed
    (SPEC §Parity).
    """
    root = engine_root if engine_root is not None else default_engine_root()
    adapter, name = _build_adapter(track)
    adapter.to(device)
    encode_inputs, _ = example_inputs(adapter, cfg, batch=1, device=device)
    _, predict_inputs = example_inputs(adapter, cfg, batch=CEM_NUM_SAMPLES, device=device)

    bench: dict = {}
    samples: dict = {}
    for precision in cfg.precisions:
        engines = engine_paths(track, precision, root, method)
        if not (engines["encoder"].exists() and engines["predictor"].exists()):
            print(
                f"[study:{name}] {precision}: engines missing under "
                f"{Path(root) / name} — skipped "
                f"(build with `src.export model={name} precision={precision}`)"
            )
            continue
        with log_gpu(run_tag(name, precision, method, "benchmark"), gpu_log_dir):
            bench[precision], samples[precision] = benchmark(
                engines, encode_inputs, predict_inputs, cfg.n_latency_iters, cfg.warmup
            )
    return name, bench, samples


def main() -> None:
    tracks = _TRACKS
    wandb_experiment = None
    out_dir = default_out_dir()  # $STABLEWM_HOME/reports/phase5 (repo-local fallback); out= overrides
    sr_overrides = None
    method = DEFAULT_CALIBRATION_METHOD
    cfg = ExportConfig()
    for a in sys.argv[1:]:
        if a.startswith("track="):
            tracks = (a.split("=", 1)[1],)
        elif a.startswith("wandb="):
            wandb_experiment = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("precision="):
            # Benchmark a SUBSET (comma-separated, `src.sr_eval`'s spelling). The method-invariant
            # fp32/fp16 need timing once, so a second method's pass runs `precision=int8,fp8` and
            # touches nothing the first recorded.
            cfg = dataclasses.replace(cfg, precisions=tuple(a.split("=", 1)[1].split(",")))
        elif a.startswith("calibration_method="):
            # Which method's engines this pass times, and the label its rows are recorded under —
            # int8/fp8 are per-method builds, so this selects a measurement, not just a stamp.
            method = check_calibration_method(a.split("=", 1)[1])
        elif a.startswith("sr="):
            # Optional: join SR + per-cycle latency from the gated eval-shim re-run — a JSON
            # file {track: {precision: {method: {success_rate, per_cycle_latencies_ms}}}}. Absent
            # -> every row stays SR-PENDING / per-cycle NaN.
            sr_overrides = json.loads(Path(a.split("=", 1)[1]).read_text())

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bench_all: dict = {}
    for track in tracks:
        name, bench, samples = run_track(
            track, cfg, device, gpu_log_dir=out_dir / "gpu_logs", method=method
        )
        bench_all[name] = bench
        # Dump the canonical per-track results BEFORE rendering, so the raw numbers persist
        # even if the (cheap) render step later changes — and so `src.report from=<out_dir>`
        # can re-render/join per-cycle latency + SR off-pod without re-running this benchmark.
        dump_track_results(name, bench, cfg, out_dir, method)
        # The samples those numbers were reduced from, beside them — likewise before rendering, so
        # an interrupted render never costs the L40S run's samples (src.stats reads this file).
        dump_track_latencies(name, samples, cfg, out_dir, method)

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
