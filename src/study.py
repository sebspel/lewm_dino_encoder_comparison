"""Phase-5 speedup-study driver — owned PLUMBING (fails LOUDLY).

Ties the finished Phase-5 workers into one L40S command: for each track it profiles the
PyTorch adapter (the FP32 baseline per-component decomposition), then benchmarks every
precision whose engines `src.export` has already built, and hands the assembled bench/prof
dicts to `src.report` for the headline tables + plots.

    uv run python -m src.study                  # both tracks, all built precisions
    uv run python -m src.study track=lewm       # one track
    uv run python -m src.study wandb=eval_lewm  # also log the headline artifacts to that
                                                # overlay's (shared) W&B project
    uv run python -m src.study out=/some/dir    # override the output dir

The headline tables (`.txt`) + plots (`.png`) are persisted to `$STABLEWM_HOME/reports/phase5/`
by default — the persistent network volume, so a completed study survives pod teardown (SPEC
§Headline-artifact durability); off-pod (no `STABLEWM_HOME`) it falls back to repo-local
`reports/phase5`. W&B logging stays additive.

Engines are NOT built here — run `uv run python -m src.export model=<t> precision=<p>` first
(after the precision-match gate). A precision whose
`engines/<track>/{encoder,predictor}.<p>.plan` is missing is skipped with a note, which is
exactly the FP16-only fallback (SPEC §Caps / PLAN §Phase-5 cap).

SR is NOT produced here: every bench row carries `success_rate=NaN`, and `src.report` flags
each unpaired row as SR-PENDING (a speed number without its SR is not a validated win). The
SR-per-precision join is the separate, owner-gated eval-shim re-run (`get_cost`/`get_action`
over the engine); its results can be fed back in via `sr=<file.json>` ({track: {precision:
SR}}) without touching code. Runs on the L40S (benchmark + profile need CUDA / TensorRT).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.interfaces import EnginePaths, ExportConfig
from src.export import _ENGINE_ROOT
from src.benchmark import benchmark
from src.profile import profile, CEM_NUM_SAMPLES
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


def dump_track_results(
    name: str, prof: dict, bench: dict, cfg: ExportConfig, out_dir: Path
) -> Path:
    """Persist one track's canonical raw results to `results.<name>.json` — the machine-
    readable benchmark + profile numbers plus this run's fairness conditions (time budget,
    batch, seed), from which the tables/plots are regenerable views (`src.report from=<dir>`).
    Written PER TRACK so lewm/dino can be benchmarked in separate pod sessions without one
    clobbering the other (CLAUDE.md §8, SPEC §Headline-artifact durability). NaN SRs serialize
    as the `NaN` token (Python's json round-trips them; `src.report.load_results` reads them
    back)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results.{name}.json"
    payload = {
        "meta": {
            "track": name,
            "precisions_built": list(bench),
            "time_budget_s": cfg.time_budget_s,
            "warmup": cfg.warmup,
            "n_profile_iters": cfg.n_profile_iters,
            "num_samples": CEM_NUM_SAMPLES,
            "seed": cfg.seed,
            "obs_shape": list(cfg.obs_shape),
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "prof": prof,
        "bench": bench,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def run_track(
    track: str,
    cfg: ExportConfig,
    device: torch.device,
    engine_root: Path = _ENGINE_ROOT,
) -> tuple[str, dict, dict]:
    """Profile one track's PyTorch adapter and benchmark every built precision.

    Returns ``(name, prof_by_precision, bench_by_precision)`` in the nested shape
    `src.report` consumes. The per-component profile times the PyTorch adapter, which is
    precision-invariant, so it is the FP32 baseline decomposition (SPEC §Speedup study — the
    optimizable fraction `(enc+pred)/total` is read off these shares); it is recorded under
    the single `fp32` key the report's component breakdown reads.
    """
    adapter, name = _build_adapter(track)
    adapter.to(device)
    encode_inputs, predict_inputs = example_inputs(adapter, cfg, device=device)

    # Profile at the CYCLE's real batches so the runtime-weighted shares are honest (the
    # profiler weights per-call times by CEM call counts): encode once at batch 1 (single
    # obs), predict at the candidate fan-out CEM_NUM_SAMPLES. The predict latent is rebuilt
    # from its own batch-1 encode inside example_inputs, so these two calls stay consistent.
    prof_encode, _ = example_inputs(adapter, cfg, batch=1, device=device)
    _, prof_predict = example_inputs(adapter, cfg, batch=CEM_NUM_SAMPLES, device=device)
    prof = {
        "fp32": profile(
            adapter, prof_encode, prof_predict, cfg.n_profile_iters, cfg.warmup
        )
    }

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
            engines, encode_inputs, predict_inputs, cfg.time_budget_s, cfg.warmup
        )
    return name, prof, bench


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
            # Optional: join SR from the gated eval-shim re-run — a JSON file
            # {track: {precision: success_rate}}. Absent -> every row stays SR-PENDING.
            import json

            sr_overrides = json.loads(Path(a.split("=", 1)[1]).read_text())

    cfg = ExportConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bench_all: dict = {}
    prof_all: dict = {}
    for track in tracks:
        name, prof, bench = run_track(track, cfg, device)
        prof_all[name] = prof
        bench_all[name] = bench
        # Dump the canonical per-track results BEFORE rendering, so the raw numbers persist
        # even if the (cheap) render step later changes — and so `src.report from=<out_dir>`
        # can re-render/join SR off-pod without re-running this benchmark.
        dump_track_results(name, prof, bench, cfg, out_dir)

    run = None
    if wandb_experiment is not None:
        from src import wandb_log

        run = wandb_log.init(
            wandb_experiment, name="phase5-study", config={"phase": "phase5-study"}
        )
    try:
        report(bench_all, prof_all, out_dir, wandb_run=run, sr_overrides=sr_overrides)
    finally:
        if run is not None:
            run.finish()
    print(f"[study] headline artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
