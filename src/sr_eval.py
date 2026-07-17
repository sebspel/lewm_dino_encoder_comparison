"""SR-per-precision eval driver — re-run the Phase-3 CEM eval on the OPTIMIZED (engine) model.

Phase-5 pairs every speed number with a Push-T success rate (SPEC "no speed number without
its task-quality counterpart"). The benchmark (`src.benchmark` / `src.study`) leaves
`success_rate=NaN`; this driver produces the SR that fills it, per precision, by re-running the
**same** platform CEM eval on the exported/quantized engines — then writes the
`{track: {precision: SR}}` JSON that `src.study` / `src.report` join back in via `sr=<file>`.

    uv run python -m src.sr_eval --config-dir conf +experiment=eval_<lewm|dino> \
        [precision=fp32,fp16,int8,fp8] [out=<dir>]

**How the engine model gets into the eval (the seam).** The CEM solver calls the world model
through ``get_cost`` — not ``encode`` / ``predict`` — so the exported engines are re-wrapped in
the owner-gated SR shim (`src.sr_shim`), a subclass of the platform model that overrides ONLY
the two engine-boundary methods and inherits ``get_cost`` / ``rollout`` / ``criterion``
byte-unchanged (cost parity by construction, proven bit-for-bit by ``src.sr_shim.sr_cost_parity``).
The vendored eval entrypoint (`scripts.plan.eval_wm.run`) builds its model internally via
``swm.wm.utils.load_pretrained(cfg.policy)`` and hands it to ``CEMSolver(model=...)`` — there is
no config seam to inject an arbitrary model object — so this driver runs it byte-unmodified and
slots the shim in by patching ``load_pretrained`` for the duration of the run. That swaps ONLY
the model object; no CEM config / seed / sample count / plan changes, so the LeWM-vs-DINO parity
(SPEC §Parity) is preserved (the SR differs from the FP32 baseline only by the engines'
quantization drift, which is exactly the signal being measured).

Runs on the L40S (the shim's engine callables need `tensorrt` + CUDA + the Push-T dataset).
A precision whose engines `src.export` has not built is skipped (reported as absent).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from src import wandb_log
from src.eval import _compose_eval_cfg, _parse_success_rate
from src.gpu_clocks import log_gpu

# The eval overlay name -> track. The overlay sets `policy=<track>/weights_epoch_<n>.pt`; the
# track selects the SR shim class and the engine directory (`engines/<track>/`). `dino_ep5` is a
# DIAGNOSTIC-ONLY track (epoch-5 DINO snapshot) namespaced apart from the headline `dino` engines
# and SR rows; it reuses the DINO shim (see `_build_shim`).
_TRACK_BY_EXPERIMENT = {
    "eval_lewm": "lewm",
    "eval_dino": "dino",
    "eval_dino_ep5": "dino_ep5",
}


def _experiment_from_argv(argv):
    """The `+experiment=eval_<lewm|dino>` override names the overlay (checkpoint, dataset,
    wandb block). Fail loud if absent — the driver has no other source for the track."""
    for a in argv:
        if a.startswith("+experiment="):
            return a.split("=", 1)[1]
    raise SystemExit(
        "src.sr_eval requires +experiment=eval_<lewm|dino> "
        "(e.g. --config-dir conf +experiment=eval_dino)"
    )


def _track_from_experiment(experiment):
    try:
        return _TRACK_BY_EXPERIMENT[experiment]
    except KeyError:
        raise SystemExit(
            f"unknown experiment {experiment!r}; expected one of "
            f"{sorted(_TRACK_BY_EXPERIMENT)}"
        )


def _split_argv(argv):
    """Separate the driver-only args (`precision=`, `out=`) from the Hydra argv passed through
    to the vendored eval composition (`--config-dir`, `+experiment=`, other overrides). The
    driver-only args are NOT valid Hydra overrides on the pusht config, so they must not reach
    `_compose_eval_cfg` (Hydra would error on them)."""
    precisions = None
    out_dir = None
    hydra_argv = []
    for a in argv:
        if a.startswith("precision="):
            precisions = tuple(p for p in a.split("=", 1)[1].split(",") if p)
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        else:
            hydra_argv.append(a)
    return precisions, out_dir, hydra_argv


def _build_shim(track, model, engines):
    """Wrap the real trained `model` + this precision's two engines in the track's SR shim.
    Imported lazily: `src.sr_shim.*.from_engines` builds `EngineRunner`s that lazy-import
    `tensorrt` and allocate CUDA buffers (pod-only)."""
    from src.sr_shim import DINOWMSRShim, LeWMSRShim

    # DINO-family tracks (`dino`, plus diagnostic snapshots like `dino_ep5`) all use the DINO
    # shim; only `lewm` uses the LeWM shim. Match on the family prefix so a namespaced diagnostic
    # track does not silently fall through to the wrong shim.
    if track.startswith("dino"):
        return DINOWMSRShim.from_engines(model, engines)
    return LeWMSRShim.from_engines(model, engines)


def _merge_sr_json(path, track, sr_by_precision):
    """Read-modify-write the merged `{track: {precision: SR}}` sr.json.

    Written as ONE merged file (the shape `src.study` / `src.report` consume via `sr=<file>`),
    but updated per track key so LeWM and DINOv3 benchmarked in **separate pod sessions** each
    touch only their own track (no clobber, CLAUDE.md §8) — the same durability contract as the
    per-track `results.<track>.json`."""
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[track] = sr_by_precision
    path.write_text(json.dumps(data, indent=2) + "\n")


def _eval_one(hydra_argv, shim):
    """Run the byte-unmodified vendored eval with `shim` slotted into `CEMSolver(model=...)`,
    and return `(success_rate, per_cycle_latencies_ms)`.

    The shim rides in by patching `load_pretrained` (the only seam — see module docstring);
    the patch is scoped to the run and restored after. `output.filename` points at a fresh,
    driver-owned results file (the entrypoint appends, so a shared file would accumulate runs).
    The **per-cycle (per-decision) latency** rides in on the SAME run via the eval overlay's
    observation-only `SolveLatencyRecorder` (`cfg.solver.callbacks`), so per-cycle latency and
    SR come from the same solves (SPEC §Interface Contracts). One record per alive episode per
    solve — NOT one per solve, which would time every episode at once. The raw per-decision list
    is returned so `src.report` can truncate to the common min-n across tracks (equal-n)."""
    import stable_worldmodel as swm

    from scripts.plan import eval_wm

    from src import eval_latency

    out_file = Path(tempfile.mkdtemp(prefix="swm_sreval_")) / "results.txt"
    cfg = _compose_eval_cfg(hydra_argv, out_file)
    eval_latency.reset_registry()
    with patch.object(swm.wm.utils, "load_pretrained", return_value=shim):
        eval_wm.run(cfg)
    latency = eval_latency.pop_records()
    if latency["n_cycles"] == 0:
        raise RuntimeError(
            "the latency callback recorded no decisions — is SolveLatencyRecorder injected "
            "via cfg.solver.callbacks in the eval overlay?"
        )
    return _parse_success_rate(out_file.read_text()), latency["latencies_ms"]


def main():
    argv = sys.argv[1:]
    experiment = _experiment_from_argv(argv)
    track = _track_from_experiment(experiment)
    precisions_arg, out_dir, hydra_argv = _split_argv(argv)

    import stable_worldmodel as swm

    from src.interfaces import ExportConfig
    from src.study import default_out_dir, engine_paths

    precisions = precisions_arg or ExportConfig().precisions
    out_dir = out_dir or default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the REAL trained checkpoint ONCE (outside the load_pretrained patch): the shim wraps
    # it, and the engine callables replace its encode/predict compute per precision. Compose the
    # eval config first only to read `cfg.policy` (the overlay's checkpoint) — one source of
    # truth, same file the eval points at.
    cfg0 = _compose_eval_cfg(
        hydra_argv, Path(tempfile.mkdtemp(prefix="swm_sreval_")) / "results.txt"
    )
    model = swm.wm.utils.load_pretrained(cfg0.policy)

    run = wandb_log.init(
        experiment,
        name=f"sr-eval-{track}",
        config={"phase": "phase5-sr-eval", "track": track},
    )
    sr_by_precision: dict = {}
    try:
        for precision in precisions:
            engines = engine_paths(track, precision)
            if not (engines["encoder"].exists() and engines["predictor"].exists()):
                print(
                    f"[sr-eval:{track}] {precision}: engines missing under "
                    f"{engines['encoder'].parent} — skipped "
                    f"(build with `src.export model={track} precision={precision}`)"
                )
                continue
            shim = _build_shim(track, model, engines)
            try:
                # Bracket the per-cycle eval-shim run with the dmon telemetry observer so the
                # unlocked GPU clock/power/temp state during the headline per-cycle solves is
                # recorded, not merely assumed (SPEC §Parity).
                with log_gpu(f"{track}.{precision}.sr_eval", out_dir / "gpu_logs"):
                    sr, per_cycle_ms = _eval_one(hydra_argv, shim)
                # Carry the RAW per-decision latencies (not pre-reduced percentiles) so src.report
                # truncates to the common min-n across tracks before taking p50/p95 (equal-n).
                sr_by_precision[precision] = {
                    "success_rate": sr,
                    "per_cycle_latencies_ms": per_cycle_ms,
                }

                import wandb

                wandb.log({f"sr/{precision}": sr, f"per_cycle_n/{precision}": len(per_cycle_ms)})
                print(
                    f"[sr-eval:{track}] {precision}: success_rate={sr} "
                    f"n_cycles={len(per_cycle_ms)}"
                )
            finally:
                # Release THIS precision's engine execution contexts before building the next
                # precision's. Each EngineRunner holds direct-cudaMalloc'd context activation
                # memory (~11 GB per engine on DINO) that TensorRT only returns to the driver
                # when the runner is collected. Without this, fp32+fp16+int8 accumulate — the
                # next `shim = _build_shim(...)` allocates while the prior shim is still bound —
                # and the third `create_execution_context` OOMs (DINO only; LeWM's ViT-Tiny
                # contexts stay under the ceiling). `empty_cache` then returns torch's retained
                # caching-allocator pool so the next engine's cudaMalloc can reuse it.
                import gc

                import torch

                del shim
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        run.finish()

    if not sr_by_precision:
        raise SystemExit(
            f"[sr-eval:{track}] no engines found for any of {precisions} under "
            f"{engine_paths(track, 'fp32')['encoder'].parent} — "
            f"run `src.export model={track} precision=<p>` first"
        )

    sr_path = out_dir / "sr.json"
    _merge_sr_json(sr_path, track, sr_by_precision)
    print(
        f"[sr-eval:{track}] wrote {sr_path} — join the SR into the headline with:\n"
        f"    uv run python -m src.report from={out_dir} sr={sr_path}"
    )


if __name__ == "__main__":
    main()
