"""SR-per-precision eval driver — re-run the Phase-3 CEM eval on the OPTIMIZED (engine) model.

Phase-5 pairs every speed number with a Push-T success rate (SPEC "no speed number without
its task-quality counterpart"). The benchmark (`src.benchmark` / `src.study`) leaves
`success_rate=NaN`; this driver produces the SR that fills it, per precision, by re-running the
**same** platform CEM eval on the exported/quantized engines — then writes the
`{track: {precision: SR}}` JSON that `src.study` / `src.report` join back in via `sr=<file>`.

    uv run python -m src.sr_eval --config-dir conf +experiment=eval_<lewm|dino> \
        [precision=fp32,fp16,int8,fp8] [calibration_method=max|entropy] [out=<dir>]

The quantized (int8/fp8) SR depends on the PTQ **calibration method** (`max` | `entropy`, a build
option for both tracks — SPEC §Parity), so each SR is tagged with it: the merged sr.json is keyed
`{track: {precision: {method: SR}}}`, and a run only touches its own (track, precision, method)
points. So `int8` @ `entropy` for a track lands BESIDE `int8` @ `max` for that track in the SAME
file — neither overwrites the other. `calibration_method` labels which method's engines this run
built/evaluated (it must match how `src.export` built them); it defaults to `max`. FP32/FP16 are
method-invariant and recorded under the `max` label.

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
from src.interfaces import DEFAULT_CALIBRATION_METHOD, check_calibration_method

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
    """Separate the driver-only args (`precision=`, `out=`, `calibration_method=`) from the Hydra
    argv passed through to the vendored eval composition (`--config-dir`, `+experiment=`, other
    overrides). The driver-only args are NOT valid Hydra overrides on the pusht config, so they
    must not reach `_compose_eval_cfg` (Hydra would error on them)."""
    precisions = None
    out_dir = None
    method = None
    hydra_argv = []
    for a in argv:
        if a.startswith("precision="):
            precisions = tuple(p for p in a.split("=", 1)[1].split(",") if p)
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("calibration_method="):
            method = a.split("=", 1)[1]
        else:
            hydra_argv.append(a)
    return precisions, out_dir, method, hydra_argv


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


def _as_method_map(existing):
    """Normalize one sr.json precision entry into a `{method: SR}` map. `None` -> empty; a legacy
    flat `{success_rate, ...}` (pre-labelling, always `max`-calibrated) -> wrapped under the
    explicit `max` label (lossless — the max data is relocated, not changed); an already-labelled
    `{method: SR}` map -> returned as-is."""
    if existing is None:
        return {}
    if isinstance(existing, dict) and "success_rate" in existing:
        return {DEFAULT_CALIBRATION_METHOD: existing}
    return existing


def _merge_sr_json(path, track, method, sr_by_precision):
    """Read-modify-write the merged sr.json, keyed `{track: {precision: {method: SR}}}`.

    ONE file across methods (the shape `src.study` / `src.report` consume via `sr=<file>`),
    updated per **(track, precision, method)** so every partial run is additive and NOTHING already
    recorded is discarded (CLAUDE.md §8):
      - LeWM and DINOv3 in separate pod sessions touch only their own track;
      - a precision subset (`precision=fp8`) touches only those precisions — the earlier whole-
        -track-block replace discarded a track's other precisions on any subset re-run;
      - a second calibration method (`int8` @ `entropy`) lands BESIDE the first (`int8` @ `max`)
        under the same precision, so `max`- and `entropy`-calibrated points coexist in one file.
    Re-running the SAME (track, precision, method) overwrites only that one point (a re-measurement).
    Legacy pre-labelling entries are folded under the `max` label on first touch (`_as_method_map`)."""
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    track_block = data.setdefault(track, {})
    for precision, val in sr_by_precision.items():
        method_map = _as_method_map(track_block.get(precision))
        method_map[method] = val
        track_block[precision] = method_map
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
    precisions_arg, out_dir, method_arg, hydra_argv = _split_argv(argv)
    method = check_calibration_method(method_arg or DEFAULT_CALIBRATION_METHOD)

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
        name=f"sr-eval-{track}-{method}",
        config={"phase": "phase5-sr-eval", "track": track, "calibration_method": method},
    )
    sr_by_precision: dict = {}
    try:
        for precision in precisions:
            # SR is method-DEPENDENT, so load the engine tagged with THIS run's method (int8/fp8);
            # fp32/fp16 are method-invariant and ignore it (study.engine_paths).
            engines = engine_paths(track, precision, method=method)
            if not (engines["encoder"].exists() and engines["predictor"].exists()):
                print(
                    f"[sr-eval:{track}] {precision} ({method}): engines missing under "
                    f"{engines['encoder'].parent} — skipped "
                    f"(build with `src.export model={track} precision={precision} "
                    f"calibration_method={method}`)"
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

                # Tag the SR with its calibration method so int8/fp8 @ max vs @ entropy are
                # distinct series (FP32/FP16 are method-invariant, logged under the run's method).
                wandb.log(
                    {f"sr/{precision}.{method}": sr, f"per_cycle_n/{precision}": len(per_cycle_ms)}
                )
                print(
                    f"[sr-eval:{track}] {precision} ({method}): success_rate={sr} "
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
    _merge_sr_json(sr_path, track, method, sr_by_precision)
    print(
        f"[sr-eval:{track}] wrote {sr_path} (method={method}) — join the SR into the headline with:\n"
        f"    uv run python -m src.report from={out_dir} sr={sr_path} calibration_method={method}"
    )


if __name__ == "__main__":
    main()
