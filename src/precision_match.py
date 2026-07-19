"""Precision-match test — engine-vs-PyTorch drift per method × precision.

For one model track, this builds the shared example inputs, computes the PyTorch reference
ONCE, then for each precision exports the two engines (`src.export`) and runs each against
its reference (`src.trt_runtime.engine_vs_reference`), reporting max abs/rel drift.

Owned PLUMBING (fails LOUDLY): input construction, the export/compare loop, the table.
OWNER-ONLY (fails SILENTLY, STOP and ask): the pass/fail decision is NOT coded — drift is
measured and printed, and the precision-match gate is an owner sign-off on that drift on the
L40S (no tolerance object, no automated gating). INT8 routes through the owner-approved
calibration set
(`src.calibrate.build_calibration_data`) + Model-Optimizer Q/DQ, so its drift row IS the
PTQ/calibration-quality signal the owner inspects here.

Runs on the L40S: `export` + engine execution need `tensorrt` + CUDA. `example_inputs`
and `reference_outputs` are pure torch and run anywhere.

    uv run python -m src.precision_match track=<lewm|dino>
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch import Tensor

from src.export import export, precision_match as engine_drift
from src.interfaces import (
    ExportConfig,
    WMStepAdapter,
    QUANTIZED_PRECISIONS,
    calibration_method_for,
)
from src.trt_runtime import engine_vs_reference

# Precision-match batches: exercise the engine at the optimization profile's min / opt / max
# — TensorRT tunes at the `opt` point, which `build_engine` pins to the example-input batch
# (docs/platform_api.md §3: CEM num_samples=300 is the max candidate fan-out). Validating a
# batch != the trace batch is load-bearing: the predictor's batch axis is a torch.export-
# specialized symbol some consumers fold to the trace batch, so a single opt-batch check
# would pass even a batch-frozen engine (TRT keeps the axis dynamic; these off-opt batches
# prove it end-to-end).
_MATCH_BATCH = 8
_MATCH_BATCHES = (1, _MATCH_BATCH, 300)

# Off-nominal HISTORY windows the platform rollout actually feeds the predictor (n_obs=1 ->
# min(n_obs, HS) grows 1, 2, 3): the batch sweep above only exercises the traced HS, but the
# predictor MIXES across the history axis and the engine is fixed-HS, so a fixed-HS-only check
# passes a hist-mismatched predict engine (it crashed only on the real SR run). These rows drive
# the shim's hist-adapt (pad-to-HS / slice) and compare it against the native-T PyTorch predict
# — the variable-window engine-vs-torch parity the SPEC requires (§Requirements — engine-fidelity
# gate). Same fix/gate for both tracks (both predictors are owner-confirmed causal).
_MATCH_HISTS = (1, 2)


def example_inputs(
    adapter: WMStepAdapter,
    cfg: ExportConfig,
    batch: int = _MATCH_BATCH,
    device: str | torch.device = "cpu",
    hist: int | None = None,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Build the SHARED example inputs both export-tracing and the reference consume:
    `encode` gets an obs tensor; `predict` gets the predictor STATE (from one cached encode)
    plus the per-track conditioning — the exact call pattern the CEM rollout drives (encode
    once, predict many). LeWM's predict is `(latent, action)`; DINO's predict is a single
    pre-assembled 404 embedding (`assemble_embedding` tiles proprio+action onto the 384
    latent — a Python step outside the compiled predict). `device` places the tensors on the
    adapter's device (CPU for tracing off-pod; CUDA for the pod reference at large batch).

    `hist` overrides the frame-axis length (default `cfg.hist` = the traced `HS`). The
    off-nominal predictor gate (`_MATCH_HISTS`) builds `hist < HS` windows to exercise the
    shim's hist-adapt against the native-`T` PyTorch predict.
    """
    from src.adapter import DINOWMAdapter

    hist = cfg.hist if hist is None else hist
    obs = torch.randn(batch, hist, *cfg.obs_shape, device=device)
    with torch.no_grad():
        latent = adapter.encode(obs)  # cache the latent — predict reuses THIS tensor
        action = torch.randn(batch, hist, cfg.action_dim, device=device)
        if isinstance(adapter, DINOWMAdapter):
            proprio = torch.randn(batch, hist, cfg.proprio_dim, device=device)
            embedding = adapter.assemble_embedding(latent, proprio, action)
            return (obs,), (embedding,)
    return (obs,), (latent, action)


def reference_outputs(
    adapter: WMStepAdapter,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
) -> dict[str, Tensor]:
    """Compute the PyTorch reference outputs the engines are compared against. Keys
    ("encoder"/"predictor") line up with `EnginePaths` so the loop can pair engine ->
    reference by name. `predict` reuses the cached latent already in `predict_inputs`
    (no re-encode), isolating predict's quantization drift from encode's."""
    adapter.eval()
    reference_dict = {}
    with torch.no_grad():
        reference_dict["encoder"] = adapter.encode(*encode_inputs)
        reference_dict["predictor"] = adapter.predict(*predict_inputs)

    return reference_dict


def precision_match_track(
    adapter: WMStepAdapter,
    name: str,
    precisions: tuple[str, ...],
    cfg: ExportConfig,
    engine_dir: Path,
) -> list[dict]:
    """Export each precision (traced ONCE at the profile opt batch) and measure engine-vs-
    PyTorch drift for both methods at each of `_MATCH_BATCHES` (profile min/opt/max), PLUS the
    off-nominal history windows `_MATCH_HISTS` (`T < HS`) the rollout feeds the predictor. The
    reference runs on CUDA when available so the max-batch check is feasible for both tracks;
    engines are built on the CPU-traced graph first, then the adapter is moved to the
    reference device (the engine already carries the same weights).

    The batch-sweep rows run the engines directly at the traced `HS`; the `_MATCH_HISTS` rows run
    through the SHIM's own hist-adapt wrappers (`build_engine_fns` / `build_lewm_engine_fns`) —
    so the gate tests the real production pad-to-`HS`/slice path against the native-`T` predict,
    not a reimplementation."""
    # Trace + build every precision from the CPU opt-batch inputs (unchanged trace behavior).
    opt_encode, opt_predict = example_inputs(adapter, cfg, batch=_MATCH_BATCH)

    # Quantized precisions (int8/fp8) need the owner-approved calibration set (drawn ONCE from the
    # real Push-T data and streamed through this adapter for the predictor). Built here — before
    # `adapter.to(device)` below — so the calibration encode runs on the same CPU graph the
    # trace/build use, matching the `src.export` CLI path. FP32/FP16 build data-free
    # (`calib_loader=None`). One shared loader across int8/fp8 (format-independent streams).
    calib_loader = None
    if any(p in QUANTIZED_PRECISIONS for p in precisions):
        from src.calibrate import build_calibration_data

        calib_loader = build_calibration_data(batch=_MATCH_BATCH)

    engines_by_precision = {
        precision: export(
            adapter,
            precision=precision,
            encode_inputs=opt_encode,
            predict_inputs=opt_predict,
            engine_dir=engine_dir / precision,
            calib_loader=calib_loader if precision in QUANTIZED_PRECISIONS else None,
            # Per-track PTQ method (max/entropy — owner-set, ADR-0002), so the drift table the
            # owner signs off is measured on the engine that SR-eval will build.
            calibration_method=calibration_method_for(name),
        )
        for precision in precisions
    }

    # Now validate drift at min/opt/max on the reference device (GPU if present).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter.to(device)

    # The shim's engine wrappers apply the hist-adapt (encoder repeat-pad; predictor causal
    # right-pad/slice) — reused here so the off-nominal rows exercise the real production path.
    from src.sr_shim import build_engine_fns, build_lewm_engine_fns

    build_fns = build_lewm_engine_fns if name == "lewm" else build_engine_fns

    rows: list[dict] = []
    for precision in precisions:
        engines = engines_by_precision[precision]
        # (1) Traced-HS batch sweep (profile min/opt/max): engines run directly.
        for batch in _MATCH_BATCHES:
            encode_inputs, predict_inputs = example_inputs(
                adapter, cfg, batch=batch, device=device
            )
            ref = reference_outputs(adapter, encode_inputs, predict_inputs)
            enc = engine_vs_reference(engines["encoder"], ref["encoder"], encode_inputs)
            pred = engine_vs_reference(
                engines["predictor"], ref["predictor"], predict_inputs
            )
            rows.append(
                {
                    "model": name,
                    "precision": precision,
                    "batch": batch,
                    "hist": cfg.hist,
                    "encode_max_abs": enc["max_abs"],
                    "encode_max_rel": enc["max_rel"],
                    "predict_max_abs": pred["max_abs"],
                    "predict_max_rel": pred["max_rel"],
                }
            )
        # (2) Off-nominal sub-HS history windows (T < HS): route through the shim's hist-adapt
        # wrappers and compare against the native-T PyTorch reference (the variable-window parity).
        encode_fn, predict_fn = build_fns(engines)
        for hist in _MATCH_HISTS:
            encode_inputs, predict_inputs = example_inputs(
                adapter, cfg, batch=_MATCH_BATCH, device=device, hist=hist
            )
            ref = reference_outputs(adapter, encode_inputs, predict_inputs)
            enc = engine_drift(ref["encoder"], encode_fn(*encode_inputs))
            pred = engine_drift(ref["predictor"], predict_fn(*predict_inputs))
            rows.append(
                {
                    "model": name,
                    "precision": precision,
                    "batch": _MATCH_BATCH,
                    "hist": hist,
                    "encode_max_abs": enc["max_abs"],
                    "encode_max_rel": enc["max_rel"],
                    "predict_max_abs": pred["max_abs"],
                    "predict_max_rel": pred["max_rel"],
                }
            )
    return rows


def _print_table(rows: list[dict]) -> None:
    # `hist < cfg.hist` rows are the off-nominal (sub-HS) predictor windows the rollout feeds;
    # `hist == cfg.hist` rows are the traced-HS batch sweep.
    hdr = f"{'model':>6} {'prec':>5} {'batch':>6} {'hist':>5} {'enc_abs':>10} {'enc_rel':>10} {'pred_abs':>10} {'pred_rel':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:>6} {r['precision']:>5} {r['batch']:>6} {r['hist']:>5} "
            f"{r['encode_max_abs']:>10.3e} {r['encode_max_rel']:>10.3e} "
            f"{r['predict_max_abs']:>10.3e} {r['predict_max_rel']:>10.3e}"
        )


# Trained checkpoints, addressed by the explicit epoch-10 .pt (the folder holds earlier
# snapshots too, so a bare run name is ambiguous — load_pretrained format-1). Same names the
# eval overlays will need for the SR-per-precision re-run.
#
# `dino_ep5` is a DIAGNOSTIC-ONLY track: the epoch-5 DINO snapshot wrapped in the same DINO
# adapter/shim, but namespaced apart so its engines (`engines/dino_ep5/`) and SR rows never
# collide with — nor replace — the headline `dino` (epoch-10) track. It is deliberately NOT in
# `study._TRACKS`, so it never enters the 2-track headline.
_CHECKPOINTS = {
    "lewm": "lewm/weights_epoch_10.pt",
    "dino": "dino/weights_epoch_10.pt",
    "dino_ep5": "dino/weights_epoch_5.pt",
}


def _build_adapter(track: str) -> tuple[WMStepAdapter, str]:
    """Materialize the REAL trained checkpoint via the platform `load_pretrained` (reusing
    the eval load path, not a hand-rolled `torch.load`) and wrap in the matching
    adapter. Drift numbers are only meaningful on trained weights, so this runs on the pod
    where the checkpoints + their DINOv3 backbone live. Fails loud."""
    import stable_worldmodel as swm

    from src.adapter import DINOWMAdapter, LeWMAdapter

    if track not in _CHECKPOINTS:
        raise SystemExit(
            f"unknown track {track!r}; expected one of {sorted(_CHECKPOINTS)}"
        )
    model = swm.wm.utils.load_pretrained(_CHECKPOINTS[track])
    adapter = LeWMAdapter(model) if track == "lewm" else DINOWMAdapter(model)
    return adapter, track


def main() -> None:
    track = "lewm"
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8", "fp8")
    for a in sys.argv[1:]:
        if a.startswith("track="):
            track = a.split("=", 1)[1]

    torch.manual_seed(0)
    adapter, name = _build_adapter(track)
    cfg = ExportConfig()
    with tempfile.TemporaryDirectory() as d:
        rows = precision_match_track(adapter, name, precisions, cfg, Path(d))
    _print_table(rows)


if __name__ == "__main__":
    main()
