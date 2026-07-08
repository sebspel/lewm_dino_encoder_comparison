"""Precision-match test (Phase 5) — engine-vs-PyTorch drift per method × precision.

For one model track, this builds the shared example inputs, computes the PyTorch reference
ONCE, then for each precision exports the two engines (`src.export`) and runs each against
its reference (`src.trt_runtime.engine_vs_reference`), reporting max abs/rel drift.

Owned PLUMBING (fails LOUDLY): input construction, the export/compare loop, the table.
OWNER-ONLY (fails SILENTLY, STOP and ask): the pass/fail TOLERANCES stay unset
(`src.export.PrecisionTolerance` — NaN), so drift is measured and printed but never gated
until owner sign-off on the L40S. INT8 needs the owner calibration set first (FP32/FP16
runnable now).

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

from src.export import export
from src.interfaces import ExportConfig, WMStepAdapter
from src.trt_runtime import engine_vs_reference

# Precision-match batch: exercise the engine at a representative candidate fan-out, not
# batch=1 — TensorRT tunes at the optimization profile's `opt` point, which `build_engine`
# pins to the example-input batch (docs/platform_api.md §3: CEM num_samples=300).
_MATCH_BATCH = 8


def example_inputs(
    adapter: WMStepAdapter, cfg: ExportConfig, batch: int = _MATCH_BATCH
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Build the SHARED example inputs both export-tracing and the reference consume:
    `encode` gets an obs tensor; `predict` gets the *cached* latent (from one encode) plus
    the per-track conditioning — the exact call pattern the CEM rollout drives (encode once,
    predict many). DINO additionally carries proprio (its predict is `(latent, proprio,
    action)`); LeWM carries only action.
    """
    from src.adapter import DINOWMAdapter

    obs = torch.randn(batch, cfg.hist, *cfg.obs_shape)
    with torch.no_grad():
        latent = adapter.encode(obs)  # cache the latent — predict reuses THIS tensor
    action = torch.randn(batch, cfg.hist, cfg.action_dim)
    if isinstance(adapter, DINOWMAdapter):
        proprio = torch.randn(batch, cfg.hist, cfg.proprio_dim)
        return (obs,), (latent, proprio, action)
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
    """Export each precision and measure engine-vs-PyTorch drift for both methods.
    Inputs + reference are built ONCE and shared across all precisions."""
    encode_inputs, predict_inputs = example_inputs(adapter, cfg)
    ref = reference_outputs(adapter, encode_inputs, predict_inputs)

    rows: list[dict] = []
    for precision in precisions:
        engines = export(
            adapter,
            precision=precision,
            encode_inputs=encode_inputs,
            predict_inputs=predict_inputs,
            engine_dir=engine_dir / precision,
        )
        enc = engine_vs_reference(engines["encoder"], ref["encoder"], encode_inputs)
        pred = engine_vs_reference(
            engines["predictor"], ref["predictor"], predict_inputs
        )
        rows.append(
            {
                "model": name,
                "precision": precision,
                "encode_max_abs": enc["max_abs"],
                "encode_max_rel": enc["max_rel"],
                "predict_max_abs": pred["max_abs"],
                "predict_max_rel": pred["max_rel"],
            }
        )
    return rows


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'model':>6} {'prec':>5} {'enc_abs':>10} {'enc_rel':>10} {'pred_abs':>10} {'pred_rel':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:>6} {r['precision']:>5} "
            f"{r['encode_max_abs']:>10.3e} {r['encode_max_rel']:>10.3e} "
            f"{r['predict_max_abs']:>10.3e} {r['predict_max_rel']:>10.3e}"
        )


# Phase-2 checkpoints, addressed by the explicit epoch-10 .pt (the folder holds earlier
# snapshots too, so a bare run name is ambiguous — load_pretrained format-1). Same names the
# eval overlays will need for the SR-per-precision re-run.
_CHECKPOINTS = {
    "lewm": "lewm/weights_epoch_10.pt",
    "dino": "dino/weights_epoch_10.pt",
}


def _build_adapter(track: str) -> tuple[WMStepAdapter, str]:
    """Materialize the REAL trained checkpoint via the platform `load_pretrained` (reusing
    the Phase-3 eval load path, not a hand-rolled `torch.load`) and wrap in the matching
    adapter. Drift numbers are only meaningful on trained weights, so this runs on the pod
    where the checkpoints + their DINOv3 backbone live. Fails loud."""
    import stable_worldmodel as swm

    from src.adapter import DINOWMAdapter, LeWMAdapter

    if track not in _CHECKPOINTS:
        raise SystemExit(f"unknown track {track!r}; expected 'lewm' or 'dino'")
    model = swm.wm.utils.load_pretrained(_CHECKPOINTS[track])
    adapter = LeWMAdapter(model) if track == "lewm" else DINOWMAdapter(model)
    return adapter, track


def main() -> None:
    track = "lewm"
    precisions: tuple[str, ...] = (
        "fp32",
        "fp16",
    )  # int8 needs the owner calibration set
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
