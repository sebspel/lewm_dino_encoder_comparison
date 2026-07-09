"""Real PyTorch -> ONNX -> TensorRT export.

Owned PLUMBING (fails LOUDLY):
  - the `torch.onnx.export(dynamo=True)` trace call, aimed per-method via thin nn.Module
    wrappers so `encode` and `predict` become **separate** ONNX graphs (2 per model, 4
    across both tracks); precision multiplies TensorRT *engines*, not graphs.
  - the TensorRT-10.7 builder invocation (FP32 default, FP16 flag, dynamic candidate-batch
    optimization profile).
  - the engine-vs-PyTorch precision-match *mechanism* (max abs/rel error).

OWNER-ONLY seams left explicit (fail SILENTLY — STOP and ask before filling):
  - INT8 calibration set + procedure  -> `INT8Calibrator` raises until owner-provided.
  - FP32/FP16/INT8 precision-match tolerances -> `PrecisionTolerance` NaN placeholders.
  - ONNX/TRT export debugging (parse/build failures surface loudly here, judgement is owner's).

Local vs pod: the torch->ONNX trace runs on any box (verified locally on dummy weights);
`build_engine` imports `tensorrt` lazily and only runs on the L40S.
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.export import Dim
from torch.nn.utils.fusion import fuse_linear_bn_eval

from src.interfaces import Precision, EnginePaths, WMStepAdapter

# The predictor engine must accept the CEM candidate fan-out: the solver expands one obs to
# num_samples=300 candidates (docs/platform_api.md §3). Headroom for num_envs > 1.
_MAX_CANDIDATE_BATCH = 512
_WORKSPACE_BYTES = 16 << 30  # 16 GiB TensorRT build scratch (L40S has 48 GiB; TRT-10
#                              default is full device memory — a 4 GiB cap under-cut valid
#                              tactics, e.g. the batch-512 ViT reshape needing ~4.9 GiB)


# --- thin method wrappers: aim the tracer at ONE method each.
class _EncodeModule(nn.Module):
    def __init__(self, adapter: WMStepAdapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, obs: Tensor) -> Tensor:
        return self.adapter.encode(obs)


# `predict` has different arity per track, so it gets an explicit-signature wrapper per
# arity (NOT one variadic `*inputs` wrapper): torch.export then sees real positional params,
# giving a flat `dynamic_shapes` and named ONNX inputs — a variadic collapses every arg into
# one pytree node, which mis-aligns with a flat dynamic_shapes spec and hides the input names.
class _Predict2Module(nn.Module):
    """Trace wrapper for the 2-arg LeWM predict: cached latent + AdaLN action."""

    def __init__(self, adapter: WMStepAdapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        return self.adapter.predict(latent, action)


class _Predict1Module(nn.Module):
    """Trace wrapper for the 1-arg DINO predict: the pre-assembled 404 embedding."""

    def __init__(self, adapter: WMStepAdapter):
        super().__init__()
        self.adapter = adapter

    def forward(self, embedding: Tensor) -> Tensor:
        return self.adapter.predict(embedding)


# Selected by predict arity (number of example predict inputs): DINO drives 1, LeWM drives 2.
_PREDICT_WRAPPERS = {1: _Predict1Module, 2: _Predict2Module}


def _batch_dynamic(n_inputs: int):
    """The `torch.export` dynamic_shapes spec handed to `torch.onnx.export(dynamo=True)` —
    one entry per positional forward arg, each declaring axis 0 (the CEM candidate batch) as
    dynamic so the ONNX graph and the TensorRT optimization profile accept a variable batch.
    Non-batch axes stay static. `encode` has 1 input; `predict` has 2 (LeWM) or 1 (DINO).

    The spec is FLAT (one entry per positional forward arg) because every trace wrapper has
    an explicit-arity `forward` — no variadic pytree collapse to nest around."""
    batch = Dim("batch", min=1, max=_MAX_CANDIDATE_BATCH)
    return tuple({0: batch} for _ in range(n_inputs))


def _fold_linear_bn_eval(module: nn.Module) -> nn.Module:
    """Fold each `Linear -> BatchNorm1d` pair into a single equivalent Linear so the ONNX
    trace never emits `_native_batch_norm_legit_no_training` — a 3-output aten node the
    torch-2.6 `dynamo=True` exporter mishandles (`'tuple' object has no attribute 'dtype'`).
    The fold is exact in eval mode (BN is a per-feature affine map) and is what TensorRT
    fuses anyway. No-op for the DINO track (no BatchNorm). The LeWM `projector` / `pred_proj`
    MLPs are the only pairs this touches."""
    for seq in module.modules():
        if isinstance(seq, nn.Sequential):
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if isinstance(a, nn.Linear) and isinstance(b, nn.BatchNorm1d):
                    seq[i], seq[i + 1] = fuse_linear_bn_eval(a, b), nn.Identity()
    return module


def export_onnx(
    module: nn.Module,
    example_inputs: tuple[Tensor, ...],
    dynamic_shapes,
    out_path: Path,
) -> Path:
    """Trace one method to ONNX via the TorchDynamo exporter (`dynamo=True`; the legacy
    TorchScript exporter is deprecated and on torch 2.6 dynamo is not yet the default, so
    it is passed explicitly). Locally runnable."""
    # deepcopy first: the fold mutates submodules in place, and the encoder/predictor traces
    # share the same underlying adapter object.
    module = _fold_linear_bn_eval(copy.deepcopy(module).eval())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            example_inputs,
            str(out_path),
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
        )
    return out_path


@dataclass(frozen=True)
class PrecisionTolerance:
    """OWNER-ONLY: FP32/FP16/INT8 precision-matching thresholds fail *silently*.
    Left NaN so pass/fail is **disabled** — error is measured
    and logged but never silently gated — until owner sign-off on the L40S."""

    rtol: float = math.nan  # TODO(owner): set after seeing real engine error on the pod
    atol: float = math.nan  # TODO(owner)


def precision_match(
    reference: Tensor,
    engine_out: Tensor,
    tol: PrecisionTolerance = PrecisionTolerance(),
) -> dict:
    """Engine-vs-PyTorch precision-match MECHANISM (owner sets the policy). Returns max
    abs/rel error; `passed` stays None until owner tolerances are set. Locally runnable.
    """
    ref = reference.detach().float()
    # Engine output lands on CUDA (EngineRunner allocates cuda buffers) while the PyTorch
    # reference is a CPU tensor; harmonize onto the reference's device before comparing.
    out = engine_out.detach().float().to(ref.device)
    diff = (ref - out).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.abs().clamp_min(1e-12)).max().item()
    passed = None
    if not (math.isnan(tol.rtol) or math.isnan(tol.atol)):
        passed = bool((diff <= tol.atol + tol.rtol * ref.abs()).all())
    return {"max_abs": max_abs, "max_rel": max_rel, "passed": passed}


class INT8Calibrator:
    """OWNER-ONLY seam. The INT8 calibration set + procedure need owner sign-off —
    a bad calib set silently degrades every INT8 number.
    FP16 is the sanctioned fallback until then."""

    def __init__(self, calib_loader):
        raise NotImplementedError(
            "INT8 calibration is OWNER-ONLY: STOP and ask for the calibration set + "
            "procedure before building an INT8 engine. FP16 is the fallback."
        )


def _build_calibrator(calib_loader):
    return INT8Calibrator(calib_loader)


def build_engine(
    onnx_path: Path,
    precision: Precision,
    out_path: Path,
    example_inputs: tuple[Tensor, ...],
    calibrator=None,
) -> Path:
    """TensorRT-10.7 builder invocation (owned plumbing). FP32 default, FP16 flag; INT8
    needs an owner-provided calibrator. Parse/build failures raise loudly (the *judgement*
    on how to fix them is owner's — ONNX/TRT debugging). Runs ONLY on the L40S (`tensorrt`
    imported lazily so this module imports off-pod)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10: explicit batch is implicit
    parser = trt.OnnxParser(network, logger)
    # Pass the model path so TRT can resolve external-data sidecars (`<name>.onnx.data`);
    # the dynamo exporter externalizes initializers, and a bare byte buffer loses the base
    # dir needed to find them (would fail: "Failed to open file: encoder.onnx.data").
    if not parser.parse(onnx_path.read_bytes(), str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")  # -> OWNER

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE_BYTES)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if calibrator is None:
            raise ValueError("int8 build requires an owner-provided INT8 calibrator")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

    # Optimization profile for the dynamic candidate-batch axis (axis 0): min 1, opt at the
    # example batch, max the CEM candidate count. Non-batch axes stay at the example shape.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        rest = tuple(example_inputs[i].shape[1:])
        profile.set_shape(
            inp.name,
            (1, *rest),
            tuple(example_inputs[i].shape),
            (_MAX_CANDIDATE_BATCH, *rest),
        )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build returned None for {onnx_path}")  # -> OWNER
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(serialized)
    return out_path


def export(
    adapter: WMStepAdapter,
    precision: Precision,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
    engine_dir: Path,
    calib_loader=None,
) -> EnginePaths:
    """PyTorch -> ONNX -> TensorRT for both methods -> {encoder, predictor} engine paths.
    `encode` and `predict` are traced and built SEPARATELY.
    """
    if precision == "int8" and calib_loader is None:
        raise ValueError("int8 export requires a calibration loader (owner-provided)")
    engine_dir.mkdir(parents=True, exist_ok=True)
    calibrator = _build_calibrator(calib_loader) if precision == "int8" else None

    predict_arity = len(predict_inputs)
    if predict_arity not in _PREDICT_WRAPPERS:
        raise ValueError(
            f"unsupported predict arity {predict_arity}; expected 1 (DINO) or 2 (LeWM)"
        )
    specs = {
        "encoder": (
            _EncodeModule(adapter),
            encode_inputs,
            _batch_dynamic(len(encode_inputs)),
        ),
        "predictor": (
            _PREDICT_WRAPPERS[predict_arity](adapter),
            predict_inputs,
            _batch_dynamic(predict_arity),
        ),
    }
    engines: dict[str, Path] = {}
    for name, (module, inputs, dyn) in specs.items():
        onnx_path = export_onnx(module, inputs, dyn, engine_dir / f"{name}.onnx")
        engines[name] = build_engine(
            onnx_path,
            precision,
            engine_dir / f"{name}.{precision}.plan",
            inputs,
            calibrator,
        )
    return EnginePaths(encoder=engines["encoder"], predictor=engines["predictor"])


# Engines are large + device-specific, so they land in a repo-local, gitignored dir
# (`*.plan`/`*.onnx` are ignored) — regenerable on the L40S, one subdir per track.
_ENGINE_ROOT = Path("engines")


def main() -> None:
    """CLI: build the encoder + predictor engines for one track at one precision on the
    L40S. Reuses the real-checkpoint loader + shared example inputs so the traced
    graph is the exact `encode`/`predict` call pattern the benchmark drives. INT8 is gated —
    it needs the owner calibration set (deferred step), so it fails loud here.

        uv run python -m src.export model=<lewm|dino> precision=<fp32|fp16|int8>
    """
    from src.interfaces import ExportConfig
    from src.precision_match import _build_adapter, example_inputs

    model = "lewm"
    precision: Precision = "fp32"
    for a in sys.argv[1:]:
        if a.startswith("model="):
            model = a.split("=", 1)[1]
        elif a.startswith("precision="):
            precision = a.split("=", 1)[1]  # type: ignore[assignment]

    cfg = ExportConfig()
    torch.manual_seed(cfg.seed)
    adapter, name = _build_adapter(model)
    encode_inputs, predict_inputs = example_inputs(adapter, cfg)

    engines = export(
        adapter,
        precision=precision,
        encode_inputs=encode_inputs,
        predict_inputs=predict_inputs,
        engine_dir=_ENGINE_ROOT / name,
    )
    for method, path in engines.items():
        print(f"[{name}/{precision}] {method}: {path}")


if __name__ == "__main__":
    main()
