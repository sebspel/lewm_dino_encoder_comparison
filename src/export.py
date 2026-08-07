"""Real PyTorch -> ONNX -> TensorRT export.

Owned PLUMBING (fails LOUDLY):
  - the `torch.onnx.export(dynamo=True)` trace call, aimed per-method via thin nn.Module
    wrappers so `encode` and `predict` become **separate** ONNX graphs (2 per model, 4
    across both tracks); precision multiplies TensorRT *engines*, not graphs.
  - the TensorRT-10.7 builder invocation (FP32 default, FP16 flag, per-component optimization
    profile pinned to each engine's production call batch — `_BATCH_PROFILE`).
  - the engine-vs-PyTorch precision-match *mechanism* (max abs/rel error).

INT8 is **explicit Q/DQ** via the NVIDIA TensorRT Model Optimizer, not a build-time TRT
calibrator: `quantize_onnx` rewrites the base FP32 ONNX into a Q/DQ-annotated ONNX with
per-tensor scales baked in from the calibration pass (run on the GPU / CUDA EP when
available), and `build_engine` then parses that
quantized graph like FP32/FP16 (TRT honors the embedded Q/DQ — no `int8_calibrator`, no
calibration profile). FP8 (E4M3) rides the EXACT same path — a second quantized format, not a
second code path: same calibration streams + `max` method, `quantize_mode="fp8"` into the Model
Optimizer, and a `BuilderFlag.FP8`+FP16 build (SPEC §Export shape). The calibration set
(OWNER-signed-off knobs: `max` method, 512 clips strided across all episodes, drawn through the
platform at matched ImageNet norm) lives in `src.calibrate`; `export` builds a per-method numpy
dict (encoder obs / predictor per-track input) keyed by ONNX input name and hands it to
`quantize_onnx`.

OWNER-ONLY seams left explicit (fail SILENTLY — STOP and ask before filling):
  - FP32/FP16/INT8 precision matching: `precision_match` reports drift only; the gate is an
    owner sign-off on the measured drift, deliberately NOT coded into a pass/fail here.
  - the Model-Optimizer quant config beyond the `max` method (Q/DQ format,
    per-channel-vs-per-tensor, op-type exclusions) — left at the tool's INT8 defaults,
    owner-confirmed against measured drift at the pod precision-match gate.
  - ONNX/Model-Optimizer/TRT export debugging (parse/build/quantize failures surface loudly
    here, judgement is owner's).

Local vs pod: the torch->ONNX trace runs on any box (verified locally on dummy weights);
`quantize_onnx` (imports `modelopt` lazily) and `build_engine` (imports `tensorrt` lazily)
only run on the L40S.
"""

from __future__ import annotations

import contextlib
import copy
import os
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.export import Dim
from torch.nn.utils.fusion import fuse_linear_bn_eval

from src.interfaces import (
    Precision,
    EnginePaths,
    WMStepAdapter,
    QUANTIZED_PRECISIONS,
    DEFAULT_CALIBRATION_METHOD,
)

# The predictor engine must accept the CEM candidate fan-out. The solver loops envs in chunks
# of batch_size and expands each to (batch_size, num_samples), so predict batch =
# batch_size * num_samples = 1 * 300 = 300 under the parity config (batch_size=1, num_samples
# =300; stable_worldmodel/solver/lagrangian.py, docs/platform_api.md §3). num_envs>1 does NOT
# enlarge this axis (envs are looped, not batched). 300 is also the only feasible ceiling for
# the DINO predictor: its (batch, 16, 588, 588) attention tensor exceeds TensorRT's 2^31
# element-volume limit above batch 388.
_MAX_CANDIDATE_BATCH = 300
# The encoder engine's batch. The CEM slices the candidate axis away BEFORE encoding
# (`PreJEPA.rollout`: `init_info_dict[k] = info[k][:, 0]`, then `.expand(...)` the latent across
# candidates; `get_cost` embeds the goal by the same `[:, 0]` path) and the vendored solver pins
# `batch_size = 1`, so the encoder is only ever called at batch 1 — the two cached, batch-1 encodes
# per decision that `ENCODER_CALLS_PER_CYCLE` counts. docs/architecture.md §6.
_ENCODER_BATCH = 1

# (min, opt, max) batch per component — the shape each engine is ACTUALLY called at. TensorRT
# selects tactics at `opt`, so this is the load-bearing knob: an engine tuned at a batch it never
# runs at is tuned for the wrong kernel. The predictor keeps `min = 1` for the profile-min rows the
# precision-match gate drives (a profile minimum costs nothing). Single source of truth for the
# convention — `build_engine` takes the triple explicitly rather than inferring it from the example
# inputs, whose batch is a TRACE property, not a build one. docs/architecture.md §6,
# SPEC §Interface Contracts (Export shape).
_BATCH_PROFILE: dict[str, tuple[int, int, int]] = {
    "encoder": (_ENCODER_BATCH, _ENCODER_BATCH, _ENCODER_BATCH),
    "predictor": (1, _MAX_CANDIDATE_BATCH, _MAX_CANDIDATE_BATCH),
}
_WORKSPACE_BYTES = 24 << 30  # 24 GiB TensorRT per-tactic scratch CEILING (not a reservation;
#                              runtime uses only what the selected tactics need). L40S has
#                              48 GiB; a 16 GiB cap still pruned a ~19.5 GiB DINO-predictor
#                              attention tactic, so widen the search to not handicap the
#                              latency study. Uniform across tracks/precisions; govern real
#                              footprint via measured peak GPU memory, not this knob.


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
    an explicit-arity `forward` — no variadic pytree collapse to nest around.

    This declares the graph's admissible RANGE; it is the BUILD-time `_BATCH_PROFILE` that pins
    each engine to its production call batch within that range."""
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


@contextlib.contextmanager
def _slice_based_splits():
    """Trace-time shim: reimplement `Tensor.chunk` and `Tensor.split` with `narrow` so the
    dynamo exporter emits `Slice` instead of `SplitToSequence`/`SequenceAt` — sequence-typed
    ONNX ops the TensorRT parser rejects (UNSUPPORTED_NODE). Both splits we hit are along a
    static axis (LeWM attention-QKV/AdaLN on the feature axis; DINOv3 RoPE prefix-vs-patch on
    the token axis), so slicing is numerically exact (verified 0.0 drift). No-op where the
    traced graph has neither op."""
    orig_chunk, orig_split = torch.Tensor.chunk, torch.Tensor.split

    def _chunk(self, chunks, dim=0):
        n = self.size(dim)
        step = (n + chunks - 1) // chunks  # torch.chunk's ceil-sized chunks
        return tuple(
            torch.narrow(self, dim, i, min(step, n - i)) for i in range(0, n, step)
        )

    def _split(self, split_size_or_sections, dim=0):
        n = self.size(dim)
        if isinstance(split_size_or_sections, int):
            s = split_size_or_sections
            sections = [min(s, n - i) for i in range(0, n, s)]
        else:
            sections = list(split_size_or_sections)
        out, start = [], 0
        for sec in sections:
            out.append(torch.narrow(self, dim, start, sec))
            start += sec
        return tuple(out)

    torch.Tensor.chunk, torch.Tensor.split = _chunk, _split
    try:
        yield
    finally:
        torch.Tensor.chunk, torch.Tensor.split = orig_chunk, orig_split


def export_onnx(
    module: nn.Module,
    example_inputs: tuple[Tensor, ...],
    dynamic_shapes,
    out_path: Path,
) -> Path:
    """Trace one method to ONNX via the TorchDynamo exporter (`dynamo=True`; the legacy
    TorchScript exporter is deprecated and on torch 2.6 dynamo is not yet the default, so
    it is passed explicitly). Locally runnable."""
    # A dim marked dynamic but handed an example extent of 1 is SILENTLY specialized: the dynamo
    # exporter emits a frozen `dim_value: 1` instead of a symbol, with no warning and no error
    # (verified on torch 2.6), and every engine built off that graph is batch-frozen. The trace
    # batch is free to differ from the build profile (`_BATCH_PROFILE`) — but it must be ≥ 2 or
    # the axis it is meant to leave open is gone. Fail here, loudly.
    for i, (example, spec) in enumerate(zip(example_inputs, dynamic_shapes)):
        if 0 in spec and example.shape[0] < 2:
            raise ValueError(
                f"input {i} of {out_path.name} declares a dynamic batch axis but is traced at "
                f"batch {example.shape[0]}: torch.export specializes a size-1 dim, freezing the "
                "axis in the ONNX graph. Trace at batch >= 2."
            )
    # deepcopy first: the fold mutates submodules in place, and the encoder/predictor traces
    # share the same underlying adapter object.
    module = _fold_linear_bn_eval(copy.deepcopy(module).eval())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), _slice_based_splits():
        torch.onnx.export(
            module,
            example_inputs,
            str(out_path),
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
        )
    return out_path


def precision_match(reference: Tensor, engine_out: Tensor) -> dict:
    """Engine-vs-PyTorch precision-match MEASUREMENT (owner-only policy is NOT coded).
    Returns max abs/rel drift only; the precision-match gate is an owner sign-off on this
    measured drift (SPEC §Requirements), deliberately not wired into a pass/fail here.
    Locally runnable.
    """
    ref = reference.detach().float()
    # Engine output lands on CUDA (EngineRunner allocates cuda buffers) while the PyTorch
    # reference is a CPU tensor; harmonize onto the reference's device before comparing.
    out = engine_out.detach().float().to(ref.device)
    diff = (ref - out).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.abs().clamp_min(1e-12)).max().item()
    return {"max_abs": max_abs, "max_rel": max_rel}


# DINO's attention exports its causal mask as an additive fill of finfo(float32).min
# (≈ -3.4e38 — torch's SDPA `-inf` sentinel, materialized as a FINITE constant on export). That
# value (a) overflows modelopt's entropy-calibration histogram (`ValueError: Too many bins for
# data range` — `2*threshold` exceeds the FP32 max) and (b) collapses `max` calibration to a
# garbage per-tensor scale. LeWM exports `is_causal=True` and never materializes such a tensor, so
# the failure is DINO-only. Neutralizing the sentinel to a mild finite fill is numerically
# equivalent for the mask — `softmax(-3e4 + logit)` underflows to 0 in FP16 and FP32 exactly as
# `-inf` does — but removes the overflow, letting BOTH calibration methods proceed with the
# attention MatMuls fully quantized (the real QK^T/scores ranges are unpolluted — the mask is added
# AFTER the score MatMul). The pass is self-targeting (edits only tensors carrying the sentinel), so
# it is a NO-OP on LeWM's graph — parity preserved, both tracks stay fully quantized.
# architecture.md §7. OWNER-approved graph edit before PTQ.
_MASK_SENTINEL_THRESHOLD = 1e30  # |x| >= this ⇒ the -3.4e38 mask sentinel, never a real activation
_MASK_FILL = -3.0e4  # softmax-equivalent to -inf; within FP16 range (<65504) + FP32 histogram edge


def _neutralize_attention_mask_sentinel(onnx_path: Path) -> Path:
    """Rewrite any finfo(float32).min mask-fill constant in the graph to `_MASK_FILL` before the PTQ
    calibration pass (see the note above). Scans initializers + `Constant` node value tensors and
    clamps every element at or beyond `-_MASK_SENTINEL_THRESHOLD`. Returns a patched sibling ONNX
    (with its external-data sidecar) when anything was rewritten, else the original path unchanged
    (LeWM — no sentinel). `onnx` imported lazily so this module still imports off-pod."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))  # pulls in the external-data sidecar if present
    n_patched = 0

    def _patch(t) -> None:
        nonlocal n_patched
        if t.data_type not in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.DOUBLE,
        ):
            return
        arr = numpy_helper.to_array(t)
        hit = arr <= -_MASK_SENTINEL_THRESHOLD  # catches -3.4e38 and any -inf (fp16-cast sentinel)
        if not hit.any():
            return
        arr = arr.copy()
        arr[hit] = _MASK_FILL
        t.CopyFrom(numpy_helper.from_array(arr, t.name))
        n_patched += int(hit.sum())

    for init in model.graph.initializer:
        _patch(init)
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                    _patch(attr.t)

    if n_patched == 0:
        return onnx_path  # LeWM: no sentinel — leave the graph byte-identical
    out = onnx_path.with_name(onnx_path.stem + ".maskfix.onnx")
    onnx.save(
        model,
        str(out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=out.name + ".data",
        size_threshold=1024,
    )
    print(f"  [maskfix] neutralized {n_patched} attention mask-sentinel value(s) -> {out.name}")
    return out


def quantize_onnx(
    onnx_path: Path,
    calibration_dict: dict,
    out_path: Path,
    calibration_method: str = "max",
    calibration_shapes: str | None = None,
    force_cpu_calibration: bool = False,
    quant_mode: str = "int8",
) -> Path:
    """NVIDIA TensorRT **Model Optimizer** PTQ invocation (owned plumbing; owner sets the
    quant config). Rewrites the base FP32 ONNX into a Q/DQ-annotated ONNX with per-tensor
    scales derived from `calibration_dict` (numpy arrays keyed by ONNX input name). Only the
    `max` calibration method is owner-pinned; the rest stay at the tool's INT8 defaults
    (owner-confirmed at the pod precision-match gate). `use_external_data_format=True`: the
    dynamo exporter externalizes initializers, so the base ONNX has a `.onnx.data` sidecar
    the quantizer must read/rewrite. Runs ONLY on the L40S (`modelopt` imported lazily so
    this module imports off-pod). Quantize failures raise loudly (judgement is owner's).

    The calibration pass runs on the **GPU (CUDA EP) when one is available**, CPU otherwise —
    modelopt's own default EP order lists `cpu` first, so this reorders it to prefer CUDA (the
    `setup.sh` CUDA-12 `onnxruntime-gpu` provides the EP) — UNLESS `force_cpu_calibration` is
    set, which pins it to CPU. The predictor path sets it: the `onnxruntime-gpu` CUDA EP
    miscomputes the predictor's dynamic-batch reshape chain (`Squeeze(Shape(latent))` feeding a
    head-split `Reshape`) — at batch 8 it fabricates a reshape target of 192 (=8x8x3) instead
    of 8 and crashes modelopt's MHA-exclusion probe, whereas the CPU EP (and native TensorRT,
    and the CUDA EP with graph-opt disabled) computes it correctly. The encoder graph lacks that
    pattern, so it keeps the faster CUDA EP. The EP only affects how fast the pass runs, not the
    derived scales (per-tensor, EP-independent), so this split is plumbing, not a quant-config
    knob. (The TensorRT EP would also be correct but is unusable here: its ORT `.so` needs the
    out-of-venv TRT libs on `LD_LIBRARY_PATH`, which the pod does not provide.)

    `calibration_shapes` pins the concrete per-input shape modelopt feeds its **ORT** sessions
    (the MHA-exclusion probe + the range pass). It is required for the LeWM predictor: that
    graph's batch axis is a `torch.export`-specialized symbol (an ignored `s == trace_batch`
    guard), which ORT's constant-folding collapses to the trace batch, so ORT executes the
    predictor as if batch were fixed. Left unset, modelopt fills the dynamic axis with 1 and
    feeds a batch-1 sample, which mismatches the folded batch and crashes a reshape. Pinning
    the batch to the trace batch makes the feed agree with the fold. This does NOT touch the
    scales (per-tensor, batch-independent) and TRT keeps the axis dynamic when it later parses
    the quantized graph (verified: the FP32 engine off the same graph runs at batch 1/8/300) —
    so it is feed plumbing, not a quant-config knob. It is likewise independent of the BUILD
    profile (`_BATCH_PROFILE`): the shape pinned here is the TRACE batch, and the profile TRT is
    later built with is free to differ (architecture.md §6)."""
    from modelopt.onnx.quantization import quantize

    # Neutralize DINO's -3.4e38 attention mask sentinel before calibration (no-op on LeWM). Both
    # calibration methods otherwise choke on it: `entropy` overflows the histogram, `max` collapses
    # the scale. See `_neutralize_attention_mask_sentinel` / architecture.md §7.
    onnx_path = _neutralize_attention_mask_sentinel(onnx_path)

    # Prefer the CUDA EP so calibration inference runs on the L40S GPU; fall back to CPU
    # off-pod / when no GPU is present ("run on GPU if available"). The predictor forces CPU
    # (`force_cpu_calibration`) to dodge the CUDA-EP dynamic-reshape miscompute described above.
    if force_cpu_calibration or not torch.cuda.is_available():
        calibration_eps = ["cpu"]
    else:
        calibration_eps = ["cuda:0", "cpu"]

    # `quant_mode` selects the 8-bit format baked into the Q/DQ graph: "int8" (integer, the
    # tool default) or "fp8" (E4M3 floating-point). Both draw the SAME calibration streams +
    # `max` method (format-independent — SPEC §Export shape); only the quantized dtype differs.
    # Threaded to modelopt's `quantize_mode` ONLY for the non-default format so the owner-signed-
    # off INT8 call stays byte-identical. 🔴 pod-verify: the `quantize_mode` kwarg name + "fp8"
    # value are the owner-set quant config (PLAN §Phase-6); an unknown kwarg fails loudly here.
    mode_kwargs = {} if quant_mode == "int8" else {"quantize_mode": quant_mode}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize(
        onnx_path=str(onnx_path),
        calibration_data=calibration_dict,
        calibration_method=calibration_method,
        calibration_eps=calibration_eps,
        calibration_shapes=calibration_shapes,
        output_path=str(out_path),
        use_external_data_format=True,
        **mode_kwargs,
    )
    if not out_path.exists():
        raise RuntimeError(f"Model Optimizer produced no quantized ONNX at {out_path}")  # -> OWNER
    return out_path


def build_engine(
    onnx_path: Path,
    precision: Precision,
    out_path: Path,
    example_inputs: tuple[Tensor, ...],
    batch_profile: tuple[int, int, int],
) -> Path:
    """TensorRT-10.7 builder invocation (owned plumbing). FP32 default, FP16 flag; INT8 and
    FP8 each set their 8-bit flag **and** FP16 and parse the **already-quantized** Q/DQ ONNX (no
    calibrator, no calibration profile — the scales are baked into the graph by `quantize_onnx`).
    FP16 is required alongside the 8-bit flag because the Model Optimizer casts the non-quantized
    remainder to FP16, so "INT8" is really INT8+FP16 and "FP8" is FP8+FP16 (see the branches
    below). Parse/build failures raise loudly (the *judgement* on how to fix them is owner's —
    ONNX/TRT debugging). Runs ONLY on the L40S (`tensorrt` imported lazily so this module imports
    off-pod).

    `batch_profile` is the `(min, opt, max)` for the dynamic batch axis — this engine's production
    call shape (`_BATCH_PROFILE`). It is REQUIRED rather than inferred from `example_inputs`, whose
    batch is a trace property: TensorRT tunes tactics at `opt`, so silently inheriting the trace
    batch would tune every engine for a batch it never runs at. `example_inputs` here supplies the
    NON-batch axes only."""
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
        # Explicit Q/DQ: the quantized ONNX carries the scales; the INT8 flag only lets TRT
        # pick INT8 tactics for the Q/DQ layers. No int8_calibrator / calibration profile.
        # FP16 is set TOO because the Model Optimizer emits a MIXED graph: it quantizes the
        # heavy MatMul/Gemm/etc. to INT8 and casts everything it did NOT quantize to FP16 (its
        # default high-precision dtype). TRT rejects an FP16-typed layer unless the FP16 flag
        # is on ("fp16 precision has been set ... but fp16 is not configured"), so both flags
        # are required. The engine is thus INT8 on the quantized layers + FP16 on the rest —
        # the realistic TRT INT8 deployment (SPEC: "INT8" == INT8+FP16).
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "fp8":
        # FP8 mirrors the INT8 branch exactly on the L40S's native FP8 Tensor Cores (Ada
        # 4th-gen): the quantized ONNX carries E4M3 Q/DQ + per-tensor scales; the FP8 flag lets
        # TRT pick FP8 tactics for the Q/DQ layers, and FP16 is set for the same reason as INT8
        # (the Model Optimizer casts the non-quantized remainder to FP16). No calibrator/profile
        # — scales are in the graph. "FP8" == FP8+FP16 (SPEC §Export shape). 🔴 pod-verify:
        # `BuilderFlag.FP8` is the owner-set build config (PLAN §Phase-6); build failures raise
        # loudly here for the owner's judgement.
        config.set_flag(trt.BuilderFlag.FP8)
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile for the dynamic batch axis (axis 0), pinned to this engine's production
    # call shape. Non-batch axes stay at the example shape.
    min_batch, opt_batch, max_batch = batch_profile
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        rest = tuple(example_inputs[i].shape[1:])
        profile.set_shape(
            inp.name,
            (min_batch, *rest),
            (opt_batch, *rest),
            (max_batch, *rest),
        )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build returned None for {onnx_path}")  # -> OWNER
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(serialized)
    return out_path


def engine_filename(
    component: str, precision: str, method: str = DEFAULT_CALIBRATION_METHOD
) -> str:
    """Engine-plan filename for one `component` (`encoder` | `predictor`). Quantized precisions
    (int8/fp8) are TAGGED with the PTQ calibration method so `int8` @ `max` and `int8` @ `entropy`
    engines coexist on the volume without overwriting each other (the SR they yield differs —
    architecture.md §7); FP32/FP16 carry no scales and are method-invariant, so they stay untagged.
    Single source of truth for the convention, shared by the writer (`export`) and the loader
    (`study.engine_paths`)."""
    if precision in QUANTIZED_PRECISIONS:
        return f"{component}.{precision}.{method}.plan"
    return f"{component}.{precision}.plan"


def export(
    adapter: WMStepAdapter,
    precision: Precision,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
    engine_dir: Path,
    calib_loader=None,
    calibration_method: str = "max",
) -> EnginePaths:
    """PyTorch -> ONNX -> TensorRT for both methods -> {encoder, predictor} engine paths.
    `encode` and `predict` are traced and built SEPARATELY. `calibration_method` (`max` | `entropy`)
    is the PTQ method — a build option for BOTH tracks, held constant across a track's int8/fp8 so
    the format delta stays clean (SPEC §Export shape); FP32/FP16 ignore it.
    """
    if precision in QUANTIZED_PRECISIONS and calib_loader is None:
        raise ValueError(f"{precision} export requires a calibration loader (owner-provided)")
    engine_dir.mkdir(parents=True, exist_ok=True)

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

    # Each engine gets its OWN calibration stream (encoder obs; predictor per-track predict
    # input) — the two graphs see different activations, so one shared calibration set would
    # mis-scale both. Built from the base ONNX per method (the predictor stream runs the clips
    # through the adapter), then the Model Optimizer bakes Q/DQ + scales into a quantized ONNX
    # that build_engine parses like FP32/FP16.
    engines: dict[str, Path] = {}
    for name, (module, inputs, dyn) in specs.items():
        onnx_path = export_onnx(module, inputs, dyn, engine_dir / f"{name}.onnx")
        if precision in QUANTIZED_PRECISIONS:
            from src.calibrate import make_calibration_dict

            batches = (
                calib_loader.encoder_batches()
                if name == "encoder"
                else calib_loader.predictor_batches(adapter)
            )
            calib_dict = make_calibration_dict(onnx_path, batches)
            # Pin each input's calibration shape to the traced example shape so modelopt's
            # ORT probe is fed the batch its constant-folding assumes (the trace batch), not
            # the dynamic-axis default of 1. `calib_dict` keys are the ONNX inputs in graph
            # order, which make_calibration_dict aligns 1:1 with `inputs` (forward order).
            shapes = ",".join(
                f"{n}:" + "x".join(str(int(d)) for d in t.shape)
                for n, t in zip(calib_dict, inputs)
            )
            onnx_path = quantize_onnx(
                onnx_path,
                calib_dict,
                # Per-(precision, method) filename so int8/fp8 × max/entropy quantized graphs
                # never collide in the same engine dir (each is a separately quantized ONNX —
                # SPEC §Export shape); mirrors the method-tagged .plan below.
                engine_dir / f"{name}.{precision}.{calibration_method}.onnx",
                # Per-track method (max for LeWM, entropy for DINO — owner-set, architecture.md §7); the
                # same for this track's int8 and fp8 so only the format differs.
                calibration_method=calibration_method,
                calibration_shapes=shapes,
                # The predictor's dynamic-batch reshape trips an onnxruntime-gpu CUDA-EP
                # miscompute in modelopt's MHA probe; calibrate it on CPU. The encoder graph
                # lacks that pattern, so it keeps the faster GPU (CUDA EP) calibration.
                force_cpu_calibration=(name == "predictor"),
                # int8 (integer) vs fp8 (E4M3) — the only per-format difference; the calibration
                # streams + method are shared.
                quant_mode=precision,
            )
        engines[name] = build_engine(
            onnx_path,
            precision,
            # Method-tagged for int8/fp8 (untagged for fp32/fp16), so a second calibration
            # method's engines are additive and never overwrite the first's (architecture.md §7).
            engine_dir / engine_filename(name, precision, calibration_method),
            inputs,
            # This component's production call batch — encoder 1, predictor the CEM candidate
            # fan-out (architecture.md §6). TRT tunes tactics at `opt`.
            _BATCH_PROFILE[name],
        )
    return EnginePaths(encoder=engines["encoder"], predictor=engines["predictor"])


def engine_root() -> Path:
    """Where the TensorRT engines are saved + loaded by default: `$STABLEWM_HOME/engines/` — the
    persistent network volume, so a pod session's built engines survive teardown and are not
    rebuilt each session (SPEC §Execution Environment). Falls back to the repo-local `engines/`
    only off-pod where `STABLEWM_HOME` is unset. Engines stay large + device-specific + gitignored
    (`*.plan`/`*.onnx` ignored) either way — regenerable on the L40S, one subdir per track."""
    home = os.environ.get("STABLEWM_HOME")
    return Path(home) / "engines" if home else Path("engines")


def main() -> None:
    """CLI: build the encoder + predictor engines for one track at one precision on the
    L40S. Reuses the real-checkpoint loader + shared example inputs so the traced
    graph is the exact `encode`/`predict` call pattern the benchmark drives. INT8 draws the
    owner calibration set and routes through the Model Optimizer (explicit Q/DQ) before the
    TRT build.

        uv run python -m src.export model=<lewm|dino> precision=<fp32|fp16|int8|fp8> \
            [calibration_method=max|entropy]

    `calibration_method` (default `max`) is the int8/fp8 PTQ method — a build option for BOTH
    tracks (SPEC §Export shape). Engines for a second method are additive: the quantized plans are
    method-TAGGED (`{encoder,predictor}.<precision>.<method>.plan`, `engine_filename`), so int8/fp8
    @ max and @ entropy coexist on the volume without overwriting (fp32/fp16 stay untagged —
    method-invariant). The SR they yield is likewise recorded under the method LABEL in sr.json
    (`src.sr_eval calibration_method=…`), so the two methods' points coexist end-to-end.
    """
    from src.interfaces import (
        DEFAULT_CALIBRATION_METHOD,
        ExportConfig,
        check_calibration_method,
    )
    from src.precision_match import _build_adapter, example_inputs

    model = "lewm"
    precision: Precision = "fp32"
    calibration_method = DEFAULT_CALIBRATION_METHOD
    for a in sys.argv[1:]:
        if a.startswith("model="):
            model = a.split("=", 1)[1]
        elif a.startswith("precision="):
            precision = a.split("=", 1)[1]  # type: ignore[assignment]
        elif a.startswith("calibration_method="):
            calibration_method = check_calibration_method(a.split("=", 1)[1])

    cfg = ExportConfig()
    torch.manual_seed(cfg.seed)
    adapter, name = _build_adapter(model)
    encode_inputs, predict_inputs = example_inputs(adapter, cfg)

    # Quantized precisions (int8/fp8) draw the calibration set through the platform (the Model
    # Optimizer derives the scales from it); `batch` is just the internal chunk that streams
    # clips through the adapter. FP32/FP16 build data-free.
    calib_loader = None
    if precision in QUANTIZED_PRECISIONS:
        from src.calibrate import build_calibration_data

        calib_loader = build_calibration_data(batch=encode_inputs[0].shape[0])

    engines = export(
        adapter,
        precision=precision,
        encode_inputs=encode_inputs,
        predict_inputs=predict_inputs,
        engine_dir=engine_root() / name,
        calib_loader=calib_loader,
        calibration_method=calibration_method,
    )
    label = f" ({calibration_method})" if precision in QUANTIZED_PRECISIONS else ""
    for method, path in engines.items():
        print(f"[{name}/{precision}{label}] {method}: {path}")


if __name__ == "__main__":
    main()
