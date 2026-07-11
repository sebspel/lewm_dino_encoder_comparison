from typing import Protocol, Literal, TypedDict, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

from torch import Tensor
from jaxtyping import Float

if TYPE_CHECKING:
    # The INT8 calibration set, drawn through the platform (src.calibrate); export turns it
    # into the numpy dict the Model-Optimizer PTQ pass (explicit Q/DQ) consumes. TYPE_CHECKING
    # -only so this foundation module stays import-light and free of a runtime cycle.
    from src.calibrate import CalibrationData

Precision = Literal["fp32", "fp16", "int8"]

# --- Dims (owner-confirmed; the 🔴 adapter-dims gate). Defined ONCE here; the platform's
# own dims are read from its config, never re-guessed. Read from the pinned installed
# source (docs/platform_api.md §2).
LATENT_DIM = 192  # LeWM single-token latent width (ViT-Tiny hidden_size)
DINO_N_PATCHES = 196  # DINOv3 patch grid after CLS + 4 register slice ((224/16)²)
DINO_LATENT_DIM = 384  # DINOv3 hidden_size
# DINO-WM predictor-input token width: 384 latent + 20 extras (proprio 10 + action 10),
# tiled and concatenated on the feature axis before `predict`. Distinct from the 384
# latent; the CausalPredictor is dim-preserving so its output width is 404 too. A wrong
# value mis-shapes the predictor engine SILENTLY — owner-confirmed against the source.
DINO_PREDICTOR_DIM = 404
ACTION_DIM = 2  # swm/PushT-v1 action_space Box(-1, 1, (2,)) — env/planner-facing action
HISTORY_SIZE = 3  # wm.history_size for both tracks
# Model-facing action width: the CEM plans env ACTION_DIM (2) actions, but the world
# models' action_encoder ingests the frameskip pack (5 × 2 = 10) — action_encoder is a
# Conv1d(10, …) in BOTH tracks (owner-confirmed on the pod, 2026-07-08). Distinct from the
# env ACTION_DIM; a wrong width mis-shapes the predictor engine SILENTLY.
MODEL_ACTION_DIM = 10
# DINO-WM proprio extra: extra_encoders['proprio'] is a Conv1d(4, 10) — proprio input is
# 4-wide, embedded to 10 and concatenated (with action's 10) onto the 384 latent -> 404.
DINO_PROPRIO_DIM = 4


class WMStepAdapter(Protocol):
    """Common two-method boundary the export/benchmark/profile plumbing binds to, so it
    treats both tracks identically and never branches per-model. `encode` and `predict`
    are separately callable (and separately exported), because the CEM rollout encodes
    the obs ONCE and calls `predict` autoregressively over the horizon for all candidates.

    The latent *shape* is model-specific; the two concrete implementations (adapter.py)
    carry the precise annotations. The variadic `*latent` below admits both without the
    plumbing knowing which:
        LeWMAdapter   -> Float[Tensor, "batch hist latent"]        (single token, D=192)
        DINOWMAdapter -> Float[Tensor, "batch hist patch latent"]  (patch grid, 196x384)
    Conditioning enters `predict` per-track and with per-track *arity* (hence the variadic
    `*conditioning`): LeWM takes `(latent, action)` — a single AdaLN-conditioning tensor
    after the latent. DINO-WM's `predict` is a faithful, dim-preserving 404->404 step over a
    *pre-assembled* embedding, so it takes a SINGLE 404-wide tensor (no separate conditioning
    arg): the 384->404 proprio/action assembly (`DINOWMAdapter.assemble_embedding`) and the
    per-step action-replacement live in the Python rollout/shim, not this call. The plumbing
    stays arity-agnostic by driving `predict(*inputs)`. The CEM planner / rollout loop stays
    in Python outside the adapter — the adapter is the unit TensorRT optimizes.
    """

    def encode(
        self,
        obs: Float[Tensor, "batch hist channel height width"],
    ) -> Float[Tensor, "batch hist *latent"]: ...

    def predict(
        self,
        latent: Float[Tensor, "batch hist *latent"],
        *conditioning: Float[Tensor, "batch hist cond"],
    ) -> Float[Tensor, "batch hist *latent"]: ...


class EnginePaths(TypedDict):
    # export traces `encode` and `predict` separately -> one engine each (two per model).
    encoder: Path
    predictor: Path


class Export(Protocol):
    def __call__(
        self,
        adapter: WMStepAdapter,
        precision: Precision,
        encode_inputs: tuple[Tensor, ...],  # example obs for the encoder graph
        # example predict inputs: LeWM (cached latent, action); DINO (assembled 404 embedding)
        predict_inputs: tuple[Tensor, ...],
        engine_dir: Path,
        # required iff precision == "int8": the Model-Optimizer PTQ input (export builds a
        # per-method numpy dict keyed by ONNX input name from it — explicit Q/DQ, not a TRT
        # build-time calibrator).
        calib_loader: "CalibrationData | None" = None,
    ) -> EnginePaths: ...


class BenchResult(TypedDict):
    # Per PREDICTOR-STEP inference latency (encode runs once per rollout and is NOT timed
    # here — the encoder asymmetry surfaces in `throughput`/`rollouts_completed` and the
    # profile, not in these percentiles). Each step syncs the CUDA stream, so for LeWM's
    # tiny ops this is a launch+sync floor, not compute — which compresses the LeWM↔DINO
    # p95 ratio (LeWM is launch-latency-bound, SPEC §Parity).
    latency_p50_ms: float
    latency_p95_ms: float
    # MODEL-ONLY rollouts in the fixed budget: encode-once + predict-over-horizon with NO CEM
    # planner in the loop (no candidate sampling / topk / elite update). This is the model
    # speedup ceiling with the planner treated as free; the REALIZED wall-clock
    # rollouts-in-budget (planner in the loop) comes from the gated eval-shim re-run. Their
    # gap is the planner floor (≈ Amdahl from the profile shares) — SPEC §dilution disclosure.
    rollouts_completed: int
    throughput: float  # model-only rollouts/sec (same planner-free caveat)
    # Sampled from cudaMemGetInfo (`torch.cuda.mem_get_info`), NOT `torch.cuda.max_memory_
    # allocated`: TensorRT's engine + execution-context device allocations bypass torch's
    # caching allocator, so the allocator would undercount exactly the optimized path
    # (SPEC §Interface Contracts). Device-level → whole-GPU used memory (benchmark pod is
    # dedicated).
    peak_mem_mb: float
    success_rate: float  # Push-T SR paired with this engine config (gated eval shim; NaN until then)


class Benchmark(Protocol):
    def __call__(
        self,
        engines: EnginePaths,  # encoder + predictor engines, driven by a Python rollout
        encode_inputs: tuple[Tensor, ...],
        predict_inputs: tuple[Tensor, ...],
        time_budget_s: float,
        warmup: int,
    ) -> BenchResult: ...


class ComponentProfile(TypedDict):
    # Raw mean time PER SINGLE CALL, measured at the cycle's real batch per component
    # (encode: batch 1 obs; predict: the candidate fan-out num_samples; planner: one CEM
    # iteration). These are NOT summable across components and NOT the cycle's time share — a
    # planning cycle calls `predict` many times and `encode` ~once, so per-call means must be
    # weighted by call count before they mean anything (see `*_cycle_ms`).
    encoder_ms: float
    predictor_ms: float
    planner_ms: float
    # Calls per planning cycle, from the CEM decomposition (docs/platform_api.md §3/§5:
    # num_samples=300, n_steps=30, horizon=5). encoder ~once (cached), predict autoregressive
    # over the horizon per iteration, planner once per iteration.
    encoder_calls: int
    predictor_calls: int
    planner_calls: int
    # Runtime-WEIGHTED per-cycle time (calls × per-call ms). THESE are the true time shares
    # and sum to the modelled cycle (within the cuda.synchronize barrier).
    encoder_cycle_ms: float
    predictor_cycle_ms: float
    planner_cycle_ms: float
    # Derived from the weighted shares: optimizable fraction p = (enc+pred)/cycle (only the
    # model is TRT-optimized; the Python planner is precision-invariant) and the Amdahl
    # end-to-end speedup ceiling 1/(1-p) it sets (SPEC §dilution disclosure).
    optimizable_fraction: float
    amdahl_ceiling: float


class Profile(Protocol):
    def __call__(
        self,
        adapter: WMStepAdapter,
        encode_inputs: tuple[Tensor, ...],
        predict_inputs: tuple[Tensor, ...],
        n_iters: int,
        warmup: int,
    ) -> ComponentProfile: ...


@dataclass(frozen=True)
class ExportConfig:
    hist: int = HISTORY_SIZE
    obs_shape: tuple[int, int, int] = (3, 224, 224)
    action_dim: int = MODEL_ACTION_DIM  # model-facing action fed to predict (frameskip pack)
    proprio_dim: int = DINO_PROPRIO_DIM  # DINO-WM proprio extra fed to predict
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8")
    warmup: int = 5
    time_budget_s: float = 10.0  # fixed wall-clock budget for the benchmark
    n_profile_iters: int = 30  # cycles timed for the per-component profile
    seed: int = 0
