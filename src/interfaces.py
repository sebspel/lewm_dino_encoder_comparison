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

# CEM planning-cycle call counts (docs/platform_api.md §5). The measured per-cycle latency is
# decomposed by weighting the isolated engine-step latencies by these counts and subtracting
# from the cycle (overhead = cycle − enc·ENC_CALLS − pred·PRED_CALLS; SPEC §Interface
# Contracts). 🔴 confirm against the installed `CEMSolver.solve` on the pod, not assumed.
CEM_NUM_SAMPLES = 300  # candidate fan-out — the batch `predict` is timed at
ENCODER_CALLS_PER_CYCLE = 2  # goal encode + initial-obs encode (both cached, batch 1)
PREDICTOR_CALLS_PER_CYCLE = 180  # (horizon 5 + 1) × n_steps 30, batched over the candidates

# CEM action-proposal shape — the distribution `predict` is ACTUALLY driven by at eval, and
# what the INT8 predictor calibration stream reproduces (SPEC §Interface Contracts —
# calibration distribution). Read from the vendored configs + `CEMSolver` source, not assumed:
#   `solver/cem.py:191-204`  candidates = randn(B, num_samples, horizon, action_dim) * var + mean
#                            -> an UNCLAMPED N(0, var_scale) about `mean` (0 at the zero-pad
#                            warm start); there is NO clamp to the action space.
#   `solver/cem.py:80`       action_dim = env_action_dim (2) * action_block (5) = 10, i.e. CEM
#                            samples the MODEL-facing 10-wide frameskip pack DIRECTLY — no
#                            env->model packing sits between the proposal and `predict`.
# Expert dataset actions are bounded by Box(-1, 1); the proposal reaches ~4 sigma. Calibrating
# on expert actions therefore under-scales the action tensor ~4x and saturates INT8 — invisible
# in FP16 (no fixed clip). 🔴 confirm against the installed solver on the pod, not assumed.
CEM_VAR_SCALE = 1.0  # scripts/plan/config/solver/cem.yaml `var_scale` — initial proposal std
CEM_HORIZON = 5  # scripts/plan/config/pusht.yaml `plan_config.horizon`
# n_obs at eval: the rollout starts from ONE encoded frame and fills the rest of the window
# with its OWN predictions, so a steady-state `predict` window holds ZERO encoder latents
# (`LeWM.rollout` lo=max(0, H+t-HS) with H=1 -> windows 1,2,3,3,…; same for `PreJEPA.rollout`).
EVAL_N_OBS = 1


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
    # HEADLINE latency — the full per-decision PLANNING CYCLE (encode + predict + overhead),
    # measured on the REAL CEM solve via the eval-latency callback over the SR eval-shim run.
    # NOT produced by `benchmark` (no planner in the harness); left NaN here and JOINED per
    # precision from that gated run, so per-cycle latency and SR come from the same solves
    # (SPEC §Interface Contracts). Equal-n across tracks (report truncates to the common
    # min-n before taking the percentiles).
    per_cycle_p50_ms: float
    per_cycle_p95_ms: float
    # COMPONENT latency — isolated per-precision engine-step p50/p95 from fixed-iteration
    # loops (warm-up dropped, equal-n). `encode_*` exposes the encoder token-count asymmetry
    # (LeWM 1 token vs DINOv3 196); `predict_*` is quantization's kernel target. Each engine
    # call syncs its stream, so for LeWM's tiny ops these sit on a launch+sync floor.
    encode_p50_ms: float
    encode_p95_ms: float
    predict_p50_ms: float
    predict_p95_ms: float
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
        engines: EnginePaths,  # encoder + predictor engines, timed as isolated step loops
        encode_inputs: tuple[Tensor, ...],  # obs at batch 1 (the cached per-cycle encode)
        predict_inputs: tuple[Tensor, ...],  # predictor state at the candidate fan-out batch
        n_iters: int,  # fixed-iteration count per step loop (equal-n percentiles)
        warmup: int,
    ) -> BenchResult: ...


@dataclass(frozen=True)
class ExportConfig:
    hist: int = HISTORY_SIZE
    obs_shape: tuple[int, int, int] = (3, 224, 224)
    action_dim: int = MODEL_ACTION_DIM  # model-facing action fed to predict (frameskip pack)
    proprio_dim: int = DINO_PROPRIO_DIM  # DINO-WM proprio extra fed to predict
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8")
    warmup: int = 10  # warm-up iters dropped before timing each engine-step loop
    n_latency_iters: int = 100  # fixed timed iters per engine-step loop (equal-n p50/p95)
    seed: int = 0
