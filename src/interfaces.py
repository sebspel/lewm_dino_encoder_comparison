from typing import Protocol, Literal, TypedDict
from pathlib import Path
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import DataLoader
from jaxtyping import Float

Precision = Literal["fp32", "fp16", "int8"]

# --- Dims (Phase-1 values, owner-confirmed; the 🔴 adapter-dims gate, SPEC
# §Implementation Boundaries). Defined ONCE here; the platform's own dims are read
# from its config, never re-guessed. Read from the pinned installed source
# (docs/platform_api.md §2).
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
        calib_loader: DataLoader | None = None,  # required iff precision == "int8"
    ) -> EnginePaths: ...


class BenchResult(TypedDict):
    latency_p50_ms: float  # per planning-step inference latency
    latency_p95_ms: float
    rollouts_completed: int  # CEM rollouts finished within the fixed time budget
    throughput: float  # rollouts/sec
    peak_mem_mb: float
    success_rate: float  # Push-T SR paired with this engine config (Phase-5 eval shim)


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
    encoder_ms: float  # mean per-cycle time in the encoder
    predictor_ms: float  # mean per-cycle time in the predictor
    planner_ms: float  # mean per-cycle time in the CEM planner (excl. model calls)


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
