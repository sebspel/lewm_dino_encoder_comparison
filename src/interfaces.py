from typing import Protocol, Literal, TypedDict
from pathlib import Path
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import DataLoader
from jaxtyping import Float
from beartype import beartype

Precision = Literal["fp32", "fp16", "int8"]


class WMStepAdapter(Protocol):
    # Shared behavioural contract the export/benchmark/profile plumbing binds to, so
    # it treats both tracks identically and never branches per-model. The latent
    # *shape* is model-specific; the two concrete implementations (adapter.py) carry
    # the precise annotations:
    #   LeWMAdapter   -> Float[Tensor, "batch latent_dim"]              (single token)
    #   DINOWMAdapter -> Float[Tensor, "batch num_patches latent_dim"]  (full patch grid)
    # The variadic `*latent` below admits both without the plumbing knowing which.
    def __call__(
        self,
        obs: Float[Tensor, "batch hist channel height width"],
        action: Float[Tensor, "batch hist action_dim"],
    ) -> Float[Tensor, "batch *latent"]: ...


class Export(Protocol):
    def __call__(
        self,
        adapter: WMStepAdapter,
        precision: Precision,
        sample_inputs: tuple[Tensor, Tensor],
        engine_dir: Path,
        calib_loader: DataLoader | None = None,  # required iff precision == "int8"
    ) -> Path: ...


class BenchResult(TypedDict):
    latency_p50_ms: float  # per planning-step inference latency
    latency_p95_ms: float
    rollouts_completed: int  # CEM rollouts finished within the fixed time budget
    throughput: float  # rollouts/sec
    peak_mem_mb: float


class Benchmark(Protocol):
    def __call__(
        self,
        engine_path: Path,
        sample_inputs: tuple[Tensor, Tensor],
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
        sample_inputs: tuple[Tensor, Tensor],
        n_iters: int,
        warmup: int,
    ) -> ComponentProfile: ...


@dataclass(frozen=True)
class ExportConfig:
    hist: int = 3
    obs_shape: tuple[int, int, int] = (3, 224, 224)
    action_dim: int = 2
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8")
    warmup: int = 5
    time_budget_s: float = 10.0  # fixed wall-clock budget for the benchmark
    n_profile_iters: int = 30  # cycles timed for the per-component profile
    seed: int = 0
