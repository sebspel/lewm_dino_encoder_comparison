"""Benchmark stub (Phase 4 tracer bullet).

The real fixed-wall-clock benchmark (p50/p95 latency, rollouts completed, throughput,
peak GPU memory, and the SR per precision) runs on the L40S in Phase 5, driving the two
TensorRT engines through the Python CEM rollout shim (SPEC §Interface Contracts). This
stub consumes both engine paths from `export`, checks they exist, and returns placeholder
metrics so the tracer-bullet flow (adapter -> export -> benchmark) closes end-to-end.
"""

from torch import Tensor

from src.interfaces import EnginePaths, BenchResult


def benchmark(
    engines: EnginePaths,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
    time_budget_s: float,
    warmup: int,
) -> BenchResult:
    for name, path in engines.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} engine missing: {path}")

    # Placeholder metrics; Phase 5 fills these from real timed runs on the engines.
    return BenchResult(
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        rollouts_completed=0,
        throughput=0.0,
        peak_mem_mb=0.0,
        success_rate=0.0,
    )
