"""Export stub (Phase 4 tracer bullet).

Real PyTorch -> ONNX -> TensorRT export (FP32/FP16/INT8) is Phase 5 and OWNER-gated
(ONNX/TRT debugging, INT8 calibration, precision matching — SPEC §Implementation
Boundaries). This stub exercises the owned boundary: it runs `encode` and `predict`
once to validate the typed shapes, then emits one placeholder engine file per method so
the benchmark stub has an `EnginePaths` to consume. `encode` and `predict` are exported
separately (one engine each) — the two-engine reality the Phase-5 exporter fills in.
"""

from pathlib import Path

from torch import Tensor

from src.interfaces import Precision, EnginePaths, WMStepAdapter


def export(
    adapter: WMStepAdapter,
    precision: Precision,
    encode_inputs: tuple[Tensor, ...],
    predict_inputs: tuple[Tensor, ...],
    engine_dir: Path,
    calib_loader=None,
) -> EnginePaths:
    if precision == "int8" and calib_loader is None:
        raise ValueError("int8 export requires a calibration loader")

    # Exercise both boundaries so a shape violation surfaces here, not at build time.
    adapter.encode(*encode_inputs)
    adapter.predict(*predict_inputs)

    engine_dir.mkdir(parents=True, exist_ok=True)
    engines: EnginePaths = {
        "encoder": engine_dir / f"encoder.{precision}.plan",
        "predictor": engine_dir / f"predictor.{precision}.plan",
    }
    for path in engines.values():
        path.write_bytes(b"")  # placeholder; real engine built on the L40S in Phase 5
    return engines
