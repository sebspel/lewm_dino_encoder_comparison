"""TensorRT engine runner (Phase 5 plumbing) — owned, fails LOUDLY (CLAUDE.md §8).

Deserializes a `.plan` built by `src.export`, runs one method's engine on CUDA torch
tensors, and returns torch tensors. This is the missing `engine_out` producer for
`src.export.precision_match` (engine-vs-PyTorch drift) and the execution primitive the
Phase-5 fixed-budget benchmark drives inside its Python CEM rollout loop.

Runs ONLY on the L40S (`tensorrt` imported lazily + CUDA buffers). The precision-match
POLICY (tolerances) stays OWNER-ONLY (`src.export.PrecisionTolerance`); this module only
produces the numbers to judge.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from src.export import PrecisionTolerance, precision_match


def _torch_dtype(trt_dtype) -> torch.dtype:
    """Map a TensorRT tensor dtype to the matching torch dtype (for allocating I/O
    buffers). FP16 engines keep FP32 network I/O unless the build sets tensor dtypes —
    `src.export.build_engine` does not — so bindings are usually FLOAT either way."""
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    return mapping[trt_dtype]


class EngineRunner:
    """Loads one serialized engine and runs it on CUDA torch tensors.

    Address-based execution: for every named I/O tensor we register a device pointer on
    the context, then `execute_async_v3` reads inputs / writes outputs at those addresses.
    """

    def __init__(self, plan_path: Path, device: str = "cuda"):
        import tensorrt as trt

        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {plan_path}")  # -> OWNER
        self.context = self.engine.create_execution_context()

        # Split the engine's named I/O tensors into inputs vs outputs (order preserved).
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def _bind_outputs(self) -> list[Tensor]:
        """Allocate + register an output buffer for each output tensor.

        name in `self.output_names`:
          1. read the now-resolved output shape: `self.context.get_tensor_shape(name)`
          2. allocate a torch buffer: `torch.empty(shape, dtype=..., device=self.device)`
             (dtype via `_torch_dtype(self.engine.get_tensor_dtype(name))`)
          3. register its pointer: `self.context.set_tensor_address(name, buf.data_ptr())`
        Collect the buffers in order and return them.
        """
        outputs = []
        for name in self.output_names:
            output_shape = self.context.get_tensor_shape(name)
            output_dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
            output_buffer = torch.empty(
                output_shape,
                dtype=output_dtype,
                device=self.device,
            )
            self.context.set_tensor_address(name, output_buffer.data_ptr())
            outputs.append(output_buffer)

        return outputs

    def run(self, inputs: tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
        """Run the engine on `inputs` (one per engine input, in order) and return the
        output tensor(s). Inputs are moved to CUDA + made contiguous before binding."""
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} inputs {self.input_names}, "
                f"got {len(inputs)}"
            )

        held: list[Tensor] = []  # keep input buffers alive until execution completes
        for name, t in zip(self.input_names, inputs):
            t = t.to(self.device).contiguous()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            held.append(t)

        outputs = self._bind_outputs()

        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")  # -> OWNER
        stream.synchronize()

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def engine_vs_reference(
    plan_path: Path,
    reference: Tensor,
    inputs: tuple[Tensor, ...],
    tol: PrecisionTolerance = PrecisionTolerance(),
) -> dict:
    """Run one engine on `inputs` and compare a single output against the PyTorch
    `reference` via `src.export.precision_match` (max abs/rel error; `passed` stays None
    until the owner sets tolerances). Convenience for the precision-match test loop."""
    engine_out = EngineRunner(plan_path).run(inputs)
    if isinstance(engine_out, tuple):
        raise ValueError("engine has multiple outputs; compare them explicitly")
    return precision_match(reference, engine_out, tol)
