"""Export build-profile contract (off-pod — no TensorRT, no CUDA).

The batch each engine is BUILT at is the knob TensorRT selects tactics on, and a wrong value is
silent: the engine still runs, just tuned for a shape it never sees. These pin the profile to the
production call shapes and keep it from drifting back to an example-input inference.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from src.export import _BATCH_PROFILE, _batch_dynamic, build_engine, engine_root, export_onnx
from src.interfaces import CEM_NUM_SAMPLES, EnginePaths


class _Sum(nn.Module):
    def forward(self, x):
        return x.sum(-1)


def test_engine_root_uses_stablewm_home(monkeypatch, tmp_path):
    """Engines live on the persistent network volume so a pod session's builds survive teardown.
    Unset raises rather than falling back — the same discipline as the report/telemetry roots."""
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    assert engine_root() == tmp_path / "engines"

    monkeypatch.delenv("STABLEWM_HOME", raising=False)
    with pytest.raises(RuntimeError, match=r"STABLEWM_HOME.*\.env"):
        engine_root()


def test_encoder_is_built_at_its_single_production_batch():
    # The CEM slices the candidate axis away before encoding and pins batch_size=1, so the encoder
    # is only ever called at batch 1 — min, opt and max alike (docs/architecture.md §5).
    assert _BATCH_PROFILE["encoder"] == (1, 1, 1)


def test_predictor_opt_is_the_cem_candidate_fan_out():
    # opt/max are the fan-out the predictor is actually called at; reading it off CEM_NUM_SAMPLES
    # rather than a loose literal means a CEM config change cannot leave the profile behind.
    lo, opt, hi = _BATCH_PROFILE["predictor"]
    assert (lo, opt, hi) == (1, CEM_NUM_SAMPLES, CEM_NUM_SAMPLES)


def test_profile_covers_exactly_the_exported_components():
    # A renamed/added component must fail here rather than KeyError deep in a pod-only build.
    assert set(_BATCH_PROFILE) == set(EnginePaths.__annotations__)


def test_build_engine_requires_an_explicit_batch_profile():
    # No default: inheriting the profile from `example_inputs` would silently tune every engine at
    # the TRACE batch, which is the one thing this contract exists to prevent.
    sig = inspect.signature(build_engine)
    assert sig.parameters["batch_profile"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        build_engine("x.onnx", "fp32", "x.plan", ())  # type: ignore[call-arg]


def test_a_size_one_trace_batch_is_rejected_not_silently_specialized(tmp_path):
    # torch.export specializes a size-1 dim: the exporter emits a frozen `dim_value: 1` with no
    # error, and every engine built off that graph is batch-frozen. The trace batch may differ
    # from the build profile, but it may not be 1.
    with pytest.raises(ValueError, match="dynamic batch axis"):
        export_onnx(_Sum(), (torch.randn(1, 4),), _batch_dynamic(1), tmp_path / "frozen.onnx")


def test_a_valid_trace_batch_leaves_the_onnx_axis_symbolic(tmp_path):
    import onnx

    path = export_onnx(_Sum(), (torch.randn(2, 4),), _batch_dynamic(1), tmp_path / "dyn.onnx")
    dim0 = onnx.load(str(path), load_external_data=False).graph.input[0].type.tensor_type.shape.dim[0]
    # A symbol, not a literal — this is what lets the build profile choose the batch.
    assert dim0.dim_param, f"batch axis specialized to {dim0.dim_value}"
