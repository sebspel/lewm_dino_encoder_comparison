"""Phase-5 study-driver plumbing — engine-path convention + orchestration (CPU).

The benchmark leg needs real engines + CUDA (pod-only), so here we drive `run_track` with a
dummy adapter and an empty engine root: profiling runs on CPU and every precision is skipped
for missing engines, exercising the orchestration + skip path without a GPU.
"""

import dataclasses

import torch

from src import study
from src.adapter import LeWMAdapter
from src.interfaces import ExportConfig
from src.smoke import build_dummy_lewm


def test_engine_paths_convention(tmp_path):
    p = study.engine_paths("dino", "fp16", engine_root=tmp_path)
    assert p["encoder"] == tmp_path / "dino" / "encoder.fp16.plan"
    assert p["predictor"] == tmp_path / "dino" / "predictor.fp16.plan"


def test_run_track_profiles_and_skips_missing_engines(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study, "_build_adapter", lambda track: (LeWMAdapter(build_dummy_lewm()), track)
    )
    cfg = dataclasses.replace(ExportConfig(), n_profile_iters=2, warmup=1)

    name, prof, bench = study.run_track(
        "lewm", cfg, torch.device("cpu"), engine_root=tmp_path
    )

    assert name == "lewm"
    assert set(prof) == {"fp32"}
    assert prof["fp32"]["encoder_ms"] > 0 and prof["fp32"]["predictor_ms"] > 0
    assert bench == {}  # no engines built -> all precisions skipped, no CUDA touched
