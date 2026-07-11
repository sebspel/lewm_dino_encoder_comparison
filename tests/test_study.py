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


def test_default_out_dir_uses_stablewm_home(monkeypatch, tmp_path):
    """Durability (SPEC §Headline-artifact durability): the study writes under
    `$STABLEWM_HOME/reports/phase5/` (persistent network volume) so it survives pod teardown,
    falling back to repo-local `reports/phase5` off-pod."""
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    assert study.default_out_dir() == tmp_path / "reports" / "phase5"

    monkeypatch.delenv("STABLEWM_HOME", raising=False)
    from pathlib import Path

    assert study.default_out_dir() == Path("reports/phase5")


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


def test_dump_track_results_roundtrips(tmp_path, monkeypatch):
    """Canonical per-track JSON (SPEC §Headline-artifact durability): the raw profile numbers
    + fairness conditions persist to `results.<track>.json` and round-trip back through
    `report.load_results` into the shape `report` consumes."""
    import json

    from src import report

    monkeypatch.setattr(
        study, "_build_adapter", lambda track: (LeWMAdapter(build_dummy_lewm()), track)
    )
    cfg = dataclasses.replace(ExportConfig(), n_profile_iters=2, warmup=1)
    name, prof, bench = study.run_track(
        "lewm", cfg, torch.device("cpu"), engine_root=tmp_path
    )

    path = study.dump_track_results(name, prof, bench, cfg, tmp_path)
    assert path == tmp_path / "results.lewm.json"

    b, p = report.load_results([path])
    assert set(b) == {"lewm"} and set(p) == {"lewm"}
    assert p["lewm"]["fp32"]["encoder_ms"] == prof["fp32"]["encoder_ms"]

    # meta records the run's fairness conditions so the numbers are self-describing later
    meta = json.loads(path.read_text())["meta"]
    assert meta["time_budget_s"] == cfg.time_budget_s
    assert meta["num_samples"] == study.CEM_NUM_SAMPLES
