"""Phase-5 study-driver plumbing — engine-path convention + orchestration (CPU).

The benchmark leg needs real engines + CUDA (pod-only), so here we drive `run_track` with a
dummy adapter and an empty engine root: every precision is skipped for missing engines,
exercising the orchestration + skip path without a GPU.
"""

import dataclasses
import json

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
    # fp32/fp16 are method-invariant -> untagged plan names.
    p = study.engine_paths("dino", "fp16", engine_root=tmp_path)
    assert p["encoder"] == tmp_path / "dino" / "encoder.fp16.plan"
    assert p["predictor"] == tmp_path / "dino" / "predictor.fp16.plan"


def test_engine_paths_quantized_are_method_tagged(tmp_path):
    """int8/fp8 plans are TAGGED with the calibration method so max/entropy engines coexist without
    overwriting (architecture.md §7); the loader selects by method."""
    mx = study.engine_paths("dino", "int8", engine_root=tmp_path, method="max")
    assert mx["encoder"] == tmp_path / "dino" / "encoder.int8.max.plan"

    ent = study.engine_paths("dino", "fp8", engine_root=tmp_path, method="entropy")
    assert ent["predictor"] == tmp_path / "dino" / "predictor.fp8.entropy.plan"
    # max and entropy resolve to DIFFERENT files -> no overwrite
    assert mx["encoder"] != study.engine_paths(
        "dino", "int8", engine_root=tmp_path, method="entropy"
    )["encoder"]


def test_engine_paths_max_falls_back_to_legacy_untagged(tmp_path):
    """Engines built before method-tagging are untagged + `max`-calibrated: a `method=max` request
    resolves to the legacy `…<precision>.plan` when the tagged file is absent (no orphaning); a
    non-default method never falls back."""
    d = tmp_path / "lewm"
    d.mkdir()
    (d / "encoder.int8.plan").write_bytes(b"legacy")  # pre-tagging max engine
    (d / "predictor.int8.plan").write_bytes(b"legacy")

    mx = study.engine_paths("lewm", "int8", engine_root=tmp_path, method="max")
    assert mx["encoder"] == d / "encoder.int8.plan"  # legacy resolved

    # entropy must NOT pick up the legacy (max) engine — it stays the tagged, here-missing name.
    ent = study.engine_paths("lewm", "int8", engine_root=tmp_path, method="entropy")
    assert ent["encoder"] == d / "encoder.int8.entropy.plan"
    assert not ent["encoder"].exists()


def test_run_track_skips_missing_engines(tmp_path, monkeypatch):
    monkeypatch.setattr(
        study, "_build_adapter", lambda track: (LeWMAdapter(build_dummy_lewm()), track)
    )
    cfg = dataclasses.replace(ExportConfig(), n_latency_iters=2, warmup=1)

    name, bench, samples = study.run_track(
        "lewm", cfg, torch.device("cpu"), engine_root=tmp_path
    )

    assert name == "lewm"
    assert bench == {}  # no engines built -> all precisions skipped, no CUDA touched
    assert samples == {}  # and no raw component samples to persist


def test_dump_track_results_roundtrips(tmp_path, monkeypatch):
    """Canonical per-track JSON (SPEC §Headline-artifact durability): the raw benchmark numbers
    + fairness conditions persist to `results.<track>.json` and round-trip back through
    `report.load_results` into the shape `report` consumes."""
    from src import report

    monkeypatch.setattr(
        study, "_build_adapter", lambda track: (LeWMAdapter(build_dummy_lewm()), track)
    )
    cfg = dataclasses.replace(ExportConfig(), n_latency_iters=2, warmup=1)
    name, bench, _ = study.run_track(
        "lewm", cfg, torch.device("cpu"), engine_root=tmp_path
    )

    path = study.dump_track_results(name, bench, cfg, tmp_path)
    assert path == tmp_path / "results.lewm.json"

    b = report.load_results([path])
    assert set(b) == {"lewm"}

    # meta records the run's fairness conditions so the numbers are self-describing later
    meta = json.loads(path.read_text())["meta"]
    assert meta["num_samples"] == study.CEM_NUM_SAMPLES
    assert meta["n_latency_iters"] == 2
    # PTQ calibration method label — a build option for both tracks (architecture.md §7), default `max`
    assert meta["calibration_method"] == "max"


def test_check_calibration_method_validates():
    """Calibration method is a build option for BOTH tracks (`max` | `entropy`); an unknown value
    fails loudly rather than mislabelling an artefact or crashing deep in modelopt (architecture.md §7)."""
    import pytest

    from src.interfaces import check_calibration_method

    assert check_calibration_method("max") == "max"
    assert check_calibration_method("entropy") == "entropy"
    with pytest.raises(SystemExit):
        check_calibration_method("minmax")  # not one of the supported ORT methods


def test_dump_track_results_is_additive_per_precision(tmp_path):
    """Benchmarking a precision subset later must NOT discard the track's other precisions — the
    canonical results file merges per precision (CLAUDE.md §8). Latency is calibration-method-
    invariant, so one file per track serves every method; the method is recorded as provenance."""
    cfg = ExportConfig()

    p = study.dump_track_results(
        "lewm", {"fp32": {"success_rate": 90.0}}, cfg, tmp_path, "max"
    )
    # A later fp8-only run (e.g. adding FP8) must leave fp32 on disk intact.
    study.dump_track_results("lewm", {"fp8": {"success_rate": 80.0}}, cfg, tmp_path, "entropy")

    data = json.loads(p.read_text())
    assert set(data["bench"]) == {"fp32", "fp8"}  # additive, no silent loss
    assert data["bench"]["fp32"]["success_rate"] == 90.0
    assert data["meta"]["calibration_method"] == "entropy"  # latest run's provenance label


def test_dump_track_latencies_roundtrips_and_records_the_loop_conditions(tmp_path):
    """The engine-step loops' RAW samples persist to `latencies.<track>.json` beside the results
    file, and load back in the shape `src.stats` consumes (SPEC §Interface Contracts). `meta` carries
    the loop conditions, so the sample is self-describing: n, the warm-up that ran untimed before
    it, and which method's engines were timed."""
    from src import stats

    cfg = dataclasses.replace(ExportConfig(), n_latency_iters=3, warmup=1)
    samples = {"fp32": {"encode_ms": [1.0, 2.0, 3.0], "predict_ms": [4.0, 5.0, 6.0]}}

    path = study.dump_track_latencies("lewm", samples, cfg, tmp_path, "entropy")
    assert path == tmp_path / "latencies.lewm.json"

    meta = json.loads(path.read_text())["meta"]
    assert (meta["n_latency_iters"], meta["warmup"]) == (3, 1)
    assert meta["calibration_method"] == "entropy"

    loaded = stats.load_component_latencies([path])
    assert loaded == {"lewm": samples}


def test_dump_track_latencies_is_additive_per_precision(tmp_path):
    """Same no-clobber discipline as `dump_track_results`: re-benchmarking one precision must not
    discard the other precisions' stored samples (CLAUDE.md §8) — losing them would cost an L40S run
    to recover, since nothing else on disk holds the raw vectors."""
    cfg = ExportConfig()
    p = study.dump_track_latencies(
        "dino", {"fp32": {"encode_ms": [1.0], "predict_ms": [2.0]}}, cfg, tmp_path
    )
    study.dump_track_latencies(
        "dino", {"fp8": {"encode_ms": [3.0], "predict_ms": [4.0]}}, cfg, tmp_path
    )

    stored = json.loads(p.read_text())["latencies"]
    assert set(stored) == {"fp32", "fp8"}
    assert stored["fp32"]["encode_ms"] == [1.0]
