"""Phase-5 SR-eval driver plumbing — pure helpers (CPU, no engines/dataset).

The eval leg needs real engines + CUDA + the Push-T dataset (pod-only), so here we exercise
the argv routing, track mapping, and the read-modify-write sr.json merge — the parts that must
be right for the pod run to hand a well-formed `{track:{precision:SR}}` file to src.report.
"""

import json

import pytest

from src import sr_eval


def test_track_from_experiment():
    assert sr_eval._track_from_experiment("eval_lewm") == "lewm"
    assert sr_eval._track_from_experiment("eval_dino") == "dino"
    with pytest.raises(SystemExit):
        sr_eval._track_from_experiment("dinov3")  # a TRAIN overlay, not an eval overlay


def test_experiment_from_argv_requires_experiment():
    assert (
        sr_eval._experiment_from_argv(["--config-dir", "conf", "+experiment=eval_dino"])
        == "eval_dino"
    )
    with pytest.raises(SystemExit):
        sr_eval._experiment_from_argv(["--config-dir", "conf"])


def test_split_argv_strips_driver_only_args():
    """`precision=`/`out=`/`calibration_method=` are NOT valid Hydra overrides on the pusht config,
    so they must be peeled off before the rest reaches `_compose_eval_cfg`; `+experiment=` stays."""
    precisions, out_dir, method, hydra_argv = sr_eval._split_argv(
        [
            "--config-dir", "conf", "+experiment=eval_dino",
            "precision=fp16,int8", "out=/tmp/x", "calibration_method=entropy",
        ]
    )
    assert precisions == ("fp16", "int8")
    assert str(out_dir) == "/tmp/x"
    assert method == "entropy"
    assert hydra_argv == ["--config-dir", "conf", "+experiment=eval_dino"]


def test_split_argv_defaults():
    precisions, out_dir, method, hydra_argv = sr_eval._split_argv(
        ["--config-dir", "conf", "+experiment=eval_lewm"]
    )
    assert precisions is None  # -> caller falls back to ExportConfig().precisions
    assert out_dir is None
    assert method is None  # -> caller falls back to DEFAULT_CALIBRATION_METHOD (max)
    assert hydra_argv == ["--config-dir", "conf", "+experiment=eval_lewm"]


def test_merge_sr_json_additive_over_track_precision_method(tmp_path):
    """The merged sr.json is keyed `{track: {precision: {method: SR}}}` and every write is additive
    at (track, precision, method) granularity (CLAUDE.md §8), so nothing already recorded is lost:
      - separate tracks don't clobber each other;
      - a precision SUBSET run keeps the track's other precisions (the bug that motivated this);
      - a second calibration METHOD lands beside the first under the same precision."""
    path = tmp_path / "sr.json"

    # First: lewm fp32 + int8 @ max.
    sr_eval._merge_sr_json(path, "lewm", "max", {"fp32": 42.0, "int8": 40.0})
    # A dino session must NOT clobber lewm.
    sr_eval._merge_sr_json(path, "dino", "max", {"fp32": 30.0})
    # A LATER fp8-only lewm run must keep lewm's fp32 + int8 on disk (subset no-clobber).
    sr_eval._merge_sr_json(path, "lewm", "max", {"fp8": 20.0})
    # int8 @ entropy for lewm must land BESIDE int8 @ max, not overwrite it.
    sr_eval._merge_sr_json(path, "lewm", "entropy", {"int8": 55.0})

    assert json.loads(path.read_text()) == {
        "lewm": {
            "fp32": {"max": 42.0},
            "int8": {"max": 40.0, "entropy": 55.0},  # both methods coexist
            "fp8": {"max": 20.0},
        },
        "dino": {"fp32": {"max": 30.0}},
    }


def test_merge_sr_json_folds_legacy_flat_entry(tmp_path):
    """A pre-labelling sr.json (flat `{precision: {success_rate, ...}}`, always `max`-calibrated)
    is losslessly folded under the explicit `max` label on first touch, so old and labelled shapes
    never mix under one precision."""
    path = tmp_path / "sr.json"
    path.write_text(
        json.dumps({"lewm": {"int8": {"success_rate": 48.0, "per_cycle_latencies_ms": [1, 2]}}})
    )

    sr_eval._merge_sr_json(path, "lewm", "entropy", {"int8": {"success_rate": 70.0}})

    data = json.loads(path.read_text())
    assert data["lewm"]["int8"]["max"]["success_rate"] == 48.0  # legacy preserved as max
    assert data["lewm"]["int8"]["entropy"]["success_rate"] == 70.0  # new method beside it
