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
    """`precision=`/`out=` are NOT valid Hydra overrides on the pusht config, so they must be
    peeled off before the rest reaches `_compose_eval_cfg`; `+experiment=` stays (it IS one)."""
    precisions, out_dir, hydra_argv = sr_eval._split_argv(
        ["--config-dir", "conf", "+experiment=eval_dino", "precision=fp16,int8", "out=/tmp/x"]
    )
    assert precisions == ("fp16", "int8")
    assert str(out_dir) == "/tmp/x"
    assert hydra_argv == ["--config-dir", "conf", "+experiment=eval_dino"]


def test_split_argv_defaults():
    precisions, out_dir, hydra_argv = sr_eval._split_argv(
        ["--config-dir", "conf", "+experiment=eval_lewm"]
    )
    assert precisions is None  # -> caller falls back to ExportConfig().precisions
    assert out_dir is None
    assert hydra_argv == ["--config-dir", "conf", "+experiment=eval_lewm"]


def test_merge_sr_json_is_per_track_no_clobber(tmp_path):
    """LeWM and DINOv3 benchmark in separate pod sessions; each session must update only its own
    track key, leaving the other intact (CLAUDE.md §8) — while the file stays the single
    `{track:{prec:SR}}` shape src.study/src.report consume via `sr=<file>`."""
    path = tmp_path / "sr.json"

    sr_eval._merge_sr_json(path, "lewm", {"fp32": 42.0, "fp16": 41.0})
    assert json.loads(path.read_text()) == {"lewm": {"fp32": 42.0, "fp16": 41.0}}

    # A second session (dino) must NOT clobber the lewm key already on disk.
    sr_eval._merge_sr_json(path, "dino", {"fp32": 30.0})
    assert json.loads(path.read_text()) == {
        "lewm": {"fp32": 42.0, "fp16": 41.0},
        "dino": {"fp32": 30.0},
    }

    # Re-running one track overwrites only that track's block.
    sr_eval._merge_sr_json(path, "lewm", {"fp32": 43.0})
    assert json.loads(path.read_text()) == {
        "lewm": {"fp32": 43.0},
        "dino": {"fp32": 30.0},
    }
