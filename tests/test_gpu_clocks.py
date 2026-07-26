"""GPU telemetry logger plumbing (CPU) — dir convention + the dmon subprocess wiring.

`nvidia-smi` is pod-only, so here we monkeypatch `shutil.which` / `subprocess.Popen` to exercise
the context manager without a GPU: it must fail loud when nvidia-smi is absent, and otherwise
open the log file, launch `dmon` with the clock/power/temp/util/mem selectors, and terminate the
observer on exit.
"""

from pathlib import Path

import pytest

from src import gpu_clocks


def test_default_log_dir_uses_stablewm_home(monkeypatch, tmp_path):
    """Durability (SPEC §Parity): telemetry logs land under
    `$STABLEWM_HOME/reports/phase5/gpu_logs/`, falling back to repo-local off-pod."""
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    assert gpu_clocks.default_log_dir() == tmp_path / "reports" / "phase5" / "gpu_logs"

    monkeypatch.delenv("STABLEWM_HOME", raising=False)
    assert gpu_clocks.default_log_dir() == Path("reports/phase5/gpu_logs")


def test_log_gpu_fails_loud_without_nvidia_smi(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_clocks.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="nvidia-smi"):
        with gpu_clocks.log_gpu("t", tmp_path):
            pass


class _FakePopen:
    def __init__(self, args, **kwargs):
        self.args = args
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):  # pragma: no cover - only on the timeout path
        pass


def test_log_gpu_launches_dmon_and_terminates(monkeypatch, tmp_path):
    launched = {}

    def _fake_popen(args, **kwargs):
        launched["proc"] = _FakePopen(args, **kwargs)
        return launched["proc"]

    monkeypatch.setattr(gpu_clocks.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu_clocks.subprocess, "Popen", _fake_popen)

    with gpu_clocks.log_gpu("lewm.fp16.benchmark", tmp_path) as path:
        assert path == tmp_path / "lewm.fp16.benchmark.dmon.log"
        assert path.exists()  # opened for writing inside the block

    proc = launched["proc"]
    assert proc.args[:2] == ["/usr/bin/nvidia-smi", "dmon"]
    assert "-s" in proc.args and gpu_clocks._DMON_METRICS in proc.args
    assert proc.terminated  # observer stopped on block exit


def test_run_tag_carries_the_calibration_method():
    """int8/fp8 are run once per method and `log_gpu` opens the log with `"w"`, so a tag without
    the method let an `entropy` re-run overwrite the `max` run's telemetry in place — and left
    the render pairing the survivor with whichever method it was rendering (CLAUDE §8)."""
    assert gpu_clocks.run_tag("dino", "int8", "max", "sr_eval") == "dino.int8.max.sr_eval"
    assert gpu_clocks.run_tag("dino", "int8", "entropy", "sr_eval") != gpu_clocks.run_tag(
        "dino", "int8", "max", "sr_eval"
    )
    # `src.clock_norm.harvest` splits the tag on "." — a composite isolation key must survive it
    assert (
        gpu_clocks.run_tag("dino", "enc-fp16+pred-int8", "entropy", "sr_eval").split(".")
        == ["dino", "enc-fp16+pred-int8", "entropy", "sr_eval"]
    )
