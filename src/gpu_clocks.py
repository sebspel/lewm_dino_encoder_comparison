"""GPU telemetry logger — wraps a timed engine run with an `nvidia-smi dmon` observer.

Owned PLUMBING (fails LOUDLY). SPEC §Parity: **GPU clocks are not locked.** This passive observer
records per-sample telemetry — SM/mem **clock (MHz)**, **power (W)**, **temperature (C)**,
utilization, and memory — alongside every timed engine run, so the actual per-run clock/thermal
state is logged rather than assumed. It is a separate `nvidia-smi` subprocess and does NOT touch
seeds / samples / the plan.

Runs ONLY where `nvidia-smi` is present (the L40S pod); the context manager fails loud if it is
absent. The logs persist to `$STABLEWM_HOME/reports/phase5/gpu_logs/` (network volume, same
durability contract as the reported artifacts).
"""

from __future__ import annotations

import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

from src.env import stablewm_home

# `nvidia-smi dmon -s` field groups: p=power+temperature, u=utilization, c=SM/mem clocks (MHz),
# m=framebuffer/BAR1 memory. `-o DT` prepends the date + time to each sample so the log is a
# timestamped time series over the run.
_DMON_METRICS = "pucm"


def default_log_dir() -> Path:
    """Where the telemetry logs land by default: `$STABLEWM_HOME/reports/phase5/gpu_logs/` — the
    persistent network volume, same durability contract as the headline artifacts (SPEC §Parity).
    `STABLEWM_HOME` comes from `.env` and is required — see `src.env.stablewm_home`."""
    return stablewm_home() / "reports" / "phase5" / "gpu_logs"


def run_tag(track: str, precision: str, method: str, run_type: str) -> str:
    """The telemetry log's basename: `<track>.<precision>.<method>.<run_type>`.

    The calibration method is part of the tag because int8/fp8 are run once per method and
    `log_gpu` opens the log with `"w"`: under an unscoped `<track>.<precision>.<run_type>` tag an
    `entropy` re-run **overwrote the `max` run's telemetry in place**, leaving `src.gpu_telemetry` to
    pair the surviving log with whichever method it happened to be rendering — a silently wrong
    normalization input. Same defect the derived tables' method-scoped filenames already guard
    against (SPEC §Parity, CLAUDE §8).

    fp32/fp16 engines are method-INVARIANT but are still tagged with the run's method: the solves
    the observer brackets are a separate run per method, so their telemetry is too.

    `src.gpu_telemetry.harvest` parses this back, and also accepts the legacy 3-part
    `<track>.<precision>.<run_type>` names already on the volume."""
    return f"{track}.{precision}.{method}.{run_type}"


@contextmanager
def log_gpu(tag: str, log_dir: Path | None = None):
    """Run `nvidia-smi dmon` for the duration of the `with` block, streaming its timestamped
    samples (clock MHz / power / temp / utilization / memory) to `<log_dir>/<tag>.dmon.log`.

    `tag` identifies the run and is the log's basename — build it with `run_tag()`, which owns the
    naming convention. `log_dir` defaults to `default_log_dir()`. The observer is terminated on
    block exit (SIGTERM, then SIGKILL if it does not stop within 5s), so it never outlives the run
    it brackets."""
    log_dir = log_dir or default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{tag}.dmon.log"

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise FileNotFoundError(
            "nvidia-smi not found on PATH — required for GPU telemetry logging (pod-only)"
        )

    with path.open("w") as fh:
        proc = subprocess.Popen(
            [nvidia_smi, "dmon", "-s", _DMON_METRICS, "-o", "DT"],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        try:
            yield path
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
