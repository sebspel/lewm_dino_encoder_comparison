"""GPU telemetry logger — wraps a timed engine run with an `nvidia-smi dmon` observer.

Owned PLUMBING (fails LOUDLY). SPEC §Parity: **GPU clocks are not locked**; the LeWM-vs-DINOv3
comparison is a *ratio* on the shared back-to-back hardware state, so any residual boost/thermal
drift applies to both tracks alike. This passive observer records per-sample telemetry — SM/mem
**clock (MHz)**, **power (W)**, **temperature (C)**, utilization, and memory — alongside every
timed engine run, so that shared-hardware-state assumption is **verifiable off the log** rather
than merely asserted. Like the `cudaMemGetInfo` peak-mem sampling in `src.benchmark`, it is a
separate `nvidia-smi` subprocess and does NOT touch seeds / samples / the plan.

Runs ONLY where `nvidia-smi` is present (the L40S pod); the context manager fails loud if it is
absent. The logs persist to `$STABLEWM_HOME/reports/phase5/gpu_logs/` (network volume, same
durability contract as the headline artifacts).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

# `nvidia-smi dmon -s` field groups: p=power+temperature, u=utilization, c=SM/mem clocks (MHz),
# m=framebuffer/BAR1 memory. `-o DT` prepends the date + time to each sample so the log is a
# timestamped time series over the run.
_DMON_METRICS = "pucm"


def default_log_dir() -> Path:
    """Where the telemetry logs land by default: `$STABLEWM_HOME/reports/phase5/gpu_logs/` — the
    persistent network volume, same durability contract as the headline artifacts (SPEC §Parity).
    Falls back to repo-local `reports/phase5/gpu_logs` off-pod where `STABLEWM_HOME` is unset."""
    home = os.environ.get("STABLEWM_HOME")
    base = Path(home) / "reports" / "phase5" if home else Path("reports/phase5")
    return base / "gpu_logs"


@contextmanager
def log_gpu(tag: str, log_dir: Path | None = None):
    """Run `nvidia-smi dmon` for the duration of the `with` block, streaming its timestamped
    samples (clock MHz / power / temp / utilization / memory) to `<log_dir>/<tag>.dmon.log`.

    `tag` identifies the run and is the log's basename (e.g. `lewm.fp16.benchmark`). `log_dir`
    defaults to `default_log_dir()`. The observer is terminated on block exit (SIGTERM, then
    SIGKILL if it does not stop within 5s), so it never outlives the run it brackets."""
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
