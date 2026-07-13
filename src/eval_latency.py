"""Owned observation-only CEM-solve-latency callback.

Recast from the initial ``solver.solve`` monkeypatch (commit ``c91d49b``) to a
``CEMSolver.Callback`` subclass, injected through the platform's own config seam
(``cfg.solver.callbacks``). The vendored eval entrypoint and the solver therefore stay
byte-untouched — no monkeypatch, no class shadow.

The base ``CEMSolver.solve`` calls ``reset()`` once at the start of the optimization body
(after ``prepare_init_action`` / ``init_action_distrib``) and ``end_solve()`` once at the
end. We bracket that span (``reset → end_solve``) with ``perf_counter`` and an optional
``torch.cuda.synchronize()`` barrier, recording **one latency per CEM solve**. This
excludes the ``prepare_init_action`` warm-start — a zero-pad for the non-``Actionable``
LeWM / DINO-WM models, hence negligible and model-independent (docs/platform_api.md §5),
so the metric is labelled *CEM-solve latency*.

Parity-safe: the callback only reads a clock and (optionally) inserts a CUDA barrier; it
never touches seeds, sample draws, or the plan. The per-CEM-step ``__call__`` hook is a
no-op — perturbing anything there would cross into the eval/CEM parity gate (owner-only).

Records only — no W&B dependency here. Because the platform (not the driver) instantiates
the callback from config, each instance registers itself in a module-level registry so the
owned eval driver (``src/eval.py``) can read the records via :func:`pop_records` and log
the median after the run.
"""

from statistics import median
from time import perf_counter

from stable_worldmodel.solver.callbacks import Callback


def _percentile(sorted_ms, q):
    """Linear-interpolated q-th percentile (q in [0,100]) over a NON-empty sorted list."""
    if len(sorted_ms) == 1:
        return sorted_ms[0]
    pos = (q / 100.0) * (len(sorted_ms) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_ms) - 1)
    return sorted_ms[lo] + frac * (sorted_ms[hi] - sorted_ms[lo])

# Registry: instances append themselves at construction (see module docstring) so the
# driver can reach the config-instantiated callback after the run.
_RECORDERS = []


class SolveLatencyRecorder(Callback):
    """Records the wall-clock latency of each CEM solve (``reset → end_solve``).

    ``sync_cuda`` brackets the timed span with ``torch.cuda.synchronize()`` so the number
    reflects GPU completion, not just kernel-launch return — a pure timing barrier that
    leaves seeds, sample draws, and the plan byte-identical.
    """

    name = "cem_solve_latency"

    def __init__(self, sync_cuda=True):
        super().__init__(reduction="none")
        self.sync_cuda = sync_cuda
        self.latencies_s = []
        self._t0 = None
        _RECORDERS.append(self)

    def _sync(self):
        if not self.sync_cuda:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset(self):
        """Start of one solve's optimization body (after the warm-start): open the bracket."""
        super().reset()
        self._sync()
        self._t0 = perf_counter()

    def __call__(self, **state):
        """Observation-only: record nothing per CEM step and touch no solver state."""
        pass

    def end_solve(self):
        """End of the solve: close the bracket and record one latency."""
        super().end_solve()
        if self._t0 is not None:
            self._sync()
            self.latencies_s.append(perf_counter() - self._t0)
            self._t0 = None

    def summary(self):
        """Per-cycle (CEM-solve) latency distribution over the recorded solves.

        Reports ``p50_ms`` / ``p95_ms`` (the Phase-5 headline, SPEC §Interface Contracts)
        and keeps ``median_ms`` (== p50) for the Phase-3 eval driver. ``latencies_ms`` is the
        RAW per-solve list so ``src.report`` can truncate to the common min-n across tracks
        before taking the percentiles (equal-n). All are ``None`` / ``[]`` on an empty run, so
        the caller can surface it instead of dividing by zero.
        """
        lat_ms = sorted(x * 1e3 for x in self.latencies_s)
        n = len(lat_ms)
        return {
            "n_solves": n,
            "median_ms": median(lat_ms) if n else None,
            "p50_ms": _percentile(lat_ms, 50) if n else None,
            "p95_ms": _percentile(lat_ms, 95) if n else None,
            "latencies_ms": lat_ms,
        }


def pop_records():
    """Summary of the most recently constructed recorder; clears the registry.

    Raises if no recorder exists — i.e. the callback was never injected via
    ``cfg.solver.callbacks`` (fails loud).
    """
    if not _RECORDERS:
        raise RuntimeError(
            "no SolveLatencyRecorder was constructed — is it injected via "
            "cfg.solver.callbacks in the eval overlay?"
        )
    summary = _RECORDERS[-1].summary()
    _RECORDERS.clear()
    return summary


def reset_registry():
    """Drop any registered recorders (driver hygiene before a run)."""
    _RECORDERS.clear()
