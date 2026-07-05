"""Owned observation-only planning-latency hook (Phase 3).

Wraps the CEM solver's ``solve`` — one CEM planning cycle (``World._get_actions →
policy.get_action → solver.solve``) — to record its wall-clock latency **without
altering the plan**. It calls the original ``solve`` unchanged and records
``perf_counter`` deltas; a ``torch.cuda.synchronize()`` bracket makes the GPU
wall-clock number accurate while leaving seeds, sample draws, and the resulting plan
byte-identical (SPEC §Implementation Boundaries — CLAUDE CODE, observation-only).

``callables=`` in the eval driver is dataset-mode env setup, not a timing seam; the
timing seam is ``solver.solve``, whose signature is
``solve(info_dict, init_action=None) -> dict`` (confirmed against
``stable_worldmodel`` 0.1.1 ``CEMSolver``). The policy invokes the solver via
``self.solver(...)``, whose ``__call__`` forwards to ``self.solve(...)`` — so shadowing
the instance's ``solve`` attribute intercepts that call. Perturbing seeds, sample
counts, or the plan would cross into the eval/CEM parity gate (OWNER-ONLY); this hook
does none of that.

This module only records. Emitting the summary to W&B is the caller's one-liner
(``src/wandb_log.py``) in the eval-run step, keeping the hook free of a W&B dependency.
"""

from contextlib import contextmanager
from statistics import median
from time import perf_counter


class LatencyHook:
    """Records the wall-clock latency of each ``solver.solve`` call.

    ``sync_cuda`` brackets the timed region with ``torch.cuda.synchronize()`` so the
    measured span reflects GPU completion, not just kernel-launch return. It is a pure
    timing barrier — numerics are identical with or without it.
    """

    def __init__(self, sync_cuda=True):
        self.latencies_s = []
        self._sync_cuda = sync_cuda
        self._solver = None
        self._orig_solve = None

    def _sync(self):
        if not self._sync_cuda:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _wrap(self, solve):
        def timed_solve(*args, **kwargs):
            self._sync()
            t0 = perf_counter()
            out = solve(*args, **kwargs)  # original, unchanged
            self._sync()
            self.latencies_s.append(perf_counter() - t0)
            return out

        return timed_solve

    def attach(self, solver):
        """Shadow ``solver.solve`` with a timing wrapper. Idempotent guard: raises if
        already attached (attach → detach → attach, never double-wrap)."""
        if self._solver is not None:
            raise RuntimeError("LatencyHook already attached; detach() first")
        self._orig_solve = solver.solve
        solver.solve = self._wrap(self._orig_solve)
        self._solver = solver
        return self

    def detach(self):
        """Restore the original ``solve``, removing the instance shadow."""
        if self._solver is None:
            return
        # solve is a class method on CEMSolver; deleting the instance attribute we set
        # re-exposes it. Fall back to reassigning the captured original if solve was an
        # instance attribute to begin with.
        try:
            del self._solver.solve
        except AttributeError:
            self._solver.solve = self._orig_solve
        self._solver = None
        self._orig_solve = None

    def summary(self):
        """Eager-baseline latency: median of the recorded ``solve`` calls (ms).

        The rigorous p50/p95 timing rig is Phase 5; here we want a single stable number
        per track. Returns ``None`` medians when nothing was recorded so the caller can
        surface an empty run instead of dividing by zero.
        """
        n = len(self.latencies_s)
        return {
            "n_calls": n,
            "median_ms": median(self.latencies_s) * 1e3 if n else None,
        }


@contextmanager
def timed_solver(solver, sync_cuda=True):
    """Attach a :class:`LatencyHook` for the duration of the block, then restore.

    Usage in the eval-run step::

        with timed_solver(solver) as hook:
            world.evaluate(...)
        wandb.log({"plan_latency_median_ms": hook.summary()["median_ms"]})
    """
    hook = LatencyHook(sync_cuda=sync_cuda).attach(solver)
    try:
        yield hook
    finally:
        hook.detach()
