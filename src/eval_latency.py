"""Owned observation-only PER-DECISION planning-latency callback.

Recast from the initial ``solver.solve`` monkeypatch (commit ``c91d49b``) to a
``CEMSolver.Callback`` subclass, injected through the platform's own config seam
(``cfg.solver.callbacks``). The vendored eval entrypoint and the solver therefore stay
byte-untouched — no monkeypatch, no class shadow.

**One record = one episode's decision, NOT one ``solve`` call** (SPEC §Interface Contracts —
per-cycle). A solve is handed *every still-alive episode at once*: at eval the 50 episodes
are 50 env instances advanced in lockstep (``world.num_envs = eval.num_eval``; dataset eval
asserts ``num_envs == len(episodes_idx)`` and freezes rather than auto-resets terminated
envs, so env *i* hosts episode *i* for the whole run), and ``policy.get_action`` passes all
of them to a single ``solver()`` call. Nothing runs in parallel: ``EnvPool.step`` loops the
envs in-process, and ``solve``'s own env loop is sequential at the pinned ``batch_size=1``
(``LeWM.criterion`` errors for B>1). One solve is therefore N independent decisions computed
back-to-back and its wall clock is their **sum**; only the ``num_samples`` candidate fan-out
inside one ``get_cost`` is batched.

So we bracket **per env**: ``start_batch()`` fires inside that loop (``cem.py:186``), once per
episode, while ``reset()`` / ``end_solve()`` sit outside it (``cem.py:148`` / ``279``). Each
span runs from one ``start_batch`` to the next, the last closing at ``end_solve``. Bracketing
``reset → end_solve`` instead would time all N episodes while ``report``'s
``ENCODER/PREDICTOR_CALLS_PER_CYCLE`` count one — a unit mismatch that inflates ``overhead_ms``
toward the whole cycle, and one the negative-overhead alarm cannot catch (it makes overhead
*more* positive, never negative).

Excluded from every span: the pre-loop ``prepare_init_action`` warm-start — a zero-pad for the
non-``Actionable`` LeWM / DINO-WM models, hence negligible and model-independent
(docs/platform_api.md §5).

Parity-safe: the callback only reads a clock and (optionally) inserts a CUDA barrier; it
never touches seeds, sample draws, or the plan. The per-CEM-step ``__call__`` hook records
nothing and returns None — it only asserts the ``batch_size==1`` invariant the per-env span
depends on. Perturbing anything else there would cross into the eval/CEM parity gate
(owner-only).

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
    """Records the wall-clock latency of each episode's decision (one ``start_batch`` span).

    ``sync_cuda`` closes each span with ``torch.cuda.synchronize()`` so the number reflects
    GPU completion, not just kernel-launch return — a pure timing barrier that leaves seeds,
    sample draws, and the plan byte-identical. One barrier per env rather than per solve; each
    waits on exactly the env whose span it closes, which is the work being attributed.
    """

    name = "per_cycle_latency"

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
        """Start of one solve, before its env loop: no span is open yet (the pre-loop
        warm-start is deliberately outside every span). The first ``start_batch`` opens one."""
        super().reset()
        self._t0 = None

    def start_batch(self):
        """Fires once per env, inside ``solve``'s env loop. Closes the previous env's span and
        opens this one off the SAME timestamp, so spans are contiguous and sum to the loop."""
        super().start_batch()
        self._sync()
        t = perf_counter()
        if self._t0 is not None:
            self.latencies_s.append(t - self._t0)
        self._t0 = t

    def __call__(self, **state):
        """Observation-only: records nothing and touches no solver state.

        Asserts the invariant the per-env span rests on — ``batch_size == 1``, so one span is
        one decision. ``candidates`` is ``(current_bs, num_samples, horizon, action_dim)``; a
        ``current_bs > 1`` would make each span cover ``current_bs`` decisions and silently
        re-open the unit mismatch against ``PREDICTOR_CALLS_PER_CYCLE``, so it fails loud.
        """
        candidates = state.get("candidates")
        if candidates is not None and getattr(candidates, "shape", None) is not None:
            if candidates.shape[0] != 1:
                raise ValueError(
                    f"per-decision latency needs solver batch_size=1, got current_bs="
                    f"{candidates.shape[0]}: each span would cover {candidates.shape[0]} "
                    "decisions while report weights by per-decision call counts "
                    "(SPEC §Interface Contracts — per-cycle)."
                )

    def end_solve(self):
        """End of the solve: close the final env's span (no ``start_batch`` follows it)."""
        super().end_solve()
        if self._t0 is not None:
            self._sync()
            self.latencies_s.append(perf_counter() - self._t0)
            self._t0 = None

    def summary(self):
        """Per-cycle (per-decision) latency distribution over the recorded decisions.

        ``n_cycles`` counts DECISIONS, not solves — a solve contributes one record per alive
        episode (module docstring). Reports ``p50_ms`` / ``p95_ms`` (the Phase-5 headline, SPEC
        §Interface Contracts) and keeps ``median_ms`` (== p50) for the Phase-3 eval driver.
        ``latencies_ms`` is the RAW per-decision list so ``src.report`` can truncate to the
        common min-n across tracks before taking the percentiles (equal-n). All are ``None`` /
        ``[]`` on an empty run, so the caller can surface it instead of dividing by zero.
        """
        raw_ms = [x * 1e3 for x in self.latencies_s]  # TEMPORAL (recording) order
        srt = sorted(raw_ms)
        n = len(raw_ms)
        return {
            "n_cycles": n,
            "median_ms": median(srt) if n else None,
            "p50_ms": _percentile(srt, 50) if n else None,
            "p95_ms": _percentile(srt, 95) if n else None,
            # RAW, in temporal order (env order within a solve, solves in run order) — src.report
            # truncates to the common min-n across tracks by taking the first n (a representative
            # chronological subset), NOT the n smallest, which would drop the upper tail
            # (equal-n, SPEC §Interface Contracts).
            "latencies_ms": raw_ms,
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
