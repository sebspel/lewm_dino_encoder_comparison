"""Phase 3: the observation-only CEM-solve-latency callback.

Core invariants under test:
  * one latency is recorded per CEM solve (``reset → end_solve``), median over solves;
  * the per-step ``__call__`` hook is a strict no-op — it records nothing and never
    raises (a regression guard: the base ``Callback.__call__`` would call ``compute`` and
    raise ``NotImplementedError``, so the override is load-bearing);
  * the module registry lets the driver reach a config-instantiated recorder.

``sync_cuda=False`` keeps the unit test hermetic (no CUDA barrier).
"""

import pytest

from src import eval_latency
from src.eval_latency import SolveLatencyRecorder


def _simulate_solve(cb, n_batches=1, n_steps=3):
    """Drive the callback through one solve exactly as ``CEMSolver.solve`` does:
    one ``reset``, then per batch a ``start_batch`` + ``n_steps`` per-step calls, then
    one ``end_solve``."""
    cb.reset()
    for _ in range(n_batches):
        cb.start_batch()
        for step in range(n_steps):
            cb(step=step, costs=None, candidates=None, mean=None, var=None)
    cb.end_solve()


def test_one_latency_recorded_per_solve():
    cb = SolveLatencyRecorder(sync_cuda=False)
    _simulate_solve(cb)
    assert cb.summary()["n_solves"] == 1
    assert cb.summary()["median_ms"] >= 0.0

    _simulate_solve(cb, n_batches=3, n_steps=5)
    assert cb.summary()["n_solves"] == 2  # median now over two solves


def test_per_step_call_is_noop_and_records_nothing():
    cb = SolveLatencyRecorder(sync_cuda=False)
    cb.reset()
    cb.start_batch()
    # Arbitrary per-step state must not raise (base would hit compute -> NotImplementedError)
    assert cb(step=0, costs="anything", extra=object()) is None
    cb.end_solve()
    # Exactly one record from the reset->end_solve bracket; the per-step call added none.
    assert cb.summary()["n_solves"] == 1


def test_empty_summary_is_none():
    s = SolveLatencyRecorder(sync_cuda=False).summary()
    assert s["n_solves"] == 0
    assert s["median_ms"] is None and s["p50_ms"] is None and s["p95_ms"] is None
    assert s["latencies_ms"] == []


def test_summary_reports_percentiles_and_raw_latencies():
    """The per-cycle headline is p50/p95 (SPEC §Interface Contracts); the raw per-solve list is
    kept so src.report can truncate to a common min-n across tracks (equal-n)."""
    cb = SolveLatencyRecorder(sync_cuda=False)
    for _ in range(5):
        _simulate_solve(cb)
    s = cb.summary()
    assert s["n_solves"] == 5
    assert len(s["latencies_ms"]) == 5
    assert s["p95_ms"] >= s["p50_ms"] >= 0.0
    assert s["median_ms"] == s["p50_ms"]  # median == p50


def test_registry_pop_returns_summary_and_clears():
    eval_latency.reset_registry()
    cb = SolveLatencyRecorder(sync_cuda=False)  # self-registers on construction
    _simulate_solve(cb)

    summary = eval_latency.pop_records()
    assert summary["n_solves"] == 1
    # Registry is now empty: a second pop fails loud.
    with pytest.raises(RuntimeError):
        eval_latency.pop_records()


def test_pop_records_raises_when_never_injected():
    eval_latency.reset_registry()
    with pytest.raises(RuntimeError):
        eval_latency.pop_records()
