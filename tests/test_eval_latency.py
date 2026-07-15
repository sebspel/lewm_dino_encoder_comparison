"""Phase 3: the observation-only PER-DECISION planning-latency callback.

Core invariants under test:
  * one latency is recorded per DECISION — i.e. per env, per ``start_batch`` span — NOT per
    ``solve`` call. A solve is handed every alive episode and plans them sequentially
    (``batch_size=1``), so a per-solve bracket would time N decisions while ``src.report``
    weights by per-decision call counts. That unit mismatch inflates ``overhead_ms`` toward the
    whole cycle and cannot trip the negative-overhead alarm, so it is pinned here
    (SPEC §Interface Contracts — per-cycle);
  * the per-step ``__call__`` hook records nothing and never raises on ordinary state (a
    regression guard: the base ``Callback.__call__`` would call ``compute`` and raise
    ``NotImplementedError``, so the override is load-bearing) — but DOES fail loud if
    ``batch_size != 1``, the invariant the per-env span rests on;
  * the module registry lets the driver reach a config-instantiated recorder.

``sync_cuda=False`` keeps the unit test hermetic (no CUDA barrier).
"""

import pytest

from src import eval_latency
from src.eval_latency import SolveLatencyRecorder


class _Candidates:
    """Stands in for the solver's `candidates` tensor — only `.shape[0]` (current_bs) is read."""

    def __init__(self, current_bs=1):
        self.shape = (current_bs, 300, 5, 10)


def _simulate_solve(cb, n_envs=1, n_steps=3, current_bs=1):
    """Drive the callback through one solve exactly as ``CEMSolver.solve`` does: one ``reset``,
    then per ENV a ``start_batch`` + ``n_steps`` per-step calls, then one ``end_solve``. At the
    pinned ``batch_size=1`` the env loop runs once per alive episode (``cem.py:152``)."""
    cb.reset()
    for _ in range(n_envs):
        cb.start_batch()
        for step in range(n_steps):
            cb(step=step, costs=None, candidates=_Candidates(current_bs), mean=None, var=None)
    cb.end_solve()


def test_one_latency_per_decision_not_per_solve():
    """THE regression guard: a 50-env solve must yield 50 records, not 1. A per-solve bracket
    would silently pair ~50 decisions' wall clock with one decision's call counts."""
    cb = SolveLatencyRecorder(sync_cuda=False)
    _simulate_solve(cb, n_envs=50)
    assert cb.summary()["n_cycles"] == 50
    assert cb.summary()["median_ms"] >= 0.0


def test_records_accumulate_across_solves():
    cb = SolveLatencyRecorder(sync_cuda=False)
    _simulate_solve(cb, n_envs=3)
    _simulate_solve(cb, n_envs=2)  # later solves are smaller as episodes terminate
    assert cb.summary()["n_cycles"] == 5  # 3 + 2 decisions, not 2 solves


def test_spans_are_contiguous_and_sum_to_the_env_loop():
    """Each span closes where the next opens, so the records partition the env loop — the
    property the pod-side reconciliation against the solver's printed solve time relies on."""
    cb = SolveLatencyRecorder(sync_cuda=False)
    from time import perf_counter

    cb.reset()
    for _ in range(4):
        cb.start_batch()
    t_first_span_open = perf_counter()
    cb.end_solve()
    total = sum(cb.latencies_s)
    assert len(cb.latencies_s) == 4
    # Sum of spans cannot exceed the wall time of the loop they partition.
    assert total <= perf_counter() - t_first_span_open + 1.0


def test_per_step_call_is_noop_and_records_nothing():
    cb = SolveLatencyRecorder(sync_cuda=False)
    cb.reset()
    cb.start_batch()
    # Arbitrary per-step state must not raise (base would hit compute -> NotImplementedError)
    assert cb(step=0, costs="anything", extra=object()) is None
    cb.end_solve()
    # Exactly one record from the single env's span; the per-step call added none.
    assert cb.summary()["n_cycles"] == 1


def test_batch_size_gt_one_fails_loud():
    """`batch_size != 1` would make each span cover several decisions — the exact silent unit
    mismatch this bracket exists to avoid. It must throw, not record a wrong number."""
    cb = SolveLatencyRecorder(sync_cuda=False)
    with pytest.raises(ValueError, match="batch_size=1"):
        _simulate_solve(cb, n_envs=1, current_bs=4)


def test_empty_summary_is_none():
    s = SolveLatencyRecorder(sync_cuda=False).summary()
    assert s["n_cycles"] == 0
    assert s["median_ms"] is None and s["p50_ms"] is None and s["p95_ms"] is None
    assert s["latencies_ms"] == []


def test_summary_reports_percentiles_and_raw_latencies():
    """The per-cycle headline is p50/p95 (SPEC §Interface Contracts); the raw per-decision list
    is kept so src.report can truncate to a common min-n across tracks (equal-n)."""
    cb = SolveLatencyRecorder(sync_cuda=False)
    for _ in range(5):
        _simulate_solve(cb)
    s = cb.summary()
    assert s["n_cycles"] == 5
    assert len(s["latencies_ms"]) == 5
    assert s["p95_ms"] >= s["p50_ms"] >= 0.0
    assert s["median_ms"] == s["p50_ms"]  # median == p50


def test_registry_pop_returns_summary_and_clears():
    eval_latency.reset_registry()
    cb = SolveLatencyRecorder(sync_cuda=False)  # self-registers on construction
    _simulate_solve(cb)

    summary = eval_latency.pop_records()
    assert summary["n_cycles"] == 1
    # Registry is now empty: a second pop fails loud.
    with pytest.raises(RuntimeError):
        eval_latency.pop_records()


def test_pop_records_raises_when_never_injected():
    eval_latency.reset_registry()
    with pytest.raises(RuntimeError):
        eval_latency.pop_records()
