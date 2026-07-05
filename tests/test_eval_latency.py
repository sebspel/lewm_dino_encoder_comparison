"""Phase 3: the observation-only planning-latency hook.

Core invariant under test — the hook must be *observation-only*: the value returned by
``solve`` is byte-identical with the hook attached, and the original ``solve`` is
restored on detach. ``sync_cuda=False`` keeps the unit test hermetic (no torch/CUDA).
"""

from src.eval_latency import LatencyHook, timed_solver


class FakeSolver:
    """Mimics CEMSolver's call surface: ``__call__`` forwards to ``solve``, and
    ``solve(info_dict, init_action=None)`` returns a deterministic plan dict."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        self.calls += 1
        return {"actions": [self.calls, info_dict["x"]], "init": init_action}


def test_plan_unchanged_and_recorded():
    solver = FakeSolver()
    baseline = solver({"x": 7}, init_action=3)

    solver = FakeSolver()  # fresh, so call counter matches the baseline
    hook = LatencyHook(sync_cuda=False).attach(solver)
    hooked = solver({"x": 7}, init_action=3)  # via __call__ -> solve
    hook.detach()

    assert hooked == baseline  # observation-only: plan byte-identical
    assert hook.summary()["n_calls"] == 1
    assert hook.summary()["median_ms"] >= 0.0


def test_detach_restores_original_solve():
    solver = FakeSolver()
    original = solver.solve
    with timed_solver(solver, sync_cuda=False) as hook:
        solver({"x": 1})
        assert solver.solve is not original  # shadowed while attached
    assert solver.solve == original  # class method re-exposed after detach
    assert hook.summary()["n_calls"] == 1


def test_double_attach_guarded():
    solver = FakeSolver()
    hook = LatencyHook(sync_cuda=False).attach(solver)
    try:
        with __import__("pytest").raises(RuntimeError):
            hook.attach(solver)
    finally:
        hook.detach()


def test_empty_summary_is_none():
    assert LatencyHook(sync_cuda=False).summary() == {"n_calls": 0, "median_ms": None}
