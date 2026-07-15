# ADR 0004 — A cycle is one decision: bracket latency per env, not per solve

**Status:** Accepted · **Date:** 2026-07-15 · **Supersedes:** the Phase-3 `reset → end_solve` bracket

## Context

`SolveLatencyRecorder` bracketed `reset → end_solve`, but both hooks sit **outside** `solve`'s env
loop (`cem.py:148` / `152` / `279`), so one record timed **every still-alive episode**, while
`ENCODER/PREDICTOR_CALLS_PER_CYCLE` count **one** decision.

`report.decompose` divides across that gap → overhead absorbs ~98% of the cycle, `p ≈ 0.02`, Amdahl
ceiling ≈ 1.02 — a false "quantization can't help". The error is **silent**: it inflates overhead, so
the negative-overhead alarm cannot fire.

### What the source actually does (traced 2026-07-15)

- `world.num_envs = eval.num_eval = 50`. Dataset-driven eval asserts `num_envs == len(episodes_idx)`
  (`world/world.py:503`) and **freezes** rather than auto-resets terminated envs (`reset_mode='wait'`),
  so env *i* hosts episode *i* for the whole run.
- `policy.py:379-404` passes every alive episode to **one** `solver()` call.
- `batch_size = 1` (pinned; `LeWM.criterion` errors for B > 1) makes that solve's env loop sequential.
- **Nothing runs in parallel.** `EnvPool.step` is a Python for-loop over in-process envs
  (`world/env_pool.py:134`). The platform's docstrings call the pool "parallel envs" and `batch_size`
  the envs "to process in parallel" — that names a **vectorized interface, not concurrency**; do not
  read it as parallel execution.

So one solve = **N independent decisions computed back-to-back, and its wall clock is their sum**. The
only genuinely batched axis is the `num_samples` candidate fan-out *within* one episode's `get_cost`.

## Decision

A **cycle is one episode's decision**. The latency callback brackets **per env** — consecutive
`start_batch` hooks, the last closing at `end_solve` — one record per decision, with a sync per span.
It never brackets `reset → end_solve`, which spans the whole batch.

Bracketing per solve while weighting components by per-decision call counts inflates `overhead_ms`
toward the entire cycle, and it would also make the headline scale with how many episodes are still
alive — SR-dependent, therefore track-dependent: **a parity break**.

**The measurement moves to the counts, not the reverse** — the constants are unchanged.

A `current_bs == 1` guard reads the per-step `__call__`'s `candidates` and fails loud if `batch_size`
ever ≠ 1, which would silently restore the mismatch.

### Rejected alternatives

- **Divide solve time by env count** — yields a mean; `p95(solve/n) ≠ p95(per-env latency)`.
- **Scale the constants** — the factor decays as episodes terminate; it is not constant.
- **Latency-only pass at `num_envs = 1`** — breaks Phase-3 eval parity and the same-solves SR pairing.

## Consequences

- `src/eval_latency.py` brackets per env; `n_solves` → `n_cycles`; consumers (`src/eval.py`,
  `src/sr_eval.py`) updated.
- W&B keys moved `cem_solve_*` → `per_cycle_*` **because the number's meaning changed** (~1/n_envs of
  the old one). The Phase-3 baseline logged under the old keys is **not comparable** to anything logged
  since.
- `tests/test_eval_latency.py`: a 50-env solve must record 50 latencies — fails against the old
  bracket, which recorded 1.
- **Pod-verify is the real gate:** the per-env spans must **sum to the solver's own printed
  `CEM solve time`** (`cem.py:282`) less the pre-loop warm-start — the reconciliation proving both
  measure the same work.
- **Accepted residuals:** each span carries one env's iterations + writeback + the *next* env's info
  expansion (the hook fires after it), so boundary attribution shifts by one — but the unit stays
  homogeneous (every env contributes exactly one of each). The last span also absorbs the trailing
  `outputs` assembly, and the pre-loop warm-start falls outside all spans (negligible and
  model-independent for non-`Actionable` models — `docs/platform_api.md` §5).
