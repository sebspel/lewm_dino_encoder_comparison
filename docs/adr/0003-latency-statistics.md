# ADR 0003 — Three statistics, three jobs: p50 compares, p95 describes, mean decomposes

**Status:** Accepted · **Date:** 2026-07-15

## Context

The study reports three latency distributions (per-cycle headline, encode-step and predictor-step
components) and also decomposes the cycle into encoder / predictor / overhead shares for an Amdahl
dilution disclosure. Both needed a statistic, and the two needs are not the same.

Two concrete defects motivated the ruling:

- `report.decompose` read `per_cycle_p50_ms` / `encode_p50_ms` / `predict_p50_ms` and subtracted. But
  the identity it asserts — `cycle = enc·2 + pred·150 + overhead` — holds only in **expectation**.
- `encode_p95_ms` / `predict_p95_ms` were computed and persisted but **rendered nowhere**, so two of
  the six SPEC-required numbers reached no table.

## Decision

**p50 — the COMPARISON basis.** The LeWM-vs-DINOv3 headline ratio and the FP32-relative speedup are
quoted at p50. The headline is a *mechanistic* claim (encoder compute asymmetry: LeWM's single token
vs DINOv3's 196-patch grid), i.e. a central-tendency question, and per-cycle n is 50–100 — p50 is the
statistic that sample supports.

**p95 — the reported tail, never a comparison basis.** Kept for all three distributions as a
descriptive figure. At n = 50–100 a p95 is roughly the 3rd-to-10th largest sample and its interval
reaches the maximum, so it is **not** load-bearing for any claim. Compounding this: the per-cycle path
has **no warm-up drop** (the vendored eval's warm-up pass is gated on `compile`, which is `false`), so
cold engine-context cost lands in exactly the samples p95 reads.

**mean — the DECOMPOSITION basis ONLY.** Never a reported headline.

### Why the decomposition must use means

`cycle = enc·calls + pred·calls + overhead` is exact for means by **linearity of expectation** —
unconditionally, for any distribution and any correlation between the terms — whereas
`p50(a + b) ≠ p50(a) + p50(b)` in general. A percentile decomposition therefore books its own
non-additivity error as "planner overhead", inflating the Amdahl denominator.

Amdahl's law is itself an expectation model, so the optimizable fraction `p` and the ceiling are
mean-derived too, and the *realized* speedup they reconcile against must be mean-based to match.

This is the one place a mean is used; nothing mean-based is ever reported as a headline.

### Why per-cycle n is 50–100, not thousands

Establish this before reading any per-cycle percentile. `eval_budget = 50` policy steps, and one plan
fills the action buffer with `receding_horizon × action_block = 25` env actions, so the solver is
called on exactly **two** steps (t=0 and t=25) and not at all in between. Decisions therefore total
`50 + alive_at_25` ≤ 100: solve 0 plans all 50 episodes, solve 1 only those not yet terminated.

Because Push-T terminates on success, **a higher-SR track contributes fewer decisions** — n is
SR-dependent hence track-dependent, which is what the equal-n truncation neutralises. This n is why
p50 carries the comparison and p95 does not.

## Consequences

- `decompose` + `dilution_disclosure` read `*_mean_ms`. `BenchResult` carries `per_cycle_mean_ms` /
  `encode_mean_ms` / `predict_mean_ms`; `benchmark` computes the step means, and
  `report._finalize_per_cycle` computes the cycle mean off the **same** equal-n-truncated sample as
  the reported p50/p95 — so the headline and the decomposition describe the same decisions.
- `tests/test_report.py::test_decompose_uses_mean_not_p50` sets the means away from the p50s, so
  reading the wrong field fails loudly.
- `per_cycle_ratio` defaults to p50; `plot_per_cycle_ratio` and the W&B
  `headline/per_cycle_p50_ratio_*` key follow it (p95 logged alongside, not plotted as the headline).
- `render_speed_table` renders all three distributions at p50 **and** p95.
- `fp32_relative` is quoted at **p50**, agreeing with the headline instead of introducing a second,
  tail-based speedup. It is distinct from `dilution_disclosure`'s mean-based
  `measured_realized_speedup` — same shape, different question; separate tables, never conflated or
  averaged.
- **Accepted residual (unquantified until the pod run):** the isolated engine loops drop warm-up iters
  but the per-cycle callback records from the first decision of the first solve, so cold-start cost
  sits in the cycle mean and **not** in the component means — the difference is booked as overhead.
  Means are outlier-sensitive, so this bites harder than it would at p50, and because it *inflates*
  overhead the negative-overhead alarm cannot catch it.
- **Open (owner):** whether to drop a per-cycle warm-up. It would shrink an already-small n and discard
  the only solve where all 50 episodes are alive.
