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
- ~~**Accepted residual (unquantified until the pod run):** the isolated engine loops drop warm-up
  iters but the per-cycle callback records from the first decision of the first solve, so cold-start
  cost sits in the cycle mean and **not** in the component means — the difference is booked as
  overhead. Means are outlier-sensitive, so this bites harder than it would at p50, and because it
  *inflates* overhead the negative-overhead alarm cannot catch it.~~
  → **CLOSED** by the warm-up amendment below (2026-07-21).
- ~~**Open (owner):** whether to drop a per-cycle warm-up. It would shrink an already-small n and
  discard the only solve where all 50 episodes are alive.~~
  → **RESOLVED** (owner, 2026-07-21): drop `k = 1`. The cost was mis-stated here — see the amendment.

---

## Amendment (2026-07-21) — report the n each percentile was computed from

**Status:** Accepted

The ruling above rests on n: p50 carries the comparison *because* n is 50–100, and the equal-n
truncation exists *because* n is SR-dependent hence track-dependent. Neither is checkable from the
rendered report — `_finalize_per_cycle` computes the common min-n, truncates, and discards the count.
A reader of `speed_table.txt` cannot tell whether a p95 came off 66 samples or 98, nor verify that
the two tracks were compared at equal n. The contract is asserted in prose and unfalsifiable from the
artefact.

**Decision:** the speed table gains an **`n` column** — the post-truncation per-cycle sample count
each row's percentiles and mean were computed from.

**The derivation is now confirmed by measurement.** The DINO component-isolation runs
(`docs/adr/0005`, `entropy`, 2026-07-21) recorded n = 66, 83, 93, 98 at SR = 70, 42, 16, 4%. Those fit
`n = 50 + alive_at_25` closely, with episodes terminating by t=25 tracking the eventual success count
— e.g. 70% SR → 66 decisions (16 alive at the second solve), 4% SR → 98 (48 alive). The n model above
is measured, not merely traced, and the SR-dependence it predicts is large: a 32-decision spread
across one track's precisions.

Because equal-n truncation takes the common minimum across tracks, the **highest-SR track sets n for
every row at that precision** — so a single strong result shrinks the sample the tail is read from.
That is another reason p95 carries no claim, and another reason the number belongs on the page.

---

## Amendment (2026-07-21) — drop a per-cycle warm-up of k = 1 decision

**Status:** Accepted (owner) · **Closes:** the "Open (owner)" item and the warm-up accepted residual

### The defect is asymmetry, not cold start

The engine-step loops drop `ExportConfig.warmup = 10` iters. The per-cycle callback drops nothing —
the vendored eval's warm-up pass is gated on `compile`, which is `false`, so `SolveLatencyRecorder`
records from the **first decision of the first solve**, including the first `execute_v2`, kernel
autotune, allocator growth, and clock ramp from idle.

Including cold start is not itself dishonest. The problem is that the report **subtracts one from the
other**: `overhead = cycle − enc·2 − pred·150` is only meaningful if both sides were measured under
the same warm-up regime. As it stood, the entire cold-start cost was booked as *planner overhead*,
which deflates `p` and lowers the Amdahl ceiling — i.e. it made quantization look **less** useful
than it is, in the study built to measure exactly that. And because it inflates overhead, the
negative-overhead alarm can never fire on it.

Two further points, neither previously recorded:

- **The bias is probably not symmetric across tracks.** DINO's execution contexts are ~11 GB
  (`src/sr_eval.py`'s teardown note — three precisions OOM'd without explicit collection); LeWM's
  ViT-Tiny contexts are trivial. So DINO's cold decision is plausibly far more expensive. The
  dilution table is a *reconciliation*, so a per-track-asymmetric bias there reads as an Amdahl-model
  failure rather than a measurement artefact.
- **Equal-n truncation actively preserved the cold sample.** `lat[:n]` keeps the temporal head, so
  truncating DINO's 66 samples to LeWM's 51 discarded 15 clean tail samples and kept the cold one.

### The cost was mis-stated

The original open item said a warm-up drop "would discard the only solve where all 50 episodes are
alive". That describes dropping the first **solve** — 50 of ~66 samples, correctly unacceptable.
Dropping the first **k decisions** costs 1 of ~66. The per-decision unit is homogeneous across solves
(ADR 0004: solve 1's decisions are the same unit, just fewer of them), so there is no reason the
drop must align to a solve boundary. The surgical option was never as expensive as this ADR implied.

### Decision

**`PER_CYCLE_WARMUP_DROP = 1`** (`src/interfaces.py`), applied in `report._finalize_per_cycle`.

- **At report time, never at record time.** `sr.json` keeps the complete raw vector, so nothing
  measured is destroyed (CLAUDE §8), the ADR-0004 span-sum reconciliation still holds against the
  untouched record, and `per_cycle_warmup=0` re-renders the undropped view off-pod with no re-run.
- **Before the equal-n truncation**, or truncation would preserve the cold decision by construction.
- **Disclosed, not hidden:** the dropped values are stashed and rendered as the speed table's
  `drop×` (worst dropped decision ÷ retained `cyc_p50`). A `drop×` near 1 says the cold decision was
  unremarkable and the correction was moot; a large one says it mattered. Either way the reader sees
  what was excluded, which pre-empts the obvious challenge that the warm-up was tuned until the
  numbers looked good.

### Consequences

- **The p50 headline does not move.** A median over n = 50-100 is robust to one head sample; pinned
  by `tests/test_report.py::test_warmup_drop_does_not_move_the_p50_headline`. This is a correction to
  the **mean**-based decomposition and dilution tables, not a result-changing intervention — which is
  what makes it defensible to apply after the data was collected.
- p95 does move (a cold sample sits near the 3rd-4th largest at n=66). p95 carries no claim.
- `k = 1` is the minimum defensible value and the default. Whether cold cost decays across several
  decisions is empirical — `sr.json` already holds the raw vectors, so comparing `latencies_ms[0]`
  (and `[:5]`) against the median of the remainder sets `k` from data. **That check has not yet been
  run**; until it is, `k = 1` is a principled default, not a measured one.
- `decompose`'s KNOWN-RESIDUAL note is retired; the residual is closed rather than accepted.
