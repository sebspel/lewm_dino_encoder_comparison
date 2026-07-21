# ADR 0005 — Component-precision isolation: attributing 8-bit SR loss to encoder or predictor

**Status:** Accepted · **Date:** 2026-07-21 · **Extends:** `docs/adr/0002`

## Context

ADR 0002 and its amendments exhausted the calibration levers: the inference-distribution fix
(recovered LeWM), the attention-mask sentinel neutralization (unblocked DINO `entropy`), and the
calibration method as a build option. What remains is a measured degradation neither lever removed —
DINO INT8/FP8 far below its ~70% FP32/FP16 baseline, and LeWM INT8 at ~76% against ~98%.

The headline tables report **that** SR fell. They do not report **which component's** quantization
caused it: `speed_table` rows are `(track, precision)`, and both engines move together. The encoder
and the predictor are separately exported, separately quantized, and separately timed — so the
attribution is measurable, and without it the study reports a symptom.

## Decision

**Re-run the SR eval with ONE component quantized and the other held at FP16**, per `(track,
precision)` cell that shows a material SR drop. Two runs per cell isolate the two sides; together
with the FP16 baseline and the pure-quantized run they close a 2×2:

| encoder | predictor | run |
|---|---|---|
| fp16 | fp16 | the FP16 baseline (already measured) |
| 8-bit | fp16 | isolates the encoder |
| fp16 | 8-bit | isolates the predictor |
| 8-bit | 8-bit | **the pure-precision run** (already measured) |

The fourth corner is not a new run — it is the pure INT8/FP8 point. So the 2×2 costs two evals per
cell, and its closure is a consistency check: if the components' damage composes, the pure corner is
predictable from the two isolated ones; if it is not, the two interact and the attribution carries a
caveat.

### Diagnostic, not a shipped configuration

A mixed-precision engine pairing is **not** a fifth precision in the study's FP32→FP16→INT8→FP8
sweep. It is never benchmarked for latency, never entered in the headline ratio, and never quoted as
a recommended configuration. Promoting one would introduce a precision outside the SPEC sweep, would
require cross-track counterparts to support any comparison, and touches the owner-gated benchmark
fairness conditions. This ADR deliberately claims less: *which component caused the ΔSR already
reported*.

Mechanically, that is enforced rather than trusted. Results land under a composite precision label
`enc-<A>+pred-<B>` which cannot collide with `fp32`/`fp16`/`int8`/`fp8`, so they are written beside
the pure points and never overwrite them; `report._PRECISIONS` stays a closed tuple, so the composite
keys reach no headline table, plot, or ratio.

### Both tracks

A DINO-only isolation would argue "the encoder is not LeWM's problem" from **absence of evidence**,
in a study whose spine is like-for-like comparison. LeWM's own 22pp INT8 drop is currently
unattributed.

ADR 0002 also makes a **falsifiable prediction** about it: under Design A the `action_encoder` sits
*inside* LeWM's predict engine, so the unbounded CEM action stressor is a predictor-side engine
input. The isolation therefore tests the ADR's stated mechanism rather than merely describing a
second track. If LeWM's damage is predictor-dominant while DINO's is encoder-dominant, the result is
symmetric and mechanistic: each track's fragility sits where its dominant stressor crosses the engine
boundary.

**Scope rule:** the isolation covers every `(track, precision)` cell with a material SR drop. Cells
with no drop need no attribution — `enc-fp16+pred-fp8 = 70.0` demonstrates that case within DINO
itself.

### `entropy` only

The isolation is run at a **single calibration method, `entropy`** (owner decision, 2026-07-21).

- The diagnostic explains a specific headline row, so it must be **method-matched to that row** —
  an `entropy` isolation cannot explain a `max`-calibrated collapse.
- SPEC §Parity requires the method held constant across INT8 and FP8 within a labelled comparison;
  one method keeps the whole 2×2 inside one labelled comparison.
- Each cell is a full CEM eval on the L40S. A second method doubles the pod cost to re-answer the
  same attribution question.

Consequence: the DINO headline rows this table explains must be rendered at `calibration_method=
entropy`, which requires the pure `entropy` INT8/FP8 corners (PLAN Phase 6) — the same runs that
close the 2×2.

## Measured — DINO, `entropy` (2026-07-21)

Against DINO's FP16 baseline of ~70%:

| config | SR | ΔSR | n_cycles |
|---|---|---|---|
| enc-fp16 + pred-fp8 | 70.0 | −0 | 66 |
| enc-fp16 + pred-int8 | 42.0 | −28 | 83 |
| enc-int8 + pred-fp16 | 16.0 | −54 | 93 |
| enc-fp8 + pred-fp16 | 4.0 | −66 | 98 |

**The encoder is the dominant failure locus, but the predictor is not innocent.** FP8 leaves the
predictor untouched (−0 pp) while INT8 costs it 28 pp; the encoder loses 54–66 pp in either format.

**The encoder is worse under FP8 than INT8 (4% vs 16%) — a mechanism signal, not noise.** E4M3 carries
3 mantissa bits against INT8's uniform 8-bit grid. A purely *range*-limited tensor would favour FP8;
this one does not, which points at **resolution of the bulk** rather than dynamic range — consistent
with ADR 0002's "a high-norm patch token pins the per-tensor amax and starves the bulk", and it
explains why `entropy`'s tail-clip did not rescue the encoder (`enc-int8@entropy = 16%` against pure
`int8@max ≈ 20%`; at 50 episodes that gap is two episodes).

`n_cycles` anti-correlates with SR exactly as ADR 0003 derives (`n = 50 + alive_at_25`): 66 at 70% SR
up to 98 at 4%. Independent confirmation of the n model, and a reminder that any latency read across
these rows must be truncated to the common n first.

**LeWM: outstanding.** Until it is measured, "the encoder is DINO's problem and not LeWM's" is an
inference from ADR 0002's mechanism, not a result.

## Consequences

- `src/sr_eval.py` already carries the mechanism (`encoder_precision=` / `predictor_precision=`,
  composite key, additive merge). No new measurement code.
- A new **isolation table** renders from the composite `sr.json` keys, placed immediately after
  `fp32_relative_table` — the answer to the question that table provokes. The headline tables, plots,
  and ratios are unchanged.
- The table carries the isolated component's **per-cycle time share** as diagnostic context. SPEC
  already notes the encoder is "wall-clock-diluted in the cycle" (2 calls against the predictor's
  150), so if the decomposition bears that out the DINO conclusion sharpens to: *the component whose
  quantization destroyed the task contributes least to the latency it was meant to save*. Pending the
  decomposition pod run — stated as a hypothesis until then.
- Per-cycle latencies **are** recorded for every isolation run (`sr.json`
  `per_cycle_latencies_ms`), and the dmon telemetry brackets them, but they are not benchmarked
  component-wise and enter no speed claim.
- **Accepted residual:** the held component sits at FP16, not FP32, so the isolation attributes
  damage relative to the FP16 baseline rather than FP32. That is the right reference — FP16 is
  lossless here (ADR 0002) — but it means the ΔSR quoted in this table is not directly the ΔSR the
  FP32-relative headline table quotes.
