# ADR 0002 — INT8 calibration must match the inference-time distribution

**Status:** Accepted · **Date:** 2026-07-15 · **Supersedes:** the expert-action calibration stream

## Context

Observed on the first INT8 run: LeWM **FP32 94% / FP16 96% / INT8 48%** success rate.

`max` calibration bakes **fixed per-tensor scales** from the largest absolute activation seen
during the calibration pass; anything outside that range **saturates** at inference. FP16 carries
no fixed clip, so a calibration/inference distribution gap is **invisible in FP16 and catastrophic
in INT8** — exactly the observed signature.

**FP32 ≈ FP16 is the expected result, not a finding.** The checkpoints are BF16-trained, and
FP16's 10-bit mantissa exceeds BF16's 7, so FP16 reproduces the trained weights' precision fully
(its only risk is range, and no overflow occurs). The 2pp is eval noise — at `eval.num_eval = 50`
it is literally one episode (47/50 vs 48/50). INT8's 48% is 24/50: a real collapse.

### Which stream is wrong

The **encoder** stream keeps the strided expert clips. At eval the env is stepped by CEM-planned
actions, so the encoder sees planner-visited states rather than expert-demo states — but its input
is a normalized image whose range is bounded by construction, so the input-side `max` scale is safe.
That argument does **not** extend to the encoder's internal activations under off-distribution
states: an accepted **lower-risk residual**, not a non-issue (LeWM's encoder INT8 drift is already
the "borderline" 0.6–1.0).

The **predictor** stream is wrong on two axes:

**1. Actions — established from the solver source, not inferred.** `CEMSolver.solve` (installed swm
0.1.1, `solver/cem.py:191-207`) draws `candidates = randn(...) * var + mean` with **no clamp to the
action space**, from `var_scale = 1.0` and `mean = 0`.

That `mean = 0` is a **conjunction of two independent conditions** — both must hold, neither is
sufficient alone (`prepare_init_action`, `solver/utils.py`):

- (i) the models are **non-`Actionable`** — neither `LeWM` nor `PreJEPA` defines `get_action`, so no
  actor generates a warm start (were one Actionable, it would fill the horizon and `mean ≠ 0`);
- **and** (ii) `horizon == receding_horizon == 5`, so the policy consumes the whole plan and `rest`
  is empty — `warm_start` (default True) never populates `_next_init`, so `init_action=None` reaches
  the solver (were `receding_horizon < horizon`, the carried plan tail would be **kept** and only the
  missing steps zero-padded → `mean = [tail, 0…] ≠ 0`).

Condition (ii) is a **config coincidence, not a property of these models**. Lowering
`receding_horizon` — e.g. to raise the per-cycle sample count — would silently make the real proposal
non-zero-mean while `calibrate._sample_cem_actions` still draws zero-mean, re-opening this exact gap
**with no error**. Both conditions are therefore parity-gated and pinned by `tests/test_sr_shim.py`.

So `predict` is driven by an **unbounded N(0,1)** proposal reaching ≈±4 across 300 samples × horizon,
while expert actions are bounded by `Box(-1, 1)`. Calibrating on expert actions under-estimates the
inference action range by **~4×** and clips most of the proposal's dynamic range. Under Design A
LeWM's `action_encoder` sits **inside** the predict engine, so the raw action tensor and every
action-encoder activation are quantized on that under-scaled range.

The clipped tensors are precisely the candidates CEM is trying to rank, so what breaks is the
**planning signal**, not merely accuracy — which is why SR collapses to near the non-planning floor
rather than degrading gracefully. Variance only shrinks across the 30 CEM iterations
(`var = topk.std`), so **iteration 0 at `var_scale` is the widest and bounds the whole run**.

**2. Latents — autoregressive drift.** `predict` is called autoregressively over the horizon: only
step 0 consumes an encoder latent; steps 1…H−1 consume the predictor's **own predicted latents**,
which drift off the encoder-latent manifold. A single-step encode→predict draw observes none of them,
so the latent-input scale is fit to the encoder range and clips the rollout, compounding down the
horizon.

## Decision

**Reproduce the distribution in the builder; do not harvest a live rollout.**

The predictor calibration stream samples actions from the CEM proposal (`randn * var_scale`, zero
mean, **unclamped** — matching the source) and rolls `predict` autoregressively over the horizon so
it consumes its own predicted latents.

It is deliberately **not** sourced from an actual CEM/eval run: that would make the INT8 scales depend
on the **eval seed and sample draws** (the clip draw is deterministic by design — no RNG) and would
couple the quantization pipeline to the CEM solver + SR shim + eval config, an owner-gated parity
surface.

The roll is driven through the **real rollout** (model-side, already fidelity-gated) rather than a
re-implementation — that is the line: **reuse the rollout, never the solver**. The proposal is
reproduced from a one-line formula read off the source, not a dependency on it.

- DINO drives the real `shim.dino_rollout` via `_CaptureAdapter` (no re-implementation of the 404
  carry); LeWM mirrors `LeWM.rollout`'s window loop (`lewm.py:94-100` — plain windowing, no carry).
- Only `T == HISTORY_SIZE` windows are kept (the engine binds static `HS`; `T < HS` transients reach
  it repeat-padded by `_predict_hist_adapt`, adding no new range).
- CEM's `action_dim` is **already** the 10-wide pack (`cem.py:80` = env 2 × action_block 5), so there
  is **no env→model packing step**.

**Sample count.** Clip coverage stays at the signed-off **512**; the roll emits the `T == HS` windows,
so the predictor stream grows ~3× to ~1536 samples. The roll's action-sequence length is `CEM_HORIZON`
— matching the real `CEMSolver` candidates tensor, which is `horizon`-long, not `n_obs + horizon` —
yielding `(horizon − n_obs) + 1` predict calls and **3** steady-state windows per clip, not 4.
Coverage is the point of the fix, so the count was allowed to grow rather than shrinking the clip draw
to hold 512.

## Consequences

- Measured off-pod: expert bound **1.0** vs proposal max **4.34** (~4×); **32.1%** of action values
  were clipped at the old scale.
- **INT8's calibration health is judged by SR, not by the drift table.** The drift table runs on
  nominal, dataset-drawn inputs and rated LeWM INT8 merely "borderline" (enc_abs ~0.6–1.0) while its
  SR collapsed to 48%. Same class of blind spot as the fixed-`HS` gate: a check drawn from the dataset
  cannot see a failure driven by the *solver's* distribution.
- **Accepted residuals:** (a) calibration rolls the FP32/torch predictor while the INT8 engine drifts
  marginally wider; (b) `max` on a Gaussian grows with draw count, so the calibration max (~4.3σ,
  measured) sits just under a full eval's (~5.5σ) — clipping ~1e-5 of action values against the 32.1%
  above. Both second-order against the ~4× gap closed.
- If matching the distribution still does not recover INT8 SR, the loss is inherent to per-tensor INT8
  on these predictors and the documented **FP16-only fallback** applies.

---

## Amendment (2026-07-19) — calibration method is a labelled build option for both tracks

**Status:** Accepted · extends the decision above to FP8 and generalizes the calibration method to a
reported dimension rather than a fixed per-track setting.

### What the distribution fix did and did not do

The distribution fix above recovered **LeWM** (INT8 48% → ~76%; FP32/FP16 ~98%). It did **not**
recover **DINO-WM**: INT8 ~20% and FP8 **2%**, against a healthy FP32/FP16 baseline of ~70% (DINO is
undertrained, but the collapse is entirely in the *quantized* path — FP16 is lossless).

The fix targeted the **action** distribution — LeWM's dominant stressor, since under Design A LeWM's
`action_encoder` sits *inside* the predict engine, so the raw unbounded CEM action is an engine input
the calibration rescaled. **That stressor does not exist at DINO's engine boundary:** the DINO action
is embedded by `extra_encoders["action"]` and tiled into just 10 of the 404 predictor-input channels
*upstream* of the engine (`adapter.assemble_embedding`, uncompiled). DINO's engine consumes the
assembled 404 embedding, **384/404 of it frozen-DINOv3 patch features** — outlier / high-norm heavy
(the phenomenon register tokens exist to mitigate).

### Why the method matters, and why it is not fixed per track

`max` (→ ORT `CalibrationMethod.MinMax` + `ActivationSymmetric`) sets each per-tensor activation scale
to the largest absolute value observed — **zero outlier rejection**. `entropy` (→ ORT
`CalibrationMethod.Entropy`) picks a KL-optimal threshold that **clips the outlier tail**. In explicit
quantization activations must be per-tensor (TensorRT), so the method is the lever, not granularity.
The ONNX int8/fp8 flow supports exactly `{entropy, max}` (`percentile` silently maps to MinMax, so it
is not offered).

The two tracks pull in opposite directions, and **which method wins is an SR question, not an
assertion**:
- **DINO** — a high-norm patch token pins the per-tensor amax and starves the bulk (where the CEM
  ranking signal lives); `entropy`'s tail-clip should help. Hypothesis, to be measured.
- **LeWM** — the wide values *are* the action signal ADR-0002 deliberately widened; `entropy`'s
  tail-clip could re-clip that range and mildly re-introduce the saturation this ADR removed, so `max`
  may remain best. Equally a hypothesis — LeWM INT8 is still ~22 pts below FP16, so `entropy` might
  instead recover ground. Measured, not assumed.

### Decision — expose both methods for both tracks; label them in reporting

- The calibration method (`max` | `entropy`) is a **build option available to both tracks**, not
  hardcoded per track.
- It is a **labelled dimension of every quantized result**: `max`- and `entropy`-calibrated engines
  coexist as separate, comparable points (track × precision × method), recorded in
  `results.<track>.json`. Cross-track comparisons are drawn **like-for-like** (same method on both
  tracks).
- It is **held constant across a track's INT8 and FP8** within any labelled comparison, so the
  INT8→FP8 step isolates the format. FP8 rides INT8's calibration then converts scales
  (`fp8_scale = int8_scale × 448/127`), so the method carries over unchanged.

### Artefact preservation

**Existing report artefacts and data points are never rewritten.** The already-collected
`max`-calibrated results stay as-is; `entropy` runs are **additive**, written as separately-labelled
points, never overwriting the `max` rows (CLAUDE §8, log-before-delete). If neither method recovers
DINO's 8-bit SR, those rows are reported **degraded vs FP32** (the FP16-only fallback named in the
original Consequences was removed in `352018f`; FP8 is now always built and reported, quoted vs FP32).

### Parity

Calibration method is a per-engine build knob surfaced as a report label, not a hidden per-track
setting. The cross-track **latency** headline is calibration-method-invariant (scale magnitudes do not
change quantized-op coverage, granularity, or TensorRT tactic selection); per-model SR is reported per
(track, precision, method).
