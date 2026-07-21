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

---

## Amendment (2026-07-19) — DINO `entropy` calibration crash: neutralize the attention mask sentinel

**Status:** Accepted · unblocks DINO `entropy` (and hardens `max`) without breaking cross-track parity.

### Symptom

Building the DINO INT8 `entropy` engine crashed in modelopt's ONNX PTQ:
`ValueError: Too many bins for data range. Cannot create 128 finite-sized bins.`, raised from
`np.histogram(..., range=(-3.4028235e+38, 3.4028235e+38))` in modelopt's `_collect_value`. It hit the
**predictor**, not the encoder, and DINO, not LeWM.

### Root cause (traced, not inferred)

DINO-WM's predictor attention (`stable_pretraining` ViT `_prepare_attn_mask`) builds an **explicit
additive causal mask** — `masked_fill(mask, float("-inf"))` fed to `scaled_dot_product_attention` as
`attn_mask`. On ONNX export that `-inf` is materialized as the **finite** constant
`finfo(float32).min` (≈ −3.4028235e+38). The mask-add tensors (the 588×588 masks and the
8×16×588×588 masked logits) therefore carry −3.4e38 as a real activation value. modelopt calibrates
those tensors, and:

- **`entropy`** — the histogram range becomes `2·threshold ≈ 6.8e38`, which **overflows FP32** when
  `np.histogram` lays out the bin edges → the `ValueError` above.
- **`max`** — no crash, but the per-tensor amax is −3.4e38, so the scale is garbage. This is a
  contributor to DINO INT8-`max`'s collapse, silent where `entropy` is loud.

**LeWM exports `is_causal=True`** (no materialized mask tensor), so neither pathology occurs — this is
DINO-only.

Ruled out along the way: it is **not** the unbounded CEM action stressor (that is the ADR-0002 body
fix, LeWM-side), **not** FP16 overflow, **not** a NaN. The FP32 unfused graph runs clean; the value is
a finite constant, not a computed non-finite. modelopt's `disable_mha_qdq` flag does **not** fix it:
its MHA-partition detector does not recognize DINO's decomposed-SDPA-with-RoPE attention, so it
excludes nothing (verified — the same 12 sentinel tensors still calibrate and it still crashes).

### Options weighed

- **A — explicit `nodes_to_exclude`** on the attention nodes: works, but leaves DINO's attention
  MatMuls in FP16 while LeWM's quantize → **parity asymmetry + a capped speedup**. Rejected.
- **C — accept it** (report DINO 8-bit degraded / unavailable, the FP16-only fallback): rejected —
  loses the DINO 8-bit rows the study is meant to report.
- **B — neutralize the mask sentinel in the exported graph (CHOSEN).**

### Decision — option B

Before the PTQ calibration pass, rewrite every `finfo(float32).min` (≥ `1e30` in magnitude) mask-fill
constant in the exported predictor graph to a mild finite fill `_MASK_FILL = −3e4`. This is
**numerically equivalent masking** — `softmax(−3e4 + logit)` underflows to 0 in FP16 and FP32 exactly
as `−inf`/`−3.4e38` does — but the histogram range collapses to ±3e4, well inside the FP32 edge and
the FP16 range (<65504), so `entropy` builds and `max` gets a sane scale. Because the mask is added
**after** the QK^T score MatMul, the real score/probability tensors are unpolluted and the attention
MatMuls quantize normally.

- Implemented as `export._neutralize_attention_mask_sentinel`, called at the top of `quantize_onnx`
  (so it runs for INT8 **and** FP8, both `max` and `entropy`). FP32/FP16 engines build from the base
  graph untouched — their owner-signed-off drift tables do not move.
- **Self-targeting ⇒ parity-preserving.** The pass edits only tensors carrying the sentinel, so it is
  a **no-op on LeWM** and both tracks stay **fully quantized** (unlike option A). It applies to both
  tracks unconditionally; only DINO's graph contains the sentinel.

### Consequences

- DINO `entropy` (and `max`) calibration completes; DINO 8-bit SR is now measurable rather than a
  build crash — the SPEC "per-tensor 8-bit loss inherent" verdict is now decided by SR, not aborted
  by an overflow.
- Owner-only ONNX/PTQ graph edit — sits behind the OWNER-ONLY boundary (SPEC §Implementation
  Boundaries); owner-approved 2026-07-19.
- The masked-logits tensors keep a −3e4 outlier, so if modelopt ever placed a Q/DQ on the softmax
  **input**, that scale would be coarse — but softmax I/O is not quantized (only the surrounding
  MatMuls are), so this is inert. Judged, like all quant health here, by **SR**.

---

## Amendment (2026-07-21) — the method label must survive into the persisted artefact

**Status:** Accepted · closes the gap between the labelling decision above and what the rendered
report actually carries.

### Symptom

The data layer honours the labelling decision: `sr.json` is keyed per `(track, precision, method)`
and int8/fp8 engine plans are method-tagged. The **render layer discards it**. All four table
renderers take only `bench`, so the method reaches no table header, no row label, and no filename;
it is printed to stdout and logged as a separate W&B key, neither of which is part of the persisted
artefact. Two consequences:

- `speed_table.txt` shows an `int8` row whose SR is method-dependent, with no way to tell which
  method produced it.
- Re-rendering the other method **overwrites the first in place** (fixed filenames, `out_dir`
  defaulting to the source dir) — the artefact-preservation rule above, broken at the last step.

Note this is not confined to the SR column: `_join_eval` pulls `success_rate` **and**
`per_cycle_latencies_ms` from the selected method's entry, so the per-cycle percentiles, and
everything derived from them (`p`, ceiling, overhead, realized speedup), are method-*sourced* too.
Kernel latency is method-invariant in principle; the measured sample is not, because n is SR-driven
and SR is method-driven.

### Decision

- **The four headline tables stay single-method**, and say so: written as `<name>.<method>.txt` with
  a `calibration_method = <m>` line in the table body. Nothing clobbers; nothing is unlabelled.
- **A fifth table, `calibration_table.txt`, carries the method comparison** — int8/fp8 only, both
  methods side by side (`SR@max`, `SR@entropy`, `Δ(entropy−max)`), plus a `headline` column naming
  which method the single-method tables were rendered at. That column is the link between the two
  artefacts and removes the ambiguity entirely.
- It reads `sr.json`'s existing per-`(track, precision, method)` keys, so **no schema change** to the
  canonical `results.<track>.json`.

### Rejected alternative — two rows per quantized precision in the headline tables

Considered and rejected: rendering `int8@max` and `int8@entropy` as sibling rows in all four tables.

- Only SR (and ΔSR) is genuinely method-dependent. The component and dilution tables derive entirely
  from step means and the cycle, so both rows would print near-identical numbers — inviting a reader
  to read run-to-run jitter as a calibration effect, against a SPEC claim that the latency headline
  *is* method-invariant.
- It removes a guard that currently works: `method` is global to a render, so one table can never mix
  `dino int8@max` with `lewm int8@entropy`. The two-row layout puts precisely that forbidden
  cross-track comparison on adjacent lines.
- It forks the keying of every cross-track consumer — `per_cycle_ratio(bench, "int8")` and the
  speed-vs-SR scatter stop being well-defined with two methods under one precision.
- It forces a method axis into `results.<track>.json`, i.e. a schema change to the durable artefact,
  where the split approach needs none.

### Consequences

- Existing `max`-labelled `.txt`/`.png` artefacts are superseded by method-scoped filenames on the
  next render; the underlying `results.<track>.json` and `sr.json` are untouched, so nothing measured
  is lost (CLAUDE §8).
- `results.<track>.json`'s `meta.calibration_method` remains a **whole-file, last-write-wins**
  provenance label while `bench` merges precisions across runs, so it can mislabel a file that mixes
  methods. Latency being method-invariant makes this provenance-only, not a wrong number — recorded
  here as a known residual rather than fixed, since the per-point truth lives in `sr.json`.
