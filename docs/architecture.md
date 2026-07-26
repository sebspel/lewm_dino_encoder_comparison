# Architecture & design rationale

**Why** the owned layer is shaped the way it is. `SPEC.md` states the contract these decisions
must satisfy; `PLAN.md` carries the execution steps; `src/interfaces.py` is the typed contract in
code. This file holds the standing design rationale for the owned export / quantization /
benchmark layer. Measured results and their interpretation live in `RESULTS.md`.

---

## 1. Layering: platform vs owned

`stable-worldmodel` provides training, the Push-T env, the CEM solver, dataset tooling, and
closed-loop MPC evaluation. The contribution starts **downstream of a trained checkpoint**:
export, quantize, benchmark.

DINOv3-WM is the platform's `prejepa` DINO-WM predictor with the reference DINOv2 backbone
swapped for a config-injected frozen DINOv3. The only code addition on the model side is one
encode-path override (`src/dino_patch.py::DINOv3PreJEPA`) that drops CLS + register tokens to
expose the true 196-patch grid. The platform wheel is never edited; the vendored entrypoints
import the override.

**Why not pool DINO's grid to one token.** It would diverge from the DINO-WM paper and erase
part of the architectural asymmetry the study exists to measure. The two tracks therefore
carry different latent ranks by design — LeWM `(B, D)`, DINO-WM `(B, N_patches, D)`.

---

## 2. Why two engines, not one fused graph

`encode` and `predict` are traced and built separately (one ONNX graph / TensorRT engine each).

The CEM rollout encodes the observation **once**, caches the latent, then calls `predict`
autoregressively over the horizon for every candidate. A single fused `obs → latent` graph
could not reproduce that call pattern: it would re-encode on every predictor call, inflating
encoder cost and erasing the encoder-cached / predictor-dominates asymmetry the study measures.

The **CEM planner is never compiled in** — it stays in Python around the engines. Only the
model (encoder + predictor) is exported.

---

## 3. The Python rollout / shim layer

**DINO-WM `predict` is a `404 → 404` reconstruction, not a slice back to 384.** The predicted
proprio must survive: the CEM criterion scores predicted proprio *and* pixels against the goal,
and the autoregressive state carried across the horizon is the full `404`. The extras
embedding, the initial `384 → 404` assembly, and the per-step action-replacement +
proprio-carry live in the Python rollout/shim (`src/shim.py`), not the compiled engine.

Because `predict` *reconstructs* the platform forward rather than calling it, a wrong `404`
assembly, orientation, or dropped proprio channel passes engine precision-match yet silently
corrupts every SR. That is why the adapter-fidelity gate (`src/fidelity.py`) validates the
adapter's `encode` + `predict` + rollout against the platform's own rollout/`get_cost` on the
real checkpoint **before** any engine is built.

**LeWM, Design A: the `action_encoder` lives inside the predict engine.** The exported engine
ingests a raw action. LeWM's action encoder is per-frame — `Conv1d(k=1)` plus a per-position
MLP, no mixing along the macro-step axis — so a per-step `predict` is numerically identical to
the platform's whole-sequence pre-encode, and the per-step engine boundary is faithful. Since
inherited `LeWM.rollout` pre-encodes the whole action sequence, the shim sets its
`action_encoder` to an Identity passthrough; rollout then windows raw actions straight into
`predict` and the engine's own per-frame `action_encoder` does the encode.

This is a **silent-failure boundary**: a temporal (kernel > 1) action-encoder config would make
the per-step boundary wrong with no error. It is guarded by a runtime check on the real
checkpoint that the action encoder is per-frame
(`action_encoder(seq)[:,t] ≈ action_encoder(seq[:,:t+1])[:,-1]`).

---

## 4. Fixed-history engines vs the rollout's growing window

The per-step `predict` engine traces a **fixed** history axis (`HS = predictor.num_frames = 3`)
with only the batch axis dynamic. The platform rollout feeds a window that **grows**
`min(n_obs, HS) → HS` — at `n_obs = 1` the lengths are 1, 2, 3, 3, … So the first steps hand the
fixed-`HS` engine a `T < HS` window it cannot bind (a negative-dim output; it surfaces loudly at
bind time, not as a wrong number).

**Fix:** right-pad the history axis up to `HS`, run the engine, slice the first `T` frames back
(`_predict_hist_adapt`) — the predictor analogue of the encoder's static-hist repeat-pad
(`_hist_adapt`), and the documented TensorRT practice (static sequence axis + dynamic batch).
It keeps the precision-match-gated engine byte-for-byte: no re-export, no re-quantize.

**Why this is exact — a model-specific, mask-free-padding exception.** It holds iff the
predictor is **causal** with **prefix positional embeddings** and the padded (tail) frames'
outputs are discarded, so no real read position ever attends a pad frame. The general case —
right-padding a causal transformer — *does* need an attention mask; this one does not, precisely
because the pad sits after every position we read. Both tracks' predictors are causal with
prefix positional embeddings, so the identical fix applies to LeWM and DINO-WM with no per-track
special-casing. Were a predictor ever *not* causal, the exception would fail and the transient
`T < HS` steps would need a torch fallback or a dynamic-hist re-export.

**Why it needs its own check.** The fixed-`HS` precision-match and SR-cost-parity checks never
exercise `T < HS`, which is why the mismatch passed every one of them yet crashed the SR run. The
boundary is proven by a variable-window (`T ∈ {1, 2}`) engine-vs-torch parity check
(`src/precision_match.py`).

The encoder's repeat-pad is exact for a simpler reason: the encoder is temporally independent
(per-frame), so padding the frame axis cannot change any frame's output.

---

## 5. SR-per-precision: the `get_cost`-only shim

The CEM solver calls the world model via **`get_cost`**, not `encode`/`predict`. To produce the
SR that pairs with each precision's speed number, the exported adapter is re-wrapped in a thin
**Python** shim exposing `get_cost` only (calling the engine's `encode`/`predict` underneath)
and slotted into the solver, letting the platform's eval logic re-run unchanged.

Both shims subclass the platform model and override the narrowest possible seam, so cost parity
holds by construction:

- `DINOWMSRShim` subclasses `DINOv3PreJEPA` and overrides **only** `_encode_image` and
  `predict`; `get_cost` / `rollout` / `criterion` / `split_embedding` / goal-encode are
  inherited byte-unchanged.
- `LeWMSRShim` subclasses `LeWM` and routes both `encode` and `predict` through injected engine
  callables, so the SR reflects the same quantized engines the benchmark times. `encode` has no
  `_encode_image` seam (`LeWM.encode` fuses backbone + info-dict bookkeeping + the
  `action_encoder` branch), so the override re-implements its body.

Parity reference is the **installed swm 0.1.1** `PreJEPA.get_cost`. Cost parity is checked at
`B = 1`: the vendored CEM pins `batch_size = 1` and `LeWM.criterion` supports one env per solve
(it broadcasts the single-env goal over candidates and errors for `B > 1`).

### `get_action` must stay absent

Two unrelated platform methods share the name: a **policy**-side `get_action(info_dict, **kwargs)`
(`WorldModelPolicy` — the replanner, not a model method at all) and the **model**-side
`Actionable.get_action(info, horizon, prefix_actions)` (only `TDMPC2` / `GCRL` define it; `LeWM`
and `PreJEPA` do not). `CEMSolver` touches the latter only via `prepare_init_action`'s
`isinstance(model, Actionable)` branch, which our non-`Actionable` tracks never take — that is
exactly what makes the warm start a zero-pad (`mean = 0`), load-bearing for both eval parity and
the INT8 calibration distribution (§7).

Because `Actionable` is `@runtime_checkable`, `isinstance` matches on **method presence, not
signature**: adding *any* `get_action` to the shim — even a policy-shaped one — silently flips it
to `Actionable` and replaces the zero-pad with a generated warm start.

### Injection seam — the one exception to the no-monkeypatch stance

`eval_wm.run` has no config seam for a model object. The SR re-run therefore uses a dedicated
driver (`src/sr_eval.py`) that slots the shim in by a **scoped patch of the checkpoint loader**
around the run, swapping only which model the loader returns. The vendored eval entrypoint and
the solver/CEM logic stay byte-unmodified, and no CEM config, seed, sample count, or plan
changes, so eval/CEM parity is preserved — the SR differs from the FP32 baseline only by the
engines' quantization drift.

Alternatives rejected: a latency-only pass at `num_envs = 1` (breaks the Phase-3 eval parity and
the same-solves SR pairing); editing the vendored entrypoint (loses byte-parity).

---

## 6. The export & quantization stack

The export/quantization stack (TensorRT + NVIDIA TensorRT Model Optimizer + its ONNX Runtime
dependency) must stay binary-compatible with the pod's CUDA 12.4 driver and the uv-locked
`cu124` torch. Several of these dependencies now default to CUDA-13 wheels, which pull
`nvidia-*-cu13` and fail to initialize against a 12.x driver.

- **TensorRT and the Model Optimizer are installed by `setup.sh`, out of uv** (cu12,
  CUDA-12.4-matched), so uv cannot pull a conflicting `libnvinfer`/CUDA stack.
- **`onnxruntime-gpu` is installed from onnxruntime's dedicated CUDA-12 feed**, before modelopt,
  so modelopt's unbounded dependency cannot re-resolve it to the cu13 PyPI default. This CUDA-12
  build is also what lets the calibration pass run on the GPU. Confirmed torch-2.6-compatible
  pins: `modelopt==0.43.0`, `onnxruntime-gpu==1.24.4`.

**Explicit Q/DQ, not build-time calibration.** FP32/FP16 build data-free from the base ONNX
(FP16 is a build flag). INT8 and FP8 are each a **separately quantized graph**: the Model
Optimizer inserts Q/DQ nodes and bakes per-tensor scales from a calibration pass, and TensorRT
honors the embedded Q/DQ instead of calibrating at build time. INT8 is integer; FP8 is E4M3,
built by the same path with `quantize_mode="fp8"` and a `BuilderFlag.FP8`+FP16 build (heavy
layers 8-bit, remainder FP16). If the FP16 cast of the non-quantized remainder ever drifts
unacceptably, the remainder can be kept FP32 via modelopt's `high_precision_dtype`.

**Calibration execution-provider (EP) split.** The encoder calibrates on the **GPU (CUDA EP)**;
the predictor calibrates on the **CPU EP**. The `onnxruntime-gpu` CUDA EP miscomputes the
predictor's dynamic-batch reshape (`Squeeze(Shape(latent))` → head-split `Reshape`), fabricating
a wrong target and crashing modelopt's MHA-exclusion probe; the CPU EP (and native TensorRT)
computes it correctly. The encoder graph lacks that pattern, so it keeps the faster CUDA EP. **EP
choice affects calibration speed only, not the derived per-tensor scales** — plumbing, not a
result-affecting decision.

---

## 7. Calibration: matching the inference-time distribution

`max` calibration bakes **fixed per-tensor scales** from the largest absolute activation seen
during the calibration pass; anything outside that range **saturates** at inference. FP16 carries
no fixed clip, so a calibration/inference distribution gap is **invisible in FP16 and
catastrophic in INT8**. The calibration set must therefore match the *inference-time* input
distribution, not the expert-demo one.

The **predictor** stream is wrong on two axes if calibrated from expert data:

1. **Actions.** `CEMSolver.solve` draws `candidates = randn(...) * var_scale + mean` with **no
   clamp** to the action space, from `var_scale = 1.0` and `mean = 0`. The `mean = 0` holds
   because our models are **non-`Actionable`** (no actor fills a warm start) **and**
   `horizon == receding_horizon`, so no carried plan tail survives — both conditions matter (§5).
   So `predict` is driven by an **unbounded ≈N(0,1)** proposal reaching ≈±4, while expert actions
   are bounded to `Box(-1, 1)`; calibrating on expert actions under-scales the action range ~4×
   and clips most of the proposal. Under Design A LeWM's `action_encoder` sits **inside** the
   predict engine, so the raw action tensor and its activations quantize on that under-scaled
   range. The clipped tensors are exactly the candidates CEM is trying to rank, so what breaks is
   the **planning signal** — SR collapses toward the non-planning floor rather than degrading
   gracefully.
2. **Latents.** `predict` runs autoregressively: only step 0 consumes an encoder latent; steps
   1…H−1 consume the predictor's **own predicted latents**, which drift off the encoder-latent
   manifold. A single-step encode→predict draw observes none of them.

**Decision: reproduce the distribution in the builder; do not harvest a live rollout.** The
predictor calibration stream samples actions from the CEM proposal (`randn * var_scale`, zero
mean, unclamped — the formula read off the solver source, not a dependency on it) and rolls
`predict` autoregressively so it consumes its own predicted latents. Sourcing it from an actual
CEM/eval run would make the scales depend on the eval seed and sample draws and would couple the
quantization pipeline to the CEM solver + SR shim + eval config. The roll is driven through the
**real rollout** (already fidelity-gated), never a re-implementation of the solver: DINO drives
`shim.dino_rollout` via a capture proxy; LeWM mirrors `LeWM.rollout`'s window loop. Only the
`T == HISTORY_SIZE` windows are kept, since the engine binds a static `HS`.

### Calibration method is a labelled build option, not a per-track setting

`max` (ORT `MinMax`) sets each per-tensor scale to the largest absolute value observed — **zero
outlier rejection**. `entropy` (ORT `Entropy`) picks a KL-optimal threshold that **clips the
outlier tail**. In explicit quantization activations must be per-tensor (TensorRT), so the method
is the lever, not granularity. The two tracks pull in opposite directions — DINO's frozen-DINOv3
patch features are outlier-heavy (a high-norm token pins the per-tensor amax and starves the
bulk, so `entropy`'s tail-clip should help), while LeWM's wide values *are* the action signal the
distribution fix widened (so `entropy` could re-clip it). Which method wins is measured per
`(track, precision)`. The method is:

- a **build option available to both tracks**, not hardcoded per track;
- a **labelled dimension of every quantized result** (`track × precision × method`), with `max`-
  and `entropy`-calibrated engines and results coexisting as separate, comparable points, and
  engine plans method-tagged on disk (`{encoder,predictor}.<precision>.<method>.plan`);
- **held constant across a track's INT8 and FP8** within any labelled comparison, so the INT8→FP8
  step isolates the format. FP8 rides INT8's calibration then converts scales
  (`fp8_scale = int8_scale × 448/127`).

The cross-track **latency** headline is calibration-method-invariant (scale magnitudes do not
change quantized-op coverage, granularity, or TensorRT tactic selection); per-model SR is
reported per `(track, precision, method)`.

### Neutralizing the attention-mask sentinel

DINO-WM's predictor attention builds an **explicit additive causal mask** —
`masked_fill(mask, float("-inf"))` fed to `scaled_dot_product_attention`. On ONNX export that
`-inf` is materialized as the **finite** constant `finfo(float32).min` (≈ −3.4e38), so the
mask-add tensors carry −3.4e38 as a real activation value. modelopt calibrates them, and
`entropy` overflows FP32 when laying out histogram bin edges (`range ≈ 6.8e38`), while `max`
bakes a garbage −3.4e38 scale. LeWM exports `is_causal=True` (no materialized mask tensor), so
this is DINO-only.

Before the PTQ pass, `export._neutralize_attention_mask_sentinel` rewrites every
`finfo(float32).min` mask-fill constant (magnitude ≥ `1e30`) to a mild finite fill
`_MASK_FILL = −3e4`. This is **numerically equivalent masking** — `softmax(−3e4 + logit)`
underflows to 0 in FP16 and FP32 exactly as `−inf` does — but the histogram range collapses to
±3e4, inside the FP32 edge and the FP16 range. Because the mask is added **after** the QK^T score
MatMul, the real score/probability tensors are unpolluted and the attention MatMuls quantize
normally. The pass edits only tensors carrying the sentinel, so it is a **no-op on LeWM** and
both tracks stay fully quantized. It runs for INT8 and FP8, `max` and `entropy`.

---

## 8. Measurement design

### What a "cycle" is — bracket per env, not per solve

A **cycle is one episode's decision**, which is *not* the span of one `CEMSolver.solve` call. A
solve plans every still-alive episode back-to-back: `batch_size = 1` (pinned; `LeWM.criterion`
errors for `B > 1`) makes the env loop sequential, and `EnvPool.step` is a Python for-loop over
in-process envs — a vectorized *interface*, not concurrency. So one solve = **N independent
decisions computed back-to-back, and its wall clock is their sum**. The only genuinely batched
axis is the `num_samples` candidate fan-out within one episode's `get_cost`.

The latency callback therefore brackets **per env** — consecutive `start_batch` hooks, the last
closing at `end_solve` — one record per decision, with a sync per span. Bracketing
`reset → end_solve` instead (both hooks sit outside the env loop) would time **all N** episodes
per record while `ENCODER/PREDICTOR_CALLS_PER_CYCLE` count **one**; weighting components by
per-decision counts across that gap inflates `overhead_ms` toward the whole cycle, and it would
make the headline scale with how many episodes are still alive — SR-dependent, therefore
track-dependent: a parity break. A `current_bs == 1` guard fails loud if `batch_size` ever ≠ 1.

### Three statistics, three jobs

- **p50 — the comparison basis.** The LeWM-vs-DINOv3 headline ratio and the FP32-relative speedup
  are quoted at p50. The headline is a *mechanistic* central-tendency claim (architectural
  asymmetry), and per-cycle n is 50–100 — p50 is the statistic that sample supports.
- **p95 — the reported tail, never a comparison basis.** Kept for all three distributions as a
  descriptive figure; at n = 50–100 it is roughly the 3rd-to-10th largest sample and carries no
  claim.
- **mean — the decomposition basis only, never a reported headline.**

**Why the decomposition must use means.** `cycle = enc·calls + pred·calls + overhead` is exact
for means by **linearity of expectation** — for any distribution and any correlation between the
terms — whereas `p50(a + b) ≠ p50(a) + p50(b)`. A percentile decomposition would book its own
non-additivity error as "planner overhead", inflating the Amdahl denominator. Amdahl's law is
itself an expectation model, so the optimizable fraction `p` and the ceiling are mean-derived,
and the *realized* speedup they reconcile against is mean-based to match. This is the one place a
mean is used.

### Why per-cycle n is 50–100 — and equal-n truncation

`eval_budget = 50` policy steps, and one plan fills the action buffer with
`receding_horizon × action_block = 25` env actions, so the solver is called on exactly **two**
steps (t=0 and t=25). Decisions total `50 + alive_at_25` ≤ 100. Because Push-T terminates on
success, **a higher-SR track contributes fewer decisions** — n is SR-dependent hence
track-dependent. The equal-n truncation neutralises this by taking the common minimum n across
tracks; the n each percentile was computed from is reported on the speed table, so the equal-n
comparison is verifiable off the artefact rather than asserted.

### Per-cycle warm-up drop

The engine-step loops drop `warmup` iters; the per-cycle callback records from the **first
decision of the first solve** (the vendored eval's warm-up pass is gated on `compile`, which is
`false`), so cold-start cost — first `execute_v2`, kernel autotune, allocator growth, clock ramp
— sits on the cycle side of `overhead = cycle − enc − pred` only. Booked as planner overhead, it
deflates `p` and the Amdahl ceiling, i.e. makes quantization look **less** useful than it is.

The fix drops a warm-up head of `k = 1` decision (`PER_CYCLE_WARMUP_DROP`), **at report time**
(the recorded vector stays complete) and **before** the equal-n truncation (which otherwise
preserves the temporal-head cold sample). The dropped value is disclosed as the speed table's
`drop×`, so the exclusion is visible rather than tuned silently. The p50 headline is robust to
one head sample and does not move — this corrects the mean-based decomposition and dilution
tables, not the comparison.

### Peak memory is sampled from the driver, not the torch allocator

TensorRT's engine and execution-context device allocations bypass torch's caching allocator, so
`torch.cuda.max_memory_allocated` would systematically undercount **exactly the optimized path** —
the one the study is trying to measure. Peak memory is sampled via `cudaMemGetInfo` /
`torch.cuda.mem_get_info` (device-level used).

### GPU clocks are not locked — the per-run clock/thermal state is recorded

GPU clocks are not locked. A passive `nvidia-smi dmon` observer (`src/gpu_clocks.py`) logs
per-sample telemetry (SM/mem clock, power, temperature, utilization, memory) alongside every timed
run — both the isolated component loops and the per-cycle eval-shim run — so the actual per-run
clock/thermal state is recorded rather than assumed. It is a separate subprocess and never touches
seeds, samples, or the plan.

### Why the decomposition subtracts rather than mirrors the solver

The full planning-cycle time is **measured on the real CEM solve**, not reconstructed from a
hand-rolled solver mirror. Encoder and predictor are timed in isolation and weighted by their
real per-cycle call counts; the remainder is `overhead_ms = cycle − encoder − predictor` — the
un-optimizable floor (CEM sampling/topk/mean-var, the criterion, the 384→404 assembly, per-step
action-replace/proprio-carry, and host/Python glue). A **negative `overhead_ms` is surfaced
loudly**, never clamped: it signals the call-count weighting or the isolated measurement is off.

**Call counts are confirmed against the installed `CEMSolver.solve`, not assumed.** Tracing
`solve → get_cost → rollout` in swm 0.1.1 shows the `candidates` tensor is `horizon`-long only,
**not** `n_obs + horizon`, so the rollout drives `(horizon − n_obs) + 1` predict calls per
rollout. With 30 CEM iterations that is
`PREDICTOR_CALLS_PER_CYCLE = ((5 − 1) + 1) × 30 = 150`; `ENCODER_CALLS_PER_CYCLE = 2` (goal +
initial-obs encode). Both counts are **per-decision**, so the measured cycle must be per-decision
too.

---

## 9. Component-precision isolation

The headline tables report **that** SR fell, not **which component's** quantization caused it:
`speed_table` rows are `(track, precision)`, and both engines move together. The encoder and the
predictor are separately exported, separately quantized, and separately timed — so the
attribution is measurable, and without it the study reports a symptom.

**Re-run the SR eval with ONE component quantized and the other held at FP16**, per
`(track, precision)` cell that shows a material SR drop. Two runs per cell isolate the two sides;
together with the FP16 baseline and the pure-quantized run they close a 2×2:

| encoder | predictor | run |
|---|---|---|
| fp16 | fp16 | the FP16 baseline (already measured) |
| 8-bit | fp16 | isolates the encoder |
| fp16 | 8-bit | isolates the predictor |
| 8-bit | 8-bit | the pure-precision run (already measured) |

The fourth corner is the pure INT8/FP8 point, so the 2×2 costs two extra evals per cell, and its
closure is a consistency check: if the components' damage composes, the pure corner is
predictable from the two isolated ones; if not, they interact and the attribution carries a
caveat.

**Diagnostic, not a shipped configuration.** A mixed pairing is not a fifth precision in the
FP32→FP16→INT8→FP8 sweep: it is never benchmarked for latency, never entered in the headline
ratio, and never quoted as a recommended configuration. Results land under a composite precision
label `enc-<A>+pred-<B>` that cannot collide with a pure precision, so they are written beside the
pure points and never overwrite them; the headline precision set stays closed, so composite keys
reach no headline table, plot, or ratio.

**Held at FP16, not FP32.** The isolation attributes damage relative to the FP16 baseline (FP16
is lossless here), so its ΔSR is not directly the ΔSR the FP32-relative headline table quotes.

**Single method (`entropy`), both tracks.** The diagnostic explains a specific headline row, so
it must be method-matched to it; one method keeps the whole 2×2 inside one labelled comparison;
and each cell is a full CEM eval, so a second method would double the cost to re-answer the same
question. It covers both tracks — a one-track isolation would argue the other track's innocence
from absence of evidence.

---

## 10. Amdahl dilution disclosure

Only encoder + predictor are quantized; the Python overhead (CEM planner + criterion + assembly +
glue) is precision-invariant. So the per-precision **wall-clock** delta is capped by the model's
share of the cycle, and reporting per-component *relative* speedup alone would hide that dilution.

The study reports, per model: the FP32 baseline per-component time shares and the derived
**optimizable fraction** `p = (encoder + predictor) / cycle`, which sets the Amdahl ceiling
`1/(1−p)`; and per precision, **both** the *model-only* speedup (overhead treated as free) and
the *realized* speedup (the measured FP32-vs-precision per-cycle ratio), whose gap is the overhead
floor and should match `1/((1−p) + p/s)`.

That the optimizable fraction is itself model-dependent — LeWM's single token is
overhead/launch-latency-bound, DINO's 196-token grid is model-bound — is what explains why the
same precision helps the two tracks differently. That is a **result, not bookkeeping**.

This whole block is **mean**-based, `p` included, so the realized speedup it reconciles against is
the mean per-cycle ratio, matching the prediction's basis. It is therefore a *different number*
from the reported p50 FP32-relative speedup, which answers the comparison question rather than the
reconciliation one. The two are rendered in separate tables and are never conflated or averaged.

---

## 11. Clock-state confound: why it is bounded rather than eliminated

GPU clocks cannot be locked on the benchmark platform — `nvidia-smi -lgc` is denied by the RunPod
virtualization layer, confirmed as root with persistence mode on. The telemetry observer
(`src/gpu_clocks.py`) therefore *records* the per-run clock state; §Parity's ask is to bound the
resulting confound, not to assume it away.

**Why it does not cancel.** A common-mode slowdown would divide out of a cross-model ratio. This
throttle is **one-sided**: the heavier track runs at 100% SM utilization against the board power
limit and trades clock for it (median SM clock 2160–2400 MHz), while the lighter track never
approaches the limit and holds the 2520 MHz boost ceiling. Because the throttled track is the
*slower* one, the confound **inflates** the measured ratio — the measured value is the pessimistic
end for DINO, which is why the bound is one-sided too (`R′ ≤ R`).

**The owner-set construction** (2026-07-25; the constants live in `src/interfaces.py`):

- **`T_ref = T × f_measured/f_ref`.** A `1/f_sm` rescaling, applied to **all** measured latency —
  per-cycle, encode-step, predict-step. It knowingly over-corrects, since memory-bound and
  host/Python time do not scale with SM clock. That is deliberate: it makes the derived number the
  **maximum plausible correction**, so the measured and derived values bracket the truth instead of
  competing as two point estimates.
- **`f_ref = 2520` MHz**, the boost ceiling the lighter track actually held. `f_ref` cancels in every
  ratio — `R′ = R × f_dino/f_lewm`, and the within-model speedups likewise — so it sets only the
  absolute derived ms, which then read as "as if unthrottled".
- **`f_measured` = the util-conditioned median SM clock** (samples at SM util ≥ 50%). `dmon` samples
  at ~1 Hz, so a short run's log is mostly idle/ramp samples and an unconditioned median can land at
  a clock nothing ran at (one log medians to 1260 MHz at 7% util). A run with fewer than 3 busy
  samples is **unmeasured** and gets no derived value; asserting a clock from an idle sample is the
  silent wrong-number failure the gate exists to prevent.

**What the three surfaces show.** The cross-model ratio moves by a low-tens-of-percent on a ratio of
two-to-three orders of magnitude, so the architectural asymmetry survives the whole bracket. The
within-model precision deltas can move in **either** direction (a speedup shrinks when the FP32
baseline was the more throttled run), so that is a caveat on the *spread*, not a signed bias. The
**overhead decomposition is the surface the confound actually damages**: `overhead = cycle − enc −
pred` differences terms measured on two different runs at two different clocks, and the term is only
resolvable where its share of the cycle `(1−p)` exceeds the clock mismatch `Δf/f_cmp`. For DINO
`1−p ≈ 0.01–0.03` against a mismatch of `0.04–0.07`, so the derived overhead flips negative — the
honest reading is that DINO's absolute overhead floor is *not resolvable* on unlocked clocks, only
bounded as small. That negative value is surfaced, never clamped.

**Why derived numbers stay additive.** Measured is canonical and remains the headline (owner
framing). The derived artifacts are separately named `*_normalized.derived.<method>.txt` +
`derived_clocks.json`, and `src/clock_norm.py` never writes `results.*.json`, `sr.json`, or a
measured table — the same read-only discipline as the `src.report` re-render (CLAUDE §8). A
clock-locked re-run remains the correct fix and is recorded as deferred future work.

---

## 12. Rejected & retired approaches

- **Standalone profiler + its CEM-iteration mirror — retired.** The PyTorch per-call timing could
  not reconcile with the engine-context cycle. The decomposition moved into
  `src/report.py::decompose`, derived from the benchmark's isolated engine-step latencies.
- **Fixed-wall-clock rollout-count run — removed.** Serial planning makes
  rollouts/sec ≈ 1/per-cycle-latency, so it is redundant with the equal-n latency measurement.
