# Architecture & design rationale

**Why** the owned code is shaped the way it is. `SPEC.md` states the contract these decisions must
satisfy; `README.md` carries how to run it; `src/interfaces.py` is the typed contract in code.

This file is deliberately restricted to what a reader could **not** derive from the code itself:
silent-failure traps, platform quirks, and places a stock library does the wrong thing. Measured
numbers do not appear here.

---

## 1. Layering: platform vs owned

`stable-worldmodel` provides training, the Push-T env, the CEM solver, dataset tooling, and
closed-loop MPC evaluation. The code here starts **downstream of a trained checkpoint**: export,
quantize, benchmark.

DINOv3-WM is the platform's `prejepa` DINO-WM predictor with the reference DINOv2 backbone swapped
for a config-injected frozen DINOv3. The only code addition on the model side is one encode-path
override (`src/dino_patch.py::DINOv3PreJEPA`) that drops CLS + register tokens to expose the true
196-patch grid. The platform wheel is never edited; the vendored entrypoints import the override.

`encode` and `predict` are traced and built as **separate** engines because the CEM rollout encodes
once, caches the latent, then calls `predict` autoregressively for every candidate. A fused
`obs → latent` graph could not reproduce that call pattern: it would re-encode on every predictor
call, inflating encoder cost and erasing the asymmetry the study measures.

---

## 2. The Python rollout / shim layer

**DINO-WM `predict` is a `404 → 404` reconstruction, not a slice back to 384.** The predicted proprio
must survive: the CEM criterion scores predicted proprio *and* pixels against the goal, and the
autoregressive state carried across the horizon is the full `404`. The extras embedding, the initial
`384 → 404` assembly, and the per-step action-replacement + proprio-carry live in the Python
rollout/shim (`src/shim.py`), not the compiled engine.

Because `predict` *reconstructs* the platform forward rather than calling it, a wrong `404` assembly,
orientation, or dropped proprio channel passes engine precision-match yet silently corrupts every SR.
That is why the adapter-fidelity gate (`src/fidelity.py`) validates the adapter's `encode` +
`predict` + rollout against the platform's own rollout/`get_cost` on the real checkpoint **before**
any engine is built.

**LeWM: the `action_encoder` lives inside the predict engine.** The exported engine ingests a raw
action. LeWM's action encoder is per-frame — `Conv1d(k=1)` plus a per-position MLP, no mixing along
the macro-step axis — so a per-step `predict` is numerically identical to the platform's
whole-sequence pre-encode, and the per-step engine boundary is faithful. Since inherited
`LeWM.rollout` pre-encodes the whole action sequence, the shim sets its `action_encoder` to an
Identity passthrough; rollout then windows raw actions straight into `predict` and the engine's own
per-frame `action_encoder` does the encode.

This is a **silent-failure boundary**: a temporal (kernel > 1) action-encoder config would make the
per-step boundary wrong with no error. It is guarded by a runtime check on the real checkpoint that
the action encoder is per-frame (`action_encoder(seq)[:,t] ≈ action_encoder(seq[:,:t+1])[:,-1]`).

---

## 3. Fixed-history engines vs the rollout's growing window

The per-step `predict` engine traces a **fixed** history axis (`HS = predictor.num_frames = 3`) with
only the batch axis dynamic. The platform rollout feeds a window that **grows** `min(n_obs, HS) → HS`
— at `n_obs = 1` the lengths are 1, 2, 3, 3, … So the first steps hand the fixed-`HS` engine a
`T < HS` window it cannot bind (a negative-dim output; it surfaces loudly at bind time, not as a
wrong number).

**Fix:** right-pad the history axis up to `HS`, run the engine, slice the first `T` frames back
(`_predict_hist_adapt`) — the predictor analogue of the encoder's static-hist repeat-pad
(`_hist_adapt`), and the documented TensorRT practice (static sequence axis + dynamic batch). It
keeps the precision-match-gated engine byte-for-byte: no re-export, no re-quantize.

**Why this is exact — a model-specific, mask-free-padding exception.** It holds iff the predictor is
**causal** with **prefix positional embeddings** and the padded (tail) frames' outputs are discarded,
so no real read position ever attends a pad frame. The general case — right-padding a causal
transformer — *does* need an attention mask; this one does not, precisely because the pad sits after
every position we read. Both tracks' predictors are causal with prefix positional embeddings, so the
identical fix applies to LeWM and DINO-WM with no per-track special-casing. Were a predictor ever
*not* causal, the exception would fail and the transient `T < HS` steps would need a torch fallback
or a dynamic-hist re-export.

**Why it needs its own check.** The fixed-`HS` precision-match and SR-cost-parity checks never
exercise `T < HS`, which is why the mismatch passed every one of them yet crashed the SR run. The
boundary is proven by a variable-window (`T ∈ {1, 2}`) engine-vs-torch parity check
(`src/precision_match.py`).

The encoder's repeat-pad is exact for a simpler reason: the encoder is temporally independent
(per-frame), so padding the frame axis cannot change any frame's output.

---

## 4. SR-per-precision: the `get_cost`-only shim

The CEM solver calls the world model via **`get_cost`**, not `encode`/`predict`. To produce the SR
that pairs with each precision's speed number, the exported adapter is re-wrapped in a thin **Python**
shim exposing `get_cost` only (calling the engine's `encode`/`predict` underneath) and slotted into
the solver, letting the platform's eval logic re-run unchanged.

Both shims subclass the platform model and override the narrowest possible seam, so cost parity holds
by construction:

- `DINOWMSRShim` subclasses `DINOv3PreJEPA` and overrides **only** `_encode_image` and `predict`;
  `get_cost` / `rollout` / `criterion` / `split_embedding` / goal-encode are inherited byte-unchanged.
- `LeWMSRShim` subclasses `LeWM` and routes both `encode` and `predict` through injected engine
  callables. `encode` has no `_encode_image` seam (`LeWM.encode` fuses backbone + info-dict
  bookkeeping + the `action_encoder` branch), so the override re-implements its body.

Cost parity is checked at `B = 1`: the vendored CEM pins `batch_size = 1` and `LeWM.criterion`
supports one env per solve (it broadcasts the single-env goal over candidates and errors for `B > 1`).

### `get_action` must stay absent

Two unrelated platform methods share the name: a **policy**-side `get_action(info_dict, **kwargs)`
(`WorldModelPolicy` — the replanner, not a model method at all) and the **model**-side
`Actionable.get_action(info, horizon, prefix_actions)` (only `TDMPC2` / `GCRL` define it; `LeWM` and
`PreJEPA` do not). `CEMSolver` touches the latter only via `prepare_init_action`'s
`isinstance(model, Actionable)` branch, which our non-`Actionable` tracks never take — that is exactly
what makes the warm start a zero-pad (`mean = 0`), load-bearing for both eval parity and the
calibration distribution (§6).

Because `Actionable` is `@runtime_checkable`, `isinstance` matches on **method presence, not
signature**: adding *any* `get_action` to the shim — even a policy-shaped one — silently flips it to
`Actionable` and replaces the zero-pad with a generated warm start. Pinned by `tests/test_sr_shim.py`.

### Injection seam — the one exception to the no-monkeypatch stance

`eval_wm.run` has no config seam for a model object. The SR re-run therefore uses a dedicated driver
(`src/sr_eval.py`) that slots the shim in by a **scoped patch of the checkpoint loader** around the
run, swapping only which model the loader returns. The vendored eval entrypoint and the solver/CEM
logic stay byte-unmodified, and no CEM config, seed, sample count, or plan changes, so eval/CEM parity
is preserved — the SR differs from the FP32 baseline only by the engines' quantization drift.

Alternatives rejected: a latency-only pass at `num_envs = 1` (breaks eval parity and the same-solves
SR pairing); editing the vendored entrypoint (loses byte-parity).

---

## 5. The export & quantization stack

The export/quantization stack (TensorRT + NVIDIA TensorRT Model Optimizer + its ONNX Runtime
dependency) must stay binary-compatible with the pod's CUDA 12.4 driver and the uv-locked `cu124`
torch. Several of these dependencies now default to CUDA-13 wheels, which pull `nvidia-*-cu13` and
fail to initialize against a 12.x driver.

- **TensorRT and the Model Optimizer are installed by `setup.sh`, out of uv** (cu12,
  CUDA-12.4-matched), so uv cannot pull a conflicting `libnvinfer`/CUDA stack.
- **`onnxruntime-gpu` is installed from onnxruntime's dedicated CUDA-12 feed**, before modelopt, so
  modelopt's unbounded dependency cannot re-resolve it to the cu13 PyPI default. This CUDA-12 build
  is also what lets the calibration pass run on the GPU. Confirmed torch-2.6-compatible pins:
  `modelopt==0.43.0`, `onnxruntime-gpu==1.24.4`.

**Calibration execution-provider (EP) split.** The encoder calibrates on the **GPU (CUDA EP)**; the
predictor calibrates on the **CPU EP**. The `onnxruntime-gpu` CUDA EP miscomputes the predictor's
dynamic-batch reshape (`Squeeze(Shape(latent))` → head-split `Reshape`), fabricating a wrong target
and crashing modelopt's MHA-exclusion probe; the CPU EP (and native TensorRT) computes it correctly.
The encoder graph lacks that pattern, so it keeps the faster CUDA EP. **EP choice affects calibration
speed only, not the derived scales** — plumbing, not a result-affecting decision.

### Optimization profiles are the production call shapes

TensorRT picks kernels/tactics for a dynamic axis at the optimization profile's **`opt`** point, so
`opt` is the load-bearing knob: an engine tuned at a batch it never runs at is tuned for the wrong
kernel. Each engine's profile is therefore the batch that engine is *actually* called at, read off the
platform source rather than assumed:

- **Encoder — `min = opt = max = 1`.** `PreJEPA.rollout` slices the candidate axis away before
  encoding (`init_info_dict[k] = info[k][:, 0]`) and `.expand(...)`s the resulting latent across
  candidates *after* the encode; `get_cost` embeds the goal by the same `[:, 0]` path. Both run at
  `B = current_bs`, and the vendored CEM pins `batch_size = 1`
  (`scripts/plan/config/solver/cem.yaml`). So the encoder sees batch 1 and nothing else.
- **Predictor — `min = 1`, `opt = max = CEM_NUM_SAMPLES` (300).** `rollout` flattens `(b n) -> ...`
  before every `self.predict`, so the predictor is called at `current_bs × num_samples = 1 × 300` on
  every horizon step of every CEM iteration. 300 is also the feasible ceiling: the DINO predictor's
  `(batch, 16, 588, 588)` attention tensor crosses TensorRT's 2^31 element-volume limit above batch
  388. `min` stays 1 because two non-headline paths drive the engine there — the precision-match
  sweep's profile-min row, and any sub-batch call the shim's pad/slice wrappers make — and a profile
  minimum costs nothing.

**The profile is a build property, decoupled from the ONNX trace batch.** The traced example batch
fixes only the non-batch axes of the graph and the concrete shape `quantize_onnx` pins for modelopt's
ORT sessions (`calibration_shapes` — required because the predictor graph's batch axis is a
`torch.export`-specialized symbol ORT constant-folds to the trace batch). TensorRT keeps the axis
dynamic when it parses that graph and honours whatever profile the build sets, and per-tensor PTQ
scales are batch-independent — so the profile never reaches the calibration set or its scales.

The one constraint the two share is a floor: **the trace batch must be ≥ 2.** `torch.export`
specializes a size-1 dim, so tracing at batch 1 emits a frozen `dim_value: 1` where the symbol should
be — silently, with no warning — and every engine built off that graph is batch-frozen regardless of
its profile. `export_onnx` raises on it rather than letting a frozen axis reach the builder.

---

## 6. Calibration: matching the inference-time distribution

A fixed per-tensor activation scale **saturates** anything outside the range the calibration pass
observed. FP16 carries no fixed clip, so a calibration/inference distribution gap is **invisible in
FP16 and catastrophic at 8 bits**. The calibration set must therefore match the *inference-time* input
distribution, not the expert-demo one.

The **predictor** stream is wrong on two axes if calibrated from expert data:

1. **Actions.** `CEMSolver.solve` draws `candidates = randn(...) * var_scale + mean` with **no clamp**
   to the action space, from `var_scale = 1.0` and `mean = 0`. The `mean = 0` holds because our models
   are **non-`Actionable`** (no actor fills a warm start) **and** `horizon == receding_horizon`, so no
   carried plan tail survives — both conditions matter (§4). So `predict` is driven by an **unbounded
   ≈N(0,1)** proposal reaching ≈±4, while expert actions are bounded to `Box(-1, 1)`; calibrating on
   expert actions under-scales the action range ~4× and clips most of the proposal. LeWM's
   `action_encoder` sits **inside** the predict engine, so the raw action tensor and its activations
   quantize on that under-scaled range. The clipped tensors are exactly the candidates CEM is trying
   to rank, so what breaks is the **planning signal** — SR collapses toward the non-planning floor
   rather than degrading gracefully.
2. **Latents.** `predict` runs autoregressively: only step 0 consumes an encoder latent; steps 1…H−1
   consume the predictor's **own predicted latents**, which drift off the encoder-latent manifold. A
   single-step encode→predict draw observes none of them.

**Decision: reproduce the distribution in the builder; do not harvest a live rollout.** Sourcing it
from an actual CEM/eval run would make the scales depend on the eval seed and sample draws and would
couple the quantization pipeline to the CEM solver + SR shim + eval config. The roll is driven through
the **real rollout** (already fidelity-gated), never a re-implementation of the solver: DINO drives
`shim.dino_rollout` via a capture proxy; LeWM mirrors `LeWM.rollout`'s window loop. Only the
`T == HISTORY_SIZE` windows are kept, since the engine binds a static `HS`.

### Why the calibration method is a labelled dimension, not a per-track setting

In explicit quantization activations must be per-tensor (TensorRT), so the method is the lever, not
granularity. The two tracks pull in opposite directions — DINO's frozen-DINOv3 patch features are
outlier-heavy (a high-norm token pins the per-tensor amax and starves the bulk, so `entropy`'s
tail-clip should help), while LeWM's wide values *are* the action signal the distribution fix widened
(so `entropy` could re-clip it). Which method wins is therefore **measured** per (track, precision),
never asserted, and every artefact is keyed by it: one method's timing run adds points rather than
replacing the other's, and a render reads only the method it names. FP32/FP16 have a single data-free
build, so they are timed once and read across labels.

### Neutralizing the attention-mask sentinel

DINO-WM's predictor attention builds an **explicit additive causal mask** —
`masked_fill(mask, float("-inf"))` fed to `scaled_dot_product_attention`. On ONNX export that `-inf`
is materialized as the **finite** constant `finfo(float32).min` (≈ −3.4e38), so the mask-add tensors
carry −3.4e38 as a real activation value. modelopt calibrates them, and `entropy` overflows FP32 when
laying out histogram bin edges (`range ≈ 6.8e38`), while `max` bakes a garbage −3.4e38 scale. LeWM
exports `is_causal=True` (no materialized mask tensor), so this is DINO-only.

Before the PTQ pass, `export._neutralize_attention_mask_sentinel` rewrites every `finfo(float32).min`
mask-fill constant (magnitude ≥ `1e30`) to a mild finite fill `_MASK_FILL = −3e4`. This is
**numerically equivalent masking** — `softmax(−3e4 + logit)` underflows to 0 in FP16 and FP32 exactly
as `−inf` does — but the histogram range collapses to ±3e4, inside the FP32 edge and the FP16 range.
Because the mask is added **after** the QK^T score MatMul, the real score/probability tensors are
unpolluted and the attention MatMuls quantize normally. The pass edits only tensors carrying the
sentinel, so it is a **no-op on LeWM** and both tracks stay fully quantized. It runs for INT8 and FP8,
`max` and `entropy`.

---

## 7. Measurement design

### What a "cycle" is — bracket per env, not per solve

A **cycle is one episode's decision**, which is *not* the span of one `CEMSolver.solve` call. A solve
plans every still-alive episode back-to-back: `batch_size = 1` (pinned; `LeWM.criterion` errors for
`B > 1`) makes the env loop sequential, and `EnvPool.step` is a Python for-loop over in-process envs —
a vectorized *interface*, not concurrency. So one solve = **N independent decisions computed
back-to-back, and its wall clock is their sum**. The only genuinely batched axis is the `num_samples`
candidate fan-out within one episode's `get_cost`.

The latency callback therefore brackets **per env** — consecutive `start_batch` hooks, the last
closing at `end_solve` — one record per decision, with a sync per span. Bracketing `reset → end_solve`
instead (both hooks sit outside the env loop) would time **all N** episodes per record while
`ENCODER/PREDICTOR_CALLS_PER_CYCLE` count **one**; weighting components by per-decision counts across
that gap inflates the residual overhead toward the whole cycle, and it would make the result scale with
how many episodes are still alive — SR-dependent, therefore track-dependent: a parity break. A
`current_bs == 1` guard fails loud if `batch_size` ever ≠ 1.

### Why per-cycle n is bounded at ~100 — and equal-n truncation

`eval_budget = 50` policy steps, and one plan fills the action buffer with
`receding_horizon × action_block = 25` env actions, so the solver is called on exactly **two** steps
(t=0 and t=25). Decisions therefore total `50 + alive_at_25` and cannot exceed 100. Because Push-T
terminates on success, **a higher-SR track contributes fewer decisions** — n is SR-dependent hence
track-dependent. The equal-n truncation neutralises this by taking the common minimum n across tracks;
the n each percentile was computed from is reported on the speed table, so the equal-n comparison is
verifiable off the artefact rather than asserted.

The **component** loops need none of this: they are fixed-iteration (100 timed, 10 warm-up dropped
before the first timed call), so their equal-n is structural rather than enforced, and their recorded
vector is already the sample any statistic runs on. Those raw vectors are **persisted** beside the
canonical results, and the percentile definition is shared across the benchmark, the report and the
stats path — one helper, so a stored p50 and a later interval around it can never be computed two
different ways.

### Per-cycle warm-up drop

The engine-step loops drop `warmup` iters; the per-cycle callback records from the **first decision of
the first solve** (the vendored eval's warm-up pass is gated on `compile`, which is `false`), so
cold-start cost — first `execute_v2`, kernel autotune, allocator growth, clock ramp — would sit on the
cycle side of `overhead = cycle − t_comp` only, and be booked entirely as planner overhead.

The fix drops a warm-up head of `k = 1` decision (`PER_CYCLE_WARMUP_DROP`), **at report time** (the
recorded vector stays complete) and **before** the equal-n truncation (which otherwise preserves the
temporal-head cold sample). The dropped value is disclosed as the speed table's `drop×`, so the
exclusion is visible rather than tuned silently. The p50 is robust to one head sample and does not
move — this corrects the mean-based decomposition, not the comparison.

### Why the decomposition subtracts rather than mirrors the solver

The full planning-cycle time is **measured on the real CEM solve**, not reconstructed from a
hand-rolled solver mirror. Encoder and predictor are timed in isolation and weighted by their real
per-cycle call counts; the remainder is the residual overhead. A **negative residual is surfaced
loudly**, never clamped: it signals the call-count weighting or the isolated measurement is off.

**Call counts are confirmed against the installed `CEMSolver.solve`, not assumed.** Tracing
`solve → get_cost → rollout` in swm 0.1.1 shows the `candidates` tensor is `horizon`-long only, **not**
`n_obs + horizon`, so the rollout drives `(horizon − n_obs) + 1` predict calls per rollout. With 30 CEM
iterations that is `PREDICTOR_CALLS_PER_CYCLE = ((5 − 1) + 1) × 30 = 150`;
`ENCODER_CALLS_PER_CYCLE = 2` (goal + initial-obs encode). Both counts are **per-decision**, so the
measured cycle must be per-decision too.

---

## 8. Component-precision isolation

The speed table reports **that** SR fell, not **which component's** quantization caused it: its rows
are `(track, precision)`, and both engines move together. The encoder and the predictor are separately
exported, separately quantized, and separately timed — so the attribution is measurable, and without
it the study reports a symptom.

**Re-run the SR eval with ONE component quantized and the other held at FP16**, per (track, precision)
cell that shows a material SR drop. Together with the FP16 baseline and the pure-quantized run the two
runs close a 2×2 whose fourth corner is already measured, so its closure is a consistency check: if the
components' damage composes, the pure corner is predictable from the two isolated ones; if not, they
interact and the attribution carries a caveat.

**Held at FP16, not FP32**, because FP16 is what the held component is running at — so an isolation row
is read against the FP16 row, not the FP32 one.

**Diagnostic, not a shipped configuration.** A mixed pairing is not a fifth precision in the sweep: it
is never benchmarked for latency and never quoted as a recommended configuration. Results land under a
composite precision label `enc-<A>+pred-<B>` that cannot collide with a pure precision, so they are
written beside the pure points and never overwrite them; the precision set stays closed, so composite
keys reach no reported table or figure.

**Both methods, both tracks.** Because the scales differ per method, the two methods' isolation runs are
**separate measurements, not a re-answer of the same question** — the attribution can legitimately
differ between them, which is precisely why the method is a labelled dimension at all (§6). Points are
keyed per (track, composite label, method) and never fall back across methods. A single-method
diagnostic would leave the other method's drops unattributed — the same absence-of-evidence objection
that forces both tracks.

---

## 9. Confidence intervals: where the stock library is wrong

The estimators themselves are standard. What is worth recording here is the four places a stock
scipy call is **not** the construction the spec asks for — each pinned by a test
in `tests/test_stats.py` whose name states the deviation, so a future refactor toward "just use scipy"
fails loudly instead of silently changing an interval.

- **`scipy.stats.permutation_test` is not the Dwass test.** Its `two-sided` is *"twice the smaller of
  the one-sided p-values"*, not `|r*| ≥ |r_obs|`. The permutation loop is therefore written directly.
  The p-value uses the add-one form `(1 + #{|r₁*| ≥ |r₁_obs|}) / (1 + B)`, which keeps the test exact
  rather than letting a zero count report `p = 0`.
- **No scipy function computes the order-statistic quantile interval.** `mstats.median_cihs`
  (Hettmansperger–Sheather) and `mstats.mquantiles_cimj` (Maritz–Jarrett) are *different estimators*
  and must not be substituted; only `scipy.stats.binom.cdf` is used.
- **The obvious rank choice under-covers.** Taking `j = binom.ppf(α/2)` and `k = binom.isf(α/2)` gives
  coverage *below* the nominal 0.95 at the sample sizes this study has, because `ppf` returns the
  smallest rank whose CDF is *at least* α/2. The conservative recipe is used instead:

  > `j−1` = largest r with `cdf(r) ≤ α/2`;  `k−1` = smallest r with `cdf(r) ≥ 1−α/2`;
  > coverage `= cdf(k−1) − cdf(j−1) ≥ 1−α`.

  Where no rank pair achieves coverage ≥ 1−α, **no interval is emitted** rather than a short one.
- **Clopper–Pearson is the one construction taken wholesale**
  (`binomtest(k, n).proportion_ci(method="exact")`), because it is exact there. The trial count is the
  one input no artefact records — it lives in `scripts/plan/config/pusht.yaml` — so successes are
  recovered as `k = round(SR/100 × n)` under a loud integrality check rather than a silent round.

Two implementation consequences worth naming:

- **The interval is computed from the same sample as the point**, via one shared helper: warm-up
  decision dropped, then equal-n truncated, byte-identical to what `_finalize_per_cycle` reduces to the
  reported p50. One residual is documented rather than fixed: the reported p50 is `torch.quantile`
  linear interpolation while the interval endpoints are order statistics, so the interval is not
  symmetric about the point and at even n the point can fall between the two central order statistics.
  Changing the point estimate to match would alter already-published numbers for no gain.
- **The independence test matters more on the component loops, not less.** It is tempting to treat a
  tight `for _ in range(100): run(); record()` loop as the clean i.i.d. case and the messy per-cycle
  sample as the suspect one. The opposite is closer to true: the component loop calls one engine
  back-to-back with no planner work between calls, so the whole sample spans seconds of sustained load
  — precisely the window in which an unlocked GPU's DVFS ramps its clock. A monotone ramp across a
  short loop is textbook positive lag-1 autocorrelation, and it makes the interval **too narrow**, in
  exactly the direction that flatters the result.

---

## 10. GPU clocks: recorded, not corrected

GPU clocks cannot be locked on the benchmark platform — `nvidia-smi -lgc` is denied by the RunPod
virtualization layer, confirmed as root with persistence mode on. A passive `nvidia-smi dmon` observer
(`src/gpu_clocks.py`) therefore *records* the per-run clock/thermal state alongside every timed run. It
is a separate subprocess and never touches seeds, samples, or the plan.

**Why no normalized number is reported.** A `T_ref = T × f_measured/f_ref` rescaling over-corrects —
memory-bound and host/Python time do not scale with SM clock — so it could only ever be a bound, and a
derived latency printed beside a measured one invites the two being read alike. The recorded telemetry
states the conditions each run was timed under; nothing is rescaled.

**What the telemetry summary must not do.** `dmon` samples at ~1 Hz, so a short run's log is mostly
idle/ramp samples and an unconditioned median can land at a clock nothing ran at. The per-run statistic
is therefore the **util-conditioned** median SM clock (samples at SM util ≥ `CLOCK_BUSY_UTIL_PCT`), and
a run with fewer than `CLOCK_MIN_BUSY_SAMPLES` busy samples is reported **unmeasured** rather than
summarised. Asserting a clock from an idle sample is the silent wrong-number failure the gate exists to
prevent.
