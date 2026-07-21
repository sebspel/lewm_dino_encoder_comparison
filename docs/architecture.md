# Architecture & design rationale

**Why** the owned layer is shaped the way it is. `SPEC.md` states the contract these
decisions must satisfy; `PLAN.md` carries the execution steps. Dated, self-contained
decisions live in `docs/adr/`; this file holds the standing structural reasoning.

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
part of the encoder-compute asymmetry the study exists to measure. The two tracks therefore
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
corrupts every SR. That is why the adapter-fidelity gate (`src/fidelity.py`) exists and runs
on the real checkpoint **before** any engine is built.

**LeWM, Design A (owner-chosen): the `action_encoder` lives inside the predict engine.** The
exported engine ingests a raw action. LeWM's action encoder is per-frame — `Conv1d(k=1)` plus a
per-position MLP, no mixing along the macro-step axis — so a per-step `predict` is numerically
identical to the platform's whole-sequence pre-encode, and the per-step engine boundary is
faithful. Since inherited `LeWM.rollout` pre-encodes the whole action sequence, the shim sets
its `action_encoder` to an Identity passthrough; rollout then windows raw actions straight into
`predict` and the engine's own per-frame `action_encoder` does the encode.

This is a **silent-failure boundary**: a temporal (kernel > 1) action-encoder config would make
the per-step boundary wrong with no error. It is guarded by a runtime assertion on the real
checkpoint (`src/fidelity.py::lewm_action_encoder_per_frame`,
`action_encoder(seq)[:,t] ≈ action_encoder(seq[:,:t+1])[:,-1]`), owner-signed-off 2026-07-11.

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
because the pad sits after every position we read. Both tracks' predictors are owner-confirmed
causal with prefix positional embeddings, so the identical fix applies to LeWM and DINO-WM with
no per-track gating. Were a predictor ever *not* causal, the exception would fail and the
transient `T < HS` steps would need a torch fallback or a dynamic-hist re-export.

**Why it needs its own gate.** The fixed-`HS` precision-match and SR-cost-parity gates never
exercise `T < HS`, which is why the mismatch passed every gate yet crashed the SR run. The
boundary is proven by a variable-window (`T ∈ {1, 2}`) engine-vs-torch parity check
(`_MATCH_HISTS` in `src/precision_match.py`; `sr_cost_parity*` also run at `n_obs = 1`).

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

Parity reference is the **installed swm 0.1.1** `PreJEPA.get_cost` — the local
`~/stable-worldmodel` checkout is 16 commits ahead / diverged and removed some of these methods.
Gates run at **B = 1**: the vendored CEM pins `batch_size = 1` and `LeWM.criterion` supports one
env per solve (it broadcasts the single-env goal over candidates and errors for B > 1).

### `get_action` must stay absent

Two unrelated platform methods share the name: a **policy**-side `get_action(info_dict, **kwargs)`
(`WorldModelPolicy` — the replanner, not a model method at all) and the **model**-side
`Actionable.get_action(info, horizon, prefix_actions)` (only `TDMPC2` / `GCRL` define it; `LeWM`
and `PreJEPA` do not). `CEMSolver` touches the latter only via `prepare_init_action`'s
`isinstance(model, Actionable)` branch, which our non-`Actionable` tracks never take — that is
exactly what makes the warm start a zero-pad (`mean = 0`), load-bearing for both eval parity and
the INT8 calibration distribution (ADR 0002).

Because `Actionable` is `@runtime_checkable`, `isinstance` matches on **method presence, not
signature**: adding *any* `get_action` to the shim — even a policy-shaped one — silently flips it
to `Actionable` and replaces the zero-pad with a generated warm start. Pinned by
`tests/test_sr_shim.py`.

### Injection seam — the one specified exception to the no-monkeypatch stance

`eval_wm.run` has no config seam for a model object. The SR re-run therefore uses a dedicated
driver (`src/sr_eval.py`) that slots the shim in by a **scoped patch of the checkpoint loader**
around the run, swapping only which model the loader returns. The vendored eval entrypoint and
the solver/CEM logic stay byte-unmodified, and no CEM config, seed, sample count, or plan
changes, so eval/CEM parity is preserved — the SR differs from the FP32 baseline only by the
engines' quantization drift.

Alternatives rejected: a latency-only pass at `num_envs = 1` (breaks Phase-3 eval parity and the
same-solves SR pairing); editing the vendored entrypoint (loses byte-parity).

---

## 6. Measurement design

### What a "cycle" is

See **ADR 0004** for the full derivation and the bracket decision. In short: a cycle is **one
episode's decision**, which is *not* the span of one `CEMSolver.solve` call — a solve plans every
still-alive episode back-to-back, so its wall clock is the sum of N decisions.

### Peak memory is sampled from the driver, not the torch allocator

TensorRT's engine and execution-context device allocations bypass torch's caching allocator, so
`torch.cuda.max_memory_allocated` would systematically undercount **exactly the optimized path** —
the one the study is trying to measure. Peak memory is therefore sampled via
`cudaMemGetInfo` / nvidia-smi (`torch.cuda.mem_get_info`, device-level used).

### GPU clocks are not locked — the shared state is recorded, not assumed

The study runs both tracks back-to-back at the same precision on the same L40S with warm-up
dropped, and the LeWM-vs-DINOv3 comparison is a **ratio** on that shared hardware state, so any
residual boost/thermal drift applies to both tracks alike rather than to one.

To make that shared-state assumption **verifiable rather than asserted**, a passive
`nvidia-smi dmon` observer (`src/gpu_clocks.py::log_gpu`) logs per-sample GPU telemetry (SM/mem
clock, power, temperature, utilization, memory) alongside every timed engine run — both the
isolated component loops and the per-cycle eval-shim run. It is a separate subprocess (like the
`cudaMemGetInfo` sampling) and never touches seeds, samples, or the plan.

### Why the decomposition subtracts rather than mirrors the solver

The full planning-cycle time is **measured on the real CEM solve**, not reconstructed from a
hand-rolled solver mirror. Encoder and predictor are timed in isolation and weighted by their real
per-cycle call counts; the remainder is `overhead_ms = cycle − encoder − predictor` — the
un-optimizable floor (CEM sampling/topk/mean-var, the criterion, the 384→404 assembly, per-step
action-replace/proprio-carry, and host/Python glue).

A **negative `overhead_ms` is surfaced loudly**, never clamped: it is a sign the call-count
weighting or the isolated measurement is off.

For that subtraction to mean anything, both sides must be measured under the same warm-up regime.
The engine loops drop warm-up iters, so the per-cycle vector drops a warm-up head too (k = 1
decision, at report time, disclosed in the table) — otherwise cold-start cost sits on one side of
the subtraction only and is booked entirely as planner overhead, a one-sided bias the
negative-overhead alarm cannot catch because it makes overhead *more* positive (ADR 0003).

**Call counts are confirmed against the installed `CEMSolver.solve`, not assumed.** Tracing
`solve → get_cost → rollout` in swm 0.1.1 (both `wm/prejepa/prejepa.py` and `wm/lewm/lewm.py`)
shows the `candidates` tensor is `horizon`-long only, **not** `n_obs + horizon`. So the rollout
drives `(horizon − n_obs) + 1` predict calls per decision, not `horizon + 1`:
`PREDICTOR_CALLS_PER_CYCLE = ((5 − 1) + 1) × 30 = 150` (an earlier unconfirmed guess had 180).
`ENCODER_CALLS_PER_CYCLE = 2` (goal + initial-obs encode) is unaffected by the horizon/n_obs split.

Both counts are **per-decision**, so the measured cycle must be per-decision too (ADR 0004).

### Attribution runs on the same seam in both directions

The encoder and predictor are separately exported, separately quantized, and separately timed. That
seam already carries the *latency* attribution (the per-component decomposition above); **ADR 0005**
runs it in the other direction for *task quality*, holding one component at FP16 while the other is
quantized, so a measured SR collapse is attributed to a component rather than reported as a symptom.

The two attributions answer the study's mechanistic claim from both sides — which component costs the
time, and which component cannot survive 8-bit. They are deliberately asymmetric in status: the
latency decomposition is a headline result, the SR isolation is a diagnostic that never enters the
FP32→FP16→INT8→FP8 sweep (ADR 0005, "Diagnostic, not a shipped configuration").

---

## 7. Amdahl dilution disclosure

Only encoder + predictor are quantized; the Python overhead (CEM planner + criterion + assembly +
glue) is precision-invariant. So the per-precision **wall-clock** delta is capped by the model's
share of the cycle, and reporting per-component *relative* speedup alone would hide that dilution.

The study therefore reports, per model: the FP32 baseline per-component time shares and the derived
**optimizable fraction** `p = (encoder + predictor) / cycle`, which sets the Amdahl ceiling
`1/(1−p)`; and per precision, **both** the *model-only* speedup (overhead treated as free) and the
*realized* speedup (the measured FP32-vs-precision per-cycle ratio), whose gap is the overhead floor
and should match the Amdahl prediction `1/((1−p) + p/s)`.

That the optimizable fraction is itself model-dependent — LeWM's single token is
overhead/launch-latency-bound, DINO's 196-token grid is model-bound — is what explains why the same
precision helps the two tracks differently. That is a **result, not bookkeeping**.

This whole block is **mean**-based, `p` included (ADR 0003), so the realized speedup it reconciles
against is the mean per-cycle ratio, matching the prediction's basis. It is therefore a *different
number* from the reported p50 FP32-relative speedup, which answers the comparison question rather
than the reconciliation one. The two are rendered in separate tables and must not be conflated or
averaged.

---

## 8. Retired approaches (history — do not re-derive)

- **Standalone profiler (`src/profile.py`) + its CEM-iteration mirror — retired.** The PyTorch
  per-call timing could not reconcile with the engine-context cycle. The decomposition moved into
  `src/report.py::decompose`, derived from the benchmark's isolated engine-step latencies.
- **Fixed-wall-clock rollout-count run — removed (owner decision).** Serial planning makes
  rollouts/sec ≈ 1/per-cycle-latency, so it is redundant with the equal-n latency measurement.
  `rollouts_completed` / `throughput` / `time_budget_s` were dropped from `BenchResult` /
  `ExportConfig`.
- **Implicit TRT INT8 calibration (`IInt8MinMaxCalibrator`) — replaced** by explicit Q/DQ via the
  Model Optimizer (ADR 0001). The calibration *data* construction was reused; only its consumer
  changed (numpy arrays keyed by ONNX input name, not CUDA pointers).
- **`SolveLatencyRecorder`'s original `reset → end_solve` bracket — replaced** by a per-env bracket
  (ADR 0004). The Phase-3 baseline logged under the old `cem_solve_*` W&B keys is **not comparable**
  to anything logged since; the keys moved to `per_cycle_*` because the number's meaning changed.
