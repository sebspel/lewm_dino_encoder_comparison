# SPEC: LeWM vs DINOv3-WM — Inference Optimization Study

The source of truth for **what** this project must satisfy. `PLAN.md` carries the ordered
execution steps and progress; `CLAUDE.md` holds behavioral rules; `src/interfaces.py` is the
typed contract in code for the owned export/benchmark layer. Design rationale that runs
longer than a couple of sentences lives in `docs/architecture.md`.

---

## Objective

Compare two world models trained on Push-T (224×224, pixels-only) and deliver the engineering
layer the platform does **not** provide. **LeWM** (scratch ViT-Tiny + SIGReg) is a reference
model from `stable-worldmodel` used as-is. **DINOv3-WM** is *this project's* variant: the
platform's DINO-WM predictor (whose reference backbone is **DINOv2**) with the frozen backbone
swapped to **DINOv3** — the same training framework, a different encoder.

1. **Inference-optimization study on an L40S:** export both models PyTorch→ONNX→TensorRT with
   the 8-bit precisions quantized **explicitly** (Q/DQ), and benchmark planning latency and peak
   GPU memory across FP32→FP16→INT8→FP8. Headline: the **LeWM-vs-DINOv3 per-cycle latency ratio**
   and the **per-model FP32→FP16→INT8→FP8 optimization delta**.
2. **FP8 as a native-L40S precision:** add FP8 (E4M3) to the sweep on the L40S's FP8 Tensor
   Cores (Ada 4th-gen / Transformer Engine — no hardware or toolchain change), exported,
   benchmarked, and reported (speed + SR, degradation quoted vs FP32) **exactly like every other
   precision** — an added precision, not a separate study or a head-to-head against INT8.

**Non-goal:** the training framework, Push-T env, CEM solver, and MPC eval are **provided by
`stable-worldmodel`** — a foundation, not the contribution. Running them, including swapping
DINOv3 into the DINO-WM predictor to set up the comparison, is foundation work; the owned
contribution is the optimization layer above a trained checkpoint.

---

## Scope & Boundaries with the Platform

Layering rationale: `docs/architecture.md` §1.

- **Use `stable-worldmodel` as the foundation.** Training, the Push-T env, the CEM solver,
  dataset tooling, and closed-loop MPC evaluation come from the package. Do not reimplement them.
- **DINOv3-WM = the platform's `prejepa` DINO-WM predictor with the encoder swapped from the
  reference DINOv2 to a config-injected, frozen DINOv3.** The only code addition is **one
  owner-approved encode-path override** that drops CLS + register tokens to expose the true
  196-patch grid; it lives in `src/` and is imported by the vendored train/eval entrypoints — no
  predictor/SIGReg changes, and the platform wheel is not edited.
- **Always DINOv3, never DINOv2.** The DINO-WM paper and the platform default to DINOv2; this
  project overrides the encoder to DINOv3 wherever DINO-WM is referenced. The wiring is otherwise
  identical (dims differ and are read from config).
- **The full patch-token grid feeds the predictor/planner unchanged.** Slice off CLS + any
  register tokens (verify the token layout first) and preserve the patch dimension. **The two
  tracks have different latent ranks** — LeWM exposes a single-token latent `(B, D)`; DINO-WM
  exposes the full patch grid `(B, N_patches, D)`. **Do not pool DINO to one token:** it would
  diverge from the paper and erase part of the architectural asymmetry the study measures.
- **LeWM = the platform's `lewm` training unchanged.** SIGReg and the scratch encoder are the
  platform's; do not reimplement or retune them beyond what training requires.
- **The contribution lives downstream of a trained checkpoint:** export, quantize, benchmark.
  That is where `src/interfaces.py` and the owned code apply.

---

## Execution Environment

- **Inference runs on one machine: L40S** — both tracks are exported and benchmarked on the same
  L40S (same hardware class as the LeWM paper, so speed numbers are comparable).
- **GPU clocks cannot be locked on the benchmark platform.** `nvidia-smi -lgc` is denied by the
  RunPod virtualization layer (confirmed as root, persistence mode on), so both tracks run at stock
  boost/throttle behaviour and the per-run clock/thermal state is *recorded* by the telemetry
  observer (§Parity), not controlled. A clock-locked re-run is unavailable here and deferred to
  future work.
- **Training hardware differs by track:** LeWM is trained on the L40S; DINOv3-WM is trained on an
  **H200 SXM**. Training hardware, like training batch size, does **not** carry into inference — the
  exported engine is built and benchmarked on the same L40S from the checkpoint weights alone — so
  the latency comparison is unaffected.
- **TensorRT engines are architecture-specific and disposable** — regenerated on the L40S,
  gitignored, never committed. They are **saved to and loaded from the network volume by
  default** (`$STABLEWM_HOME/engines/<track>/{encoder,predictor}.<precision>.plan`, the quantized
  int8/fp8 plans additionally tagged with the calibration method `…<precision>.<method>.plan` so
  the `max`/`entropy` engines coexist; repo-local `engines/` fallback only off-pod where
  `STABLEWM_HOME` is unset), so a pod session's built engines survive teardown. Living on the volume
  does not make them durable *artifacts* — they stay regenerable and are re-derived if the L40S
  class changes.
- **Durable artifacts persist on the persistent network volume, never in git:** datasets,
  checkpoints, exports, and the Phase-5 headline reports all live under the persistent root
  (`STABLEWM_HOME`, e.g. `$STABLEWM_HOME/reports/`). The root must point at the network volume
  or a multi-hour run's checkpoints are lost on pod restart.
- Datasets are loaded via the platform (streamed from object storage or cached under the root).
- Runtime secrets/env (`WANDB_API_KEY`, `HF_TOKEN`, `STABLEWM_HOME`) are passed at runtime.

The pinned dependency set (CUDA-12 export stack, TensorRT/modelopt out of uv) and the reasons for
those pins are recorded in `docs/architecture.md` §6; the executable form is
`pyproject.toml` + `setup.sh`. Concrete run commands live in `PLAN.md`.

### W&B logging discipline

Every phase logs to the **same** W&B project. Training uses the platform's Lightning logger; the
non-training phases (eval/benchmark) open the run through an owned helper that reads the
project/entity from the same config block — no second source of truth.

---

## Interface Contracts (the owned layer)

`src/interfaces.py` is the single source of truth **in code** for the boundaries the project
owns, runtime-checked via jaxtyping + beartype. This section states the **behavioral**
requirements those signatures must satisfy; it does not restate the signatures.

### Export shape

- **Export produces separate encoder and predictor engines.** The adapter's `encode` and
  `predict` are traced and built **separately** (one ONNX graph / TensorRT engine each), because
  the CEM rollout encodes once, caches the latent, then calls `predict` autoregressively for
  every candidate. FP32 and FP16 share the base graph (FP16 is a build flag); **INT8 and FP8 are
  each a separately quantized graph** with Q/DQ and per-tensor scales baked in from a calibration
  pass (INT8 integer, FP8 E4M3 floating-point).
  Only the **model** (encoder + predictor) is exported; the **CEM planner is never compiled in**
  — it stays in Python around the engines. (Rationale: `docs/architecture.md` §2.)
- **"INT8" means INT8 + FP16.** The Model Optimizer quantizes only the heavy layers
  (MatMul/Gemm/Conv) to INT8 and casts the non-quantized remainder to FP16, so the engine builds
  with both flags set. This makes the INT8-vs-FP16 delta the marginal benefit of pushing the
  heavy layers to INT8 on the same FP16 backbone. Owner-accepted trade-off: the FP16 cast of the
  remainder can under/overflow a few initializers; keeping the remainder FP32 is the documented
  fallback if that drift proves unacceptable.
- **"FP8" means FP8 + FP16, and follows the exact INT8 pattern.** The Model Optimizer quantizes
  the same heavy layers to FP8 (E4M3) and casts the remainder to FP16, so its engine builds with
  the FP8 **and** FP16 flags set. FP8 rides the same Q/DQ export, calibration, precision-match,
  and SR machinery as INT8 — it is another precision, not a second code path. Its speed and SR are
  reported and degradation-quoted against FP32, like every other precision.
- **The INT8 calibration set must match the *inference-time* input distribution, not the expert
  one.** The predictor calibration stream samples actions from the CEM proposal (`randn *
  var_scale`, zero mean, **unclamped**) and rolls `predict` autoregressively so it consumes its
  own predicted latents; it is reproduced in the builder, never harvested from a live CEM/eval
  run. **INT8's calibration health is judged by SR, not by the drift table.**
  (Full derivation, the two failure axes, and accepted residuals: `docs/architecture.md` §7.)
- **FP8 draws the identical calibration *streams* as INT8** (format-independent — the same
  encoder/predictor clips) and reuses whichever calibration *method* the engine was built with, so
  the INT8→FP8 step isolates the format. The **calibration method (`max` | `entropy`) is a build
  option available to both tracks** and a **labelled dimension of every quantized result**
  (`docs/architecture.md` §7); the FP8 path computes the INT8 scales with that method, then converts them to
  E4M3, so the choice carries over unchanged. Calibration health is likewise judged by SR, not the
  drift table.

### Latency & profiling

- **Every speed result carries the SR for that engine config** — no speed number is reported
  without its task-quality counterpart. **Per-cycle planning latency is the headline speed
  measure**, reported p50/p95 and **compared at p50**. There is no fixed-wall-clock rollout-count
  run.
- **Three statistics, three jobs — never substituted for one another** (owner ruling, 2026-07-15;
  `docs/architecture.md` §8): **p50** is the comparison basis (headline ratio, FP32-relative speedup);
  **p95** is the reported tail and carries no claim; **mean** is the decomposition basis **only**
  and is never a reported headline.
- **Every reported ABSOLUTE SR and ABSOLUTE per-cycle p50 carries a 95% confidence interval; no
  difference or ratio carries one.** SR uses the **Clopper–Pearson** exact binomial interval over the
  50 eval episodes (exact at the 0/50 and 50/50 boundaries the study actually hits); per-cycle p50
  uses the **exact binomial order-statistic** interval, computed from the *same* warm-up-dropped,
  equal-n-truncated sample the reported p50 is computed from — never a re-derived one. ΔSR, the
  FP32-relative p50 speedup, the cross-model per-cycle ratio, and Δ(entropy−max) get **no** interval,
  and the mean-based decomposition/dilution tables get none either. The order-statistic interval's
  **i.i.d. premise is tested, not assumed**: each run's truncated sample gets a two-sided **Dwass
  Monte-Carlo permutation test on its lag-1 autocorrelation** (α = 0.05, 50,000 permutations, no
  Student-t transform). **The test decision is the UNADJUSTED p-value** — intervals are reported
  separately per engine eval run, so no family-wise correction governs the reported flag. Holm
  step-down adjusted p-values are computed and persisted as **secondary reporting only**, never
  driving a flag or a table, so "did multiplicity matter here" stays answerable off the artefact
  rather than argued. A rejected test is **disclosed as a flag beside the interval, never silently
  corrected** — serial correlation makes the interval too narrow, which reads as a stronger result
  than the sample supports. **The interval artefact is self-describing**: every point records the
  **n it was computed over** (episodes for SR, cycles for p50), the observed lag-1 statistic, **both**
  the pre-permutation asymptotic p-value **and** the permutation p-value, the permutation null's
  own lag-1 summary, and the **fixed, recorded permutation seed** — so an interval can be re-derived
  and audited off the artefact rather than taken on trust. (`docs/architecture.md` §12.)
- **A "cycle" is ONE episode's decision, not the span of one `CEMSolver.solve` call.** A solve
  plans every still-alive episode sequentially, so its wall clock is the sum of N decisions. The
  latency callback therefore brackets **per env** (consecutive `start_batch` hooks, the last
  closing at `end_solve`) and **never `reset → end_solve`**. Bracketing per solve while weighting
  by per-decision call counts inflates `overhead_ms` silently and would make the headline scale
  with how many episodes are alive — SR-dependent, therefore track-dependent: a parity break.
  (`docs/architecture.md` §8.)
- **Per-cycle n is 50–100 per track per precision, not thousands** — establish this before
  reading any per-cycle percentile. It is SR-dependent hence track-dependent, which is what the
  equal-n truncation neutralises. This n is why p50 carries the comparison and p95 does not.
  **The n each percentile was computed from is reported in the artefact**, not merely asserted: the
  equal-n truncation must be verifiable off the table rather than taken on trust (`docs/architecture.md` §8).
- **Three latency distributions, each REPORTED as p50/p95**, mapping to the three profile slices.
  A mean is never *reported* for any of them.
  1. **per-cycle** — the **headline**, compared at p50: full per-decision planning latency
     (encode + predict + overhead), measured on the real solve.
  2. **encode-step** — a component exposing the LeWM-vs-DINOv3 encoder token-count asymmetry
     (wall-clock-diluted in the cycle because encode is cached and runs ~twice).
  3. **predictor-step** — a component: quantization's kernel target. Any unqualified
     "p50/p95 latency" means **predictor-step**.
  Percentiles are harvested from **fixed-iteration** loops (100 timed iters per step, 10 warm-up
  iters dropped), so n is equal across tracks and the tail is not boundary-censored. encode-/
  predict-step ride isolated per-precision engine loops; per-cycle rides the observation-only
  per-decision latency callback over the SR eval-shim run, so **per-cycle latency and SR come
  from the same solves**. The per-cycle vector likewise **drops a warm-up head (k = 1 decision by
  default) before truncation**, so the cycle and the engine steps are measured under the same
  warm-up regime and `overhead = cycle − enc − pred` subtracts like from like; the drop is applied
  at **report** time (the recorded vector stays complete) and the excluded decision is **disclosed**
  in the speed table, never silently removed (`docs/architecture.md` §8). Per-cycle samples are then truncated
  to a common minimum n for an equal-n comparison; the reported p50/p95 **and** the decomposition
  mean are taken off that *same* reduced sample. A dedicated latency-only pass is **not** an
  alternative — it would break the same-solves pairing.
- **Peak memory is sampled from the driver/runtime** (`cudaMemGetInfo`/nvidia-smi), **not** the
  torch caching allocator, which would systematically undercount the optimized path.
- **The per-component profile slices are mutually exclusive and additive, by subtraction from the
  measured cycle — and the decomposition runs on MEANS, not percentiles.** The cycle is measured
  on the real CEM solve; encoder and predictor are timed in isolation and weighted by their real
  per-cycle call counts (**confirmed against the installed `CEMSolver.solve`, not assumed** —
  `ENCODER_CALLS_PER_CYCLE = 2`, `PREDICTOR_CALLS_PER_CYCLE = ((horizon − n_obs) + 1) × n_steps =
  150`). **Those counts are per-decision, so the measured cycle must be per-decision too.** The
  remainder is `overhead_ms = cycle − encoder − predictor` — the un-optimizable floor (CEM
  sampling/topk/mean-var, the criterion, the 384→404 assembly, per-step action-replace/
  proprio-carry, and host/Python glue). A negative `overhead_ms` is **surfaced loudly** — never
  clamped. (`docs/architecture.md` §6, `docs/architecture.md` §8.)

### Adapter, rollout & shim

- **One adapter Protocol, two concrete tracks.** A common two-method `encode`/`predict` interface
  (not a fused `__call__`) so export and benchmark treat both tracks identically; the latent shape
  and how the action enters `predict` differ per track (LeWM: a separate AdaLN-conditioning
  argument; DINO-WM: concatenated onto the predictor embedding — see Constants).
- **DINO-WM `predict` is a faithful, dim-preserving `404 → 404` reconstruction of the platform
  predictor.** It is **not** sliced back to 384: the predicted **proprio** must survive. The
  extras embedding, the initial `384 → 404` assembly, and the per-step action-replacement +
  proprio-carry live in the **Python rollout/shim**, not the compiled engine.
  (`docs/architecture.md` §3.)
- **LeWM `predict`'s action conditioning is per-frame, so the per-step engine boundary is
  faithful** — the action encoder has no receptive field along the macro-step axis, so it may live
  inside the compiled per-step engine. This is an **owner-gated silent-failure boundary**: a
  temporal (kernel > 1) action-encoder config would make the per-step boundary wrong with no
  error, so it is guarded by a runtime assertion on the real checkpoint.
- **The per-step `predict` engine traces a FIXED history axis, but the platform `rollout` feeds a
  GROWING history window.** The shim serves a `T < HS` window by right-padding the history axis up
  to `HS`, running the engine, and slicing the first `T` frames back — keeping the
  precision-match-gated engine byte-for-byte (no re-export / re-quantize). This is **exact only
  under a model-specific mask-free-padding exception**: it holds iff the predictor is **causal**
  with **prefix positional embeddings** and the padded tail frames' outputs are discarded. Because
  that exactness is a silent-failure assumption, the boundary is **owner-gated** and must be
  **proven by a variable-window (`T ∈ {1, 2}`) engine-vs-torch parity check** — the fixed-`HS`
  gates never exercise `T < HS`. **Both tracks' predictors are owner-confirmed causal with prefix
  positional embeddings, so the identical fix applies to LeWM and DINO-WM — no per-track gating.**
  (`docs/architecture.md` §4.)
- **SR-per-precision re-runs the platform eval on the optimized model.** The CEM solver calls the
  world model via **`get_cost`**, not `encode`/`predict`, so the exported adapter is re-wrapped in
  a thin **Python** shim exposing **`get_cost` only** and slotted into the CEM solver, letting the
  platform's eval logic re-run unchanged. The shim stays in Python; only `encode`/`predict` lives
  in the engine. For DINO-WM the shim must reproduce the platform rollout faithfully (full `404`
  carry, per-step action-replace, proprio+pixels cost) or the SR is not comparable to the Phase-3
  baseline — a silent parity break.
  - **`get_action` must stay ABSENT from the shim.** `Actionable` is `@runtime_checkable`, so
    `isinstance` matches on **method presence, not signature**: adding *any* `get_action` — even a
    policy-shaped one — silently flips the shim to `Actionable` and replaces the CEM zero-pad warm
    start with a generated one, breaking both eval parity and the INT8 calibration premise. Pinned
    by `tests/test_sr_shim.py`. (`docs/architecture.md` §5.)
  - **Injection seam (the one specified exception to the no-monkeypatch eval stance).** Model
    injection has no platform config seam, so the SR re-run uses a dedicated driver that slots the
    shim in by a **scoped patch of the checkpoint loader** around the run — swapping only which
    model the loader returns. The vendored eval entrypoint and the solver/CEM logic stay
    byte-unmodified, and no CEM config, seed, sample count, or plan changes. Because it touches
    the model boundary the eval runs on, this seam is **owner-gated**.

**Constants** are defined once in `src/interfaces.py` (platform-native dims are read from config,
not re-guessed) and are **owner-gated** because a wrong value fails silently: `LATENT_DIM = 192`
(LeWM single-token latent), the DINO-WM patch-grid latent `(N_patches, D) = (196, 384)`,
`ACTION_DIM = 2`, and the DINO-WM predictor-input token width `404 = 384 + 20` extras. The `404`
is the width on **both** the predictor's input and output (dim-preserving; not sliced to 384).

---

## Parity & Fairness Contracts (load-bearing — never vary silently)

- **Same trained-task comparison conditions:** both tracks evaluated with the same CEM config
  (300 samples, 30 elites, horizon 5, init variance 1, 10–30 iterations), same action budget, goal
  encoding, eval seeds, and identical input normalization (ImageNet stats). Mostly enforced by the
  platform's eval; confirm they are not varied between tracks.
- **Matched export/benchmark conditions:** both models exported and benchmarked at the **same
  precision** on the **same L40S**, same env/goal, and the **same shared inference batch size**.
  There is no fixed-wall-clock budget run; **latency is the headline** and the model is the only
  difference, so the **per-cycle latency gap is the measured result**. Latency percentiles are
  measured in equal-n fixed-iteration loops (§Interface Contracts).
- **GPU clocks are not locked.** A passive `nvidia-smi dmon` observer logs per-sample GPU
  telemetry (SM/mem clock, power, temperature, utilization, memory) alongside every timed engine
  run — both the isolated component loops and the per-cycle eval-shim run — so the actual per-run
  clock/thermal state is recorded rather than assumed. Its logs persist to
  `$STABLEWM_HOME/reports/phase5/gpu_logs/` (same durability contract as the headline artifacts).
  The observer never touches seeds, samples, or the plan. (`docs/architecture.md` §6.)
- **The clock-state confound is bounded and disclosed, not assumed away.** The observed throttle is
  **differential** — the heavier track power-throttles to a lower SM clock while the lighter track
  holds the boost ceiling — so it does **not** cancel in the ratio. It is also **endogenous**: the
  throttle is driven by the benchmarked workload's own power draw, not by ambient conditions, so the
  measured numbers are what that workload experiences on stock hardware, while the normalized bound
  isolates the architecture at a common clock. The logged telemetry is used to
  quantify it on the three surfaces it touches — the cross-model per-cycle ratio, the within-model
  precision deltas, and the component/overhead decomposition — and the result is reported as a
  limitation alongside the measured numbers. (`docs/architecture.md` §11.)
- **Clock-normalized figures are DERIVED and a BOUND.** Any clock-normalized latency is a `1/f_sm`
  rescaling that **over-corrects** — memory-bound and host-overhead time do not scale with SM clock
  — so it is reported as the **maximum plausible correction**, a bound beside the measured value,
  never a point estimate and never a replacement. Its construction — the scaling model and reference
  clock — is **owner-gated** (§Implementation Boundaries). The measured numbers stay **canonical and remain
  the headline**; normalized numbers are **additive, labelled `derived`, and never overwrite**
  `results.*.json` or the measured tables (CLAUDE §8).
- **Confidence intervals are RE-ANALYSIS of the stored samples, never a new measurement.** They are
  computed off `sr.json` alone — no eval, benchmark, or export run — and are **additive**: they land
  in their own `stats.json` plus columns/error bars on the regenerable views, and `results.*.json`
  and `sr.json` stay **byte-unchanged**, the same read-only discipline the derived clock render obeys.
- Training batch size is held equal across tracks (128, LeWM's paper value); training **hardware**
  is not (LeWM on the L40S, DINOv3-WM on an H200 SXM). Neither carries into inference — the exported
  engine is built and benchmarked on the same L40S from the checkpoint weights alone.
- **Every speed figure is reported with its SR**, and FP16/INT8/FP8 results quote **SR and
  latency degradation relative to FP32** — a precision that is faster but degrades task quality
  must be visible.
- **The speedup is mechanistic, not configuration.** The LeWM-vs-DINOv3 gap is an architectural
  asymmetry between the two models — LeWM's tiny scratch ViT-Tiny exposing a single latent token vs
  DINOv3's large backbone exposing the full patch-token grid. Batch and precision are held equal by
  construction, so neither may confound it; **GPU clock/thermal state cannot be held equal — clocks
  are unlockable (§Execution Environment) — so it is a named confound to the ratio's exact
  magnitude, bounded and disclosed (§Parity), not eliminated, and does not flip the qualitative
  direction**. Which component carries the gap is a result the profile reports, not one the spec
  fixes in advance — the encoder/predictor/overhead profile must attribute the gap to the right
  component.
- **FP8 rides the identical parity conditions as the other precisions** — same CEM config, seeds,
  normalization, L40S, and shared inference batch — so its degradation vs FP32 is the format alone,
  **at a fixed calibration method** (held constant across INT8 and FP8 for any labelled comparison —
  `docs/architecture.md` §7).
- **Calibration method (`max` | `entropy`) is a build option for both tracks, surfaced as a report
  label — not a hidden per-track setting.** Every quantized result records its method in
  `results.<track>.json`, so `max`- and `entropy`-calibrated points coexist and cross-track
  comparisons are drawn **like-for-like** (same method on both tracks), held constant across INT8 and
  FP8 within a labelled comparison so the format delta stays clean. **Existing artefacts are never
  rewritten** — a new method's runs are additive, separately-labelled points (CLAUDE §8). The
  cross-track **latency** headline is calibration-method-invariant regardless (scale values do not
  change quantized-op coverage, granularity, or TensorRT tactic selection); per-model SR is a
  quality-retention measure reported per (track, precision, method).
  **The label must survive into the persisted artefact, never stdout-only:** each single-method table
  is method-scoped by filename and states its method in the table body, and a dedicated
  `calibration_table.txt` carries both methods' SR side by side plus which method the headline was
  rendered at. A rendered table that does not name its method is not a valid artefact
  (`docs/architecture.md` §7).

---

## Implementation Boundaries (ownership by failure mode)

**OWNER-ONLY** — fails *silently* (plausible wrong number). Claude Code must STOP and ask before
touching:

- ONNX / Model-Optimizer PTQ / TensorRT export debugging.
- INT8/FP8 calibration set + Model-Optimizer PTQ config (quant mode, calibration method, Q/DQ
  format, per-channel-vs-per-tensor, op-type exclusions) + procedure; the FP32/FP16/INT8/FP8
  precision matching.
- The benchmark fairness conditions (matched precision, env/goal).
- The model adapter dims (the Constants above).
- Any change to the platform's eval/CEM config that would break the LeWM-vs-DINO parity.
- Whether the headline is the measured stock-hardware ratio (with disclosure) or a reframed
  presentation (e.g. deployment-cost), and which number is *the* headline — it touches the
  parity/mechanistic framing.
- The clock-normalization construction — the scaling model (`time ∝ 1/f_sm` → `T_ref = T ×
  f_measured/f_ref`), the reference clock `f_ref`, the per-run clock statistic feeding it, and which
  time is treated as clock-bound. A wrong choice is a plausible wrong corrected number (silent);
  CLAUDE Code wires the harvest/apply/render only once the owner has fixed the formula.
- The **confidence-interval construction** — which estimator carries each quantity (Clopper–Pearson
  for SR, the exact binomial order-statistic interval for p50), the rank convention, α, the
  permutation count, the choice of lag-1 autocorrelation as the independence statistic, the ruling
  that the test decision is the **unadjusted** p-value with Holm kept as secondary reporting, and
  what a rejected independence test licenses about the reported number. A wrong choice here is a
  plausible wrong interval — silent. CLAUDE Code wires the computation and the render once the owner
  has fixed the construction.
- **The clock-confound limitations write-up itself.** Deciding how the confound is framed, what the
  bound licenses, and which caveats are load-bearing is an *interpretation* of the measured result,
  not plumbing over it — the same reason the headline framing is owner-gated. CLAUDE Code produces
  the derived numbers, tables, and plot the write-up draws on; the owner writes the prose.

**CLAUDE CODE** — fails *loudly* (throws when wrong). Owns freely:

- Dockerfile, compose, uv/pyproject scaffolding.
- Hydra / W&B wiring around the platform entrypoints, including the owned W&B helper and the
  **observation-only** per-decision latency **callback** — injected through the platform's own
  config seam, so the vendored eval and solver stay byte-untouched. It may only read/record timing
  (bracketing each env's solve span with an optional `cuda.synchronize` barrier) and must leave
  seeds, sample draws, and the plan byte-identical; perturbing any of those crosses into the
  eval/CEM parity gate above and is OWNER-ONLY. **What the callback measures is a unit, not a free
  choice** — it must match the per-decision call counts the report weights by.
- The DINOv3 encoder config (model string; dims read from config).
- Export-script and benchmark-harness *plumbing* (trace call, Model-Optimizer PTQ invocation
  wiring — owner sets the quant config — TensorRT builder invocation, percentile timing, memory
  logging, the passive `nvidia-smi dmon` GPU-telemetry logging, table/plot runners).
- The tracer-bullet smoke script.
- The confidence-interval *computation and render plumbing* — evaluating the owner-set estimators
  and permutation test over the stored samples, the Holm secondary values, the `stats.json` write,
  the interval columns on the tables where SR and p50 appear, and the error bars on the speed-vs-SR
  plot. Off-pod over the existing canonical artifacts, read-only, like the derived clock render.
- The clock-confound derived-render *plumbing* — harvesting the per-run clock statistic from the
  telemetry, applying the **owner-set** normalization formula to the canonical latencies, and
  rendering the derived tables + throttle plot — off-pod over the existing canonical artifacts, like
  the current decoupled render. The normalization's construction, and the limitations write-up that
  interprets the output, are both OWNER-ONLY (above).

---

## Requirements

What the finished project must satisfy (ordered build steps live in `PLAN.md`).

- **Foundation runs:** `stable-worldmodel` installed and pinned; both reference trainings run on
  Push-T and produce checkpoints; DINOv3 confirmed to slot into the platform predictor cleanly.
- **Task baseline:** Push-T success rate + planning latency for both tracks via the platform's CEM
  evaluation, under matched conditions — the comparison baseline the optimization study builds on.
- **Integration (tracer bullet):** a checkpoint flows through the owned adapter → export stub →
  benchmark stub end-to-end on random/dummy weights, with typed checks passing at every owned
  boundary. This is the **sole pre-optimization integration check** and is kept strict.
- **Adapter-fidelity gate (before export):** because DINO-WM `predict` *reconstructs* the platform
  forward rather than calling it, the adapter's `encode` + `predict` + rollout/shim is validated
  against the platform's own rollout/`get_cost` on the **real checkpoint** (short-horizon drift
  within tolerance). A wrong `404` assembly, orientation, or a dropped proprio channel passes
  engine precision-match yet silently corrupts every SR — this gate catches it before any engine
  is built.
- **Engine-fidelity gate (before benchmarking):** exported FP32/FP16/INT8/FP8 engines are
  precision-matched against the PyTorch reference on the **real checkpoints** before any
  profiling/benchmark builds on them (INT8 and FP8 after their Q/DQ graphs are built from the
  calibration set).
  Drift is measured and reported only; the pass/fail is an **owner sign-off on the measured drift
  table** — deliberately **not** coded into a tolerance object or automated gate. The match must
  exercise the **off-nominal history windows the rollout actually feeds** (`T ∈ {1, 2}`, not only
  the traced `HS`). It is measured on **nominal, dataset-drawn inputs**, so it does **not**
  exercise the unbounded CEM action proposal that drives INT8 saturation — **INT8's calibration
  health is judged by SR, not by the drift table** (`docs/architecture.md` §7).
- **Speedup study:** both models exported PyTorch→ONNX→TensorRT with explicit-Q/DQ INT8 and FP8
  (FP32→FP16→INT8→FP8), benchmarked on the L40S as three equal-n p50/p95 latency distributions
  (**per-cycle headline**, encode-step + predictor-step components), plus peak GPU memory **and SR
  per precision**, with encoder/predictor/overhead profiled separately to locate bottlenecks. Only
  the model is TRT-optimized; the CEM planner stays in Python around it. Headline: the
  LeWM-vs-DINOv3 **per-cycle latency ratio at p50** (p95 reported alongside as the tail) +
  per-model FP32→FP16→INT8→FP8 delta in **both speed and SR** (degradation quoted vs FP32 — the p50
  speedup and the SR delta in the **same row**; speed plotted against SR). FP8 is a second
  explicitly-quantized 8-bit precision, native to the L40S; it augments every headline recording,
  table, and plot rather than forming a separate study.
  - **Headline-artifact durability.** The headline tables (serialized to text) **and** plots (PNG)
    are persisted to the persistent network volume, the same durability contract as checkpoints
    and engines. W&B logging is **additive, never the sole copy**. The **canonical** artifact is
    the raw **per-track results** (benchmark + profile numbers plus the run's fairness conditions —
    batch, seed); tables and plots are regenerable **views** of it. It is written **per track** so
    LeWM and DINOv3 can be benchmarked in separate pod sessions without clobbering each other, and
    it decouples the expensive pod-only benchmark from the cheap render — the report re-renders
    **off-pod** from the saved results, which is how the separately-gated SR-per-precision is
    joined in without re-running the L40S benchmark. A single-track render omits the two
    cross-track ratio plots, which need both tracks.
    - **Committed display copy (`reports/figs/`).** A small curated set of the rendered headline
      plots (PNG) is checked into the repo under `reports/figs/`. This is the **one exception to the
      never-in-git rule for artifacts** — and it holds only because these figures are a
      **regenerable, display-only view**, never the canonical artifact: the durable per-track
      results, tables, and full plot set stay on the network volume under `$STABLEWM_HOME/reports/`.
      The committed copy is refreshed by re-copying from a render, never hand-edited.
  - **Dilution disclosure (Amdahl).** Because only encoder+predictor are quantized and the Python
    overhead is precision-invariant, the per-precision wall-clock delta is capped by the model's
    share of the cycle. The study reports, per model: the FP32 baseline per-component time shares
    and the derived **optimizable fraction** `p = (encoder+predictor)/cycle`, which sets the
    Amdahl ceiling `1/(1−p)`; and per precision, **both** the *model-only* speedup and the
    *realized* speedup (measured FP32-vs-precision per-cycle ratio), whose gap is the overhead
    floor and should match `1/((1−p) + p/s)`. **This whole block is mean-based**, `p` included, so
    it is a **different number** from the reported p50 FP32-relative speedup — rendered in separate
    tables, never conflated or averaged. (`docs/architecture.md` §7, `docs/architecture.md` §8.)
- **FP8 delta:** FP8 (E4M3) built and benchmarked like INT8 on the L40S's native FP8 Tensor
  Cores, its speed/SR degradation quoted vs FP32, and its rows/points folded into the same
  headline recordings, tables, and plots as the other precisions.
- **Component-precision isolation (diagnostic, both tracks, `entropy`).** Where a quantized precision
  shows a material SR drop, the study must attribute it to the **encoder or the predictor**, not
  merely report it: the SR eval is re-run with ONE component quantized and the other held at FP16,
  two runs per affected (track, precision) cell. It is run at a **single calibration method
  (`entropy`)**, method-matched to the headline row it explains, and covers **both tracks** — a
  one-track isolation would argue the other track's innocence from absence of evidence.
  **It is a diagnostic, not a fifth precision:** mixed pairings are never benchmarked for latency,
  never entered in the headline ratio or the FP32→FP16→INT8→FP8 sweep, and never quoted as a
  recommended configuration. Results are recorded under composite `enc-<A>+pred-<B>` keys that cannot
  collide with a pure precision, so they are additive and the headline is unchanged by construction.
  Rendered as its own table immediately after the FP32-relative table. (`docs/architecture.md` §9.)
- **Uncertainty quantification.** Every reported absolute SR and absolute per-cycle p50 carries a 95%
  confidence interval — Clopper–Pearson for SR, the exact binomial order-statistic interval for p50 —
  computed from the samples already stored in `sr.json`, with **no additional eval, benchmark, or
  export run**. The p50 interval's i.i.d. premise is tested per eval run by a two-sided Dwass
  Monte-Carlo permutation test on lag-1 autocorrelation (α = 0.05, 50,000 permutations, fixed seed),
  the decision taken on the unadjusted p-value with Holm values persisted as secondary reporting. No
  interval is placed on any difference or ratio. The numbers land in a **`stats.json` on the
  persistent network volume** (same durability contract as the other headline artifacts) and are
  surfaced as interval columns in the tables where SR and p50 appear and as error bars on the
  speed-vs-SR plot. `results.*.json` and `sr.json` are **read-only** to this analysis and stay
  byte-unchanged. (`docs/architecture.md` §12.)
- **Clock-state confound disclosure.** Because GPU clocks cannot be locked (§Execution Environment)
  and the observed throttle is differential — one-sided on the heavier track, so it does not cancel
  in the ratio — the study discloses the confound and bounds it from the logged telemetry: the
  cross-model ratio plus its clock-normalized bound; the within-model precision deltas re-expressed
  at a common clock; and the overhead decomposition recomputed with components and cycle at matched
  clock. On the overhead surface a corrected value only exists where the overhead's measured share
  of the cycle (1−p) exceeds the cycle-vs-component clock mismatch; below that threshold the
  required disclosure is the **resolvability verdict** — not resolvable under unlocked clocks,
  bounded as small — never a corrected number. The derived numbers, tables, and throttle plot are
  **CLAUDE-owned plumbing**; the
  **limitations write-up that interprets them is OWNER-authored** (§Implementation Boundaries), since
  how the confound is framed is a judgement about the result, not a render of it. Both are **durable
  artifacts** persisted to the network volume like the other headline outputs (never git-only), and
  the normalized numbers are additive and labelled `derived` (§Parity). Unlike the headline outputs
  they are **not** mirrored to W&B — the volume copy is the only copy, so there is no second place
  for a derived number to be read as a measured one. A clock-locked re-run is deferred future work,
  unavailable on the current platform.

---

## Execution Rules

The general engineering rules (debugging cap, log-before-delete, never run git, tick-before-advance)
live in `CLAUDE.md` and govern here too. Project-specific caps:

- **Training is epoch-capped** — 10 epochs for both tracks, batch size 128 — not wall-clock-capped.
- **Lean on the platform; don't reimplement it.** If a need looks like training, env, CEM, or eval,
  it's the platform's — wire to it.
