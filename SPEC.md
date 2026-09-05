# SPEC: LeWM vs DINOv3-WM — Inference Optimization Study

The source of truth for **what** this project must satisfy. `README.md` carries how to run it;
`CLAUDE.md` holds behavioral rules; `src/interfaces.py` is the typed contract in code for the
owned export/benchmark layer. Non-obvious design rationale lives in `docs/architecture.md`.

---

## Objective

Compare two world models trained on Push-T (224×224, pixels-only) under reduced-precision
inference. **LeWM** (scratch ViT-Tiny + SIGReg) is a reference model from `stable-worldmodel`
used as-is. **DINOv3-WM** is *this project's* variant: the
platform's DINO-WM predictor (whose reference backbone is **DINOv2**) with the frozen backbone
swapped to **DINOv3** — the same training framework, a different encoder. The two differ on the
axes the study is built to separate: whether the encoder is co-trained with the predictor, and the
token granularity the predictions are made at.

Both models are exported PyTorch→ONNX→TensorRT with the 8-bit precisions quantized **explicitly**
(Q/DQ) and evaluated closed-loop on an L40S across FP32→FP16→FP8→INT8, at both calibration
methods. FP8 (E4M3) runs on the L40S's native FP8 Tensor Cores (Ada 4th-gen — no hardware or
toolchain change) and is an added precision, not a separate study or a head-to-head against INT8.

Four objectives:

1. **The effect of reduced precision on each model** — success rate and per-cycle planning
   latency, per engine configuration.
2. **Whether those effects generalise between the two models**, or are model-dependent.
3. **What the per-cycle latency decomposes into** — the encoder's and predictor's contributions
   and the residual overhead neither accounts for.
4. **Which component any task degradation is attributable to** — a model's encoder or its
   predictor, measured rather than inferred.

The training framework, Push-T env, CEM solver and MPC eval come from `stable-worldmodel` and
are used as they are, including for swapping DINOv3 into the DINO-WM predictor to set up the
comparison. The code written here starts at a trained checkpoint: export, quantization,
benchmarking, and the statistics computed over the recorded samples.

---

## Scope & Boundaries with the Platform

Layering rationale: `docs/architecture.md`.

- **Training, the Push-T env, the CEM solver, dataset tooling, and closed-loop MPC evaluation
  come from `stable-worldmodel`.** Wire to them; do not reimplement them.
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
  diverge from the reference DINO-WM construction and erase part of the architectural asymmetry
  the study measures.
- **LeWM = the platform's `lewm` training unchanged.** SIGReg and the scratch encoder are the
  platform's; do not reimplement or retune them beyond what training requires.
- **The owned code begins downstream of a trained checkpoint:** export, quantize, benchmark.
  That is where `src/interfaces.py` applies.

---

## Execution Environment

- **Inference runs on one machine: L40S** — both tracks are exported and benchmarked on the same
  L40S (same hardware class as the LeWM paper, so speed numbers are comparable).
- **GPU clocks cannot be locked on the benchmark platform.** `nvidia-smi -lgc` is denied by the
  RunPod virtualization layer (confirmed as root, persistence mode on), so both tracks run at stock
  boost/throttle behaviour and the per-run clock/thermal state is *recorded* by the telemetry
  observer (§Parity), not controlled. No latency is rescaled to a reference clock: the measured
  numbers are the only latency numbers.
- **Training hardware differs by track:** LeWM is trained on the L40S; DINOv3-WM is trained on an
  **H200 SXM**. Training hardware, like training batch size, does **not** carry into inference — the
  exported engine is built and benchmarked on the same L40S from the checkpoint weights alone — so
  the latency comparison is unaffected.
- **TensorRT engines are architecture-specific and disposable** — regenerated on the L40S,
  gitignored, never committed. They are **saved to and loaded from the network volume by
  default** (`$STABLEWM_HOME/engines/<track>/{encoder,predictor}.<precision>.plan`, the quantized
  int8/fp8 plans additionally tagged with the calibration method `…<precision>.<method>.plan` so
  the `max`/`entropy` engines coexist), so a pod session's built engines survive teardown. Living on the volume
  does not make them durable *artifacts* — they stay regenerable and are re-derived if the L40S
  class changes.
- **Durable artifacts persist on the persistent network volume, never in git:** datasets,
  checkpoints, exports, the rendered reports, and the **stored raw samples every later
  statistic is re-derived from** (`sr.json`'s per-cycle vectors and the per-track component-latency
  artefact) all live under the persistent root (`STABLEWM_HOME`, e.g. `$STABLEWM_HOME/reports/`).
  The root must point at the network volume or a multi-hour run's checkpoints are lost on pod restart.
- Datasets are loaded via the platform (streamed from object storage or cached under the root).
- **Runtime configuration lives in an uncommitted `.env`** at the repo root (`STABLEWM_HOME`,
  `WANDB_API_KEY`, `HF_TOKEN`; template in `.env.example`, gitignored), never typed into a shell.
  It is read once on `import src` and once on `import scripts` — the vendored train/eval
  entrypoints never import `src`, so the second hook is what keeps training off the platform's
  ephemeral default — and `setup.sh` reads it under the same rule, so a pod session, a
  `src.pipeline` subprocess, a training run and an off-pod render all resolve the same paths and
  secrets. A real environment variable still wins over the file, in the shell and in Python alike.
  **`STABLEWM_HOME` is required and raises when unset** — the platform's own default is the
  ephemeral container filesystem, so a silent fallback would put a multi-hour run's artifacts where
  a pod restart loses them.

The pinned dependency set (CUDA-12 export stack, TensorRT/modelopt out of uv) and the reasons for
those pins are recorded in `docs/architecture.md`; the executable form is
`pyproject.toml` + `setup.sh`. Concrete run commands live in `README.md`.

### W&B logging discipline

Every phase logs to the **same** W&B project. Training uses the platform's Lightning logger; the
non-training phases (eval/benchmark) open the run through an owned helper that reads the
project/entity from the same config block — no second source of truth.

---

## Interface Contracts (the owned code)

`src/interfaces.py` is the single source of truth **in code** for the boundaries the project
owns, runtime-checked via jaxtyping + beartype. This section states the **behavioral**
requirements those signatures must satisfy; it does not restate the signatures.

### Export shape

- **Export produces separate encoder and predictor engines.** The adapter's `encode` and
  `predict` are traced and built **separately** (one ONNX graph / TensorRT engine each), because
  the CEM rollout encodes once, caches the latent, then calls `predict` autoregressively for
  every candidate. FP32 and FP16 share the base graph (FP16 is a build flag); **INT8 and FP8 are
  each a separately quantized graph** with Q/DQ and scales baked in from a calibration pass
  (INT8 integer, FP8 E4M3 floating-point). **Activations are quantized per tensor, weights per
  channel** — weights have static ranges and need no calibration run, so their per-channel scales
  come from each channel's maximum absolute value, while activations must be per-tensor under
  TensorRT's explicit quantization.
  Only the **model** (encoder + predictor) is exported; the **CEM planner is never compiled in**
  — it stays in Python around the engines.
- **Each engine's optimization profile is its production call shape.** TensorRT selects tactics at
  the profile's `opt` point, so the **encoder** is built at batch **1** (`min = opt = max = 1`) —
  the CEM encodes the initial obs and the goal once per decision at `batch_size = 1` and expands
  the latent across candidates *after* encoding — and the **predictor** at `min = 1`,
  `opt = max = CEM_NUM_SAMPLES` (300), the candidate fan-out `batch_size × num_samples` every
  `predict` call carries. The profile is a **build** property, independent of the ONNX trace batch
  (which fixes only the shape the calibration pass feeds its ORT sessions); per-tensor PTQ scales
  are batch-independent, so the calibration path is unaffected by it.
  (`docs/architecture.md`.)
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
- **The calibration set must match the *inference-time* input distribution, not the expert one.**
  It is **512 clips, approximately uniformly spaced** across the Push-T dataset, so the activation
  distributions observed during the calibration run are representative of those at inference. The
  predictor stream additionally samples actions from the CEM proposal (`randn * var_scale`, zero
  mean, **unclamped** — a 10-dimensional `N(0, I)` vector per horizon step, `H = 5`) and rolls
  `predict` autoregressively so it consumes its own predicted latents; it is reproduced in the
  builder, never harvested from a live CEM/eval run. **Calibration health is judged by SR, not by
  the drift table.** (Full derivation and the two failure axes: `docs/architecture.md`.)
- **FP8 draws the identical calibration *streams* as INT8** (format-independent — the same
  encoder/predictor clips) and reuses whichever calibration *method* the engine was built with, so
  the INT8→FP8 step isolates the format. The **calibration method (`max` | `entropy`) is a build
  option available to both tracks** and a **labelled dimension of every quantized result**
  (`docs/architecture.md`). The FP8 path derives its scale from the clipping threshold `α` that
  method's INT8 run produced, as `α / 448` (the largest representable E4M3 value), so the choice
  carries over unchanged. Calibration health is likewise judged by SR, not the drift table.

### Latency & profiling

- **Every speed result carries the SR for that engine config** — no speed number is reported
  without its task-quality counterpart. **Per-cycle planning latency is the headline speed
  measure**, reported p50/p95 and **compared at p50**.
- **Three statistics, three jobs — never substituted for one another** (`docs/architecture.md`):
  **p50** is the comparison basis; **p95** is the reported tail and carries no claim; **mean** is
  the decomposition basis **only** and is never a comparison.
- **Every reported ABSOLUTE SR and ABSOLUTE per-cycle p50 carries a 95% confidence interval; no
  difference or ratio carries one — the mean decomposition's residual overhead is the single named
  exception.** SR uses the **Clopper–Pearson** exact binomial interval over the 50 eval episodes
  (exact at the 0/50 and 50/50 boundaries the study actually hits); the per-cycle p50 uses the
  **exact binomial order-statistic** interval, computed from the *same* warm-up-dropped,
  equal-n-truncated sample the reported p50 is computed from — never a re-derived one. ΔSR, any
  speedup, and any cross-model ratio get **no** interval, and the p95 of any distribution gets none
  (it carries no claim).
  The order-statistic interval's **i.i.d. premise is tested, not assumed**: every sample it is
  computed over gets a two-sided **Dwass Monte-Carlo permutation test on its lag-1
  autocorrelation** (α = 0.05, 50,000 permutations, no Student-t transform). A rejected test is
  **disclosed as a flag beside the interval, never silently corrected** — serial correlation makes
  the interval too narrow, which reads as a stronger result than the sample supports. **The interval
  artefact is self-describing**: every point records the **n it was computed over**, the observed
  lag-1 statistic, both the asymptotic and the permutation p-value, the permutation null's own
  lag-1 summary, and the **fixed, recorded permutation seed** — so an interval can be re-derived
  and audited off the artefact rather than taken on trust. (`docs/architecture.md`.)
- **The five MEAN latency quantities carry a 95% non-parametric BOOTSTRAP interval, per
  configuration (track × precision × calibration method).** The **two components are reported per
  ENGINE CALL — unweighted, on the scale their fixed-iteration loops time them**: `enc =
  mean(encode-step)`, `pred = mean(predictor-step)`, so each cell states the latency of the thing it
  names. The CEM call counts enter only the **three per-cycle composites**: `t_comp =
  ENCODER_CALLS × enc + PREDICTOR_CALLS × pred`, the **measured cycle mean**, and the residual
  `overhead = cycle − t_comp`. The table's two scales are **labelled in the header and the body**,
  never inferred. The estimator is **scipy's `bootstrap`, percentile method, 3,000 resamples,
  `paired=False`, α = 0.05, fixed recorded seed**, over the **same stored samples the p50 intervals
  use**: the fixed-iteration loop vectors as recorded for the components — **that configuration's
  own method's**, since the quantized engines differ per method — and the warm-up-dropped,
  equal-n-truncated sample for the cycle. `paired=False` is load-bearing and correct — the three
  samples come from different runs of different length and carry no pairing, so each is resampled
  independently.
  **The residual `overhead` is the ONE interval on a difference this spec permits**, and only
  because it is the decomposition of measured absolute times into an absolute floor, not a
  comparison claim; it is never used to argue a difference *between* configurations, and the
  exception generalizes to nothing else.
  **No new independence test.** `enc`, `pred` and `cycle` carry the flag of their constituent
  sample's already-run lag-1 test (same vector, same seed, same result), reported **beside the
  interval in the same cell**; `t_comp` and `overhead` carry no marker of their own, because a flag
  belongs to a sample and these are functions of two and three of them. The bootstrap rests on the
  *same* i.i.d. premise, so a flagged interval here is anti-conservative for the same reason and is
  likewise disclosed, never corrected. Composite `enc-<A>+pred-<B>` isolation labels get no row —
  they are an SR diagnostic and are never benchmarked for latency.
  Persisted as a `points_means` section in `stats.json` (self-describing: the n of each sample, the
  call counts, the bootstrap method/B/seed) and rendered as `latency_means_table.txt` — **method-
  unscoped**, because its config column names the method (`FP32`, `INT8 (max)`, `FP8 (entropy)`).
  (`docs/architecture.md`.)
- **A "cycle" is ONE episode's decision, not the span of one `CEMSolver.solve` call.** A solve
  plans every still-alive episode sequentially, so its wall clock is the sum of N decisions. The
  latency callback therefore brackets **per env** (consecutive `start_batch` hooks, the last
  closing at `end_solve`) and **never `reset → end_solve`**. Bracketing per solve while weighting
  by per-decision call counts inflates the residual overhead silently and would make the headline
  scale with how many episodes are alive — SR-dependent, therefore track-dependent: a parity break.
  (`docs/architecture.md`.)
- **Per-cycle n is 50–100 per track per precision, not thousands** — establish this before
  reading any per-cycle percentile. It is SR-dependent hence track-dependent, which is what the
  equal-n truncation neutralises. This n is why p50 carries the comparison and p95 does not.
  **The n each percentile was computed from is reported in the artefact**, not merely asserted: the
  equal-n truncation must be verifiable off the table rather than taken on trust.
- **Three latency distributions**, mapping to the three profile slices. The per-cycle distribution
  is **reported as p50/p95**; the two components are reported as **means** on the decomposition
  surface (above), which is the scale their contribution to a cycle is defined on. A mean is never
  *compared*.
  1. **per-cycle** — the **headline**, compared at p50: full per-decision planning latency
     (encode + predict + residual overhead), measured on the real solve.
  2. **encode-step** — a component exposing the LeWM-vs-DINOv3 encoder token-count asymmetry
     (diluted in the cycle because encode is cached and runs twice).
  3. **predictor-step** — a component: quantization's kernel target, and the call the CEM rollout
     makes 150× per decision.
  The component distributions are harvested from **fixed-iteration** loops (100 timed iters per
  step, 10 warm-up iters dropped), so n is equal across tracks and neither loop is
  boundary-censored. **Those loops' raw per-call latencies are PERSISTED, not just their summary** —
  a durable per-track artefact beside the canonical results, keyed by **(precision, calibration
  method)** like every other quantized result and merged per cell, exactly as the per-cycle vectors
  are persisted in `sr.json` — so a statistic over the component samples is re-derivable and
  auditable off the artefact rather than taken on trust, and computing one needs no L40S.
  Encode-/predict-step ride isolated per-precision engine loops; per-cycle rides the
  observation-only per-decision latency callback over the SR eval-shim run, so **per-cycle latency
  and SR come from the same solves**. The per-cycle vector **drops a warm-up head (k = 1 decision by
  default) before truncation**, so the cycle and the engine steps are measured under the same
  warm-up regime and `overhead = cycle − t_comp` subtracts like from like; the drop is applied at
  **report** time (the recorded vector stays complete) and the excluded decision is **disclosed**
  in the speed table, never silently removed (`docs/architecture.md`). Per-cycle samples are then
  truncated to a common minimum n for an equal-n comparison; the reported p50/p95 **and** the
  decomposition mean are taken off that *same* reduced sample. A dedicated latency-only pass is
  **not** an alternative — it would break the same-solves pairing.
- **The per-component profile slices are mutually exclusive and additive, by subtraction from the
  measured cycle — and the decomposition runs on MEANS, not percentiles.** The cycle is measured
  on the real CEM solve; encoder and predictor are timed in isolation and weighted by their real
  per-cycle call counts (**confirmed against the installed `CEMSolver.solve`, not assumed** —
  `ENCODER_CALLS_PER_CYCLE = 2`, `PREDICTOR_CALLS_PER_CYCLE = ((horizon − n_obs) + 1) × n_steps =
  150`). **Those counts are per-decision, so the measured cycle must be per-decision too.** The
  remainder is the residual `overhead = cycle − t_comp` — the un-optimizable floor (CEM
  sampling/topk/mean-var, the criterion, the 384→404 assembly, per-step action-replace/
  proprio-carry, and host/Python glue). A negative residual is **surfaced loudly** — never clamped.
  (`docs/architecture.md`.)

### Adapter, rollout & shim

- **One adapter Protocol, two concrete tracks.** A common two-method `encode`/`predict` interface
  (not a fused `__call__`) so export and benchmark treat both tracks identically; the latent shape
  and how the action enters `predict` differ per track (LeWM: a separate AdaLN-conditioning
  argument; DINO-WM: concatenated onto the predictor embedding — see Constants).
- **DINO-WM `predict` is a faithful, dim-preserving `404 → 404` reconstruction of the platform
  predictor.** It is **not** sliced back to 384: the predicted **proprio** must survive. The
  extras embedding, the initial `384 → 404` assembly, and the per-step action-replacement +
  proprio-carry live in the **Python rollout/shim**, not the compiled engine.
  (`docs/architecture.md`.)
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
  (`docs/architecture.md`.)
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
    by `tests/test_sr_shim.py`. (`docs/architecture.md`.)
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
  (300 samples, 30 elites, horizon 5, init variance 1, 30 iterations), same action budget, goal
  encoding, eval seeds, and identical input normalization (ImageNet stats). Mostly enforced by the
  platform's eval; confirm they are not varied between tracks.
- **Matched export/benchmark conditions:** both models exported and benchmarked at the **same
  precision** on the **same L40S**, same env/goal, and the **same per-component inference batch on
  both tracks** — encoder 1, predictor `CEM_NUM_SAMPLES` — matching the optimization profile each
  engine is built at (§Interface Contracts), so batch cannot confound the cross-track gap.
  The model is the only difference, so the **per-cycle latency gap is the measured result**. The
  component distributions are measured in equal-n fixed-iteration loops (§Interface Contracts).
- **GPU clocks are not locked.** A passive `nvidia-smi dmon` observer logs per-sample GPU
  telemetry (SM/mem clock, power, temperature, utilization, memory) alongside every timed engine
  run — both the isolated component loops and the per-cycle eval-shim run — so the actual per-run
  clock/thermal state is recorded rather than assumed. Its logs persist to
  `$STABLEWM_HOME/reports/phase5/gpu_logs/` (same durability contract as the reported artifacts).
  The observer never touches seeds, samples, or the plan. (`docs/architecture.md`.)
- **Confidence intervals are RE-ANALYSIS of the stored samples, never a new measurement.** They are
  computed off the stored sample artefacts alone — `sr.json` (SR + per-cycle vectors) and the
  per-track component-latency artefact — with no eval, benchmark, or export run, and are
  **additive**: they land in their own `stats.json` (intervals on the absolute SR/p50s **and** on the
  five mean latency quantities) plus columns/error bars on the regenerable views and the
  method-unscoped `latency_means_table.txt`,
  and `results.*.json`, `sr.json` and the component-latency artefact stay **byte-unchanged**.
- Training batch size is held equal across tracks (128, the LeWM baseline's value); training **hardware**
  is not (LeWM on the L40S, DINOv3-WM on an H200 SXM). Neither carries into inference — the exported
  engine is built and benchmarked on the same L40S from the checkpoint weights alone.
- **Every speed figure is reported with its SR**, and reduced-precision results are read against
  FP32 in **both** speed and task quality — a precision that is faster but degrades task quality
  must be visible on the same row.
- **The cross-model gap is mechanistic, not configuration.** Any LeWM-vs-DINOv3 difference is an
  architectural asymmetry between the two models — LeWM's tiny scratch ViT-Tiny exposing a single
  latent token vs DINOv3's large backbone exposing the full patch-token grid. Batch and precision
  are held equal by construction, so neither may explain it. GPU clock and thermal state cannot be
  held equal (§Execution Environment), so they bear on the exact magnitude; the telemetry records
  what each run ran at. Which component carries the gap is a result the profile reports, not one
  the spec fixes in advance.
- **FP8 rides the identical parity conditions as the other precisions** — same CEM config, seeds,
  normalization, L40S, and shared inference batch — so its degradation vs FP32 is the format alone,
  **at a fixed calibration method** (held constant across INT8 and FP8 for any labelled comparison —
  `docs/architecture.md`).
- **Calibration method (`max` | `entropy`) is a build option for both tracks, surfaced as a report
  label — not a hidden per-track setting.** **Every quantized result is keyed by (track, precision,
  method) — the SR in `sr.json` and, equally, the measured latencies in `results.<track>.json` and
  the component-latency artefact** — so `max`- and `entropy`-calibrated points coexist and
  cross-track comparisons are drawn **like-for-like** (same method on both tracks), held constant
  across INT8 and FP8 within a labelled comparison so the format delta stays clean. **Existing
  artefacts are never rewritten** — a new method's runs are additive, separately-labelled points
  (CLAUDE §8). An INT8/FP8 engine is a **per-method build**, so the latencies timed off it are that
  method's measurement and **never stand in for the other method's**: a render selects one method's
  points, and a configuration that method never timed is reported absent rather than filled from the
  other. FP32/FP16 build data-free — one engine, timed once — so their numbers are read across method
  labels rather than duplicated per method. Per-model SR is a quality-retention measure reported per
  (track, precision, method).
  **The label must survive into the persisted artefact, never stdout-only:** each single-method table
  is method-scoped by filename and states its method in the table body; the method-unscoped mean
  latency table names the method on every row instead. A rendered table that does not name its
  method is not a valid artefact (`docs/architecture.md`).

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
- Which number carries a claim, and how a result is framed — it touches the parity/mechanistic
  argument, not the plumbing under it.
- The **confidence-interval construction** — **which quantities carry an interval at all** (the
  absolute SR, the per-cycle p50, and the five mean latency quantities — `enc`, `pred`, `t_comp`,
  `cycle`, `overhead`; never a p95, and no difference or ratio other than the named `overhead`
  decomposition), which estimator carries each (Clopper–Pearson for SR, the exact binomial
  order-statistic interval for the p50, the non-parametric **percentile bootstrap** — B = 3,000,
  `paired=False`, α = 0.05, fixed seed — for every mean), **the scale each mean is reported on**
  (the two components per engine call, unweighted; the three composites call-count-weighted onto the
  per-cycle scale), the ruling that the mean intervals **inherit** their constituent sample's lag-1
  flag rather than opening a new test family and that the two composites (`t_comp`, `overhead`)
  carry no flag of their own, **which sample each interval is computed over**
  (per-cycle: warm-up-dropped and equal-n-truncated; component: the fixed-iteration loop sample as
  recorded), the rank convention, α, the permutation count, the choice of lag-1 autocorrelation as
  the independence statistic, and what a rejected independence test licenses about the reported
  number. A wrong choice here is a plausible wrong interval — silent. CLAUDE Code wires the
  computation and the render once the owner has fixed the construction.

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
  wiring — owner sets the quant config — TensorRT builder invocation, percentile timing, the passive
  `nvidia-smi dmon` GPU-telemetry logging, table/plot runners).
- The tracer-bullet smoke script.
- The confidence-interval *computation and render plumbing* — evaluating the owner-set estimators
  and permutation test over the stored samples, the bootstrap over the mean quantities, the
  `stats.json` write, the interval columns on the tables where SR and the p50 appear, the
  `latency_means_table.txt` render, and the error bars on the speed-vs-SR figure. Off-pod over the
  existing canonical artifacts, read-only.
- **Retaining and persisting the timing loops' raw per-call latencies** (the component-latency
  artefact + its merge/no-clobber discipline). Record-and-write plumbing only: it must not change
  what is timed, the iteration count, the warm-up, the batches, or the reported percentiles — those
  are the benchmark methodology above. A capture that altered the measurement would fail the
  percentile agreement check, loudly.
- The GPU-telemetry *render plumbing* — parsing the `dmon` logs into per-run summaries and
  rendering the throttle diagnostic beside them, off-pod over the existing artifacts. It derives no
  corrected latency.

---

## Requirements

What the finished project must satisfy; `README.md` carries the ordered run commands.

- **Platform runs:** `stable-worldmodel` installed and pinned; both reference trainings run on
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
  the traced `HS`), and each engine at **its own profile points** — the encoder at its single
  production batch, the predictor at profile min, the trace batch, and `opt`/`max`.
  It is measured on **nominal, dataset-drawn inputs**, so it does **not**
  exercise the unbounded CEM action proposal that drives INT8 saturation — **INT8's calibration
  health is judged by SR, not by the drift table** (`docs/architecture.md`).
- **Precision study:** both models exported PyTorch→ONNX→TensorRT with explicit-Q/DQ INT8 and FP8
  (FP32→FP16→FP8→INT8), benchmarked on the L40S as the equal-n per-cycle p50/p95 distribution plus
  the two engine-step component distributions, **and SR per precision**, with encoder/predictor/
  residual overhead profiled separately to locate where the cycle time goes. Only the model is
  TRT-optimized; the CEM planner stays in Python around it. Every precision is read against FP32 in
  **both speed and SR** — the p50 and the SR of a configuration in the **same row**; speed plotted
  against SR. FP8 is a second explicitly-quantized 8-bit precision, native to the L40S; it augments
  every recording, table, and figure rather than forming a separate study.
  - **Artifact durability.** The reported tables (serialized to text) **and** the figure (PNG) are
    persisted to the persistent network volume, the same durability contract as checkpoints
    and engines. W&B logging is **additive, never the sole copy**. The **canonical** artifact is
    the raw **per-track results** (benchmark + profile numbers plus the run's fairness conditions —
    batch, seed); tables and figures are regenerable **views** of it. A **companion per-track
    component-latency artefact** carries the engine-step loops' raw per-call samples — written
    **beside** the results file, never replacing it or folded into it, so the schema every consumer
    parses stays summary-numeric while the samples stay re-analysable. It is written **per track** so
    LeWM and DINOv3 can be benchmarked in separate pod sessions without clobbering each other, and —
    like the results file — keyed **per (precision, calibration method)** and merged per cell, so
    timing one method's quantized engines lands beside the other's rather than over it. It
    decouples the expensive pod-only benchmark from the cheap render — the report re-renders
    **off-pod** from the saved results, which is how the separately-gated SR-per-precision is
    joined in without re-running the L40S benchmark.
    - **Committed display copy (`reports/figs/`).** The rendered speed-vs-SR figure (PNG) is checked
      into the repo under `reports/figs/`. This is the **one exception to the never-in-git rule for
      artifacts** — and it holds only because it is a **regenerable, display-only view**, never the
      canonical artifact: the durable per-track results and tables stay on the network volume under
      `$STABLEWM_HOME/reports/`. The committed copy is refreshed by re-copying from a render, never
      hand-edited.
- **FP8 delta:** FP8 (E4M3) built and benchmarked like INT8 on the L40S's native FP8 Tensor
  Cores, its speed/SR degradation read against FP32, and its rows/points folded into the same
  recordings, tables, and figure as the other precisions.
- **Component-precision isolation (diagnostic, both tracks, both calibration methods).** Where a
  quantized precision shows a material SR drop, the study must attribute it to the **encoder or the
  predictor**, not merely report it: the SR eval is re-run with ONE component quantized and the
  other held at FP16, two runs per affected (track, precision, **method**) cell. It is run at **both
  calibration methods (`max` and `entropy`)** and covers **both tracks**. Each run is method-matched
  to the row it explains and the isolation table is method-scoped like every other single-method
  table, so a row only explains a row rendered at the **same** method: isolation points are
  quantized results, never method-invariant, and never fall back across methods. A single-method
  isolation would leave the other method's drops unattributed — the same argument that forbids a
  one-track isolation, which would argue the other track's innocence from absence of evidence.
  **It is a diagnostic, not a fifth precision:** mixed pairings are never benchmarked for latency,
  never entered in the FP32→FP16→FP8→INT8 sweep, and never quoted as a recommended configuration.
  Results are recorded under composite `enc-<A>+pred-<B>` keys that cannot collide with a pure
  precision, so they are additive and the reported sweep is unchanged by construction.
  Rendered as its own table immediately after the speed table. (`docs/architecture.md`.)
- **Uncertainty quantification.** Every reported absolute SR and absolute per-cycle p50 carries a
  95% confidence interval — Clopper–Pearson for SR, the exact binomial order-statistic interval for
  the p50 — computed from the samples already stored on the volume (`sr.json` for SR + per-cycle,
  the per-track component-latency artefact for the two engine-step distributions, each per (track,
  precision, method)), with **no additional eval, benchmark, or export run**. Every p50 interval's
  i.i.d. premise is tested on its own sample by a two-sided Dwass Monte-Carlo permutation test on
  lag-1 autocorrelation (α = 0.05, 50,000 permutations, fixed seed), and a rejection is disclosed
  as a flag, never silently corrected.
  **The mean latencies carry intervals too** — the per-call component means `enc` and `pred` and the
  call-count-weighted per-cycle composites `t_comp = ENCODER_CALLS × enc + PREDICTOR_CALLS × pred`,
  `cycle`, and the residual `overhead = cycle − t_comp`, per (track, precision, method), by
  non-parametric **percentile bootstrap** (3,000 resamples, `paired=False`, fixed seed) over those
  same stored samples, with the constituent sample's lag-1 flag carried into the cell and no new
  test family opened. Apart from that named `overhead` decomposition, no interval is placed on any
  difference or ratio, or on any p95. The numbers land in a **`stats.json` on the persistent network
  volume** (same durability contract as the other artifacts) and are surfaced as interval columns in
  the tables where SR and the p50 appear, as the method-unscoped **`latency_means_table.txt`**, and
  as error bars on the speed-vs-SR figure. `results.*.json`, `sr.json` and the component-latency
  artefact are **read-only** to this analysis and stay byte-unchanged. (`docs/architecture.md`.)
- **GPU telemetry.** Because clocks cannot be locked and no thermal steady state is established
  (§Execution Environment), the per-run clock and thermal state is recorded: a passive
  `nvidia-smi dmon` observer logs alongside every timed run, and the logs are reduced to per-run
  summaries and a throttle diagnostic. **No clock-normalized latency is reported** — the measured
  numbers are the only latency numbers (§Parity). The same unlocked state is why the intervals'
  i.i.d. premise is tested rather than assumed. Telemetry and its diagnostic persist to the network
  volume like the other outputs and are **not** mirrored to W&B.

---

## Execution Rules

The general engineering rules (debugging cap, log-before-delete, never run git, tick-before-advance)
live in `CLAUDE.md` and govern here too. Project-specific caps:

- **Training is epoch-capped** — 10 epochs for both tracks, batch size 128 — not wall-clock-capped.
- **Lean on the platform; don't reimplement it.** If a need looks like training, env, CEM, or eval,
  it's the platform's — wire to it.
