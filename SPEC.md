# SPEC: LeWM vs DINOv3-WM — Inference Optimization & QLoRA Study

The source of truth for **what** this project must satisfy. `PLAN.md` carries the ordered
execution steps and progress; `CLAUDE.md` holds behavioral rules; `src/interfaces.py` is the
typed contract in code for the owned export/benchmark/QLoRA layer. Design rationale that runs
longer than a couple of sentences lives in `docs/` (see `docs/adr/`).

---

## Objective

Compare two world models trained on Push-T (224×224, pixels-only) and deliver the engineering
layer the platform does **not** provide. **LeWM** (scratch ViT-Tiny + SIGReg) is a reference
model from `stable-worldmodel` used as-is. **DINOv3-WM** is *this project's* variant: the
platform's DINO-WM predictor (whose reference backbone is **DINOv2**) with the frozen backbone
swapped to **DINOv3** — the same training framework, a different encoder.

1. **Inference-optimization study on an L40S:** export both models PyTorch→ONNX→TensorRT with
   INT8 quantized **explicitly** (Q/DQ), and benchmark planning latency, throughput, and peak
   GPU memory across FP32→FP16→INT8. Headline: the **LeWM-vs-DINOv3 speedup ratio** and the
   **per-model FP32→FP16→INT8 optimization delta**.
2. **QLoRA delta on the DINOv3-WM backbone:** fine-tune the frozen DINOv3 backbone with QLoRA
   on Push-T, re-run the task-quality metric, and report the delta vs the frozen baseline.

**Non-goal:** the training framework, Push-T env, CEM solver, and MPC eval are **provided by
`stable-worldmodel`** — a foundation, not the contribution. Running them, including swapping
DINOv3 into the DINO-WM predictor to set up the comparison, is foundation work; the owned
contribution is the optimization + QLoRA layer above a trained checkpoint.

---

## Scope & Boundaries with the Platform

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
  diverge from the paper and erase part of the encoder-compute asymmetry the study measures.
- **LeWM = the platform's `lewm` training unchanged.** SIGReg and the scratch encoder are the
  platform's; do not reimplement or retune them beyond what training requires.
- **The contribution lives downstream of a trained checkpoint:** export, quantize, benchmark,
  and QLoRA-tune. That is where `src/interfaces.py` and the owned code apply.

---

## Execution Environment

- **Single machine: L40S** — train and benchmark on the same instance (same hardware class as
  the LeWM paper, so speed numbers are comparable).
- **TensorRT engines are architecture-specific and disposable** — regenerated on the L40S,
  never committed.
- **Durable artifacts persist on the persistent network volume, never in git:** datasets,
  checkpoints, exports, and the Phase-5 headline reports all live under the persistent root
  (`STABLEWM_HOME`, e.g. `$STABLEWM_HOME/reports/`). The root must point at the network volume
  or a multi-hour run's checkpoints are lost on pod restart.
- Datasets are loaded via the platform (streamed from object storage or cached under the root).
- Runtime secrets/env (`WANDB_API_KEY`, `HF_TOKEN`, `STABLEWM_HOME`) are passed at runtime.

The pinned dependency set (CUDA-12 export stack, TensorRT/modelopt out of uv) and the reasons
for those pins are recorded in `docs/adr/0001-cuda12-export-stack.md`; the executable form is
`pyproject.toml` + `setup.sh`. Concrete run commands live in `PLAN.md`.

### W&B logging discipline

Every phase logs to the **same** W&B project. Training uses the platform's Lightning logger;
the non-training phases (eval/benchmark/QLoRA) open the run through an owned helper that reads
the project/entity from the same config block — no second source of truth.

---

## Interface Contracts (the owned layer)

`src/interfaces.py` is the single source of truth **in code** for the boundaries the project
owns, runtime-checked via jaxtyping + beartype. This section states the **behavioral**
requirements those signatures must satisfy; it does not restate the signatures.

- **Export produces separate encoder and predictor engines.** The adapter's `encode` and
  `predict` are traced and built **separately** (one ONNX graph / TensorRT engine each). The CEM
  rollout encodes the observation **once**, caches the latent, then calls `predict`
  autoregressively over the horizon for every candidate — a single fused `obs → latent` graph
  could not reproduce that call pattern and would re-encode on every predictor call, inflating
  encoder cost and erasing the encoder-cached / predictor-dominates asymmetry the study measures.
  FP32 and FP16 share the base graph (FP16 is a build flag); **INT8 is a separately quantized
  graph** with Q/DQ and per-tensor scales baked in from a calibration pass. Only the **model**
  (encoder + predictor) is exported; the **CEM planner is never compiled in** — it stays in
  Python around the engines.
- **"INT8" means INT8 + FP16.** The Model Optimizer quantizes only the heavy layers
  (MatMul/Gemm/Conv) to INT8 and casts the non-quantized remainder to FP16, so the engine builds
  with both flags set. This is the realistic TRT INT8 deployment, and it makes the INT8-vs-FP16
  delta the marginal benefit of pushing the heavy layers to INT8 on the same FP16 backbone.
  Owner-accepted trade-off: the FP16 cast of the remainder can under/overflow a few initializers;
  keeping the remainder FP32 is the documented fallback if that drift proves unacceptable.
- **Every speed result carries the SR for that engine config** — no speed number is reported
  without its task-quality counterpart. Rollouts-completed under a fixed wall-clock budget is the
  headline speed measure.
- **Peak memory is sampled from the driver/runtime** (`cudaMemGetInfo`/nvidia-smi), **not** the
  torch caching allocator: TensorRT's engine and execution-context device allocations bypass
  torch's allocator, so `torch.cuda.max_memory_allocated` would systematically undercount exactly
  the optimized path.
- **The per-component profile slices are mutually exclusive and additive.** `planner_ms` is the
  pure CEM/Python overhead with the encode/predict call time subtracted, so encoder + predictor +
  planner sum to the measured cycle within the `cuda.synchronize` barrier. Only then are the FP32
  baseline **time shares** meaningful — and they are load-bearing for the dilution disclosure
  below.
- **One adapter Protocol, two concrete tracks.** A common two-method `encode`/`predict` interface
  (not a fused `__call__`) so export and benchmark treat both tracks identically; the latent shape
  and how the action enters `predict` differ per track (LeWM: a separate AdaLN-conditioning
  argument; DINO-WM: concatenated onto the predictor embedding — see Constants).
- **DINO-WM `predict` is a faithful, dim-preserving `404 → 404` reconstruction of the platform
  predictor.** It is **not** sliced back to 384: the predicted **proprio** must survive, because
  the CEM criterion scores predicted proprio *and* pixels against the goal and the autoregressive
  state carried across the horizon is the full `404`. The extras embedding, the initial
  `384 → 404` assembly, and the per-step action-replacement + proprio-carry live in the **Python
  rollout/shim**, not the compiled engine.
- **LeWM `predict`'s action conditioning is per-frame, so the per-step engine boundary is
  faithful.** LeWM's action encoder has no receptive field along the macro-step axis, so a
  per-step `predict` is numerically identical to the platform's whole-sequence pre-encode, and the
  action encoder may live inside the compiled per-step engine. This is an **owner-gated
  silent-failure boundary**: a temporal (kernel > 1) action-encoder config would make the per-step
  boundary wrong with no error, so it is guarded by a runtime assertion on the real checkpoint.
- **SR-per-precision re-runs the platform eval on the optimized model.** The CEM solver calls the
  world model via `get_cost`/`get_action`, not `encode`/`predict`. So to produce the SR that pairs
  with each precision's speed number, the exported adapter is re-wrapped in a thin **Python** shim
  exposing `get_cost`/`get_action` (which call the engine's `encode`/`predict` underneath) and
  slotted into the CEM solver, letting the platform's eval logic re-run unchanged. The shim stays
  in Python; only `encode`/`predict` lives in the engine. For DINO-WM the shim must reproduce the
  platform rollout faithfully (full `404` carry, per-step action-replace, proprio+pixels cost) or
  the SR is not comparable to the Phase-3 baseline — a silent parity break.
  - **Injection seam (the one specified exception to the no-monkeypatch eval stance).** Model
    injection has no platform config seam, so the SR re-run uses a dedicated driver that slots the
    shim in by a **scoped patch of the checkpoint loader** around the run — swapping only which
    model the loader returns. The vendored eval entrypoint and the solver/CEM logic stay
    byte-unmodified, and no CEM config, seed, sample count, or plan changes, so eval/CEM parity is
    preserved (the SR differs from the FP32 baseline only by the engines' quantization drift).
    Because it touches the model boundary the eval runs on, this seam is **owner-gated** (see
    Implementation Boundaries — eval/CEM parity).

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
  precision** on the **same L40S** under the **same fixed wall-clock budget**, same env/goal, and
  the **same shared inference batch size**. Within that budget, compare per-step inference latency
  (**p50 and p95**) and the **number of CEM rollouts completed** — rollout count is the intended
  degree of freedom; the model is the only other difference. Training batch size is held equal
  across tracks (128, LeWM's paper value) and does **not** carry into inference. **Every speed
  figure is reported with its SR**, and FP16/INT8 results quote **SR and latency degradation
  relative to FP32** — a precision that is faster but degrades task quality must be visible.
- **The speedup is mechanistic, not configuration.** The LeWM-vs-DINOv3 gap comes from the
  encoder-compute asymmetry — LeWM's tiny scratch ViT-Tiny exposing a single latent token vs
  DINOv3's large backbone exposing the full patch-token grid (so the predictor and planner also
  operate over `N_patches` tokens for DINO vs one for LeWM). No batch or precision mismatch may
  confound it; the encoder/predictor/planner profile must attribute the gap to the right component.
- **QLoRA comes after the frozen baseline**, and the delta is reported against frozen DINOv3-WM.

---

## Implementation Boundaries (ownership by failure mode)

**OWNER-ONLY** — fails *silently* (plausible wrong number). Claude Code must STOP and ask before
touching:

- ONNX / Model-Optimizer PTQ / TensorRT export debugging.
- INT8 calibration set + Model-Optimizer PTQ config (calibration method, Q/DQ format,
  per-channel-vs-per-tensor, op-type exclusions) + procedure; the FP32/FP16/INT8 precision
  matching.
- QLoRA targeting (which DINOv3 modules, rank, what stays frozen — the predictor is unfrozen and
  co-trained, so only backbone targeting is open).
- The benchmark fairness conditions (matched precision, fixed time budget, env/goal).
- The model adapter dims (the Constants above).
- Any change to the platform's eval/CEM config that would break the LeWM-vs-DINO parity.

**CLAUDE CODE** — fails *loudly* (throws when wrong). Owns freely:

- Dockerfile, compose, uv/pyproject scaffolding.
- Hydra / W&B wiring around the platform entrypoints, including the owned W&B helper and the
  **observation-only** CEM-solve-latency **callback** — injected through the platform's own config
  seam, so the vendored eval and solver stay byte-untouched. It may only read/record timing
  (bracketing one CEM solve with an optional `cuda.synchronize` barrier) and must leave seeds,
  sample draws, and the plan byte-identical; perturbing any of those crosses into the eval/CEM
  parity gate above and is OWNER-ONLY.
- The DINOv3 encoder config (model string; dims read from config).
- Export-script and benchmark-harness *plumbing* (trace call, Model-Optimizer PTQ invocation
  wiring — owner sets the quant config — TensorRT builder invocation, percentile timing, memory
  logging, table/plot runners).
- The QLoRA training-loop wiring (owner specifies the targeting config).
- The tracer-bullet smoke script.

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
- **Engine-fidelity gate (before benchmarking):** exported FP32/FP16/INT8 engines are
  precision-matched against the PyTorch reference on the **real checkpoints** before any
  profiling/benchmark builds on them (INT8 after its Q/DQ graph is built from the calibration set).
  Drift is measured and reported only; the pass/fail is an **owner sign-off on the measured drift
  table** — deliberately **not** coded into a tolerance object or automated gate.
- **Speedup study:** both models exported PyTorch→ONNX→TensorRT with explicit-Q/DQ INT8
  (FP32→FP16→INT8), benchmarked on the L40S under a fixed wall-clock budget (latency p50/p95,
  rollouts completed, throughput, peak GPU memory, **and SR per precision**), with
  encoder/predictor/planner profiled separately to locate bottlenecks. Only the model is
  TRT-optimized; the CEM planner stays in Python around it. Headline: LeWM-vs-DINOv3
  rollouts-in-budget + p95-latency ratio + per-model FP32→FP16→INT8 delta in **both speed and SR**
  (degradation quoted vs FP32; speed plotted against SR).
  - **Headline-artifact durability.** The headline tables (serialized to text) **and** plots (PNG)
    are persisted to the persistent network volume, the same durability contract as checkpoints and
    engines, so a completed study survives pod teardown. W&B logging is **additive, never the sole
    copy**. The **canonical** artifact is the raw **per-track results** (benchmark + profile numbers
    plus the run's fairness conditions — time budget, batch, seed); tables and plots are regenerable
    **views** of it. It is written **per track** so LeWM and DINOv3 can be benchmarked in separate
    pod sessions without clobbering each other, and it decouples the expensive pod-only benchmark
    from the cheap render — the report re-renders **off-pod** from the saved results, which is how
    the separately-gated SR-per-precision is joined in without re-running the L40S benchmark. A
    single-track render omits the two cross-track ratio plots, which need both tracks.
  - **Dilution disclosure (Amdahl).** Because only encoder+predictor are quantized and the Python
    planner is precision-invariant, the per-precision **wall-clock** delta is capped by the model's
    share of the cycle. So the study also reports, per model: the **FP32 baseline per-component time
    shares** and the derived **optimizable fraction** `p = (encoder+predictor)/total`, which sets
    the Amdahl ceiling on end-to-end speedup `1/(1−p)`; and — per precision — **both** the
    *model-only* speedup (planner treated as free) **and** the *realized* wall-clock speedup
    (rollouts-in-budget), whose gap is the planner floor and should match the Amdahl prediction
    `1/((1−p) + p/s)`. Reporting per-component *relative* speedup alone hides this dilution. That
    the optimizable fraction is itself model-dependent (LeWM's single token is
    planner/launch-latency-bound, DINO's 196-token grid is model-bound) is what explains why the
    same precision helps the two tracks differently — a result, not bookkeeping.
- **QLoRA delta:** the task-quality metric re-run on a QLoRA-tuned DINOv3 backbone (backbone
  QLoRA-adapted, **predictor unfrozen and co-trained**), reported as a delta against the frozen
  baseline, with adapters confirmed to target real modules.

---

## Execution Rules

The general engineering rules (debugging cap, log-before-delete, never run git, tick-before-advance)
live in `CLAUDE.md` and govern here too. Project-specific caps:

- **TensorRT/INT8 export is time-capped with an explicit FP16-only fallback** — surface when
  approaching the cap rather than iterating silently.
- **Training is epoch-capped** — 10 epochs for both tracks, batch size 128 — not wall-clock-capped.
- **Lean on the platform; don't reimplement it.** If a need looks like training, env, CEM, or eval,
  it's the platform's — wire to it.
