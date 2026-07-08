# SPEC: LeWM vs DINOv3-WM — Inference Optimization & QLoRA Study

The source of truth for this project. `PLAN.md` is generated from this spec and
carries execution progress. `CLAUDE.md` holds behavioral rules and points here.
`interfaces.py` is the contract in code for the parts I own (the export/benchmark
and QLoRA layer).

---

## Objective

Take the two reference world models from `stable-worldmodel` — **LeWM** (scratch
ViT-Tiny + SIGReg, `scripts/train/lewm.py`) and **DINOv3-WM** (frozen DINOv3
backbone + predictor, `scripts/train/prejepa.py` with a DINOv3 encoder config) —
trained on Push-T (224x224, pixels-only), and deliver the engineering layer the
platform does **not** provide:

1. **Inference-optimization study on an L40S:** export both models
   PyTorch -> ONNX -> TensorRT (FP32 -> FP16 -> INT8) and benchmark planning latency,
   throughput, and peak GPU memory. Headline: the **LeWM-vs-DINOv3 speedup ratio**
   (reproduces/stresses the paper's ~48x claim) and the **per-model
   FP32->FP16->INT8 optimization delta**.
2. **QLoRA delta on the DINOv3-WM backbone:** fine-tune the frozen DINOv3 backbone
   with QLoRA on Push-T, re-run the task-quality metric, and report the delta vs
   the frozen baseline.

The model training and the LeWM-vs-DINOv3 task comparison are **provided by
`stable-worldmodel`** — they are a foundation, not the contribution. The owned
contribution is the optimization + QLoRA layer above.

---

## Scope & Boundaries with the Platform

- **Use `stable-worldmodel` as the foundation.** Training (`lewm.py`,
  `prejepa.py`), the Push-T env (`swm/PushT-v1`), the CEM solver, dataset tooling,
  and closed-loop MPC evaluation come from the package. Do not reimplement them.
- **DINOv3-WM = `prejepa.py` with a DINOv3 encoder.** The backbone is config-injected
  and frozen (`encoder.eval(); requires_grad_(False)`), so the DINOv3 track is a
  config override (DINOv2->DINOv3 model string + `patch_size=16`) plus **one
  owner-approved encode-path override**: a `PreJEPA` subclass that drops CLS + the 4
  register tokens (`last_hidden_state[:, 5:, :]`) to expose the true 196-patch grid
  (Phase-1 §6). No predictor/SIGReg changes, and the platform wheel is not edited — the
  override lives in `src/` and is imported by the vendored train + eval entrypoints. **Always DINOv3,
  never DINOv2:** the DINO-WM paper and the platform's `prejepa.py` both default to
  DINOv2 — this project overrides the encoder to DINOv3 wherever DINO-WM is referenced;
  the wiring is otherwise identical (dims differ and are read from config). Verify
  DINOv3 exposes `config.hidden_size` + `last_hidden_state` so the **full patch-token
  grid** (patch tokens only — slice off **CLS + any register tokens**; DINOv3 prepends
  register tokens, so verify the token layout before slicing) feeds the
  predictor/planner unchanged, matching DINO-WM (`prejepa.py`). **The two tracks have different latent
  ranks:** LeWM exposes a single-token latent `(B, D)`; DINO-WM exposes the full patch
  grid `(B, N_patches, D)`. Pooling DINO to one token would both diverge from the paper
  and erase part of the encoder-compute asymmetry the study measures, so the patch dim
  is preserved.
- **LeWM = `lewm.py` unchanged.** SIGReg and the scratch encoder are the platform's;
  I do not reimplement or retune them beyond what training requires.
- **The contribution lives downstream of a trained checkpoint:** export, quantize,
  benchmark, and QLoRA-tune. That is where `interfaces.py` and the owned code apply.

---

## Tech Stack (pinned)

- Python 3.10+ (stable-worldmodel / jaxtyping requirement)
- **`stable-worldmodel`** (`pip install stable-worldmodel`) + `stable-pretraining` —
  training, env, CEM, eval. Pin the version (`uv.lock`); APIs change between minor versions.
- Runtime: a **RunPod L40S pod** on a **CUDA 12.4** base. RunPod pods cannot build
  Docker images in-pod (no Docker daemon), so dependencies install at pod start via
  `setup.sh`, not from a locally-built image.
- **uv** for dependency management — `pyproject.toml` + `uv.lock` committed. **torch**
  is uv-managed from the **cu124** wheel index (matches the pod's CUDA 12.4). **TensorRT**
  is installed by `setup.sh` (cu12, CUDA-12.4-matched) and kept OUT of uv (do not pin
  `tensorrt` in uv) so it can't pull a conflicting `libnvinfer`/CUDA stack.
- Hydra (config — the platform uses it), Weights & Biases (logging)
- jaxtyping + beartype (contracts for the owned export/QLoRA boundaries, runtime-checked)
- onnx (export stage); TensorRT installed by `setup.sh` (the export/benchmark stage)
- transformers / timm (DINOv3 + ViT-Tiny backbones), peft / bitsandbytes (QLoRA)
- Docker + docker-compose — **reproducibility image composed at project end, off-pod;
  not part of the dev loop**

---

## Execution Environment

- **Single machine: L40S** — train and benchmark on the same instance (same
  hardware class as the LeWM paper, so speed numbers are comparable). Training
  hardware doesn't affect results; one image, one host.
- **TensorRT engines built locally on the L40S** — engines are architecture-specific
  and disposable; regenerate from the export script (gitignored).
- **No in-pod image build** (RunPod has no Docker daemon): dev runs directly on the pod
  via `setup.sh` + `uv run`; a reproducibility image is composed at the end, off-pod.
  Datasets, checkpoints, and exports live on the pod's **persistent network volume**
  (RunPod mounts it at `/workspace`) — never committed.
- **Persistent root (`STABLEWM_HOME`):** the platform caches everything under this root —
  datasets in `$STABLEWM_HOME/datasets`, checkpoints in `$STABLEWM_HOME/checkpoints/<run_name>/`
  (`save_pretrained`). It defaults to `~/.stable_worldmodel` on the **ephemeral** container
  fs, so it MUST be set to the network volume (e.g. `STABLEWM_HOME=/workspace/.stablewm`)
  or a multi-hour run's checkpoints are lost on pod restart. `setup.sh` validates it.
- **Datasets:** the official Push-T data is loaded via the platform; it can stream
  from HF object storage (no local download needed) or cache under `$STABLEWM_HOME/datasets`.
- **Secrets / runtime env** (`WANDB_API_KEY`, `HF_TOKEN` if needed, `STABLEWM_HOME`) passed at runtime via env.

---

## Commands

Training/eval use the platform's entrypoints; the owned layer adds export/benchmark.
Run on the pod via `uv run`; `setup.sh` provisions the environment (uv + deps + TensorRT).

- Train LeWM:        `uv run python -m scripts.train.lewm --config-dir conf +experiment=lewm`
- Train DINOv3-WM:   `uv run python -m scripts.train.prejepa --config-dir conf +experiment=dinov3`
- Evaluate (MPC):    `uv run python -m src.eval --config-dir conf +experiment=eval_<lewm|dino>`
  — a thin **owned** driver (`src/eval.py`) that composes and runs the vendored eval
  entrypoint (`eval_wm.run`, **byte-unmodified**) and opens/logs the W&B run.
  **CEM-solve latency** is measured by an owned observation-only `CEMSolver` **callback**
  injected purely through config (`cfg.solver.callbacks`) — the platform's own per-solve
  extension seam, so neither the vendored file nor the solver logic is touched (no monkeypatch,
  no vendored edit). Success rate comes from `World.evaluate`'s return value (captured
  observation-only); both land in one W&B run. The `eval_<lewm|dino>` overlay selects the
  trained checkpoint (`policy=<ckpt>`), the eval dataset, the `wandb:` block, and the callback
  injection; the **training** overlays (`lewm`/`dinov3`) are not reused for eval — they carry
  only training keys, so composing them would leave `policy=random`.
  **Latency scope:** the callback brackets the CEM optimization body (`reset → end_solve`),
  which excludes the `prepare_init_action` warm-start. For LeWM and DINO-WM that warm-start is
  a zero-pad (neither model is `Actionable`), so the exclusion is negligible and
  model-independent — the metric is labelled *CEM-solve latency*, not full planning-cycle
  latency. (If an actor were ever added, making a model `Actionable`, this would need
  revisiting — but no phase does so.)
- Export/benchmark:  `uv run python -m src.export model=<lewm|dino> precision=<fp32|fp16|int8>`
- QLoRA tune:        `uv run python -m src.qlora`
- Smoke (tracer bullet): `uv run python -m src.smoke`

On the pod: `bash setup.sh`, then `uv run pytest -v`.

**W&B logging discipline (all phases, one shared project).** Every phase logs to the
**same** W&B project. Training logs via the platform's Lightning `WandbLogger` (driven by
the `wandb:` block in `conf/experiment/`). The non-training phases (eval/benchmark/QLoRA)
have no Lightning `Trainer`, so they open the run via a small **owned** helper
(`src/wandb_log.py`): its `init()` reads the project (and entity) from the same
`conf/experiment/` `wandb:` block — no second source of truth — then phases log with
`wandb.log` directly against that run. This makes "logged to W&B" an established path, not
just a PLAN verify assertion.

---

## Project Structure

- `src/`          — the owned layer: interfaces.py, export, benchmark, qlora, smoke,
  the owned W&B helper, the observation-only CEM-solve-latency callback, and the thin
  Phase-3 eval driver (`eval.py`, runs the byte-unmodified `eval_wm.run` + logs SR/latency)
- `conf/`         — Hydra configs incl. the DINOv3 encoder config (COMMITTED)
- `scripts/train/`— platform training entrypoints (lewm.py, prejepa.py) as used
- `scripts/plan/` — platform eval entrypoint (eval_wm.py) + its config group
  (pusht.yaml, solver/cem.yaml), vendored as used (provenance in `scripts/plan/VENDORED.md`)
- `tests/`        — pytest
- `data/`         — Push-T dataset cache (GITIGNORED, mounted volume or HF stream)
- `checkpoints/`  — trained weights, `lewm/` vs `dino/` (GITIGNORED, mounted volume)
- `exports/`      — ONNX / TensorRT artifacts (GITIGNORED, built on L40S)
- `setup.sh`     — pod bootstrap: uv + deps + TensorRT, run on each pod load (COMMITTED)
- `pyproject.toml`, `uv.lock` — dependency pins (COMMITTED)
- `PLAN.md`       — generated from this spec; carries progress + artifact links

---

## Interface Contracts (the owned layer)

`interfaces.py` is the single source of truth for the boundaries I own — the
export/benchmark/QLoRA layer that sits on top of a trained platform model.
Runtime-checked via jaxtyping + beartype with shared named axes.

- `export(adapter, precision) -> {encoder, predictor} engine paths` — PyTorch -> ONNX ->
  TensorRT. The adapter's `encode` and `predict` are traced and built **separately** (one
  ONNX graph / TensorRT engine each), because the CEM rollout encodes once and calls
  predict many times over the cached latent — a single fused `obs -> latent` graph could
  not reproduce that call pattern (see the adapter bullet). ONNX/TensorRT does not require
  a single fused forward: each method is exported by pointing the tracer at it with its own
  example inputs. Tracing uses the **TorchDynamo exporter** (`torch.onnx.export(dynamo=True)`
  — the legacy TorchScript exporter is deprecated; on the pinned torch 2.6 `dynamo=True` is
  passed explicitly since it is not the default until 2.9), aimed at each method via a thin
  `nn.Module` whose `forward` calls it (shapes come from the example inputs + `dynamic_shapes`
  for the variable candidate batch). Across both tracks this is **4 ONNX graphs** (encode +
  predict × LeWM + DINO-WM); precision (FP32/FP16/INT8) is a TensorRT engine-build setting,
  not additional graphs, so it multiplies engines, not ONNX. Only the **model** (encoder +
  predictor) is exported; the CEM planner is not.
- `benchmark(engine, time_budget) -> {latency_p50, latency_p95, rollouts_completed,
  throughput, peak_mem, success_rate}` — fixed wall-clock budget; rollouts is the
  headline speed measure, and **every speed result carries the SR for that engine
  config** (no speed number without its task-quality counterpart).
- `profile(adapter, ...) -> {encoder_ms, predictor_ms, planner_ms}` — per-component
  breakdown to locate the bottleneck (encoder vs predictor vs CEM planner)
- `plan_latency(model, obs, goal) -> seconds` — one CEM planning cycle, timed
- A thin adapter exposing each platform model behind a common **two-method**
  `encode` / `predict` signature — **not** a single fused `__call__(obs, action) -> latent`
  — so export and benchmark treat both tracks identically. The two methods are **separately
  callable and separately exported**: the CEM rollout encodes the obs **once**, caches the
  latent, then calls `predict` autoregressively over the horizon for all candidates (the
  platform's `rollout`). `encode` and `predict` must therefore stay distinct — a fused
  `obs -> latent` step would re-encode on every predictor call, inflating encoder cost and
  erasing the encoder-cached / predictor-dominates asymmetry the study measures. Export
  produces **two engines per model** (encoder + predictor), each traced from its own
  example inputs (`encode`: obs; `predict`: cached latent + action), and the Python rollout
  drives both at benchmark time.
  **One shared Protocol, two concrete implementations** (`LeWMAdapter`, `DINOWMAdapter`):
  identical method signatures so the plumbing never branches, but the latent shape differs
  by model (LeWM `(B, D)`, DINO-WM `(B, N_patches, D)`) and the action enters `predict`
  differently per track (LeWM: a separate AdaLN-conditioning argument; DINO-WM: concatenated
  onto the feature axis inside the adapter, widening the predictor tokens to 404 — see
  Constants). **The adapter is the unit TensorRT optimizes; the CEM rollout loop runs in
  Python around it** — the planner is never compiled into the engine.
- **Re-entering the platform eval on the optimized model (Phase-5 SR-per-precision).**
  The CEM solver calls the world model via `get_cost` / `get_action` — not `encode` /
  `predict` directly. So to produce the SR that pairs with each precision's speed number,
  the exported/quantized adapter is re-wrapped in a thin **Python** shim exposing
  `get_cost` / `get_action` (which call the engine's `encode` / `predict` underneath) and
  slotted into `CEMSolver(model=...)`, letting the Phase-3 owned eval driver re-run
  unchanged on the optimized model. The shim stays in Python; only `encode` / `predict`
  lives inside the engine — the planner is still never compiled in.

Constants (`LATENT_DIM = 192` for LeWM's single-token latent, the DINO-WM patch-grid latent
shape `(N_patches, D) = (196, 384)`, `ACTION_DIM = 2`, and the DINO-WM **predictor-input
token width** `404 = D + Σ(extra encoding dims) = 384 + 20`) are defined ONCE here; the
platform's own dims are read from its config, not re-guessed. The 404 width is distinct from
the 384 latent because the action/proprio embeddings are tiled and concatenated onto the
feature axis before `predict` (`prejepa.encode`), so the DINO-WM `predict` boundary is
404-wide on input, not 384. It is a **silently-failing** dim (a wrong value mis-shapes the
predictor engine with no error), so it is owner-confirmed against the instantiated predictor
alongside the Phase-1 dims.

---

## Parity & Fairness Contracts (load-bearing — never vary silently)

- **Same trained-task comparison conditions:** both tracks evaluated with the same
  CEM config (300 samples, 30 elites, horizon 5, init variance 1, 10-30 iterations),
  same action budget, same goal encoding, same eval seeds, identical input
  normalization (ImageNet stats — the platform applies these). These are mostly
  enforced by the platform's eval; confirm they are not varied between tracks.
- **Matched export/benchmark conditions:** both models exported and benchmarked at
  the **same precision** on the **same L40S** under the **same fixed wall-clock time
  budget**, same env/goal, and the **same shared inference batch size**. Within that
  budget we compare per-step inference latency (**p50 and p95**) and the **number of CEM
  rollouts completed** — rollout count is the intended degree of freedom; the only other
  difference is the model itself. **Training batch size is held equal across tracks (128,
  LeWM's paper value) and does not carry into inference** — inference uses the shared batch
  size above, so no training-time batch difference can confound the benchmark. **Every
  speed figure is reported with its SR**, and per-model FP16/INT8 results quote the
  **SR and latency degradation relative to FP32** (a precision that is faster but
  degrades task quality must be visible, not hidden behind throughput).
- **The speedup is mechanistic, not configuration:** the LeWM-vs-DINOv3 gap comes
  from the encoder-compute asymmetry — LeWM's tiny scratch ViT-Tiny exposing a single
  latent token vs DINOv3's large backbone exposing the full patch-token grid, so the
  predictor and CEM planner also operate over `N_patches` tokens for DINO vs one for
  LeWM — surfaced as how many more rollouts LeWM fits in the budget. Do not let a batch
  or precision mismatch confound it. The encoder/predictor/planner profile attributes
  the gap to the right component.
- **QLoRA comes after the frozen baseline**, never from the outset — the delta is
  reported against frozen DINOv3-WM.

---

## Implementation Boundaries (ownership by failure mode)

**OWNER-ONLY** — fails *silently* (plausible wrong number). Claude Code must STOP
and ask before touching:
- ONNX / TensorRT export debugging (reading the failure output is the judgment-heavy part)
- INT8 calibration set + procedure; the FP32/FP16/INT8 precision matching
- QLoRA targeting (which DINOv3 modules, rank, what stays frozen — note the predictor
  is unfrozen and co-trained, so only backbone targeting is open)
- the benchmark fairness conditions (matched precision, fixed time budget, env/goal)
- the model adapter dims (`LATENT_DIM`, DINO-WM patch-grid latent shape `(N_patches, D)`,
  the DINO-WM predictor-input token width `404 = 384 + 20 extras`, `ACTION_DIM`)
- any change to the platform's eval/CEM config that would break the LeWM-vs-DINO parity

**CLAUDE CODE** — fails *loudly* (throws when wrong). Owns freely:
- Dockerfile, compose, uv/pyproject scaffolding, `.dockerignore`
- Hydra / W&B wiring around the platform entrypoints, incl. the owned W&B helper for the
  non-training phases and the **observation-only** CEM-solve-latency **callback** — an owned
  `CEMSolver.Callback` subclass injected through config (`cfg.solver.callbacks`), the
  platform's own per-solve extension seam, so the vendored `eval_wm.py` and the solver stay
  byte-untouched (no monkeypatch, no vendored edit). It may only read/record timing: it
  brackets one CEM solve (`reset → end_solve`) with `perf_counter` and an optional
  `torch.cuda.synchronize()` barrier — the barrier blocks the CPU for an accurate GPU
  wall-clock number but leaves seeds, sample draws, and the plan byte-identical. It brackets
  the optimization body only, excluding the `prepare_init_action` warm-start (a zero-pad for
  these non-`Actionable` models, hence negligible and model-independent — labelled *CEM-solve
  latency*). Perturbing seeds, sample counts, or the plan crosses into the eval/CEM parity
  gate above and is OWNER-ONLY.
- the DINOv3 encoder config for `prejepa.py` (model string, dims read from config)
- export-script and benchmark-harness *plumbing* (ONNX trace call, TensorRT builder
  invocation, percentile timing, memory logging, the speedup-table runner)
- the QLoRA training-loop wiring (owner specifies the targeting config)
- the tracer-bullet smoke script

---

## Requirements

What the finished project must satisfy (ordered build steps live in `PLAN.md`):

- **Foundation runs:** `stable-worldmodel` installed and pinned; both reference
  trainings (`lewm.py`, `prejepa.py` with a DINOv3 encoder) run on Push-T and
  produce checkpoints. DINOv3 confirmed to slot into `prejepa.py` cleanly.
- **Task baseline:** Push-T success rate + planning latency for both tracks via the
  platform's CEM evaluation, under matched conditions — the comparison baseline the
  optimization study builds on.
- **Integration (tracer bullet):** a trained checkpoint flows through the owned
  adapter -> export stub -> benchmark stub end-to-end on random/dummy weights in the
  container, typed checks passing at every owned boundary. Sole pre-optimization
  integration check.
- **Engine fidelity gate (before benchmarking):** exported FP32/FP16 engines are
  precision-matched against the PyTorch reference on the **real checkpoints** before any
  profiling/benchmark builds on them — the export-stage analogue of the tracer bullet, so a
  silently-diverging quantized engine is caught before it poisons every downstream SR. Drift
  (max abs/rel) is measured; the FP32/FP16/INT8 tolerance policy is owner-set (a
  silent-failure boundary), so drift is logged-not-gated until sign-off.
- **Speedup study:** both models exported PyTorch->ONNX->TensorRT (FP32->FP16->INT8),
  benchmarked on the L40S under a fixed wall-clock time budget (latency p50/p95,
  rollouts completed, throughput, peak GPU memory, **and SR per precision**), with
  encoder/predictor/planner profiled separately to locate bottlenecks. Only the model
  is TRT-optimized; the CEM planner stays in Python around it. Headline: LeWM-vs-DINOv3
  rollouts-in-budget + p95-latency ratio + per-model FP32->FP16->INT8 delta in **both
  speed and SR** (degradation quoted vs FP32; speed plotted against SR).
- **QLoRA delta:** the task-quality metric re-run on a QLoRA-tuned DINOv3 backbone
  (backbone QLoRA-adapted, **predictor unfrozen and co-trained**), reported as a delta
  against the frozen baseline (adapters confirmed to target real modules).

---

## Execution Rules

- **Hard caps.** The TensorRT/INT8 export (unsupported-op failures, fiddly calibration)
  is time-capped with an explicit fallback (FP16-only); surface when approaching the cap
  rather than iterating silently. Training is bounded by a fixed **epoch budget** (10 epochs
  for both tracks, batch size 128), not a wall-clock cap.
- **Lean on the platform; don't reimplement it.** If a need looks like training,
  env, CEM, or eval, it's the platform's — wire to it, don't rebuild it.
- **Tracer bullet is the sole pre-optimization integration check.** Keep it strict —
  every owned boundary typed/asserted, on a real (or dummy) checkpoint.
- **PLAN.md progress:** every completed step records a checkbox + artifact name
  (commit hash added by me when I commit).
- **Tick before advancing:** a step's checkbox should be ticked before the next begins.
- **Debugging cap:** after 3+ failed attempts at the same fix, stop, summarize, ask.
- **Log before delete:** never overwrite a run, checkpoint, or config representing
  completed work without confirming it's logged to W&B or committed.
- **Never run git:** output the files to stage and the commit message; I run git myself.