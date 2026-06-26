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
  `data/`, `checkpoints/`, `exports/` live on the pod's persistent volume — never committed.
- **Datasets:** the official Push-T data is loaded via the platform; it can stream
  from HF object storage (no local download needed) or cache to the data volume.
- **Secrets** (`WANDB_API_KEY`, `HF_TOKEN` if needed) passed at runtime via env.

---

## Commands

Training/eval use the platform's entrypoints; the owned layer adds export/benchmark.
Run on the pod via `uv run`; `setup.sh` provisions the environment (uv + deps + TensorRT).

- Train LeWM:        `uv run python -m scripts.train.lewm --config-dir conf +experiment=lewm`
- Train DINOv3-WM:   `uv run python -m scripts.train.prejepa --config-dir conf +experiment=dinov3`
- Evaluate (MPC):    via the platform's `World.evaluate` (CEM solver)
- Export/benchmark:  `uv run python -m src.export model=<lewm|dino> precision=<fp32|fp16|int8>`
- QLoRA tune:        `uv run python -m src.qlora`
- Smoke (tracer bullet): `uv run python -m src.smoke`

On the pod: `bash setup.sh`, then `uv run pytest -v`.

---

## Project Structure

- `src/`          — the owned layer: interfaces.py, export, benchmark, qlora, smoke
- `conf/`         — Hydra configs incl. the DINOv3 encoder config (COMMITTED)
- `scripts/train/`— platform training entrypoints (lewm.py, prejepa.py) as used
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

- `export(model, precision) -> engine_path` — PyTorch -> ONNX -> TensorRT. Only the
  **model** (encoder + predictor, via the adapter) is exported; the CEM planner is not.
- `benchmark(engine, time_budget) -> {latency_p50, latency_p95, rollouts_completed,
  throughput, peak_mem, success_rate}` — fixed wall-clock budget; rollouts is the
  headline speed measure, and **every speed result carries the SR for that engine
  config** (no speed number without its task-quality counterpart).
- `profile(adapter, ...) -> {encoder_ms, predictor_ms, planner_ms}` — per-component
  breakdown to locate the bottleneck (encoder vs predictor vs CEM planner)
- `plan_latency(model, obs, goal) -> seconds` — one CEM planning cycle, timed
- A thin adapter exposing each platform model behind a common
  `encode / predict` signature so export and benchmark treat both tracks identically.
  **One shared Protocol, two concrete implementations** (`LeWMAdapter`,
  `DINOWMAdapter`): identical call signature so the plumbing never branches, but the
  latent shape differs by model (LeWM `(B, D)`, DINO-WM `(B, N_patches, D)`).
  **The adapter is the unit TensorRT optimizes; the CEM rollout loop runs in Python
  around it** — the planner is never compiled into the engine.

Constants (`LATENT_DIM` for LeWM's single-token latent, the DINO-WM patch-grid latent
shape `(N_patches, D)`, `ACTION_DIM = 2`) are defined ONCE here; the platform's own
dims are read from its config, not re-guessed.

---

## Parity & Fairness Contracts (load-bearing — never vary silently)

- **Same trained-task comparison conditions:** both tracks evaluated with the same
  CEM config (300 samples, 30 elites, horizon 5, init variance 1, 10-30 iterations),
  same action budget, same goal encoding, same eval seeds, identical input
  normalization (ImageNet stats — the platform applies these). These are mostly
  enforced by the platform's eval; confirm they are not varied between tracks.
- **Matched export/benchmark conditions:** both models exported and benchmarked at
  the **same precision** on the **same L40S** under the **same fixed wall-clock time
  budget**, same env/goal. Within that budget we compare per-step inference latency
  (**p50 and p95**) and the **number of CEM rollouts completed** — rollout count is the
  intended degree of freedom; the only other difference is the model itself. **Every
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
- the model adapter dims (`LATENT_DIM`, DINO-WM patch-grid latent shape `(N_patches, D)`, `ACTION_DIM`)
- any change to the platform's eval/CEM config that would break the LeWM-vs-DINO parity

**CLAUDE CODE** — fails *loudly* (throws when wrong). Owns freely:
- Dockerfile, compose, uv/pyproject scaffolding, `.dockerignore`
- Hydra / W&B wiring around the platform entrypoints
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

- **Hard caps.** The two highest-risk efforts — the TensorRT/INT8 export
  (unsupported-op failures, fiddly calibration) and, if it overruns, scratch-LeWM
  training — are time-capped with explicit fallbacks (FP16-only; reference LeWM
  hyperparameters). Surface when approaching a cap rather than iterating silently.
- **Lean on the platform; don't reimplement it.** If a need looks like training,
  env, CEM, or eval, it's the platform's — wire to it, don't rebuild it.
- **Tracer bullet is the sole pre-optimization integration check.** Keep it strict —
  every owned boundary typed/asserted, on a real (or dummy) checkpoint.
- **PLAN.md progress:** every completed step records a checkbox + W&B run URL +
  artifact name (commit hash added by me when I commit).
- **Tick before advancing:** a step's checkbox should be ticked before the next begins.
- **Debugging cap:** after 3+ failed attempts at the same fix, stop, summarize, ask.
- **Log before delete:** never overwrite a run, checkpoint, or config representing
  completed work without confirming it's logged to W&B or committed.
- **Never run git:** output the files to stage and the commit message; I run git myself.