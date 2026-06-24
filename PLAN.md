# PLAN.md — LeWM vs DINOv3-WM: Inference Optimization & QLoRA Study

> Generated from `SPEC.md`. Carries execution progress. `CLAUDE.md` holds the
> behavioral rules; `src/interfaces.py` is the typed contract for the owned layer.
> Every completed step records: `[x]` + W&B run URL + artifact name (commit hash
> added by the owner). Tick a box before starting the next step.

## Context

**Why this project exists.** `stable-worldmodel` already trains two Push-T world
models — **LeWM** (scratch ViT-Tiny + SIGReg) and **DINOv3-WM** (frozen DINOv3
backbone + predictor) — and already compares them on task success via its CEM/MPC
evaluation. That is the *foundation*, not the contribution. The owned contribution
is the **engineering layer the platform does not provide**: (1) a PyTorch→ONNX→
TensorRT inference-optimization study (FP32→FP16→INT8) on an L40S that quantifies
the **LeWM-vs-DINOv3 speedup ratio** (the encoder-compute asymmetry: LeWM's tiny
scratch ViT-Tiny vs DINOv3's large backbone, which attends over hundreds of patch
tokens *internally* even though both tracks expose a single **CLS-token** latent)
and the **per-model precision delta**; and (2) a **QLoRA delta** on the DINOv3
backbone vs its frozen baseline.

**Current repo state (greenfield).** Only `CLAUDE.md`, `SPEC.md`,
`src/interfaces.py`, `.gitignore`, `LICENSE`, `README.md` exist. There is **no**
`pyproject.toml`, `conf/`, `scripts/`, `tests/`, and **`stable-worldmodel` /
`stable-pretraining` are not installed**. Consequence: the platform's real API
cannot be read yet — Phase 1 installs it and reads it before any wiring (CLAUDE.md
§10). Adapter dims (`LATENT_DIM`, DINOv3 pooled dim, `ACTION_DIM`) are read from the
platform config, never guessed.

**Execution model (confirmed).** The **L40S is the only execution target**; local
WSL is for editing only. Every run command is `docker compose run app …` on the
L40S. All five SPEC requirement bundles are covered below as phases; near-term
phases are detailed, GPU- and owner-gated steps are marked.

**Legend.** 🟢 CLAUDE-CODE owns (fails loud). 🔴 OWNER-ONLY — STOP and ask before
touching (fails silently / plausible wrong number; see SPEC "Implementation
Boundaries"). 🖥️ runs on the L40S GPU. ⏱️ hard-capped effort with a stated fallback.

---

## Phase 0 — Scaffolding & pinned dependencies  🟢

Stand up the container and dependency layer so everything else runs reproducibly on
the L40S. No model code.

- [ ] `pyproject.toml` + `uv.lock` pinning: `stable-worldmodel`, `stable-pretraining`,
  `hydra-core`, `wandb`, `jaxtyping`, `beartype`, `onnx`, `transformers`,
  `timm`, `peft`, `bitsandbytes`. **Torch AND TensorRT come from the base image** —
  configure uv not to reinstall either (mark as provided / constraint). Do **not** pin
  `tensorrt` in uv: the NGC image's bundled, CUDA-matched TRT must win, or pip's TRT
  pulls conflicting `libnvinfer`/CUDA libs. Pin versions (APIs shift between minors).
- [ ] `Dockerfile` on the **NGC PyTorch image** (`nvcr.io/nvidia/pytorch:<tag>`, which
  bundles a matched TensorRT) whose CUDA ≤ the L40S driver (`nvidia-smi` on host).
- [ ] `docker-compose.yml`: single `app` service handling all roles by entrypoint
  command; mounts `data/ checkpoints/ exports/` as volumes (never baked in); passes
  `WANDB_API_KEY` / `HF_TOKEN` from runtime env. `.dockerignore`.
- [ ] Skeleton dirs: `conf/` (Hydra), `tests/` (pytest).

**Verify:** `docker build` succeeds; `docker compose run app uv sync`;
`docker compose run app python -c "import stable_worldmodel, stable_pretraining, tensorrt, peft"`.

---

## Phase 1 — Read the real platform API  🟢 → 🔴 (dims sign-off)

Gate before any wiring (CLAUDE.md §10). The platform is young/fast-moving; do not
call it from memory.

- [ ] In-container, read the **installed source** (`.venv/.../stable_worldmodel`,
  `.../stable_pretraining`) and the training entrypoints `scripts/train/lewm.py`,
  `scripts/train/prejepa.py` (obtain "as used" from the platform's examples; record
  provenance). Capture the true signatures for: the `World` object + `World.evaluate`
  (CEM/MPC), the CEM solver config, the Push-T env id (`swm/PushT-v1`), the
  CLS-token extraction path, and how the backbone is config-injected and frozen
  (`encoder.eval(); requires_grad_(False)`).
- [ ] Confirm **DINOv3 exposes `config.hidden_size` + `last_hidden_state`** so the
  **CLS-token** path (`last_hidden_state[:, 0]`) applies unchanged (SPEC §Scope).
- [ ] Record findings (a short `docs/platform_api.md` or module docstring): exact call
  shapes feeding the adapter and the dims, **and where one CEM planning cycle decomposes
  into encoder / predictor / planner calls** (needed for the Phase-5 per-component profile).

**🔴 OWNER gate:** confirm `LATENT_DIM` (LeWM CLS token), DINOv3 CLS-token dim, and
`ACTION_DIM = 2` against the real config **before** they are hard-coded once in
`src/interfaces.py` / the adapter.

**Verify:** introspection commands run clean in-container; DINOv3 attribute check
passes; dims written down and owner-confirmed.

---

## Phase 2 — Foundation trainings  🟢 wiring · 🔴 config slot-in · 🖥️ ⏱️

Produce the two reference checkpoints. Training is the platform's — wire to it, don't
rebuild it.

- [ ] Vendor `scripts/train/lewm.py` and `scripts/train/prejepa.py` as used; wire
  Hydra + W&B around them (no changes to SIGReg, the scratch encoder, or the
  predictor — SPEC §Boundaries / CLAUDE.md §8).
- [ ] 🔴 `conf/` **DINOv3 encoder config** for `prejepa.py` (DINOv2→DINOv3 model
  string; dims **read from config**, not guessed). Owner confirms it slots in cleanly.
- [ ] 🖥️ Train LeWM: `docker compose run app python -m scripts.train.lewm`
  → `checkpoints/lewm/`.
- [ ] 🖥️ Train DINOv3-WM: `docker compose run app python -m scripts.train.prejepa backbone=dinov3`
  → `checkpoints/dino/`.
- [ ] ⏱️ **Hard cap on scratch-LeWM training**; fallback = reference LeWM
  hyperparameters. Surface on approaching the cap; don't iterate silently.

**Verify:** two checkpoints exist; both W&B runs logged (URLs recorded here).
**Log-before-delete:** never overwrite a checkpoint/run without confirming it's in
W&B or committed (CLAUDE.md §7).

---

## Phase 3 — Task baseline (platform CEM/MPC eval)  🟢 wiring · 🔴 parity · 🖥️

The trained-task comparison the optimization study builds on — produced by the
platform's eval, under matched conditions.

- [ ] 🖥️ Run `World.evaluate` (CEM solver) for **both** tracks: Push-T **success rate**
  + **planning latency**.
- [ ] 🔴 **Parity (fairness, load-bearing):** same CEM config (300 samples, 30 elites,
  horizon 5, init var 1, 10–30 iters), same action budget, same goal encoding, same
  eval seeds, identical ImageNet normalization — confirm **not varied between tracks**
  (do not silently change the platform eval/CEM config). *(Task-quality eval keeps a
  fixed iteration budget; the inference benchmark in Phase 5 uses a fixed time budget —
  see note there.)*

**Verify:** success-rate + latency for both tracks, logged to W&B; parity conditions
recorded as identical.

---

## Phase 4 — Owned adapter + tracer bullet  🟢 · (sole pre-optimization check)

Wire the owned `src/` layer to `interfaces.py` and prove the path end-to-end on
dummy/random weights. Keep it strict.

- [ ] Implement the **`WMStepAdapter`** for both tracks behind a common
  `encode`/`predict` signature (so export & benchmark treat both identically), typed
  per `src/interfaces.py`. **The adapter is the only thing TensorRT optimizes** — it
  wraps the model (encoder + predictor) so the CEM planner / rollout loop stays in
  Python, *outside* TRT, and calls the optimized model through this boundary.
- [ ] Constants (`LATENT_DIM`, DINOv3 CLS-token dim, `ACTION_DIM`) defined **once** in
  `interfaces.py`, from the Phase-1 owner-confirmed values; platform dims read from
  config.
- [ ] Implement `export()` and `benchmark()` **stubs** conforming to the `Export` /
  `Benchmark` Protocols + `ExportConfig` (real TRT comes in Phase 5).
- [ ] `src/smoke.py`: dummy checkpoint → adapter → export-stub → benchmark-stub, with
  jaxtyping + beartype assertions at **every owned boundary**.
- [ ] `tests/` covering the adapter shapes and the typed boundaries.

**Verify:** `docker compose run app python -m src.smoke` passes;
`docker compose run app pytest -v` green; a shape/precision violation actually raises.

---

## Phase 5 — Speedup study: export, profile & fixed-budget benchmark  🔴 OWNER-heavy · 🖥️ ⏱️

The headline deliverable. Owner makes the silent-failure calls; Claude owns plumbing
(trace call, builder invocation, percentile timing, memory logging, profiler hooks,
table runner).

**Benchmark methodology.** Only the **model** (encoder + predictor, via the
`WMStepAdapter`) is TensorRT-optimized; the **CEM planner / rollout loop stays in
Python**, wrapping the optimized model — so the comparison is of the model, not a
re-implemented planner. Evaluate both models under a **fixed wall-clock time budget**
and compare **(a) inference latency** (per planning step, **p50 and p95**) and **(b)
number of CEM rollouts completed** within that budget. This expresses the
encoder-compute asymmetry the way it actually matters for planning: how much more
search each model fits in the same time. **Every speed number is paired with a
success rate (SR):** each precision is also run through the Phase-3 platform SR eval,
so no speed figure stands without its task-quality counterpart. (Matches SPEC §Parity
and `src/interfaces.py`.)

- [ ] 🔴 Real export **PyTorch→ONNX→TensorRT**, **FP32→FP16→INT8**, per model:
  `docker compose run app python -m src.export model=<lewm|dino> precision=<fp32|fp16|int8>`.
  ONNX/TRT export debugging, INT8 **calibration set + procedure**, and FP32/FP16/INT8
  **precision matching** are OWNER-ONLY — STOP and ask.
- [ ] 🖥️ **Per-component profiling** — profile **encoder, predictor, and planner (CEM)
  separately** to identify bottlenecks: where each model spends its time per planning
  cycle and what dominates (encoder token count vs predictor rollout vs planner
  sampling/sorting). Use the Phase-1 cycle decomposition; emit a per-component breakdown
  (`src/profile.py` or a `benchmark` mode) for both models × precisions.
- [ ] 🖥️ **Fixed-time-budget benchmark** on the L40S: for a fixed budget per model ×
  precision, record **rollouts completed**, **per-step inference latency p50/p95**,
  throughput (rollouts/sec), **peak GPU memory**, **and the SR for that engine config**
  (Phase-3 eval re-run on the optimized model). Same env/goal/precision/budget across
  models — only the model differs; rollout count is the thing allowed to vary.
- [ ] Headline outputs, as tables **and plots**: **LeWM-vs-DINOv3 rollouts-in-budget
  ratio** + **p95 latency** ratio (stresses the paper's ~48×); **per-model
  FP32→FP16→INT8 delta** reported as **both speed and SR, with SR/latency degradation
  quoted relative to FP32** (e.g. INT8 SR drop vs FP32); **speed-vs-SR plotted** so a
  precision that wins on throughput while losing task quality is visible; and the
  **per-component (encoder/predictor/planner) bottleneck breakdown**.
- [ ] ⏱️ **Hard cap on TensorRT/INT8** (unsupported-op / calibration); fallback =
  **FP16-only**. Debugging cap: after 3 failed attempts at the same fix, stop,
  summarize, ask (CLAUDE.md §6).

**Interface note:** `src/interfaces.py` declares the targets for this phase —
`BenchResult.rollouts_completed`, the fixed `time_budget_s` on `Benchmark` /
`ExportConfig`, and `ComponentProfile` / `Profile` for the encoder/predictor/planner
breakdown.

**Verify:** engines built on the L40S (gitignored, regenerable); fixed-budget comparison
(rollouts + p95 latency **+ SR per precision**, with FP32-relative degradation quoted)
and the encoder/predictor/planner profile tables produced and logged to W&B.

---

## Phase 6 — QLoRA delta on DINOv3-WM  🔴 targeting · 🖥️

Reported **against the frozen baseline**, never from the outset.

- [ ] 🔴 **OWNER specifies QLoRA targeting:** which DINOv3 modules, rank, what stays
  frozen. Claude owns only the training-loop wiring.
- [ ] 🖥️ QLoRA fine-tune the DINOv3 backbone on Push-T (`peft` + `bitsandbytes`):
  `docker compose run app python -m src.qlora`. The **predictor is unfrozen and
  co-trained** (not held fixed) so it tracks the shifting backbone latents — avoids the
  representation-drift mismatch a frozen predictor would suffer. Confirm adapters target
  **real** modules (introspect, don't assume).
- [ ] 🖥️ Re-run the Phase-3 task-quality metric on the tuned backbone; report the
  **delta vs frozen DINOv3-WM**.

**Verify:** tuned checkpoint produced; task-metric delta vs frozen reported and logged
to W&B; adapter target modules confirmed real.

---

## Critical files

- `src/interfaces.py` — typed contract (declares the fixed-budget benchmark +
  per-component profile; dim constants filled in Phase 4 from Phase-1 owner-confirmed
  values).
- `src/adapter.py`, `src/export.py`, `src/benchmark.py`, `src/profile.py`,
  `src/qlora.py`, `src/smoke.py` — the owned layer (Phases 4–6).
- `conf/` — Hydra configs incl. the DINOv3 encoder config (COMMITTED).
- `scripts/train/lewm.py`, `scripts/train/prejepa.py` — platform entrypoints, as used.
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`.
- `tests/` — pytest for the owned boundaries.

## Cross-cutting rules (from CLAUDE.md / SPEC)

- **Owner gates:** anything 🔴 (export/INT8 debugging, precision matching, QLoRA
  targeting, benchmark methodology, adapter dims, eval/CEM parity) → STOP and ask.
- **Git:** never run git. On completing a unit of work, output the files to stage and
  a `type(scope): summary` commit message; the owner runs git.
- **Progress:** each `[x]` records W&B URL + artifact name; tick before advancing.
- **Caps:** TRT/INT8 and LeWM training are time-capped with fallbacks; 3-attempt
  debugging cap; log-before-delete.

## End-to-end verification

1. `docker build` + `docker compose run app uv sync` + import check (Phase 0).
2. Platform API introspection in-container; DINOv3 `config.hidden_size`/
   `last_hidden_state` confirmed (Phase 1).
3. Two checkpoints + W&B runs (Phase 2); both-track success-rate + latency under
   matched CEM config (Phase 3).
4. `python -m src.smoke` + `pytest -v` green on dummy weights (Phase 4).
5. `src.export` builds TRT engines on the L40S; fixed-budget benchmark emits
   rollouts-in-budget + latency comparison and the encoder/predictor/planner profile
   tables (Phase 5).
6. `src.qlora` produces a tuned backbone; task-metric delta vs frozen reported
   (Phase 6).
