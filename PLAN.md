# PLAN.md — LeWM vs DINOv3-WM: Inference Optimization & QLoRA Study

> Generated from `SPEC.md` (rationale lives there; this file is execution steps only).
> Carries execution progress. `CLAUDE.md` holds behavioral rules; `src/interfaces.py` is
> the typed contract for the owned layer. Every completed step records: `[x]` + W&B run
> URL + artifact name (commit hash added by the owner). Tick a box before the next step.

## Context

**Execution model.** The **L40S RunPod pod is the only execution target**; local WSL is
edit-only. Every run command is `uv run …` on the pod (provisioned by `setup.sh`). The
five SPEC requirement bundles map to the Phases below.

**Legend.** 🟢 CLAUDE-CODE owns (fails loud). 🔴 OWNER-ONLY — STOP and ask (see SPEC
"Implementation Boundaries"). 🖥️ runs on the L40S GPU. ⏱️ capped effort with a stated fallback.

---

## Phase 0 — Scaffolding & pinned dependencies  🟢

- [x] `pyproject.toml` + `uv.lock` pinning: `stable-worldmodel`, `stable-pretraining`,
  `hydra-core`, `wandb`, `jaxtyping`, `beartype`, `onnx`, `transformers`, `timm`,
  `peft`, `bitsandbytes`, **torch (cu124 wheel index)** — all uv-managed. **TensorRT NOT
  in uv** (installed by `setup.sh`). Versions pinned.
- [x] `setup.sh` — pod bootstrap, idempotent, run on each pod load: installs **uv**, runs
  `uv sync`, then installs **TensorRT (cu12, CUDA-12.4)** outside the lock. Secrets
  (`WANDB_API_KEY` / `HF_TOKEN`) from the pod's runtime env.
- [x] Skeleton dirs: `conf/` (Hydra), `tests/` (pytest).
- [x] **Deferred to project end:** `Dockerfile` + `docker-compose.yml` (off-pod reproducibility image).

**Verify (on the pod):** `bash setup.sh` succeeds; `uv run python -c "import
stable_worldmodel, stable_pretraining, tensorrt, peft, torch"`; `uv run pytest -v`.

---

## Phase 1 — Read the real platform API  🟢 → 🔴 (dims sign-off)

Gate before any wiring (CLAUDE.md §10).

> **Status: COMPLETE (2026-06-26).** Findings in `docs/platform_api.md` (provenance:
> swm 0.1.1 / sp 0.1.7 sdists + GitHub tag 0.1.1). In-pod introspection confirmed dims:
> DINOv3 `hidden_size=384`, `patch_size=16`, `num_register_tokens=4`,
> `last_hidden_state=(1,201,384)` → **N_patches=196** after CLS+register slice; LeWM
> `hidden_size=192`, CLS `(1,192)`; PushT `action_space=Box(-1,1,(2,))` → `ACTION_DIM=2`.
> 🔴 OWNER gate resolved: **slice CLS+registers**, stay on `dinov3_small` (doc §6).
> Phase-4 dims: `LATENT_DIM=192`, DINO-WM `(N_patches,D)=(196,384)`, `ACTION_DIM=2`.

- [x] Read the **installed source** + entrypoints `scripts/train/{lewm,prejepa}.py`
  (record provenance). Capture signatures for: `World` + `World.evaluate` (CEM/MPC), the
  CEM solver config, the Push-T env id (`swm/PushT-v1`), the latent extraction path (LeWM
  single token vs DINO-WM patch grid), and the config-injected frozen backbone
  (`encoder.eval(); requires_grad_(False)`).
- [x] Confirm encoder is **DINOv3, not DINOv2**, exposing `config.hidden_size` +
  `last_hidden_state`; verify token layout, slice **CLS + register tokens**, record
  `N_patches` and `D`. Confirm LeWM single-token latent `(B, D)`.
- [x] Record findings in `docs/platform_api.md`: adapter call shapes, dims, and the CEM
  planning-cycle decomposition into encoder / predictor / planner (for the Phase-5 profile).

**🔴 OWNER gate — CLEARED:** `LATENT_DIM=192`, DINO-WM `(N_patches, D)=(196, 384)`,
`ACTION_DIM=2`. Hard-coded once in `src/interfaces.py` / the adapter in Phase 4.

**Verify — PASSED:** introspection ran clean; DINOv3 attribute check passed; dims written
to `docs/platform_api.md` and owner-confirmed.

---

## Phase 2 — Foundation trainings  🟢 wiring · 🔴 config slot-in · 🖥️ ⏱️

Produce the two reference checkpoints.

- [ ] Vendor `scripts/train/lewm.py` and `scripts/train/prejepa.py` (GitHub tag `0.1.1`,
  not in the wheel — provenance in `scripts/train/VENDORED.md`) as used; wire Hydra + W&B.
- [ ] 🔴 **DINOv3 config + register-slice subclass** for `prejepa.py` (owner-approved,
  Phase-1 §6) — wiring done, on-pod slot-in confirmation pending:
  - `conf/experiment/dinov3.yaml` overlay (via `--config-dir conf +experiment=dinov3`):
    `backbone.name/type=dinov3_small`, `patch_size=16`.
  - `src/dino_patch.py::DINOv3PreJEPA` — `PreJEPA` subclass overriding `_encode_image`
    slice `[:, 1:, :]` → `[:, 1+num_reg:, :]`, injected via `model._target_`; reused by
    Phase-3 eval. Owner confirms slot-in on the pod (import + one forward).
- [ ] **Pre-flight before any GPU run:** `STABLEWM_HOME` points at the persistent network
  volume (not `~`); Push-T dataset resolves and one batch streams via HF.
- [ ] 🖥️ Train LeWM:
  `uv run python -m scripts.train.lewm --config-dir conf +experiment=lewm` → `$STABLEWM_HOME/checkpoints/lewm/`.
- [ ] 🖥️ Train DINOv3-WM:
  `uv run python -m scripts.train.prejepa --config-dir conf +experiment=dinov3` → `$STABLEWM_HOME/checkpoints/dino/`.
- [ ] ⏱️ **Training is epoch-capped** — LeWM 10, DINO-WM 100 (set in the conf overlays);
  no wall-clock cap.

**Verify:** two checkpoints exist; both W&B runs logged (URLs recorded here); encode
sanity confirms Phase-1 latent dims (LeWM `(B, 192)`, DINO-WM `(B, 196, 384)`).
**Log-before-delete:** confirm logged to W&B or committed before overwriting (CLAUDE.md §7).

---

## Phase 3 — Task baseline (platform CEM/MPC eval)  🟢 wiring · 🔴 parity · 🖥️

- [ ] 🖥️ Run `World.evaluate` (CEM solver) for **both** tracks: Push-T **success rate** +
  **planning latency**.
- [ ] 🔴 **Parity (load-bearing):** same CEM config (300 samples, 30 elites, horizon 5,
  init var 1, 10–30 iters), same action budget, same goal encoding, same eval seeds,
  identical ImageNet normalization — confirm **not varied between tracks** (do not change
  the platform eval/CEM config).

**Verify:** success-rate + latency for both tracks, logged to W&B; parity conditions
recorded as identical.

---

## Phase 4 — Owned adapter + tracer bullet  🟢 · (sole pre-optimization check)

- [ ] Implement **`WMStepAdapter`** as two classes (`LeWMAdapter` single-token latent,
  `DINOWMAdapter` patch-grid latent) behind a common `encode`/`predict` signature, typed
  per `src/interfaces.py`. The adapter wraps the model (encoder + predictor); the CEM
  planner / rollout loop stays in Python outside it.
- [ ] Constants (`LATENT_DIM`, DINO-WM patch-grid `(N_patches, D)`, `ACTION_DIM`) defined
  **once** in `interfaces.py` from the Phase-1 values; platform dims read from config.
- [ ] Implement `export()` and `benchmark()` **stubs** conforming to the `Export` /
  `Benchmark` Protocols + `ExportConfig`.
- [ ] `src/smoke.py`: dummy checkpoint → adapter → export-stub → benchmark-stub, with
  jaxtyping + beartype assertions at **every owned boundary**.
- [ ] `tests/` covering adapter shapes and the typed boundaries.

**Verify:** `uv run python -m src.smoke` passes; `uv run pytest -v` green; a
shape/precision violation actually raises.

---

## Phase 5 — Speedup study: export, profile & fixed-budget benchmark  🔴 OWNER-heavy · 🖥️ ⏱️

Owner makes the silent-failure calls; Claude owns plumbing (trace call, builder
invocation, percentile timing, memory logging, profiler hooks, table runner).

**Benchmark methodology.** Only the **model** (encoder + predictor, via `WMStepAdapter`)
is TensorRT-optimized; the **CEM planner stays in Python**. Evaluate both under a **fixed
wall-clock time budget**; compare **(a) per-step inference latency (p50/p95)** and **(b)
CEM rollouts completed**. **Every speed number is paired with an SR** (Phase-3 eval per
precision). (See SPEC §Parity, `src/interfaces.py`.)

- [ ] 🔴 Real export **PyTorch→ONNX→TensorRT**, **FP32→FP16→INT8**, per model:
  `uv run python -m src.export model=<lewm|dino> precision=<fp32|fp16|int8>`. ONNX/TRT
  debugging, INT8 **calibration set + procedure**, and FP32/FP16/INT8 **precision
  matching** are OWNER-ONLY — STOP and ask.
- [ ] 🖥️ **Per-component profiling** — encoder, predictor, and planner (CEM) separately,
  per planning cycle, for both models × precisions (`src/profile.py` or a `benchmark`
  mode). Use the Phase-1 cycle decomposition.
- [ ] 🖥️ **Fixed-time-budget benchmark** on the L40S: per model × precision, record
  **rollouts completed**, **per-step latency p50/p95**, throughput (rollouts/sec), **peak
  GPU memory**, **and SR** (Phase-3 eval re-run on the optimized model). Same
  env/goal/precision/budget across models; only the model differs.
- [ ] Headline outputs (tables **and plots**): **LeWM-vs-DINOv3 rollouts-in-budget ratio**
  + **p95 latency ratio**; **per-model FP32→FP16→INT8 delta** in **both speed and SR,
  degradation quoted vs FP32**; **speed-vs-SR plotted**; **per-component
  (encoder/predictor/planner) bottleneck breakdown**.
- [ ] ⏱️ **Cap on TensorRT/INT8** (unsupported-op / calibration); fallback = **FP16-only**.
  3-attempt debugging cap (CLAUDE.md §6).

**Interface note:** `src/interfaces.py` declares `BenchResult.rollouts_completed`, the
fixed `time_budget_s` on `Benchmark` / `ExportConfig`, and `ComponentProfile` / `Profile`.

**Verify:** engines built on the L40S (gitignored, regenerable); fixed-budget comparison
(rollouts + p95 latency **+ SR per precision**, FP32-relative degradation quoted) and the
encoder/predictor/planner profile tables produced and logged to W&B.

---

## Phase 6 — QLoRA delta on DINOv3-WM  🔴 targeting · 🖥️

- [ ] 🔴 **OWNER specifies QLoRA targeting:** which DINOv3 modules, rank, what stays
  frozen. Claude owns the training-loop wiring only.
- [ ] 🖥️ QLoRA fine-tune the DINOv3 backbone on Push-T (`peft` + `bitsandbytes`):
  `uv run python -m src.qlora`. **Predictor unfrozen and co-trained.** Confirm adapters
  target **real** modules (introspect, don't assume).
- [ ] 🖥️ Re-run the Phase-3 task-quality metric on the tuned backbone; report the **delta
  vs frozen DINOv3-WM**.

**Verify:** tuned checkpoint produced; task-metric delta vs frozen reported and logged to
W&B; adapter target modules confirmed real.

---

## Critical files

- `src/interfaces.py` — typed contract (declares the fixed-budget benchmark +
  per-component profile; dim constants filled in Phase 4 from Phase-1 values).
- `src/adapter.py`, `src/export.py`, `src/benchmark.py`, `src/profile.py`,
  `src/qlora.py`, `src/smoke.py` — the owned layer (Phases 4–6).
- `conf/` — owned Hydra overlays (incl. `conf/experiment/{lewm,dinov3}.yaml`).
- `scripts/train/lewm.py`, `scripts/train/prejepa.py` + `scripts/train/config/` —
  vendored platform entrypoints/configs, as used (provenance in `scripts/train/VENDORED.md`).
- `pyproject.toml`, `uv.lock`, `setup.sh` (pod bootstrap). `Dockerfile` +
  `docker-compose.yml` composed at project end (off-pod).
- `tests/` — pytest for the owned boundaries.

## Cross-cutting rules

- **Owner gates:** anything 🔴 (export/INT8 debugging, precision matching, QLoRA
  targeting, benchmark methodology, adapter dims, eval/CEM parity) → STOP and ask.
- **Git:** never run git. On completing a unit of work, output the files to stage and a
  `type(scope): summary` commit message; the owner runs git.
- **Progress:** each `[x]` records W&B URL + artifact name; tick before advancing.
- **Caps:** TRT/INT8 is time-capped with a fallback; training is epoch-capped; 3-attempt
  debugging cap; log-before-delete.

## End-to-end verification

1. `bash setup.sh` (uv + deps + TensorRT) + import check (Phase 0).
2. Platform API introspection in-container; DINOv3 `config.hidden_size`/
   `last_hidden_state` confirmed (Phase 1).
3. Two checkpoints + W&B runs (Phase 2); both-track success-rate + latency under matched
   CEM config (Phase 3).
4. `python -m src.smoke` + `pytest -v` green on dummy weights (Phase 4).
5. `src.export` builds TRT engines on the L40S; fixed-budget benchmark emits
   rollouts-in-budget + latency comparison and the encoder/predictor/planner profile
   tables (Phase 5).
6. `src.qlora` produces a tuned backbone; task-metric delta vs frozen reported (Phase 6).
