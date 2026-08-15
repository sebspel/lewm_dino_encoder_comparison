# PLAN.md — LeWM vs DINOv3-WM: Inference Optimization Study

> Execution steps only. **What** the project must satisfy → `SPEC.md`. **Why** a design is
> shaped this way → `docs/architecture.md`. Behavioral rules → `CLAUDE.md`.
> Typed contract → `src/interfaces.py`.
> Every completed step records `[x]` + artifact name (commit hash added by the owner).
> Tick a box before the next step.

## Context

**Execution model.** Inference (export + benchmark) and LeWM training run on the **L40S RunPod
pod**; **DINOv3-WM training runs on an H200 SXM** (SPEC §Execution Environment). Local WSL is
edit-only. Every run command is `uv run …` on the pod (provisioned by `setup.sh`). The five
SPEC requirement bundles map to the Phases below.

**Legend.** 🟢 CLAUDE-CODE owns (fails loud). 🔴 OWNER-ONLY — STOP and ask (SPEC §Implementation
Boundaries). 🖥️ runs on the L40S GPU. ⏱️ capped effort with a stated fallback.

**Tick state.** A box is `[x]` when the artifact it records stands. Boxes recording landed **code**
or an owner **decision** are `[x]` once that lands; boxes whose artifact is an **engine plan, a
measured number, or a rendered view** are `[ ]` while that artifact is outstanding — Phase 10 is
what produces them.

---

## Phase 0 — Scaffolding & pinned dependencies  🟢

- [x] `pyproject.toml` + `uv.lock` pinning: `stable-worldmodel`, `stable-pretraining`,
  `hydra-core`, `wandb`, `jaxtyping`, `beartype`, `onnx`, `transformers`, `timm`,
  **torch (cu124 wheel index)** — all uv-managed. **TensorRT NOT in uv**
  (installed by `setup.sh`). Versions pinned. → `docs/architecture.md` §6
- [x] `setup.sh` — pod bootstrap, idempotent: installs **uv**, runs `uv sync`, then installs
  **TensorRT (cu12, CUDA-12.4)** outside the lock. Secrets from the pod's runtime env.
- [x] Skeleton dirs: `conf/` (Hydra), `tests/` (pytest).
- [x] **Deferred to project end:** `Dockerfile` + `docker-compose.yml`.

**Verify (on the pod):** `bash setup.sh` succeeds; `uv run python -c "import stable_worldmodel,
stable_pretraining, tensorrt, torch"`; `uv run pytest -v`.

---

## Phase 1 — Read the real platform API  🟢 → 🔴 (dims sign-off)

> **Status: COMPLETE (2026-06-26).** Findings in `docs/platform_api.md` (provenance: swm 0.1.1 /
> sp 0.1.7 sdists + GitHub tag 0.1.1).

- [x] Read the **installed source** + entrypoints `scripts/train/{lewm,prejepa}.py` (record
  provenance). Capture signatures for `World` + `World.evaluate`, the CEM solver config, the
  Push-T env id (`swm/PushT-v1`), the latent extraction path, and the config-injected frozen
  backbone. → `docs/platform_api.md`
- [x] Confirm encoder is **DINOv3, not DINOv2**; verify token layout, slice **CLS + register
  tokens**, record `N_patches` and `D`. Confirm LeWM single-token latent `(B, D)`.
- [x] Record adapter call shapes, dims, and the CEM planning-cycle decomposition in
  `docs/platform_api.md`.

**🔴 OWNER gate — CLEARED:** `LATENT_DIM=192`, DINO-WM `(N_patches, D)=(196, 384)`,
`ACTION_DIM=2`; slice CLS+registers, stay on `dinov3_small` (`docs/platform_api.md` §6).
Hard-coded once in `src/interfaces.py` in Phase 4.

**Verify — PASSED:** introspection clean; DINOv3 attribute check passed; dims owner-confirmed.

---

## Phase 2 — Foundation trainings  🟢 wiring · 🔴 config slot-in · 🖥️ ⏱️

- [x] Vendor `scripts/train/lewm.py` + `scripts/train/prejepa.py` (GitHub tag `0.1.1`; provenance
  in `scripts/train/VENDORED.md`); wire Hydra + W&B.
- [x] 🔴 **DINOv3 config + register-slice subclass** (owner-approved, Phase-1 §6):
  - `conf/experiment/dinov3.yaml` (via `--config-dir conf +experiment=dinov3`):
    `backbone.name/type=dinov3_small`, `patch_size=16`.
  - `src/dino_patch.py::DINOv3PreJEPA` — `PreJEPA` subclass overriding `_encode_image`
    (`[:, 1+num_reg:, :]`), injected via `model._target_`; reused by Phase-3 eval.
- [x] **Pre-flight before any GPU run:** `STABLEWM_HOME` points at the persistent network volume;
  Push-T expert dataset pre-downloaded to `$STABLEWM_HOME/datasets/` (`hf download
  galilai-group/lewm-pusht --repo-type dataset --include "pusht_expert_train.lance/*"`, wired into
  `setup.sh` §8); resolves and one batch loads.
- [x] 🖥️ Train LeWM: `uv run python -m scripts.train.lewm --config-dir conf +experiment=lewm`
  → `$STABLEWM_HOME/checkpoints/lewm/`.
- [x] 🖥️ Train DINOv3-WM (**on an H200 SXM**, not the L40S — SPEC §Execution Environment):
  `uv run python -m scripts.train.prejepa --config-dir conf +experiment=dinov3` →
  `$STABLEWM_HOME/checkpoints/dino/`.
- [x] ⏱️ Epoch-capped: **10 epochs both tracks**, batch size 128 (set in the conf overlays).

**Verify:** two checkpoints exist; both W&B runs logged; `uv run python -m scripts.verify_encode`
passes (LeWM CLS `(B, 192)`, DINO-WM grid `(B, T, 196, 384)`; override 196 vs base 200,
`num_reg=4`). Expected tail:
```
[DINO-WM] override grid (2, 3, 196, 384) vs base (2, 3, 200, 384) (D=384, num_reg=4) -> 196 patches OK
[LeWM] CLS latent (2, 192) -> single token, D=192 OK
encode sanity: PASS
```
Valid pre- or post-training; also satisfies the §2 slot-in "import + one forward".

---

## Phase 3 — Task baseline (platform CEM/MPC eval)  🟢 wiring · 🔴 parity · 🖥️

- [x] **Pod-confirm** `scripts/plan/eval_wm.py` is absent from the installed wheel before
  vendoring.
- [x] Vendor the eval entrypoint **as used**: `scripts/plan/eval_wm.py` + `scripts/plan/config/
  {pusht.yaml, solver/cem.yaml, launcher/local.yaml}` (tag `0.1.1`; provenance in
  `scripts/plan/VENDORED.md`). **Stays byte-unmodified.**
- [x] Owned **W&B helper** `src/wandb_log.py` — `init()` reads project/entity from the
  `conf/experiment/` `wandb:` block. Reused by Phases 5–6.
- [x] Owned **observation-only CEM latency callback** `src/eval_latency.py` (+
  `tests/test_eval_latency.py`) — `CEMSolver.Callback` subclass injected via `cfg.solver.callbacks`.
  → **Bracket superseded in Phase 5** (`docs/architecture.md` §8); this box records Phase 3 as shipped.
- [x] Owned **eval driver** `src/eval.py` — opens the W&B run, invokes byte-unmodified
  `eval_wm.run`, captures `World.evaluate`'s metrics, logs SR + latency.
- [x] Owned **eval overlays** `conf/experiment/eval_{lewm,dino}.yaml` (`@package _global_`): set
  `policy=<ckpt run_name>`, `eval.dataset_name`, the `wandb:` block, and inject the latency
  callback via `cfg.solver.callbacks`. The training overlays are **not** reused for eval (they
  leave `policy=random`).
- [x] 🖥️ Run both tracks: `uv run python -m src.eval --config-dir conf
  +experiment=eval_<lewm|dino>`. Pod-confirm: the checkpoint's `model._target_` reconstructs the
  register-slice subclass (196-grid); SR unchanged vs a callback-free run.
- [x] 🔴 **Parity (load-bearing):** same CEM config (300 samples, 30 elites, horizon 5, init var 1,
  10–30 iters), action budget, goal encoding, eval seeds, ImageNet normalization — confirm **not
  varied between tracks**.

**Verify:** SR + latency for both tracks logged to W&B; SR identical with/without the callback;
parity conditions recorded as identical.

---

## Phase 4 — Owned adapter + tracer bullet  🟢 · 🔴 (adapter-dims sign-off)

- [x] **Reconcile `src/interfaces.py` to the two-method adapter** — `encode(obs)` +
  `predict(latent, action)` replacing the fused `__call__`; `Export`/`Benchmark` Protocols target
  `encode` and `predict` separately and yield both engine paths.
- [x] Constants defined **once** in `interfaces.py`: `LATENT_DIM=192`, `(N_patches, D)=(196, 384)`,
  `ACTION_DIM=2`, predictor-input width `404`.
- [x] 🔴 **OWNER gate — adapter dims:** confirm the DINO-WM `predict` boundary widths by
  introspecting the real predictor on the pod before hard-coding.
- [x] Implement **`WMStepAdapter`** as `LeWMAdapter` + `DINOWMAdapter` behind the common
  `encode`/`predict` signature. → `src/adapter.py`
- [x] Implement `export()` / `benchmark()` **stubs** conforming to the Protocols + `ExportConfig`.
- [x] `src/smoke.py`: dummy checkpoint → adapter → export-stub → benchmark-stub, with
  jaxtyping + beartype assertions at **every** owned boundary.
- [x] `tests/` covering adapter shapes (both methods, both tracks) and the typed boundaries.

**Verify:** `uv run python -m src.smoke` passes; `uv run pytest -v` green; a shape/precision
violation raises at both the `encode` and `predict` boundaries.

---

## Phase 5 — Speedup study: export, profile & latency benchmark  🔴 OWNER-heavy · 🖥️ ⏱️

Owner makes the silent-failure calls; Claude owns plumbing. Methodology → SPEC §Interface
Contracts; statistic split → `docs/architecture.md` §8; cycle definition → `docs/architecture.md` §8.

- [x] 🟢 **Checkpoint → adapter loader** — materialize each checkpoint via the platform
  `load_pretrained` (reuse the Phase-3 load path), wrap in `LeWM`/`DINOWMAdapter`.
  → `src/precision_match.py::_build_adapter` (pod-verify pending: needs real checkpoints)
- [x] 🔴 **DINO-WM `predict` — faithful `404 → 404` reconstruction**; extras embedding, `384 → 404`
  assembly, per-step action-replace + proprio-carry moved to the Python rollout/shim.
  → `src/adapter.py` (`predict`, `assemble_embedding`), `src/shim.py` (`_replace_action`,
  `dino_rollout`); `interfaces.py` predict I/O + example inputs updated.
  → verify: predict output width == 404 ✔
- [x] 🟢 **Adapter-fidelity gate (before export):** adapter `encode` + `predict` + shim rollout +
  criterion reproduces the platform `rollout`/`get_cost` within tolerance on the real checkpoint.
  → `src/fidelity.py` + `tests/test_fidelity.py` (bit-for-bit, max_abs 0.0 on a dummy
  `DINOv3PreJEPA`). Pod: `uv run python -m src.fidelity` on the real checkpoint.
  → LeWM per-frame `action_encoder` boundary owner-signed-off 2026-07-11
  (`src/fidelity.py::lewm_action_encoder_per_frame`).
- [x] 🔴 **Real export PyTorch→ONNX→TensorRT:** `uv run python -m src.export model=<lewm|dino>
  precision=<fp32|fp16|int8>`. `torch.onnx.export(dynamo=True)` aimed at a thin `nn.Module`
  wrapper per method → **4 base ONNX graphs** (2 methods × 2 models). FP32/FP16 build data-free;
  INT8 deferred to the quantization step below.
  → `src/export.py`: `_Predict1Module` (DINO) / `_Predict2Module` (LeWM) explicit-arity trace
  wrappers; `model=`/`precision=` CLI; writes to `engines/<track>/`.
  → **per-component TRT optimization profile** `export._BATCH_PROFILE` — encoder `(1, 1, 1)`,
  predictor `(1, CEM_NUM_SAMPLES, CEM_NUM_SAMPLES)`; `build_engine(batch_profile=)` takes it
  explicitly (never inferred from the trace batch, which fixes only the non-batch axes and the
  modelopt feed shape). `export_onnx` raises on a size-1 trace batch — `torch.export` specializes
  it and freezes the ONNX axis silently. `docs/architecture.md` §6, `tests/test_export.py`.
  → verify (off-pod ✔): all 4 ONNX graphs trace with a dynamic batch axis; `pytest` green.
  TRT build + precision matching are pod-only 🔴.
  → verify (pod ✔ 2026-08-08): 24 plans built; deserialized profiles carry the intended
  `_BATCH_PROFILE` on every one — encoder `opt=1`, predictor `opt=300`, both tracks.
- [x] 🔴 **INT8 explicit quantization (Model Optimizer PTQ):** base FP32 ONNX →
  `modelopt.onnx.quantization` (Q/DQ + per-tensor scales) → quantized ONNX per method →
  `build_engine` (no `int8_calibrator`, no calibration profile). Sequenced **before** the
  precision-match gate. → `docs/architecture.md` §6
  → `setup.sh` installs `nvidia-modelopt[onnx]` + CUDA-12 `onnxruntime-gpu` (out of uv) and
  sanity-opens an ORT CUDA-EP session. Pins: `modelopt==0.43.0`, `onnxruntime-gpu==1.24.4`.
  → `src/calibrate.py`: clip draw + per-method streams kept; `make_calibrator` →
  `make_calibration_dict` (numpy dict keyed off the base ONNX input names).
  → `src/export.py`: `quantize_onnx` (`calibration_method="max"`, `use_external_data_format=True`);
  `build_engine` calibrator/profile branch dropped. `interfaces.py` re-documents `calib_loader`.
  → **calibration EP split:** encoder on CUDA EP, predictor on CPU EP
  (`quantize_onnx(force_cpu_calibration=name=="predictor")`) — `docs/architecture.md` §6.
  → verify (off-pod ✔): `pytest` green. 🔴 owner-confirm on pod: non-`max` modelopt knobs left at
  INT8 defaults; keeping the TRT INT8 flag for a Q/DQ graph; `onnx` lock harmonization (modelopt
  caps 1.19.1 vs locked 1.22.0).
  → verify (pod): quantized ONNX carries QuantizeLinear + INT8 engine builds; encoder binds CUDA
  EP, predictor binds CPU EP.

- [x] 🔴🖥️ **Calibration-distribution fix — INT8 SR collapse** (reopens the INT8 box above).
  Observed: lewm FP32 94% / FP16 96% / **INT8 48%**. Diagnosis + decision → `docs/architecture.md` §7.
  - [x] `src/interfaces.py` — `CEM_VAR_SCALE=1.0`, `CEM_HORIZON=5`, `EVAL_N_OBS=1` beside
    `CEM_NUM_SAMPLES`. 🔴 confirm vs source on pod.
  - [x] `src/calibrate.py::predictor_batches` — `_sample_cem_actions` (`randn * CEM_VAR_SCALE`,
    zero mean, unclamped, seeded) replaces expert `a`; rolls `predict` over `CEM_HORIZON`
    (`_ROLL_FRAMES = CEM_HORIZON`) capturing its own inputs; keeps only `T == HISTORY_SIZE`
    windows. DINO drives the real `shim.dino_rollout` via `_CaptureAdapter`; LeWM mirrors
    `LeWM.rollout`'s window loop. `encoder_batches` unchanged.
    → verify (off-pod ✔): `pytest` 70 passed; expert bound 1.0 vs proposal max **4.34** (~4×),
    **32.1%** of action values clipped at the old scale.
  - [x] `src/probe_ranges.py` — read-only range probe: `uv run python -m src.probe_ranges
    [track=<lewm|dino>] [n_clips=<int>]`. Diagnostic only (INPUT tensors; a clean result does not
    exonerate internals), **not** a post-fix check — SR is the post-fix verifier.
  - [x] 🖥️ **Run the probe on the pod, both tracks** (real checkpoints + dataset).
    → verify: action ratio ≈ 4× → mechanism confirmed, proceed. Ratio ≈ 1 → the drawn actions are
    not box-bounded, the ~4× premise is wrong → **STOP and re-diagnose**. Latent ratio sizes the
    second axis.
  - [x] 🖥️ Re-run INT8 PTQ + rebuild engines, both tracks: `uv run python -m src.export
    model=<lewm|dino> precision=int8`.
    → verify: quantized ONNX carries QuantizeLinear; INT8 engine builds.
  - [x] 🖥️🔴 Re-run `uv run python -m src.precision_match track=<lewm|dino>` → new INT8 drift rows;
    owner sign-off. **Not the arbiter** (nominal inputs — SPEC §Requirements).
  - [x] 🖥️ Re-run `uv run python -m src.sr_eval --config-dir conf +experiment=eval_<lewm|dino>
    precision=int8`, both tracks.
    → verify (**the real gate**): lewm INT8 SR recovers toward FP16 (96%). Still collapsed →
    record the INT8 row as degraded and advance.

- [x] 🖥️🔴 **Precision-match gate (before profiling/benchmark):** `uv run python -m
  src.precision_match track=<lewm|dino>` on the **real** FP32+FP16+INT8 engines →
  engine-vs-PyTorch drift table. **No coded pass/fail** — the gate is the owner's sign-off on the
  drift table. Each engine is exercised at **its own** profile points — `_MATCH_BATCHES` carries
  `(encoder batch, predictor batch)` pairs and the table has `enc_b`/`pred_b` columns (SPEC
  §Requirements — engine-fidelity gate).
  → **OWNER SIGN-OFF (2026-07-10):** drift judged on **abs** only (`max_rel` is a
  near-zero-denominator artifact — disregarded). FP32 engines faithful (lewm enc_abs ≤7e-3, dino
  ≤5.6e-2) → export/assembly/register-slice/reshape sound. **FP32 trusted**; **FP16
  trusted-provisional** (pending SR); **INT8 recorded-but-flagged degraded** (lewm borderline
  ~0.6–1.0, dino ~3–4), SR is the arbiter. All rows kept.
  → **Gap found + fixed (2026-07-14):** `_MATCH_BATCHES` varied the batch axis only at the traced
  hist, so the gate never caught the fixed-`HS` predict engine failing on `T < HS` windows. Added
  off-nominal history rows `_MATCH_HISTS=(1,2)` routed through the shim's hist-adapt wrappers
  (a `hist` table column); `sr_cost_parity*` now also run at `n_obs=1`. Both tracks. 🔴 owner-gated
  (`docs/architecture.md` §4).

- [x] 🖥️ **Per-component decomposition** — encoder / predictor / **overhead** per cycle, both
  models × precisions, derived in `src/report.py::decompose` from the benchmark's isolated
  engine-step **means** × the CEM per-cycle call counts; `overhead = cycle − enc − pred`; negative
  overhead **surfaced loudly**. Reports the shares + `p=(enc+pred)/cycle` + Amdahl ceiling
  `1/(1-p)`.
  → **landed (pod-run pending):** `src/profile.py` + its CEM mirror **retired**
  (`docs/architecture.md` §8); `ENCODER/PREDICTOR_CALLS_PER_CYCLE` moved to `interfaces.py`.
  **Call counts confirmed against source (2026-07-15):** 180 (unconfirmed guess) → **150**
  (`((5−1)+1)×30`); `ENCODER_CALLS_PER_CYCLE=2` unaffected. `test_profile.py` removed, covered by
  `tests/test_report.py`.
  → **statistic fixed to the MEAN** (`docs/architecture.md` §8): `decompose` + `dilution_disclosure` read
  `*_mean_ms`; `BenchResult` carries `per_cycle_mean_ms`/`encode_mean_ms`/`predict_mean_ms`;
  `report._finalize_per_cycle` computes the cycle mean off the same truncated sample as p50/p95.
  `tests/test_report.py::test_decompose_uses_mean_not_p50`.
  → 🔴 open: whether to drop a per-cycle warm-up (`docs/architecture.md` §8, accepted residual).

- [x] 🔴 **Per-decision latency bracket** (`docs/architecture.md` §8).
  - [x] `src/eval_latency.py` — bracket **per env** via consecutive `start_batch` hooks (last
    closing at `end_solve`), one record per decision, sync per span; `current_bs == 1` guard;
    `n_solves` → `n_cycles`. Consumers `src/eval.py` + `src/sr_eval.py` updated (W&B
    `cem_solve_*` → `per_cycle_*`). `tests/test_eval_latency.py`: a 50-env solve must record 50
    latencies. Stale "per-solve" wording corrected in `src/report.py`, `src/benchmark.py`,
    `conf/experiment/eval_{lewm,dino}.yaml`.
    → verify (off-pod ✔): `pytest` 73 passed.

- [x] 🖥️ **Latency benchmark** on the L40S, per model × precision — three equal-n p50/p95
  distributions: **per-cycle** (headline) off the per-decision callback over the SR eval-shim run;
  **encode-step** + **predictor-step** off isolated per-precision engine loops (`n_latency_iters=100`
  timed, `warmup=10` dropped). **Peak GPU memory** via `cudaMemGetInfo`/nvidia-smi. GPU clocks
  **not** locked; `src/gpu_clocks.py::log_gpu` brackets each timed run (both `src.benchmark`
  loops and the `src.sr_eval` run) → `$STABLEWM_HOME/reports/phase5/gpu_logs/<track>.<precision>.
  <phase>.dmon.log`. **SR** comes from the Phase-3 eval re-run on the engine-backed **`get_cost`-only**
  shim, so per-cycle latency and SR come from the **same solves**.
  → **instrument + labeling fixes (landed, pod-run pending):** `src/benchmark.py` `peak_mem_mb`
  samples `torch.cuda.mem_get_info` (device-level used), not `max_memory_allocated`.
  → **latency-methodology rework (landed, pod-run pending):** fixed-wall-clock loop **removed**;
  `rollouts_completed`/`throughput`/`time_budget_s` dropped. Per-cycle truncated to common min-n in
  `src/report.py`. `interfaces.BenchResult` carries the three distributions + means + peak mem.
  `tests/test_benchmark.py`, `tests/test_eval_latency.py`.
  → **owner-gated SR shim (DINO-WM) landed:** `src/sr_shim.py::DINOWMSRShim` subclasses
  `DINOv3PreJEPA`, overrides ONLY `_encode_image` + `predict`. `build_engine_fns(engines)` = pod
  EngineRunner callables; `.from_adapter` = torch path for the gate. Gate
  `src/sr_shim.py::sr_cost_parity` (+ `tests/test_sr_shim.py`): bit-for-bit vs `PreJEPA.get_cost`
  (max_abs 0.0, n_obs∈{1,3}); pod runs `uv run python -m src.sr_shim` on the real checkpoint.
  Encoder static-hist resolved shim-side via `_hist_adapt` (no re-export).
  → **LeWM shim landed (engine-backed, Design A):** `src/sr_shim.py::LeWMSRShim` routes `encode`
  and `predict` through engine callables; `action_encoder` set to Identity passthrough so rollout
  windows raw actions into the engine (`docs/architecture.md` §3). `build_lewm_engine_fns` +
  `from_engines`. Gate `sr_cost_parity_lewm` (+ `build_dummy_lewm_model`): bit-for-bit vs
  `LeWM.get_cost` (max_abs 0.0, n_obs∈{1,3}), run at **B=1**.
  → **static-hist PREDICT fix (2026-07-14, both tracks):** `_predict_hist_adapt` (right-pad to
  `HS`, run, slice `[:, :T]`) wraps the predictor callable in `build_engine_fns` (DINO, 1-input)
  and `build_lewm_engine_fns` (LeWM, 2-input). Engine byte-unchanged, no re-export. 🔴 owner-gated
  (`docs/architecture.md` §4).
  → **SR-per-precision driver landed (pod-run pending):** `src/sr_eval.py` (`uv run python -m
  src.sr_eval --config-dir conf +experiment=eval_<lewm|dino> [precision=fp32,fp16,int8]
  [out=<dir>]`) re-runs byte-unmodified `scripts.plan.eval_wm.run` on the shim per built precision
  and writes `{track:{precision:{success_rate, per_cycle_latencies_ms}}}` `sr.json` that
  `src.study`/`src.report` join via `sr=<file>` (read-modify-write per track key → no clobber,
  CLAUDE.md §8). Missing engines → precision skipped. `tests/test_sr_eval.py`
  (CPU; the eval leg is pod-only). 🔴 owner-confirm on pod: the `load_pretrained` patch-seam
  (`docs/architecture.md` §5).

- [x] Headline outputs (tables **and** plots): **LeWM-vs-DINOv3 per-cycle latency ratio at p50**
  (p95 alongside); **per-model FP32→FP16→INT8 delta** in **both speed and SR**, degradation quoted
  vs FP32 (p50 speedup + SR delta in the same row); **speed-vs-SR plotted**; **per-component
  breakdown** with all three p50/p95 distributions; per model × precision **both** the model-only
  and realized speedup alongside `p` (SPEC §Speedup study — dilution disclosure).
  → **statistic ruling landed (pod-run pending):** `per_cycle_ratio` defaults to **p50**;
  `plot_per_cycle_ratio` + W&B `headline/per_cycle_p50_ratio_*` follow it. `render_speed_table`
  renders all three distributions at p50 **and** p95 (the encode/predict p95s previously reached no
  table). `docs/architecture.md` §8.
  → **`fp32_relative` wired in (was dead code):** now quoted at **p50**, `base` guarded with
  `.get`, rendered as `fp32_relative_table.txt` (speedup + ΔSR in one row). Distinct from
  `dilution_disclosure`'s mean-based `measured_realized_speedup`. `tests/test_report.py` asserts
  the table *renders*.
  → **table runner (landed, pod-run pending):** `src/study.py` (`uv run python -m src.study
  [track=<lewm|dino>] [wandb=<eval overlay>]`) loads the engines from
  `engines/<track>/{encoder,predictor}.<prec>.plan`, benchmarks each built precision, dumps
  `results.<track>.json`, calls `src.report`. Missing engines → precision skipped; per-cycle + SR
  left NaN for the gated eval-shim join. `tests/test_study.py`.
  → **honesty fixes in `src/report.py` (landed):** Amdahl dilution table (p, ceiling, model-only vs
  realized); every unpaired-SR row flagged **SR-PENDING**; `sr=<json>` / `report(sr_overrides=)`
  joins the gated SR + per-cycle latency without code edits.
  → **canonical per-track results JSON + decoupled render (landed):** `src/study.py
  ::dump_track_results` writes each track's raw numbers + fairness conditions (num_samples, seed,
  obs_shape) to `results.<track>.json` **before** rendering. `src/report.py` gains `load_results` +
  `from=<dir|file>` (`uv run python -m src.report from=$STABLEWM_HOME/reports/phase5 [out=<dir>]
  [sr=<f.json>] [wandb=<ov>]`) to re-render **off-pod**. Single-track render **skips** the two
  cross-track ratio plots.
  → verify: `results.{lewm,dino}.json` round-trip through `report.load_results`; a one-track render
  emits no `*_ratio.png`.
  - [x] **Persist headline artifacts to network storage (pending):** `src/report.py` serializes
    each table to `.txt` (currently stdout + W&B HTML only); `src/study.py` defaults `out_dir`
    under `$STABLEWM_HOME` (env-derived, repo-local fallback).
    → verify: after a study run the three table `.txt` + four plot `.png` files exist under
    `$STABLEWM_HOME/reports/phase5/`.

**Verify:** engines built on the L40S (gitignored, regenerable); precision-match gate passed
(drift owner-signed-off) **before** benchmarking; the three equal-n p50/p95 distributions + peak
mem + SR per precision (FP32-relative degradation quoted) and the encoder/predictor/overhead
tables produced and logged to W&B.

---

## Phase 6 — FP8 precision (L40S)  🔴 quant config · 🟢 wiring · 🖥️ ⏱️

FP8 (E4M3) is a second explicitly-quantized 8-bit format, built and benchmarked exactly like
INT8 on the L40S's native FP8 Tensor Cores (Ada 4th-gen / Transformer Engine) — no toolchain or
hardware change. Owner sets the silent-failure quant config; Claude owns the wiring. It extends
the Phase-5 sweep to FP32→FP16→INT8→FP8, so every Phase-5 recording, table, and plot gains the
FP8 rows/points, reported vs FP32 like the others. Methodology → SPEC §Interface Contracts;
statistic split → `docs/architecture.md` §8.

- [x] 🟢 **Precision plumbing:** `Precision` literal + `ExportConfig.precisions` gain `fp8`
  (`src/interfaces.py`). Audit `src/report.py`, `src/study.py`, `src/benchmark.py`,
  `src/sr_eval.py` for any hard-coded `{fp32,fp16,int8}` set so FP8 flows through the recordings,
  tables, and plots off the precision tuple — not a fourth special case.
  → **landed:** `QUANTIZED_PRECISIONS=("int8","fp8")` added to `interfaces.py` (single source
  for the calib-required set); `Precision` literal + `ExportConfig.precisions` gain `fp8`.
  `report._PRECISIONS` + the speed-vs-SR marker map gain `fp8` (`D`); `precision_match` default
  tuple + calib-loader branch generalized to `QUANTIZED_PRECISIONS`. `study`/`sr_eval` iterate
  `cfg.precisions` (auto), `benchmark` is precision-agnostic (untouched), `calibrate` reused
  unchanged (format-independent).
  → verify (off-pod ✔): `pytest` green; precision loops iterate the config tuple incl. `fp8`.

- [x] 🔴 **FP8 export/build wiring (owner sets quant config):** `src/export.py` — `quantize_onnx`
  gains a quant-mode arg and passes `quantize_mode="fp8"` to the Model Optimizer (E4M3 Q/DQ,
  `calibration_method="max"` kept); `build_engine` gains an `fp8` branch setting
  `BuilderFlag.FP8` + `BuilderFlag.FP16` (heavy layers FP8, remainder FP16, mirroring INT8+FP16);
  the `precision == "int8"` gates in `export`/`main` generalize to the quantized set
  `{int8, fp8}`. Calibration streams (`src/calibrate.py`) reused unchanged (format-independent).
  → `docs/architecture.md` §6
  → **landed:** `quantize_onnx(quant_mode=)` threads `quantize_mode` to modelopt ONLY for the
  non-default format (the owner-signed-off INT8 call stays byte-identical); `build_engine` `fp8`
  branch (`BuilderFlag.FP8`+FP16); quantized-ONNX filename per precision
  (`{name}.{precision}.onnx`, no int8/fp8 collision); all `int8` gates → `QUANTIZED_PRECISIONS`.
  🔴 pod-verify: the modelopt `quantize_mode` kwarg + `BuilderFlag.FP8` are the owner-set config,
  unverifiable off-pod (modelopt/tensorrt install pod-only) — fail loudly on the L40S if the API
  differs.
  → verify (off-pod ✔): `pytest` green; export imports (TRT/modelopt lazy). TRT build is
  pod-only 🔴.

- [x] 🔴 **Calibration method as a labelled build option — DINO INT8/FP8 SR collapse**
  (`docs/architecture.md` §7). Observed: DINO FP32/FP16 ~70% / **INT8 ~20% / FP8 2%**; LeWM ~98% /
  INT8 ~76%. The architecture.md §7 distribution fix recovered LeWM (action stressor, in-engine) but not DINO
  (outlier-heavy frozen-DINOv3 activations; `max` per-tensor amax saturates them). `entropy`
  (tail-clip) is the candidate lever — which method wins per track is an SR question, measured.
  - [x] 🔴 **Decision recorded** — calibration method (`max` | `entropy`) is a build option for
    **both** tracks and a **labelled result dimension** (track × precision × method); held constant
    across INT8+FP8 within a labelled comparison; existing `max` artefacts preserved (additive, never
    rewritten). → `docs/architecture.md` §7, SPEC §Parity + §Interface Contracts (Export shape).
  - [x] 🟢 **Plumb `calibration_method` (`max` | `entropy`)** as a build option in `src.export` and a
    label recorded in `results.<track>.json` (`src.study::dump_track_results`); a new method's runs
    are additive — do **not** overwrite existing `max`-labelled points (CLAUDE §8).
    → **landed (off-pod ✔):** `interfaces.calibration_method_for` (hidden per-track map) **retired** →
    `CALIBRATION_METHODS`+`DEFAULT_CALIBRATION_METHOD`+`check_calibration_method`; `calibration_method=`
    is a CLI build option for BOTH tracks in `src.export`/`src.precision_match`/`src.study`/`src.sr_eval`
    (default `max`). SR is method-labelled + additive: `sr.json` re-keyed `{track:{precision:{method:SR}}}`,
    `_merge_sr_json` now merges per **(track, precision, method)** — this also fixes the observed bug where
    a precision-subset run (e.g. `precision=fp8`) **replaced the whole track block** and dropped the
    other precisions. `results.<track>.json` merges per precision + records the method label (latency is
    method-invariant). `src.report calibration_method=<max|entropy>` selects which method's int8/fp8 SR to
    render like-for-like (legacy flat sr.json folded under `max`). **Engine plans method-TAGGED:** int8/fp8
    `.plan`+quantized `.onnx` now `…<precision>.<method>.plan` (`export.engine_filename`, single source
    shared with `study.engine_paths`), so a second method's engines are additive on the volume;
    `study`/`sr_eval` load by method (fp32/fp16 untagged/method-invariant; `method=max` falls back to the
    legacy untagged plan so pre-tagging engines aren't orphaned). `tests/{test_sr_eval,test_study,test_report}.py`.
  - [x] 🖥️ **Measure `entropy`, both tracks (INT8 first, additive):** rebuild INT8 with
    `calibration_method=entropy`, re-run `src.sr_eval +experiment=eval_<lewm|dino> precision=int8` →
    new `entropy`-labelled SR points beside the existing `max` ones.
    → verify: DINO-`entropy` INT8 moves off the ~20% floor; LeWM-`entropy` vs LeWM-`max` ~76% decides
    LeWM's headline method. Neither recovers DINO → per-tensor 8-bit loss inherent; report DINO 8-bit
    degraded vs FP32.
  - [x] 🖥️ **Extend the winning method(s) to FP8** and fold the labelled points into the headline;
    keep the existing `max`-labelled INT8/FP8 rows intact.

- [x] 🖥️ **Component-precision isolation — both tracks, both methods (`max` + `entropy`)**
  (`docs/architecture.md` §9).
  Attributes each material 8-bit SR drop to the encoder or the predictor. Diagnostic: composite
  `enc-<A>+pred-<B>` keys, never in the headline sweep. Two runs per affected
  (track, precision, **method**) cell — up to **16 evals** (2 sides × 2 precisions × 2 tracks ×
  2 methods).
  A row only explains a headline row rendered at the SAME method, so both methods' renders are
  covered.
  - [x] 🖥️ **Pure corners, both methods** — the 2×2's both-quantized corner AND the headline row the
    diagnostic explains: `uv run python -m src.sr_eval --config-dir conf
    +experiment=eval_<lewm|dino> precision=int8,fp8 calibration_method=entropy`; the `max` corners
    come from the default-method sweep (`precision=fp32,fp16,int8,fp8 calibration_method=max`,
    Phases 5–6 and the pipeline's `sr_eval` stage).
    → verify: `sr.json` carries `{track}.{int8,fp8}.{max,entropy}`; a render at **either**
    `calibration_method` shows no PEND in the int8/fp8 SR column.
  - [x] 🖥️ **LeWM `entropy` engines** (DINO's already built): `uv run python -m src.export
    model=lewm precision=<int8|fp8> calibration_method=entropy`.
    → verify: `engines/lewm/{encoder,predictor}.<p>.entropy.plan` exist; quantized ONNX carries
    QuantizeLinear.
    → **both methods' engines already stand** — the isolation needs no new export: the 24 plans
    confirmed in Phase 10 are 2 tracks × 2 components × {fp32, fp16, int8.max, int8.entropy,
    fp8.max, fp8.entropy}.
  - [x] 🖥️ **DINO isolation runs — `entropy`** (2026-07-21): int8 + fp8, both sides.
    → enc-fp16+pred-fp8 70.0 · enc-fp16+pred-int8 42.0 · enc-int8+pred-fp16 16.0 ·
    enc-fp8+pred-fp16 4.0 (FP16 baseline ~70). Encoder-dominant; predictor FP8-clean,
    INT8-sensitive. Recorded in `sr.json` under composite keys.
  - [x] 🖥️ **LeWM isolation runs — `entropy`** — `uv run python -m src.sr_eval --config-dir conf
    +experiment=eval_lewm encoder_precision=int8 predictor_precision=fp16
    calibration_method=entropy` and the reverse. FP8 only if LeWM FP8 also drops.
    → verify: composite keys land beside the pure points; pure SRs unchanged. architecture.md §7 predicts
    **predictor**-dominant damage (Design A puts `action_encoder` inside the predict engine) — a
    contrary result reopens that mechanism.
  - [x] 🖥️ **DINO isolation runs — `max`** — the same four runs at `calibration_method=max`:
    `uv run python -m src.sr_eval --config-dir conf +experiment=eval_dino
    encoder_precision=<int8|fp8> predictor_precision=fp16 calibration_method=max` and the reverse.
    → verify (✔): `sr.json` carries `dino.enc-<A>+pred-<B>.max` beside the `entropy` points; the
    `entropy` points are byte-unchanged (additive merge per (track, label, method)).
    → the attribution may legitimately DIFFER from the `entropy` result — different scales,
    different saturation (`docs/architecture.md` §7, §9). It is a second measurement, not a re-run.
    → 🔴 **the two methods' attributions differ on LeWM** — `docs/architecture.md` §7's
    predictor-dominant prediction (PLAN §Phase-6 isolation) is not what the `max` render shows.
    Which mechanism that reopens is an OWNER reading of the result, not a plumbing fix.
  - [x] 🖥️ **LeWM isolation runs — `max`** — the same four runs at `calibration_method=max`.
    → verify (✔): composite `max` keys land beside the pure `max` points; the `entropy` composites
    and all pure SRs unchanged.
  - [x] 🟢 **Isolation table** in `src/report.py`, rendered from the composite `sr.json` keys and
    placed after `fp32_relative_table`. Columns: track · precision · component quantized (other held
    fp16) · SR · ΔSR vs that track's FP16 · that component's per-cycle time share.
    → **landed (off-pod ✔):** `render_isolation_table` + `_parse_isolation_key` / `_ISOLATION_KEY`;
    written as `isolation_table.<method>.txt` only when isolation runs exist.
    `tests/test_report.py::test_isolation_table_attributes_component`,
    `::test_isolation_keys_never_reach_the_headline` (headline `.txt` byte-identical with and
    without composite keys). `pytest` 93 passed.
    → **already method-parametric — no code change for the second method:**
    `render_isolation_table(bench, overrides, method, …)` selects per (track, composite label,
    method) with **no** cross-method fallback (composite keys are not method-invariant), so a `max`
    render emits `isolation_table.max.txt` as soon as the `max` runs exist. Until then a `max`
    render emits no isolation table at all — the gap the runs above close.

- [x] 🟢 **Report labelling + provenance** (`docs/architecture.md` §7, `docs/architecture.md` §8).
  - [x] **Method-scoped headline tables:** the four single-method tables written as
    `<name>.<method>.txt`, each carrying a `calibration_method = <m>` line in the table body.
    → **landed (off-pod ✔):** `_method_line` + `render_*(bench, method)` (method optional, defaults
    to `max` so existing callers are unchanged); `report` writes the method-scoped names.
    `::test_headline_tables_are_method_scoped_and_labelled`.
  - [x] **`calibration_table.txt`** — int8/fp8 only, both methods side by side: track · prec ·
    `SR@max` · `SR@entropy` · `Δ(entropy−max)` · `headline` (which method the single-method tables
    were rendered at). Reads `sr.json`'s per-(track, precision, method) keys — **no**
    `results.<track>.json` schema change.
    → **landed (off-pod ✔):** `render_calibration_table`; unscoped filename (spans both methods);
    absent when no quantized SR exists. `::test_calibration_table_shows_both_methods`,
    `::test_calibration_table_absent_without_quantized_sr`.
  - [x] **`n` column on the speed table** — the post-truncation per-cycle sample count each row's
    percentiles and mean were computed from (`report._finalize_per_cycle`).
    → **landed (off-pod ✔):** `_finalize_per_cycle` stashes `_per_cycle_n`; `cyc_n` column.
    `::test_speed_table_reports_equal_n`.
  - [x] 🔴 **Per-cycle warm-up drop `k=1`** (owner-approved 2026-07-21; `docs/architecture.md` §8
    closes the "Open (owner)" item). The engine loops drop `warmup` iters, the per-cycle callback
    dropped none, so cold-start sat on one side of `overhead = cycle − enc − pred` only.
    → **landed (off-pod ✔):** `interfaces.PER_CYCLE_WARMUP_DROP = 1`; applied in
    `report._finalize_per_cycle(warmup_drop=)` **before** the equal-n truncation and at **report**
    time (sr.json's raw vector untouched → architecture.md §8 span-sum reconciliation still valid);
    `src.report per_cycle_warmup=0` re-renders the undropped view. Excluded decisions stashed
    (`_per_cycle_dropped_ms`) and DISCLOSED as the speed table's `drop×`.
    `::test_per_cycle_warmup_drops_cold_decision_and_discloses_it`,
    `::test_warmup_drop_does_not_move_the_p50_headline` (the p50 headline is unmoved — this
    corrects the mean-based tables only).
    → ⏳ **`k` is a principled default, not yet measured.** Run the head-vs-median check on
    `sr.json`'s raw vectors (`latencies_ms[0]` and `[:5]` vs the median of the remainder, per track
    per precision) to confirm `k=1` suffices; read `drop×` off the rendered table once real data is
    joined.
  - [x] **Canonical results never rewritten by a render** — `src.report` writes only `.txt`/`.png`;
    `results.<track>.json` + `sr.json` are read-only to it even when `out_dir` holds them
    (`src.study` also dumps BEFORE rendering). `::test_report_never_rewrites_canonical_results`.
  - [x] **Method-invariant SR join fixed (required by the `entropy` render):** `_select_method` gains
    a `precision` arg — fp32/fp16 fall back across method labels (they build data-free, so their SR
    cannot depend on a PTQ method), quantized + composite keys never do. Without it an `entropy`
    render left fp32/fp16 SR-PENDING and NaN'd every FP32-relative ΔSR purely from a label.
    `::test_method_invariant_precisions_join_across_methods`.

- [x] 🖥️ **Build FP8 engines, both tracks:** `uv run python -m src.export model=<lewm|dino>
  precision=fp8` → `engines/<track>/{encoder,predictor}.fp8.plan`.
  → verify: quantized ONNX carries QuantizeLinear (E4M3); FP8 engine builds; **FP8 tactics are
  actually selected** (verbose build / layer inspection), not a silent FP16 no-op.

- [x] 🖥️🔴 **Precision-match gate — FP8 rows:** `uv run python -m src.precision_match
  track=<lewm|dino>` gains FP8 drift rows vs the PyTorch reference (nominal + off-nominal hist
  `T ∈ {1,2}`). No coded pass/fail — owner sign-off on the drift table. **Not the arbiter**
  (nominal inputs; SR is — SPEC §Requirements).

- [x] 🖥️ **SR-per-precision — FP8:** `uv run python -m src.sr_eval --config-dir conf
  +experiment=eval_<lewm|dino> precision=fp8`, both tracks. Writes the `fp8` key into the
  per-track `sr.json` (read-modify-write → no clobber, CLAUDE.md §8).
  → verify: FP8 SR recorded per track, paired with its per-cycle latencies (same solves).

- [x] 🖥️ **Per-component benchmark — FP8:** `src/benchmark.py` times the FP8 encode-step +
  predictor-step distributions (equal-n p50/p95, warm-up dropped) + peak mem; `src/report.py
  ::decompose` derives the FP8 encoder/predictor/overhead split from the FP8 engine-step means ×
  the CEM call counts. `src/gpu_clocks.py` brackets the FP8 runs.
  → verify: FP8 per-component + peak-mem rows populated; negative overhead surfaced loudly.

- [x] **FP8 in the headline artifacts (tables + plots):** the FP32→FP16→INT8→FP8 speed table (all
  three distributions at p50 **and** p95), the FP32-relative table (p50 speedup + ΔSR per row),
  the Amdahl dilution table (model-only vs realized at FP8), and the speed-vs-SR plot all gain the
  FP8 row/point; the canonical per-track `results.<track>.json` records the FP8 numbers + fairness
  conditions. `src/report.py from=… [sr=…]` re-renders off-pod with FP8 into
  `$STABLEWM_HOME/reports/phase5/`.
  → verify: `results.{lewm,dino}.json` round-trips FP8 through `report.load_results`; the rendered
  tables + speed-vs-SR plot show the FP8 row/point; FP8 `.txt`/`.png` persisted under
  `$STABLEWM_HOME/reports/phase5/`.

**Verify:** FP8 engines built on the L40S (gitignored, regenerable); FP8 precision-match
owner-signed-off; FP8 SR + latency + peak mem recorded and joined; every Phase-5 table/plot and
the per-track results JSON carry the FP8 row/point (quoted vs FP32); all logged to W&B.
Every rendered table names its calibration method and no render clobbers another method's artefacts;
the speed table carries `n`; the isolation table attributes each material 8-bit SR drop to a
component on **both** tracks at **both** calibration methods (`isolation_table.max.txt` **and**
`isolation_table.entropy.txt`, neither borrowing the other's rows).

---

## Phase 7 — Clock-state confound: bound (off-pod)  🔴 normalization + framing · 🟢 render

Off-pod: reads the saved `results.{lewm,dino}.json` + `gpu_logs/*.dmon.log` — no L40S. **Bounds** the
differential-throttle confound; canonical artifacts untouched, normalized numbers additive and
labelled `derived`. (SPEC §Execution Environment, §Parity, §Requirements "Clock-state confound
disclosure", §Implementation Boundaries.)

- [x] 🔴 **Lock-denial recorded (Execution Environment).** `nvidia-smi -lgc` denied on the L40S pod
  (root, persistence mode on) → clocks unlockable; recorded in SPEC §Execution Environment. A
  clock-locked re-run is deferred (future work).
  → verify: SPEC §Execution Environment states the denial; no pod phase attempts a clock lock.

- [x] 🔴 **OWNER GATE — normalization construction + headline framing.** Owner fixes (a) the scaling
  model (`T_ref = T × f_measured/f_ref`), reference clock `f_ref`, the per-run clock statistic feeding
  `f_measured` (e.g. util-conditioned median SM clock), and which time is clock-bound; and (b) whether
  the headline stays the measured stock-hardware ratio (with disclosure) or is reframed. CLAUDE builds
  the render only after sign-off.
  → verify: owner sign-off on the formula + `f_ref` + statistic + framing recorded **before** any
  `*_normalized.derived.*` is written. (Sign-off 2026-07-25, recorded in `docs/architecture.md` §11 +
  the `src.clock_norm` docstring; headline stays the measured ratio, with disclosure. First derived
  render 2026-07-26.)

- [x] 🟢 **Throttle diagnostic.** `src/clock_norm.py` parses `gpu_logs/*.dmon.log` → per-run telemetry
  summary (SM/mem clock, power, temp, util medians + util-conditioned clock) and a differential-throttle
  plot (LeWM vs DINO, per precision) → `$STABLEWM_HOME/reports/phase5/gpu_logs/*_clock_diag.png`.
  → verify: plot renders from the saved `dmon` logs; the summary shows the lighter track at the boost
  ceiling and the heavier track throttled below it.

- [x] 🟢 **Harvest → `derived_clocks.json`.** `src/clock_norm.py` writes the owner-set per-run clock
  statistic + power per (track, precision, {sr_eval, benchmark}) to
  `$STABLEWM_HOME/reports/phase5/derived_clocks.json`. Read-only over `gpu_logs/`; `results.*.json`
  untouched.
  → verify: round-trips; one entry per (track, precision, run-type); values match the diagnostic.

- [x] 🟢 **Apply normalization → three surfaces.** `src/clock_norm.py` applies the owner-set formula to
  the canonical latencies (`results.*.json` via `report.load_results` + `derived_clocks.json`):
  (a) cross-model per-cycle ratio `R` and normalized `R′`; (b) within-model FP32→FP16→INT8→FP8 deltas
  at a common clock; (c) overhead recomputed with component-loop and cycle latencies at a matched clock.
  Written as `*_normalized.derived.txt`, never overwriting `results.*.json` or the `.entropy.txt` tables.
  → verify: `R′ ≤ R` (bound brackets); canonical artifacts byte-unchanged (extend
  `test_report_never_rewrites_canonical_results`); each derived table names itself `derived` + the
  `f_ref`/statistic used.

- [x] 🟢 **Throttle plot → committed display copy.** `src/clock_norm.py` copies the cycle-run
  throttle plot to `reports/figs/` (display-only exception); the canonical copy stays under
  `$STABLEWM_HOME/reports/phase5/gpu_logs/`.
  → verify: `reports/figs/sr_eval_clock_diag.png` matches the volume copy byte-for-byte.

**Verify:** owner sign-off on the normalization construction + framing recorded; `derived_clocks.json`
+ `*_normalized.derived.*` written additively with `results.*.json` / `.entropy.txt` untouched; the
`R`/`R′` bound reported on surfaces (a)/(b) and, on surface (c), the bound **or its resolvability
verdict** where the overhead share is below the clock mismatch; throttle plot committed to
`reports/figs/`.

---

## Phase 8 — Confidence intervals & the independence premise (off-pod)  🔴 construction · 🟢 compute + render

Off-pod: reads the saved `sr.json` — no L40S, **no eval/benchmark/export re-run**. Adds a 95% CI to
every reported **absolute** SR and **absolute** per-cycle p50, and tests the i.i.d. premise the p50
interval rests on. Canonical artifacts stay byte-unchanged; intervals are additive.
(SPEC §Interface Contracts, §Parity, §Requirements "Uncertainty quantification",
§Implementation Boundaries; `docs/architecture.md` §12.)

- [x] 🔴 **OWNER GATE — interval construction.** Owner fixes: Clopper–Pearson for SR; the exact
  binomial order-statistic interval for p50 (rank convention, α = 0.05) over the warm-up-dropped,
  equal-n-truncated sample; the Dwass MC permutation test on lag-1 autocorrelation (two-sided,
  50,000 permutations, no Student-t transform, fixed seed); the decision taken on the **unadjusted**
  p-value with **Holm as secondary reporting only**; no interval on any difference or ratio.
  → verify: recorded in `docs/architecture.md` §12 + the `src.stats` docstring **before** any
  `stats.json` is written. (Signed off 2026-08-05.)

- [x] 🟢 **Declare the stats dependencies.** `scipy` + `numpy` into `pyproject.toml`
  `[project.dependencies]` (present transitively today, so a future resolve could drop them).
  ⚠️ **Owner runs `uv lock` — NOT `uv sync` on the pod:** TensorRT lives outside the lock by design
  (`setup.sh`), and a sync removes it. Off-pod work needs no sync.
  → **landed:** `pyproject.toml` + `uv lock` (scipy 1.17.1, numpy; 150 packages resolved).
  → verify (✔): both appear in `[project.dependencies]`; `uv run python -c "import scipy, numpy"`.

- [x] 🟢 **`src/stats.py` + `tests/test_stats.py`.** Clopper–Pearson (scipy `binomtest
  .proportion_ci("exact")`), `order_statistic_ci`, `lag1_autocorr`, `dwass_permutation_test`, `holm`,
  `compute`, `write_stats_json`. The per-cycle sample rule is **shared with `src/report.py`**
  (`per_cycle_samples`), not duplicated, so the CI's sample is byte-identical to the p50's.
  → **landed:** `src/stats.py` + `tests/test_stats.py` (19 tests); `report._finalize_per_cycle`'s
  sample rule extracted to `report.per_cycle_samples`, shared with `stats.compute` so the composite
  isolation labels get the same truncation without entering `bench`.
  → verify (✔): `pytest` 134 passed; the four scipy-deviation guards pinned (permutation_test's
  two-sided convention, the under-covering naive rank choice, `false_discovery_control` ≠ Holm, the
  Maritz-Jarrett / Hettmansperger-Sheather near-misses).

- [x] 🟢 **Wire intervals into `src/report.py`.** `cyc_p50_CI95` + `ac` + `SR_CI95` on the speed
  table, `SR_CI95` on the isolation table, `CI95@<method>` on the calibration table; asymmetric
  `xerr`/`yerr` error bars on `speed_vs_sr*.png`. Ratio/difference/mean tables and
  `per_cycle_ratio.png` / `component_breakdown_*.png` **untouched**.
  → **landed:** `_stats_lookup` mirrors `_select_method`'s method-invariant fallback (without it an
  `entropy` render leaves fp32/fp16 intervals blank purely from a label — `::test_method_invariant
  _intervals_join_across_methods`). Every table cell stays ONE whitespace-delimited token (`[lo,hi]`
  unspaced, `ac` never empty) so the artefacts remain `split()`-parseable.
  → verify (✔): ratio/difference/mean tables unchanged; `speed_vs_sr.png` carries error bars on
  **both** axes — the latency bars are 0.02–1.9% of the panel x-span, i.e. narrower than the marker,
  which is the measurement, not a plotting fault.

- [x] 🟢 **Generate `stats.json` — network storage by default, no re-run of anything.** `uv run
  python -m src.stats` (and `src.report`'s re-render) default their out dir to
  `study.default_out_dir()` = `$STABLEWM_HOME/reports/phase5/`, so on the pod it lands on the
  persistent volume; `from=`/`out=` override it for an off-pod run. It reads `sr.json` only —
  **no `src.study`, no `benchmark.py`, no `sr_eval`, no engines, no GPU**.
  → verify (✔ 2026-08-05): `stats.json` carries **18** SR intervals and **18** per-cycle p50
  intervals (composite isolation keys included), each with its n, both p-values, and the recorded
  seed; `sha256sum` of `sr.json` + both `results.*.json` unchanged.

- [x] 🟢 **Refresh the committed display copies** `reports/figs/speed_vs_sr{,.titled}.png` from the
  render (display-only exception, re-copied never hand-edited).
  → verify (✔): repo copies match the volume copies byte-for-byte (`cmp`).

**Verify:** owner sign-off on the construction recorded before any `stats.json`; 18 SR + 18 p50
intervals persisted with their n, observed lag-1, asymptotic **and** permutation p-values, Holm
secondary values, and the fixed seed; the tables where SR and p50 appear carry their intervals and
the speed-vs-SR plot carries error bars; no interval on any difference or ratio; `sr.json` and
`results.*.json` byte-unchanged.

---

## Phase 9 — Per-component uncertainty: raw-latency capture + intervals  🔴 construction · 🟢 capture/compute/render · 🖥️ benchmark re-run

Extends Phase 8's intervals to the two **component** p50s (encode-step, predictor-step). Their samples
are not on disk — `src.benchmark` reduces each fixed-iteration loop to p50/p95/mean and discards the raw
list — so the loops must persist them, which costs one benchmark run per track. **Loop conditions are
unchanged**: `n_latency_iters=100` timed, `warmup=10` dropped, same batches and call counts.
(SPEC §Interface Contracts, §Parity, §Requirements "Uncertainty quantification";
`docs/architecture.md` §12 "Component p50s ride the same construction".)

- [x] 🔴 **OWNER GATE — component-interval construction.** Owner fixes: the exact binomial
  order-statistic interval on the encode-/predict-step **p50 only** (never a p95, a mean, a derived
  share, a difference or a ratio); the sample is the **fixed-iteration loop vector as recorded** — no
  equal-n truncation, no report-time warm-up drop; the same Dwass lag-1 permutation test (α = 0.05,
  B = 50,000, fixed seed, decision on the **unadjusted** p-value); **Holm scoped per measurement
  surface**, per-cycle and component families never pooled; `n_latency_iters` stays **100**; intervals
  rendered as columns on the existing speed table.
  → verify: recorded in SPEC §Interface Contracts + `docs/architecture.md` §12 **before** any component
  interval is written (`src.stats` docstring lands with the compute step below). (Signed off 2026-08-06.)

- [x] 🟢 **Capture the raw component samples.** `src/benchmark.py::benchmark` returns
  `(BenchResult, ComponentSamples)` — `_time_loop` already builds the lists; stop discarding them.
  `ComponentSamples` + the `Benchmark` Protocol arity in `src/interfaces.py`; `BenchResult` unchanged.
  `_percentiles_ms` computes on `float64` to match `report._percentile_ms` exactly. `src/smoke.py`
  follows the new arity.
  → **landed (off-pod ✔):** `interfaces.ComponentSamples`; `benchmark` returns the pair;
  `_percentiles_ms` on float64. `tests/test_benchmark.py::test_benchmark_returns_the_raw_samples
  _beside_the_summary`, `::test_stored_p50_matches_the_percentile_the_interval_is_built_on`.
  → verify (✔): each component vector is `n_iters` long with the warm-up iters excluded; the
  returned p50 equals `report._percentile_ms(sample, 0.5)` (the anti-drift guard).

- [x] 🟢 **Persist them.** `src/study.py::dump_track_latencies` writes
  `latencies.<track>.json` beside `results.<track>.json` — `{meta: {track, n_latency_iters, warmup,
  calibration_method, methods, seed, written}, latencies: {precision: {method: {encode_ms: [...],
  predict_ms: [...]}}}}` — merged **per (precision, method)** with `dump_track_results`' no-clobber
  discipline, called BEFORE rendering. Keyed by method because the quantized plans are per-method
  builds, so a second method's timing pass is additive (SPEC §Parity, CLAUDE §8).
  → **landed (off-pod ✔):** `run_track` returns `(name, bench, samples)`; `main` dumps results then
  latencies before the render. `tests/test_study.py::test_dump_track_latencies_roundtrips_and
  _records_the_loop_conditions`, `::test_dump_track_latencies_is_additive_per_precision`.
  → verify (✔): round-trips through `stats.load_component_latencies`; a later single-precision run
  preserves the other precisions; `results.<track>.json` schema unchanged.

- [x] 🟢 **`src/stats.py::compute_components`.** Per (track, precision, method, component):
  `order_statistic_ci`
  + `dwass_permutation_test` over the stored vector — existing helpers, no new estimator. Own Holm
  family, separate from the per-cycle one. New top-level `points_components` section
  (`[track][precision][method][encode|predict]`); `points` untouched. CLI auto-discovers
  `latencies.*.json` beside `sr.json`, plus explicit `latencies=<dir|file>`; absent → section omitted.
  `meta` records the component sample rule, both family sizes, and `holm_scope`.
  → **landed (off-pod ✔):** `compute(component_latencies=)` threads it through;
  `load_component_latencies` + `component_latency_paths`. `tests/test_stats.py`
  `::test_component_interval_uses_the_recorded_vector_as_the_sample`,
  `::test_component_points_carry_p50_only_never_p95_or_mean`,
  `::test_component_holm_family_is_separate_from_the_per_cycle_family`,
  `::test_component_section_omitted_without_stored_samples`,
  `::test_component_latency_paths_finds_the_track_files`.
  → verify (✔): the per-cycle `points` block is **byte-identical** with and without a component
  section (pins the per-surface ruling); every component point records its n, both p-values, the
  ranks/coverage and the seed.

- [x] 🟢 **Render on the speed table.** `render_speed_table` gains `enc_p50_CI95`, `enc_ac`,
  `pred_p50_CI95`, `pred_ac` beside the existing component columns (reuse `_ci` / `_ac_flag`; every
  cell stays ONE whitespace-delimited token). `_component_stats_lookup` walks `points_components` at
  the rendered method (`report.method_key`: fp32/fp16 read across labels, quantized never do).
  `src.report from=` loads `latencies.*.json` from the source dir.
  → **landed (off-pod ✔):** `report(component_latencies=)`, picked up automatically by the
  `from=` re-render. `tests/test_report.py::test_speed_table_carries_component_intervals_and_stays
  _parseable`, `::test_component_intervals_do_not_touch_the_derived_tables`,
  `::test_report_never_rewrites_canonical_results` extended to `latencies.*.json`.
  → verify (✔): 19 tokens on every row, header and rows alike; the ratio/FP32-relative/component/
  dilution tables **and** all three plots are byte-identical with and without the component payload;
  a track with no stored samples renders `—`, never a borrowed interval. `pytest` 145 passed.

- [x] 🖥️ **Archive before the re-run (CLAUDE §8 — log before you delete).** `src.study` merges into
  `results.<track>.json` and `gpu_clocks.log_gpu` opens each `*.benchmark.dmon.log` with `"w"`, so the
  run supersedes both. Copy `results.*.json`, `gpu_logs/*.benchmark.dmon.log`, `derived_clocks.json`,
  `stats.json` and the rendered `.txt`/`.png` → `$STABLEWM_HOME/reports/phase5/archive/2026-08-07/`.
  → **done (2026-08-07):** 43 files, 6.4M, each verified byte-identical to its source (`cmp`).
  Pre-run `sha256sum` of `sr.json` + both `results.*.json` recorded in the archive's
  `PRE_RUN_SHA256.txt` (`sr.json` = `eb78dc8e…`).
  → verify (✔): every superseded file has an archive copy; `sha256sum sr.json` recorded for the
  post-run comparison.

- [x] 🖥️ **Re-run the per-component benchmark, both tracks:** `uv run python -m src.study
  track=<lewm|dino> calibration_method=entropy` — one pass per track, recording that method's cells
  (the `max` cells are the separate pass below).
  → **done (2026-08-07):** both tracks, all four precisions. `latencies.{lewm,dino}.json` carry
  **100 values per component per precision at `entropy`** (1600 raw values); `meta` records
  `n_latency_iters=100, warmup=10, calibration_method=entropy, seed=0`.
  → **`sr.json` re-baselined, not violated:** two LeWM FP8 isolation evals (`enc-fp8+pred-fp16`,
  `enc-fp16+pred-fp8` @ `entropy`) ran BEFORE the study, so `sr.json` moved
  `eb78dc8e…` → `b5ad422e…` by that additive merge. `src.study` itself left it byte-unchanged.
  → **telemetry is method-TAGGED** (`run_tag` = `<track>.<precision>.<method>.benchmark`), so this
  run wrote NEW `…entropy.benchmark.dmon.log` files; the untagged pre-Phase-9 logs survive beside them.
  → verify (✔): 100 values per component per built precision; a fresh `*.benchmark.dmon.log` per
  precision.

- [x] 🟢 **Regenerate downstream, in order (off-pod).** Component **means** moved, so everything derived
  from them must be rebuilt: (1) `src.stats` → `stats.json` with both interval sections; (2) `src.report
  from=… sr=… calibration_method=entropy` → widened speed table + the mean-based component/dilution
  tables + `component_breakdown_fp32.png`; (3) `src.clock_norm` → `derived_clocks.json` +
  `*_normalized.derived.*` re-harvested from the **new** benchmark telemetry (Phase-7 surfaces (b)/(c)
  read component clocks — stale pairing would normalize new latencies with old clocks); (4) refresh
  `reports/figs/` by re-copying from the render.
  → **done (2026-08-07):** (1) `stats.json` — **20** per-cycle/SR points + **16** component points,
  Holm families 20 / 16 kept separate (`holm_scope` per surface). (2) speed table widened to
  `enc_p50_CI95`/`enc_ac`/`pred_p50_CI95`/`pred_ac`, **19 tokens on header and every row**
  (`split()`-parseable). (3) `src.clock_norm` **re-run at `calibration_method=entropy`** — the first
  pass defaulted to `max` and wrote `*.derived.max.txt`, leaving the entropy tables stale at the
  pre-re-run telemetry; the corrected pass refreshed `derived_clocks.json` +
  `*_normalized.derived.entropy.txt`. (4) `reports/figs/` re-copied, all five `cmp`-clean.
  → **disclosed:** 13 of 16 component cells REJECT the lag-1 independence test at the unadjusted p
  (`dino fp32 encode` ac=+0.76, p=2e-05) — flagged `*` in the table, intervals anti-conservative
  (too narrow). Back-to-back timing loops correlate; what a rejection licenses is OWNER-authored
  (SPEC §Implementation Boundaries).
  → verify (✔): two family sizes present; derived tables cite the new telemetry; `reports/figs/`
  matches the volume copies byte-for-byte.
  → the per-cycle/SR family grew **20 → 28** when the `max` isolation points landed (+4 composite
  points per track, 2026-08-10); the component family stays **16** — isolation is an SR diagnostic
  and is never benchmarked for latency.

### Mean latencies + overhead — bootstrap intervals (off-pod, no GPU)

Extends the intervals to the **mean-based decomposition surface** (SPEC §Interface Contracts, "the five
MEAN per-cycle latency quantities"). Re-analysis of the samples already on the volume — `sr.json` +
`latencies.{lewm,dino}.json` — with **no** eval/benchmark/export run.

- [ ] 🔴 **OWNER GATE — mean-interval construction** (2026-08-14). Five quantities per (track,
  precision, method), all call-count-weighted onto the per-cycle scale: `enc_cyc = 2 × mean(encode)`,
  `pred_cyc = 150 × mean(predict)`, `t_comp = enc_cyc + pred_cyc`, `cycle = mean(per-cycle sample)`,
  `overhead = cycle − t_comp`. Estimator: `scipy.stats.bootstrap`, `method="percentile"`,
  `n_resamples=3000`, `paired=False`, `confidence_level=0.95`, fixed seed. Samples: the component loop
  vectors as recorded; `report.per_cycle_samples` for the cycle. `enc_cyc`/`pred_cyc`/`cycle` inherit
  their sample's existing lag-1 flag — **no new test, no third Holm family**; `t_comp`/`overhead` carry
  no flag. Composite `enc-<A>+pred-<B>` labels excluded.
  → verify: recorded in SPEC §Interface Contracts + `docs/architecture.md` §12 + the `src.stats`
  docstring **before** any mean interval is written.

- [ ] 🟢 **Constants** — `BOOTSTRAP_RESAMPLES = 3000`, `BOOTSTRAP_SEED = 0` in `src/interfaces.py`,
  beside the existing CI block.
  → **landed (off-pod ✔).**

- [ ] 🟢 **`src/stats.py::compute_means`** → new top-level `points_means` section
  (`[track][precision][method]`) + `bootstrap_mean_ci` (one wrapper, all five quantities). `meta`
  records the estimator, B, `paired`, seed and the two call counts; `compute` threads it through;
  `main` reports the count.
  → **landed (off-pod ✔):** means taken with `statistics.fmean` and grouped as `cycle − (enc + pred)`,
  matching `benchmark._mean_ms` / `report.decompose` bit-for-bit; method-invariant precisions collapse
  to one row (`max`-first), composites skipped. `tests/test_stats.py` — `::test_mean_points_are
  _additive_and_bracketed`, `::test_mean_cycle_uses_the_same_sample_as_the_p50_interval`,
  `::test_mean_flags_are_inherited_never_re_tested`, `::test_mean_labels_name_the_method_only_where
  _it_applies`, `::test_mean_section_leaves_the_other_surfaces_byte_identical`,
  `::test_mean_intervals_are_reproducible_at_the_fixed_seed`,
  `::test_mean_section_omitted_without_component_samples`, `::test_isolation_composites_get_no_mean
  _row`, `::test_mean_construction_is_recorded_in_meta`.
  → verify (off-pod ✔): `points` + `points_components` byte-identical with and without the section;
  both Holm family sizes unchanged; each mean point records its per-sample `n`; two runs at the fixed
  seed give identical bounds; `t_comp == enc_cyc + pred_cyc` and `overhead == cycle − t_comp`.

- [ ] 🟢 **`src/report.py::render_latency_means_table`** → `latency_means_table.txt`, **unscoped**
  (the config column carries `FP32` / `INT8 (max)` / `FP8 (entropy)`, so it spans both methods like
  `calibration_table.txt`). One token per VALUE cell — `value[lo,hi]` + the inherited `*`/`-` marker;
  the config column is the one that may split, so the five values are read from the END of the row.
  Printed after the component table, logged to W&B.
  → **landed (off-pod ✔):** `_mean_cell` + `_mean_flag`; a pure walk of `points_means` (no
  recomputation off `bench`, so rendered and persisted numbers cannot drift). `tests/test_report.py`
  — `::test_latency_means_table_matches_decompose`, `::test_latency_means_table_is_parseable_and
  _carries_its_markers`, `::test_latency_means_table_is_unscoped_and_identical_across_methods`,
  `::test_mean_table_does_not_touch_the_other_artifacts`. `pytest` 180 passed.
  → verify (off-pod ✔): point estimates equal `report.decompose`'s `enc_cyc_ms`/`pred_cyc_ms`/
  `model_cyc_ms`/`cycle_ms`/`overhead_ms` (anti-drift guard); a `max` and an `entropy` render write
  byte-identical files; the four method-scoped tables, the isolation/calibration tables and all three
  plots byte-identical with and without the section.

- [ ] 🟢 **Regenerate off-pod:** `uv run python -m src.stats from=<reports/phase5>`, then
  `uv run python -m src.report from=<reports/phase5> sr=<…/sr.json> calibration_method=<max|entropy>`.
  → verify: `points_means` carries **12** entries (2 tracks × {FP32, FP16, INT8 (max), INT8 (entropy),
  FP8 (max), FP8 (entropy)}); `sha256sum` of `sr.json`, `results.*.json` and `latencies.*.json`
  unchanged. Phase 10 needs no new stage — `stages=stats,report` already covers it.

### Component latencies per calibration method — the `max` int8/fp8 timing pass

The quantized engines are per-method BUILDS, so their measured latencies are keyed
`(track, precision, method)` like their SR, and each method's plans are timed on their own
(SPEC §Parity). The `entropy` cells stand; this closes the `max` ones. **No `sr_eval` run** — SR and
the per-cycle sample are already recorded at both methods; only the component timing is outstanding.

- [ ] 🟢 **Method axis on the measured artefacts.** `results.<track>.json` `bench` and
  `latencies.<track>.json` `latencies` keyed `{precision: {method: …}}`, merged per cell
  (`study._merge_by_method`); `report.method_key`/`select_by_method` is the ONE selection rule
  (fp32/fp16 read across labels, quantized never fall back) and `report.load_results(paths, method)`
  collapses to the render's method; `stats.compute_components` gains the method axis and
  `compute_means` weights each row with its own method's vectors, stamping `component_method`;
  `src.study precision=<list>` benchmarks a subset. A legacy flat entry folds under its file's
  `meta.calibration_method`, so the recorded `entropy` cells keep their own label.
  → **landed (off-pod ✔):** `tests/test_study.py::test_a_second_methods_benchmark_lands_beside
  _the_first`, `::test_a_quantized_precision_never_borrows_the_other_methods_numbers`;
  `tests/test_stats.py::test_component_points_are_keyed_per_calibration_method`,
  `::test_a_quantized_mean_row_needs_its_own_methods_components`,
  `::test_a_method_invariant_mean_row_records_which_run_timed_it`;
  `tests/test_report.py::test_component_intervals_are_selected_by_calibration_method`.
  `pytest` 186 passed.

- [ ] 🖥️ **Archive first (CLAUDE §8):** `uv run python -m src.pipeline stages=archive`.
  → verify: every superseded file has a `cmp`-verified copy; `PRE_RUN_SHA256.txt` written.

- [ ] 🖥️ **Time the `max` int8/fp8 engines, both tracks:** `uv run python -m src.study
  track=<lewm|dino> precision=int8,fp8 calibration_method=max out=$STABLEWM_HOME/reports/phase5
  sr=$STABLEWM_HOME/reports/phase5/sr.json` (the plans already stand — `{encoder,predictor}
  .<int8|fp8>.max.plan`; fp32/fp16 are one data-free build and are NOT re-timed).
  → verify: `results.{lewm,dino}.json` + `latencies.{lewm,dino}.json` carry `int8`/`fp8` under
  **both** `max` and `entropy`, 100 values per component per cell; every pre-existing `entropy` cell
  byte-unchanged against the archive; a fresh `<track>.<precision>.max.benchmark.dmon.log` per run.

- [ ] 🟢 **Regenerate downstream, in order (off-pod):** (1) `src.stats` → `stats.json` with the
  `max` component + mean points beside the `entropy` ones; (2) `src.report from=… sr=…
  calibration_method=<max|entropy>` — both renders; (3) `src.clock_norm` at both methods (the `max`
  quantized rows now normalize with the `max` benchmark telemetry); (4) refresh `reports/figs/`.
  → verify: the `max` speed table's int8/fp8 rows carry component intervals instead of `—`, the
  `entropy` tables are byte-unchanged, and `sr.json` stays byte-unchanged throughout.

**Verify:** owner sign-off recorded before any component interval; `latencies.{lewm,dino}.json` on the
volume with the loops' raw samples, keyed per (precision, method) and covering both methods' quantized
engines; every benchmarked speed-table row carries a component p50 interval — its OWN method's, never
the other's — and an `ac` flag; the five mean per-cycle quantities carry bootstrap intervals +
inherited flags in
`latency_means_table.txt`; no interval on any p95, on the dilution shares/speedups, or on any
difference or ratio other than the named `overhead` decomposition; `sr.json`, `results.*.json` and
`latencies.*.json` byte-unchanged by the analysis.

---

## Phase 10 — Full-pipeline driver  🟢 wiring · 🖥️ ⏱️

One command carries both tracks from engines to rendered artifacts: the Phase-5→9 drivers run in
order, each in its **own subprocess** (process isolation for the CUDA contexts, TensorRT engine
arenas and Hydra global state the stages allocate). Every step is that stage's own documented
command — this driver sequences the study, it does not reimplement any part of it. Diagnostics are
opt-in. `out` resolves `study.default_out_dir()` = `$STABLEWM_HOME/reports/phase5/` **once** and is
passed to every child; engines keep `export.engine_root()`. Trainings (Phase 2) and the Phase-3
torch baseline are outside its scope — neither runs through an engine.

- [x] 🟢 `src/pipeline.py` + `tests/test_pipeline.py` — stage graph (`archive → export →
  precision_match → sr_eval → isolation → benchmark → stats → report → clock_norm → figs`, with
  `pytest`/`verify_encode`/`smoke`/`fidelity`/`sr_shim`/`probe_ranges`/`precision_match` behind
  `diagnostics=true`), `dry_run=`/`stages=`/`tracks=`/`out=` CLI, fail-fast with a resume line, and
  `<out>/pipeline_manifest.json` written after every step.
  → **isolation and benchmark at both methods landed (off-pod ✔):** both stages loop
  `CALIBRATION_METHODS` as `report`/`clock_norm` already do; the `_ISOLATION_METHOD` and
  `_BENCHMARK_METHOD` constants are retired. `benchmark` follows `sr_eval`'s shape — the default
  method's pass covers every precision, a further method only the quantized ones (fp32/fp16 are one
  data-free build, timed once). All engines already exist → `export` unchanged.
  → verify (off-pod ✔): `uv run python -m src.pipeline dry_run=true` lists **43** steps — 12 export,
  4 sr_eval, **16** isolation, **4** benchmark, 1 stats, 2 report, 2 clock_norm + archive + figs;
  `tests/test_pipeline.py::test_isolation_holds_one_component_at_fp16_per_run` asserts 16 labels,
  `per_method` under each of the two methods, exactly one side quantized per run;
  `::test_benchmark_times_each_methods_engines_in_its_own_process` asserts the 4 passes and the
  quantized-only second method. `pytest` 186 passed.

- [x] 🖥️ **Archive first (CLAUDE §8 — log before you delete).** `uv run python -m src.pipeline
  stages=archive` → `$STABLEWM_HOME/reports/phase5/archive/<UTC date>/`.
  → verify: every superseded file has a `cmp`-verified copy; `PRE_RUN_SHA256.txt` written.

- [x] 🖥️ **Run it, both tracks:** `uv run python -m src.pipeline`.
  → the 8 `max` isolation evals are the only new measurement — engines, pure corners and the
  component benchmark are unchanged — so `stages=isolation,stats,report,clock_norm,figs` is the
  cheap path where the rest still stands.
  → **run as the cheap path (2026-08-10):** the 8 `max` isolation evals had already landed in
  `sr.json`, so only `stages=stats,report,clock_norm,figs` was needed; all 6 steps `rc=0` in
  `pipeline_manifest.json`. Archived first (`stages=archive` → `archive/2026-08-10/`, 103 files,
  each `cmp`-verified, `PRE_RUN_SHA256.txt` written) per CLAUDE §8.
  → verify (✔): **24** engine plans under `$STABLEWM_HOME/engines/`; `sr.json` carries every
  (track, precision, method) point plus **16** composite points (8 `enc-<A>+pred-<B>` keys ×
  2 methods);
  `results.{lewm,dino}.json` + `latencies.{lewm,dino}.json` refreshed (100 values per component per
  precision); `stats.json` carries both interval families; the method-scoped tables,
  `derived_clocks.json` + `*_normalized.derived.<method>.txt`, and `reports/figs/` all regenerated.
  → **`isolation_table.max.txt` now stands beside `isolation_table.entropy.txt`** — the Phase-6
  verify (PLAN §Phase-6, "neither borrowing the other's rows") is met at both methods.
  → **canonical artefacts byte-unchanged by the render** (`cmp` vs the archive): `sr.json`,
  `results.*.json`, `latencies.*.json`. Every measured table re-rendered byte-identical — no
  latency moved, as expected for an SR-only diagnostic. `derived_clocks.json` grew **additively**
  by the 8 `max` composite telemetry entries, no existing value altered; the three
  `*_normalized.derived.*` pairs are byte-identical (isolation never enters a latency surface).
  `reports/figs/` re-copied, all 5 `cmp`-clean.

- [x] 🖥️🔴 **Owner sign-off on the drift table:** `uv run python -m src.pipeline
  stages=precision_match diagnostics=true`. No coded pass/fail (SPEC §Requirements).
  → verify: `enc_b` = 1 on every row, `pred_b` ∈ {1, 8, 300}; FP32 drift stays at the order the
  gate has always judged it at (a materially different FP32 row means the build reached numerics —
  STOP and ask).

- [x] 🟢 **`reports/figs/` refreshed** by re-copying from the render (display-only exception).
  → verify: `cmp` clean against the volume copies. All 5 display copies `cmp`-clean (2026-08-08).

**Verify:** one `src.pipeline` invocation reproduces every Phase-5→9 artifact; `dry_run=true`
prints the stage plan and touches nothing; a failed stage names itself and the `stages=` to resume
from; `pipeline_manifest.json` records every step's command, exit code and duration.

---

## Critical files

- `src/interfaces.py` — typed contract; dim constants + CEM per-cycle call counts.
- `src/adapter.py`, `src/shim.py`, `src/export.py` (incl. the Model-Optimizer INT8 Q/DQ step),
  `src/calibrate.py`, `src/probe_ranges.py`, `src/benchmark.py`, `src/report.py`, `src/study.py`,
  `src/smoke.py` — the owned layer (Phases 4–6). From Phase 9 `src/benchmark.py` also returns the
  engine-step loops' raw per-call samples and `src/study.py` persists them to
  `latencies.<track>.json`, beside (never inside) `results.<track>.json`. Both files are keyed
  `{precision: {calibration method: …}}` and merged per cell, so each method's engines are timed and
  recorded on their own (`report.method_key` is the single selection rule the renders use).
- `src/precision_match.py`, `src/fidelity.py`, `src/sr_shim.py`, `src/sr_eval.py`,
  `src/trt_runtime.py` — the gates + engine-backed SR path.
- `src/pipeline.py` — Phase-10 end-to-end driver: sequences the Phase-5→9 drivers as isolated
  subprocesses (archive → export → sr_eval → isolation → benchmark → stats → report → clock_norm →
  figs; diagnostics opt-in), resolving `study.default_out_dir()` once for every child and recording
  each step in `<out>/pipeline_manifest.json`.
- `src/wandb_log.py` — owned W&B helper for the non-training phases (Phase 3+).
- `src/gpu_clocks.py` — passive `nvidia-smi dmon` GPU-telemetry observer →
  `$STABLEWM_HOME/reports/phase5/gpu_logs/`.
- `src/clock_norm.py` — Phase-7 clock-confound render (off-pod): harvests per-run clock stats from
  `gpu_logs/`, applies the owner-set normalization, writes `derived_clocks.json` +
  `*_normalized.derived.*` + the throttle plot; reads canonical results via `report.load_results`.
  Writes **no** disclosure prose (SPEC §Implementation Boundaries).
- `src/stats.py` — Phase-8/9 confidence intervals (off-pod): Clopper–Pearson SR + exact binomial
  order-statistic p50 intervals over the stored `sr.json` samples **and the component p50 intervals
  over `latencies.<track>.json`, per (track, precision, method)** (Holm scoped per measurement
  surface), the Dwass lag-1 permutation
  test, Holm secondary values, **and the percentile-bootstrap intervals on the five mean per-cycle
  quantities** (`points_means`, rendered as `latency_means_table.txt` by `src/report.py`) →
  `stats.json`. Shares the per-cycle sample rule with `src/report.py`.
  Writes **no** interpretation of a rejected independence test — that is owner-authored
  (SPEC §Implementation Boundaries).
- `src/eval_latency.py` — observation-only **per-decision** latency callback (one record per
  episode's decision, NOT per solve — `docs/architecture.md` §8).
- `src/eval.py` — Phase-3 eval driver over the byte-unmodified `eval_wm.run`.
- `src/dino_patch.py` — `DINOv3PreJEPA` register-slice subclass.
- `conf/` — owned Hydra overlays (`{lewm,dinov3}.yaml` train; `eval_{lewm,dino}.yaml` eval).
- `scripts/train/{lewm,prejepa}.py` + `scripts/train/config/` — vendored platform entrypoints
  (`scripts/train/VENDORED.md`).
- `scripts/plan/eval_wm.py` + `scripts/plan/config/` — vendored eval entrypoint
  (`scripts/plan/VENDORED.md`).
- `scripts/verify_encode.py` — Phase-2 encode sanity (owned, fails loud).
- `pyproject.toml`, `uv.lock`, `setup.sh`. `Dockerfile` + `docker-compose.yml` at project end.
- `tests/` — pytest for the owned boundaries.
- `SPEC.md` (contract) · `docs/architecture.md` (rationale) · `docs/platform_api.md`
  (Phase-1 findings).

## Cross-cutting rules

- **Owner gates:** anything 🔴 (export/INT8/FP8/Model-Optimizer-PTQ debugging, precision matching,
  benchmark methodology, adapter dims, eval/CEM parity) → STOP and ask.
- **Git:** never run git. On completing a unit of work, output the files to stage and a
  `type(scope): summary` commit message; the owner runs git.
- **Progress:** each `[x]` records artifact name; tick before advancing.
- **Caps:** training is epoch-capped; 3-attempt debugging cap; log-before-delete.

## End-to-end verification

1. `bash setup.sh` (uv + deps + TensorRT) + import check (Phase 0).
2. Platform API introspection in-container; DINOv3 `config.hidden_size`/`last_hidden_state`
   confirmed (Phase 1).
3. Two checkpoints + W&B runs + `scripts.verify_encode` PASS (Phase 2); both-track SR + latency
   under matched CEM config (Phase 3).
4. `python -m src.smoke` + `pytest -v` green on dummy weights (Phase 4).
5. `src.export` builds TRT engines on the L40S; the latency benchmark emits the three p50/p95
   distributions (per-cycle headline) + SR per precision and the encoder/predictor/overhead
   tables (Phase 5).
6. `src.export` builds FP8 engines on the L40S; FP8 rows appear in the SR/latency/peak-mem
   recordings, tables, and plots, quoted vs FP32 like the other precisions (Phase 6).
7. `src.clock_norm` renders the clock-normalized derived tables + throttle plot off-pod from the
   saved results + `gpu_logs/`; the confound is bounded with canonical artifacts untouched
   (Phase 7).
8. `stats.json` lands in `$STABLEWM_HOME/reports/phase5/` by default — the network volume, same
   durability contract as the other headline artifacts — written by the **cheap render path**
   (`src.stats`, or `src.report`'s re-render) off the saved `sr.json`. It requires **no `src.study`
   run**: no `benchmark.py`, no `sr_eval`, no engines, no GPU. Every reported absolute SR and
   absolute per-cycle p50 carries its 95% interval in the tables and as error bars on the
   speed-vs-SR plot, the lag-1 independence test is reported per run, and no interval sits on a
   difference or ratio (Phase 8).
9. The engine-step loops persist their raw per-call samples to `latencies.<track>.json` on the volume,
   per (precision, calibration method) so both methods' quantized engines are timed and kept;
   `src.stats` extends the intervals to the encode-/predictor-step p50s off those samples (own Holm
   family, no truncation, no report-time drop) and the speed table carries that method's component
   interval + `ac` flag on every benchmarked row; the same off-pod pass adds the five mean per-cycle quantities
   (`enc_cyc`, `pred_cyc`, `t_comp`, `cycle`, `overhead`) with percentile-bootstrap intervals and
   inherited independence markers to `stats.json` + `latency_means_table.txt`, with `sr.json` /
   `results.*.json` / `latencies.*.json` byte-unchanged (Phase 9).
10. `uv run python -m src.pipeline dry_run=true` prints the whole stage plan off-pod without
    touching the volume, and `uv run python -m src.pipeline` reproduces every artifact of steps
    5–9 on the L40S in one invocation, recording each step in `pipeline_manifest.json` (Phase 10).
