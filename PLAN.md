# PLAN.md — LeWM vs DINOv3-WM: Inference Optimization Study

> Execution steps only. **What** the project must satisfy → `SPEC.md`. **Why** a design is
> shaped this way → `docs/architecture.md` + `docs/adr/`. Behavioral rules → `CLAUDE.md`.
> Typed contract → `src/interfaces.py`.
> Every completed step records `[x]` + artifact name (commit hash added by the owner).
> Tick a box before the next step.

## Context

**Execution model.** The **L40S RunPod pod is the only execution target**; local WSL is
edit-only. Every run command is `uv run …` on the pod (provisioned by `setup.sh`). The five
SPEC requirement bundles map to the Phases below.

**Legend.** 🟢 CLAUDE-CODE owns (fails loud). 🔴 OWNER-ONLY — STOP and ask (SPEC §Implementation
Boundaries). 🖥️ runs on the L40S GPU. ⏱️ capped effort with a stated fallback.

---

## Phase 0 — Scaffolding & pinned dependencies  🟢

- [x] `pyproject.toml` + `uv.lock` pinning: `stable-worldmodel`, `stable-pretraining`,
  `hydra-core`, `wandb`, `jaxtyping`, `beartype`, `onnx`, `transformers`, `timm`,
  **torch (cu124 wheel index)** — all uv-managed. **TensorRT NOT in uv**
  (installed by `setup.sh`). Versions pinned. → `docs/adr/0001`
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
- [x] 🖥️ Train DINOv3-WM: `uv run python -m scripts.train.prejepa --config-dir conf
  +experiment=dinov3` → `$STABLEWM_HOME/checkpoints/dino/`.
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
  → **Bracket superseded in Phase 5** (`docs/adr/0004`); this box records Phase 3 as shipped.
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
Contracts; statistic split → `docs/adr/0003`; cycle definition → `docs/adr/0004`.

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
  → verify (off-pod ✔): all 4 ONNX graphs trace with a dynamic batch axis; `pytest` green.
  TRT build + precision matching are pod-only 🔴.
- [x] 🔴 **INT8 explicit quantization (Model Optimizer PTQ):** base FP32 ONNX →
  `modelopt.onnx.quantization` (Q/DQ + per-tensor scales) → quantized ONNX per method →
  `build_engine` (no `int8_calibrator`, no calibration profile). Sequenced **before** the
  precision-match gate. → `docs/adr/0001`
  → `setup.sh` installs `nvidia-modelopt[onnx]` + CUDA-12 `onnxruntime-gpu` (out of uv) and
  sanity-opens an ORT CUDA-EP session. Pins: `modelopt==0.43.0`, `onnxruntime-gpu==1.24.4`.
  → `src/calibrate.py`: clip draw + per-method streams kept; `make_calibrator` →
  `make_calibration_dict` (numpy dict keyed off the base ONNX input names).
  → `src/export.py`: `quantize_onnx` (`calibration_method="max"`, `use_external_data_format=True`);
  `build_engine` calibrator/profile branch dropped. `interfaces.py` re-documents `calib_loader`.
  → **calibration EP split:** encoder on CUDA EP, predictor on CPU EP
  (`quantize_onnx(force_cpu_calibration=name=="predictor")`) — `docs/adr/0001`.
  → verify (off-pod ✔): `pytest` green. 🔴 owner-confirm on pod: non-`max` modelopt knobs left at
  INT8 defaults; keeping the TRT INT8 flag for a Q/DQ graph; `onnx` lock harmonization (modelopt
  caps 1.19.1 vs locked 1.22.0).
  → verify (pod): quantized ONNX carries QuantizeLinear + INT8 engine builds; encoder binds CUDA
  EP, predictor binds CPU EP.

- [x] 🔴🖥️ **Calibration-distribution fix — INT8 SR collapse** (reopens the INT8 box above).
  Observed: lewm FP32 94% / FP16 96% / **INT8 48%**. Diagnosis + decision → `docs/adr/0002`.
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
  drift table.
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
  → **statistic fixed to the MEAN** (`docs/adr/0003`): `decompose` + `dilution_disclosure` read
  `*_mean_ms`; `BenchResult` carries `per_cycle_mean_ms`/`encode_mean_ms`/`predict_mean_ms`;
  `report._finalize_per_cycle` computes the cycle mean off the same truncated sample as p50/p95.
  `tests/test_report.py::test_decompose_uses_mean_not_p50`.
  → 🔴 open: whether to drop a per-cycle warm-up (`docs/adr/0003`, accepted residual).

- [x] 🔴 **Per-decision latency bracket** (`docs/adr/0004`).
  - [x] `src/eval_latency.py` — bracket **per env** via consecutive `start_batch` hooks (last
    closing at `end_solve`), one record per decision, sync per span; `current_bs == 1` guard;
    `n_solves` → `n_cycles`. Consumers `src/eval.py` + `src/sr_eval.py` updated (W&B
    `cem_solve_*` → `per_cycle_*`). `tests/test_eval_latency.py`: a 50-env solve must record 50
    latencies. Stale "per-solve" wording corrected in `src/report.py`, `src/benchmark.py`,
    `conf/experiment/eval_{lewm,dino}.yaml`.
    → verify (off-pod ✔): `pytest` 73 passed.
  - [x] 🖥️ **Pod-verify (the real gate):** the per-env spans must **sum to the solver's printed
    `CEM solve time`** (`cem.py:282`) less the pre-loop warm-start.
    → also verify: records per solve == alive-env count; `overhead_ms` lands at a believable
    fraction, not ~98%.

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
  table). `docs/adr/0003`.
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
statistic split → `docs/adr/0003`.

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
  → `docs/adr/0001`
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
  (`docs/adr/0002` amendment). Observed: DINO FP32/FP16 ~70% / **INT8 ~20% / FP8 2%**; LeWM ~98% /
  INT8 ~76%. The ADR-0002 distribution fix recovered LeWM (action stressor, in-engine) but not DINO
  (outlier-heavy frozen-DINOv3 activations; `max` per-tensor amax saturates them). `entropy`
  (tail-clip) is the candidate lever — which method wins per track is an SR question, measured.
  - [x] 🔴 **Decision recorded** — calibration method (`max` | `entropy`) is a build option for
    **both** tracks and a **labelled result dimension** (track × precision × method); held constant
    across INT8+FP8 within a labelled comparison; existing `max` artefacts preserved (additive, never
    rewritten). → `docs/adr/0002`, SPEC §Parity + §Interface Contracts (Export shape).
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

- [x] 🖥️ **Component-precision isolation — both tracks, `entropy` only** (`docs/adr/0005`).
  Attributes each material 8-bit SR drop to the encoder or the predictor. Diagnostic: composite
  `enc-<A>+pred-<B>` keys, never in the headline sweep. Two runs per affected (track, precision) cell.
  - [x] 🖥️ **Pure `entropy` corners** — the 2×2's both-quantized corner AND the headline row the
    diagnostic explains: `uv run python -m src.sr_eval --config-dir conf
    +experiment=eval_<lewm|dino> precision=int8,fp8 calibration_method=entropy`.
    → verify: `sr.json` carries `{track}.{int8,fp8}.entropy`; a render at
    `calibration_method=entropy` shows no PEND in the int8/fp8 SR column.
  - [x] 🖥️ **LeWM `entropy` engines** (DINO's already built): `uv run python -m src.export
    model=lewm precision=<int8|fp8> calibration_method=entropy`.
    → verify: `engines/lewm/{encoder,predictor}.<p>.entropy.plan` exist; quantized ONNX carries
    QuantizeLinear.
  - [x] 🖥️ **DINO isolation runs** (2026-07-21, `entropy`): int8 + fp8, both sides.
    → enc-fp16+pred-fp8 70.0 · enc-fp16+pred-int8 42.0 · enc-int8+pred-fp16 16.0 ·
    enc-fp8+pred-fp16 4.0 (FP16 baseline ~70). Encoder-dominant; predictor FP8-clean,
    INT8-sensitive. Recorded in `sr.json` under composite keys.
  - [x] 🖥️ **LeWM isolation runs** — `uv run python -m src.sr_eval --config-dir conf
    +experiment=eval_lewm encoder_precision=int8 predictor_precision=fp16
    calibration_method=entropy` and the reverse. FP8 only if LeWM FP8 also drops.
    → verify: composite keys land beside the pure points; pure SRs unchanged. ADR-0002 predicts
    **predictor**-dominant damage (Design A puts `action_encoder` inside the predict engine) — a
    contrary result reopens that mechanism.
  - [x] 🟢 **Isolation table** in `src/report.py`, rendered from the composite `sr.json` keys and
    placed after `fp32_relative_table`. Columns: track · precision · component quantized (other held
    fp16) · SR · ΔSR vs that track's FP16 · that component's per-cycle time share.
    → **landed (off-pod ✔):** `render_isolation_table` + `_parse_isolation_key` / `_ISOLATION_KEY`;
    written as `isolation_table.<method>.txt` only when isolation runs exist.
    `tests/test_report.py::test_isolation_table_attributes_component`,
    `::test_isolation_keys_never_reach_the_headline` (headline `.txt` byte-identical with and
    without composite keys). `pytest` 93 passed.

- [x] 🟢 **Report labelling + provenance** (`docs/adr/0002` 3rd amendment, `docs/adr/0003` amendment).
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
  - [x] 🔴 **Per-cycle warm-up drop `k=1`** (owner-approved 2026-07-21; `docs/adr/0003` amendment
    closes the "Open (owner)" item). The engine loops drop `warmup` iters, the per-cycle callback
    dropped none, so cold-start sat on one side of `overhead = cycle − enc − pred` only.
    → **landed (off-pod ✔):** `interfaces.PER_CYCLE_WARMUP_DROP = 1`; applied in
    `report._finalize_per_cycle(warmup_drop=)` **before** the equal-n truncation and at **report**
    time (sr.json's raw vector untouched → ADR-0004 span-sum reconciliation still valid);
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
component on **both** tracks.

---

## Critical files

- `src/interfaces.py` — typed contract; dim constants + CEM per-cycle call counts.
- `src/adapter.py`, `src/shim.py`, `src/export.py` (incl. the Model-Optimizer INT8 Q/DQ step),
  `src/calibrate.py`, `src/probe_ranges.py`, `src/benchmark.py`, `src/report.py`, `src/study.py`,
  `src/smoke.py` — the owned layer (Phases 4–6).
- `src/precision_match.py`, `src/fidelity.py`, `src/sr_shim.py`, `src/sr_eval.py`,
  `src/trt_runtime.py` — the gates + engine-backed SR path.
- `src/wandb_log.py` — owned W&B helper for the non-training phases (Phase 3+).
- `src/gpu_clocks.py` — passive `nvidia-smi dmon` GPU-telemetry observer →
  `$STABLEWM_HOME/reports/phase5/gpu_logs/`.
- `src/eval_latency.py` — observation-only **per-decision** latency callback (one record per
  episode's decision, NOT per solve — `docs/adr/0004`).
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
- `SPEC.md` (contract) · `docs/architecture.md` + `docs/adr/` (rationale) · `docs/platform_api.md`
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
