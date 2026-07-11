# PLAN.md — LeWM vs DINOv3-WM: Inference Optimization & QLoRA Study

> Generated from `SPEC.md` (rationale lives there; this file is execution steps only).
> Carries execution progress. `CLAUDE.md` holds behavioral rules; `src/interfaces.py` is
> the typed contract for the owned layer. Every completed step records: `[x]` + artifact
> name (commit hash added by the owner). Tick a box before the next step.

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

- [x] Vendor `scripts/train/lewm.py` and `scripts/train/prejepa.py` (GitHub tag `0.1.1`,
  not in the wheel — provenance in `scripts/train/VENDORED.md`) as used; wire Hydra + W&B.
- [x] 🔴 **DINOv3 config + register-slice subclass** for `prejepa.py` (owner-approved,
  Phase-1 §6) — wiring done, on-pod slot-in confirmation pending:
  - `conf/experiment/dinov3.yaml` overlay (via `--config-dir conf +experiment=dinov3`):
    `backbone.name/type=dinov3_small`, `patch_size=16`.
  - `src/dino_patch.py::DINOv3PreJEPA` — `PreJEPA` subclass overriding `_encode_image`
    slice `[:, 1:, :]` → `[:, 1+num_reg:, :]`, injected via `model._target_`; reused by
    Phase-3 eval. Owner confirms slot-in on the pod (import + one forward).
- [x] **Pre-flight before any GPU run:** `STABLEWM_HOME` points at the persistent network
  volume (not `~`); Push-T expert dataset pre-downloaded to `$STABLEWM_HOME/datasets/`
  (`hf download galilai-group/lewm-pusht --repo-type dataset --include
  "pusht_expert_train.lance/*"`, wired into `setup.sh` §8) — the bare `.lance` name does
  not auto-stream from HF; resolves and one batch loads.
- [x] 🖥️ Train LeWM:
  `uv run python -m scripts.train.lewm --config-dir conf +experiment=lewm` → `$STABLEWM_HOME/checkpoints/lewm/`.
- [x] 🖥️ Train DINOv3-WM:
  `uv run python -m scripts.train.prejepa --config-dir conf +experiment=dinov3` → `$STABLEWM_HOME/checkpoints/dino/`.
- [x] ⏱️ **Training is epoch-capped** — **10 epochs for both tracks**, batch size 128
  (LeWM's paper value, applied to both; set in the conf overlays); no wall-clock cap.

**Verify:** two checkpoints exist; both W&B runs logged; encode
sanity `uv run python -m scripts.verify_encode` passes — confirms Phase-1 latent dims
(LeWM CLS `(B, 192)`, DINO-WM grid `(B, T, 196, 384)`) and the register-slice
differential (override 196 vs base `PreJEPA` 200, `num_reg=4`). Expected tail:
```
[DINO-WM] override grid (2, 3, 196, 384) vs base (2, 3, 200, 384) (D=384, num_reg=4) -> 196 patches OK
[LeWM] CLS latent (2, 192) -> single token, D=192 OK
encode sanity: PASS
```
Dims are set by the encoder config (not training), so this is valid pre- or
post-training; it also satisfies the §2 slot-in "import + one forward".
**Log-before-delete:** confirm logged to W&B or committed before overwriting (CLAUDE.md §7).

---

## Phase 3 — Task baseline (platform CEM/MPC eval)  🟢 wiring · 🔴 parity · 🖥️

**Prerequisites (implicit in "Run World.evaluate" — established here, not assumed):**

- [x] **Pod-confirm** `scripts/plan/eval_wm.py` is absent from the installed wheel before
  vendoring (`python -c "import stable_worldmodel, os; print(os.path.dirname(stable_worldmodel.__file__))"`
  then check for a shipped `plan`/eval module).
- [x] Vendor the platform eval entrypoint **as used** (same as Phase-2 train vendoring):
  `scripts/plan/eval_wm.py` + its config group `scripts/plan/config/{pusht.yaml, solver/cem.yaml,
  launcher/local.yaml}` from GitHub tag `0.1.1`; provenance in `scripts/plan/VENDORED.md`.
  Unmodified — register-slice flows in via the checkpoint's saved `model._target_`
  (`load_pretrained`). **Stays byte-unmodified in Phase 3:** latency rides in via a
  config-injected CEM callback and SR/W&B via the owned driver (below), never by editing it.
- [x] Owned **W&B helper** (`src/wandb_log.py`): `init()` opens the run with project/entity
  read from the `conf/experiment/` `wandb:` block; phases log via `wandb.log` (SPEC
  §W&B logging discipline). Reused by Phases 5–6.
- [x] Owned **observation-only CEM-solve-latency callback** (`src/eval_latency.py` +
  `tests/test_eval_latency.py`) — **recast** from the initial `solver.solve` wrapper (commit
  `c91d49b`) to a `CEMSolver.Callback` subclass, since the callback rides in through the
  platform's own config seam and needs no monkeypatch/vendored edit. Brackets one CEM solve
  (`reset → end_solve`) with `perf_counter` + optional `torch.cuda.synchronize()` and records
  per-solve latency; the owned driver reads the records and logs the median (records only —
  no W&B dependency in the callback). Injected via `cfg.solver.callbacks`. Must only
  read/record — no effect on seeds, sample counts, or the plan (else → 🔴 OWNER, SPEC parity
  gate). Measures **CEM-solve latency** — excludes `prepare_init_action` warm-start, a
  zero-pad for non-`Actionable` LeWM/DINO-WM (`get_cost` only, no `get_action` — confirmed in
  `stable_worldmodel` 0.1.1), hence negligible and model-independent. Eager baseline (median
  of a few solves); the rigorous p50/p95 rig is Phase 5.

- [x] Owned **eval driver** (`src/eval.py`): thin driver that (a) opens the W&B run via the
  owned helper, (b) invokes the vendored `eval_wm.run` (byte-unmodified), (c) captures
  `World.evaluate`'s returned metrics (observation-only) and logs SR + the callback's
  per-solve latency median to that run. No monkeypatch, no class shadow — the latency
  callback rides in via config (below).
- [x] Owned **eval overlays** `conf/experiment/eval_{lewm,dino}.yaml` (`@package _global_`):
  set `policy=<ckpt run_name>` (`lewm` / `dino`), `eval.dataset_name` (trained dataset), the
  `wandb:` block (shared project), and inject the latency callback via `cfg.solver.callbacks`.
  The training overlays (`lewm`/`dinov3`) carry only training keys and are **not** reused for
  eval — composing one leaves `policy=random` (random policy, not the trained WM).
- [x] 🖥️ Run the eval driver for **both** tracks:
  `uv run python -m src.eval --config-dir conf +experiment=eval_<lewm|dino>` → Push-T
  **success rate** + **CEM-solve latency**. Pod-confirm: the checkpoint's saved
  `model._target_` reconstructs the register-slice subclass (196-grid, `src` importable),
  and SR is unchanged vs a callback-free run (callbacks feed `outputs['callbacks']` only,
  never the optimization).
- [x] 🔴 **Parity (load-bearing):** same CEM config (300 samples, 30 elites, horizon 5,
  init var 1, 10–30 iters), same action budget, same goal encoding, same eval seeds,
  identical ImageNet normalization — confirm **not varied between tracks** (do not change
  the platform eval/CEM config).

**Verify:** success-rate + CEM-solve latency for both tracks, logged to W&B by the eval
driver (owned helper, shared project); SR identical with/without the callback; parity
conditions recorded as identical.

---

## Phase 4 — Owned adapter + tracer bullet  🟢 · 🔴 (adapter-dims sign-off) · (sole pre-optimization check)

- [x] **Reconcile `src/interfaces.py` to the two-method adapter** (SPEC §Interface
  Contracts): replace the single fused `WMStepAdapter.__call__(obs, action) -> latent`
  with separately-callable `encode(obs) -> latent` + `predict(latent, action) -> latent`,
  so each exports to its own engine and the rollout can encode-once / predict-many. Adjust
  the `Export` (and `Benchmark`) Protocols to the two-engine reality — `export` targets
  `encode` and `predict` separately (per-method example inputs: `encode` obs; `predict`
  cached latent + action) and yields the encoder + predictor engine paths; the benchmark
  consumes both.
- [x] Constants defined **once** in `interfaces.py` from the Phase-1 values: `LATENT_DIM=192`,
  DINO-WM patch-grid `(N_patches, D)=(196, 384)`, `ACTION_DIM=2`, and the DINO-WM
  predictor-input token width `404` (`=384+20` extras concatenated on the feature axis,
  distinct from the 384 latent); platform dims read from config.
- [x] 🔴 **OWNER gate — adapter dims:** confirm the DINO-WM `predict` boundary widths
  (input `404 = 384 latent + 20 extras`; output width per the instantiated predictor) by
  introspecting the real predictor on the pod before hard-coding — a wrong width mis-shapes
  the predictor engine **silently** (SPEC §Implementation Boundaries).
- [x] Implement **`WMStepAdapter`** as two classes (`LeWMAdapter` single-token latent,
  `DINOWMAdapter` patch-grid latent) behind the common `encode`/`predict` signature, typed
  per `src/interfaces.py`. Each wraps the model's encoder + predictor; action enters
  `predict` per-track (LeWM: separate AdaLN arg; DINO-WM: concatenated on the feature axis
  inside the adapter). The CEM planner / rollout loop stays in Python outside it.
- [x] Implement `export()` and `benchmark()` **stubs** conforming to the `Export` /
  `Benchmark` Protocols + `ExportConfig` — `export` stub emits an encoder + a predictor
  engine path per model; `benchmark` stub consumes both.
- [x] `src/smoke.py`: dummy checkpoint → adapter (`encode`, then `predict` on the cached
  latent) → export-stub → benchmark-stub, with jaxtyping + beartype assertions at **every
  owned boundary**.
- [x] `tests/` covering adapter shapes (both `encode` and `predict`, both tracks) and the
  typed boundaries.

**Verify:** `uv run python -m src.smoke` passes; `uv run pytest -v` green; a
shape/precision violation actually raises at both the `encode` and `predict` boundaries.

---

## Phase 5 — Speedup study: export, profile & fixed-budget benchmark  🔴 OWNER-heavy · 🖥️ ⏱️

Owner makes the silent-failure calls; Claude owns plumbing (trace call, builder
invocation, percentile timing, memory logging, profiler hooks, table runner).

**Benchmark methodology.** Only the **model** (encoder + predictor, via `WMStepAdapter`)
is TensorRT-optimized; the **CEM planner stays in Python**. Evaluate both under a **fixed
wall-clock time budget**; compare **(a) per-step inference latency (p50/p95)** and **(b)
CEM rollouts completed**. **Every speed number is paired with an SR** (Phase-3 eval per
precision). (See SPEC §Parity, `src/interfaces.py`.)

**Prerequisite (checkpoint loader — established here, not assumed):**

- [x] 🟢 **Checkpoint → adapter loader** (`_build_adapter` real path): materialize each
  trained checkpoint via the platform `load_pretrained` (reuse the Phase-3 eval load path,
  not a hand-rolled `torch.load`) and wrap in `LeWM`/`DINOWMAdapter`. Fails loud.
  → `src/precision_match.py::_build_adapter` (pod-verify pending: needs real checkpoints).

- [x] 🔴 **DINO-WM `predict` — faithful `404 → 404` reconstruction:** revise
  `DINOWMAdapter.predict` to mirror `PreJEPA.predict` (dim-preserving; do **not** slice to
  384 — keep the predicted proprio). Move the extras embedding + initial `384 → 404` assembly
  and the per-step action-replacement + proprio-carry (`replace_action_in_embedding`) into
  the Python rollout/shim. Update `interfaces.py` predict I/O (`404 → 404`) + the `predict`
  example inputs (SPEC §Interface Contracts). → verify: predict output width == 404.
  → `src/adapter.py` (predict 404→404 + `assemble_embedding`), `src/shim.py`
  (`_replace_action` + `dino_rollout`); example inputs updated; verified predict out == 404.
- [x] 🟢 **Adapter-fidelity gate (before export):** on the real checkpoint, assert the
  adapter's `encode` + `predict` + shim rollout + criterion reproduces the platform's
  `rollout` / `get_cost` within tolerance — `predict` *reconstructs* the platform forward, so
  a wrong `404` assembly/carry passes engine precision-match yet corrupts SR. → verify: max
  abs drift vs `get_cost` within tolerance on real weights.
  → `src/fidelity.py` (+ `tests/test_fidelity.py`): DINO-WM shim vs `DINOv3PreJEPA.rollout`,
  bit-for-bit (max_abs 0.0) on a dummy `DINOv3PreJEPA`; pod runs `python -m src.fidelity` on
  the real checkpoint. LeWM: `action_encoder` (`Embedder`) confirmed **per-frame**
  (`Conv1d(k=1)` + per-position MLP, no T-axis mixing) vs installed swm 0.1.1 → per-step
  `LeWMAdapter.predict` == `LeWM.rollout` whole-sequence act-encode (SPEC §Interface
  Contracts). Remaining 🔴: owner sign-off + a self-guarding runtime assert
  (`action_encoder(seq)[:,t] ≈ action_encoder(seq[:,:t+1])[:,-1]`) in `src/fidelity.py`.

- [x] 🔴 Real export **PyTorch→ONNX→TensorRT**, per model:
  `uv run python -m src.export model=<lewm|dino> precision=<fp32|fp16|int8>`. Trace via
  `torch.onnx.export(dynamo=True)` (legacy TorchScript exporter deprecated; pass
  `dynamo=True` explicitly on torch 2.6), aiming a thin `nn.Module` forward-wrapper at each
  method — `encode` + `predict` traced separately → **4 base ONNX graphs** total (2 methods ×
  2 models). FP32/FP16 share the base graph (FP16 = a build flag); **INT8 is a separate,
  explicitly-quantized ONNX** from the Model Optimizer (step below), one Q/DQ graph per
  method. **FP32 + FP16 build data-free here; INT8 is deferred to the explicit-quantization
  step below** — the Model Optimizer inserts Q/DQ + derives scales from a calibration pass,
  so the INT8 (quantized-ONNX) engine cannot exist until that runs. ONNX/Model-Optimizer/TRT
  debugging and FP32/FP16/INT8 **precision matching** are OWNER-ONLY — STOP and ask.
  → `src/export.py`: explicit-arity predict trace wrappers (`_Predict1Module` DINO /
  `_Predict2Module` LeWM, selected by predict arity) so torch.export sees real params (flat
  `dynamic_shapes`, named ONNX inputs) instead of a variadic `*inputs` pytree collapse;
  added the `model=/precision=` CLI (reuses `precision_match` `_build_adapter` +
  `example_inputs`, writes to `engines/<track>/`). Verified locally: all 4 ONNX graphs trace
  with a dynamic batch axis; `pytest` green. TensorRT engine build + precision matching are
  pod-only (🔴 owner).
- [x] 🔴 **INT8 explicit quantization (Model Optimizer PTQ) + calibration set (before the
  gate):** switch INT8 from the implicit TRT-calibrator path to **explicit Q/DQ** — base FP32
  ONNX (dynamo, above) → **`modelopt.onnx.quantization`** inserts Q/DQ + derives per-tensor
  scales from a calibration pass → quantized ONNX per method → `build_engine` (TensorRT honors
  the baked-in Q/DQ; **no** `int8_calibrator`, no calibration profile). The calibration
  **data** (representative Push-T through the platform, matched ImageNet norm; two streams —
  encoder obs; predictor `404` via the real adapter) is reused; only its **consumer** changes
  (numpy arrays keyed by ONNX input name for the Model Optimizer, not CUDA pointers for a TRT
  calibrator). Sequenced **before** the precision-match gate so INT8 earns a drift row (its
  drift *is* the PTQ/calibration-quality signal). OWNER-ONLY silent-failure: owner sets the
  sample source/count **and the Model-Optimizer quant config** (calibration method —
  MinMax→`max` is the direct analogue — Q/DQ format, per-channel-vs-tensor, op-type
  exclusions). ⏱️ capped with FP16-only fallback.
  → reopened from the implicit calibration implementation (commits `540a27b`, `b10b495`):
    - `setup.sh`: add `nvidia-modelopt[onnx]` (alongside TensorRT, out of uv, CUDA-12.4).
    - `src/calibrate.py`: **keep** the clip draw (`draw_calibration_clips`, 512 clips strided
      across `pusht_expert_train.lance`, vendored `eval_wm.img_transform`) + the per-method
      streams; **replace** `make_calibrator` (`IInt8MinMaxCalibrator`) with a per-method
      numpy-dict producer keyed by ONNX input name for `modelopt`.
    - `src/export.py`: add a `quantize_onnx` step (Model-Optimizer PTQ on the base FP32 ONNX)
      in the INT8 path; **drop** the `build_engine` INT8 branch (`int8_calibrator` +
      `set_calibration_profile`) — INT8 parses the quantized ONNX like FP32/FP16.
    - `interfaces.py`: re-document `Export.calib_loader` as the Model-Optimizer PTQ input.
  → done: `setup.sh` installs `nvidia-modelopt[onnx]` + a **CUDA-12** `onnxruntime-gpu` (from
    onnxruntime's cu12 feed, pinned before modelopt so its unbounded dep isn't re-resolved to
    the cu13 PyPI default — which fails `cudnnCreate` on the 12.x driver), out of uv, and
    sanity-opens a real ORT CUDA-EP session. Confirmed torch-2.6-compatible pins:
    `modelopt==0.43.0` (latest 0.45 needs a cu13 torch 2.13) + `onnxruntime-gpu==1.24.4` (cu12,
    what 0.43.0 pins) + torch pinned to the cu124 build during the install (a newer-torch
    modelopt then fails loud, not a silent cu124->cu13 swap). modelopt 0.43.0 also caps
    `onnx==1.19.1` vs the locked 1.22.0 — lock-harmonization TBD (owner);
    `src/calibrate.py` keeps the clip draw/streams, `make_calibrator` → `make_calibration_dict`
    (numpy dict keyed off the base ONNX graph's real input names); `src/export.py` adds
    `quantize_onnx` (`modelopt.onnx.quantization.quantize`, `calibration_method="max"`,
    `use_external_data_format=True`) in the INT8 path and drops the `build_engine` calibrator +
    calibration-profile branch (INT8 keeps only the INT8 flag, parses the quantized ONNX);
    `interfaces.py` re-documents `calib_loader`. `pytest` green off-pod (28, incl. the
    numpy-dict producer). 🔴 owner-confirm on pod: the non-`max` Model-Optimizer knobs
    (Q/DQ format, per-channel-vs-tensor, op-exclusions) left at INT8 defaults; keeping the TRT
    INT8 flag for a Q/DQ graph.
  → **calibration EP split (fix, 2026-07-10):** encoder calibrates on the **GPU (CUDA EP)**,
    predictor on the **CPU EP** (`quantize_onnx(force_cpu_calibration=name=="predictor")`). The
    `onnxruntime-gpu` CUDA EP miscomputes the predictor's dynamic-batch reshape and crashes
    modelopt's MHA probe (`Reshape` wants `{192,3,16,64}` from `{8,3,1024}`); CPU EP is correct.
    EP affects calibration speed only, not scales (rationale in SPEC). Verify: quantized ONNX
    carries QuantizeLinear + INT8 engine builds; encoder calibration binds CUDA EP, predictor
    binds CPU EP — pod-only (needs dataset + `tensorrt` + `modelopt`).
- [x] 🖥️🔴 **Precision-match gate (before profiling/benchmark):** run
  `uv run python -m src.precision_match track=<lewm|dino>` on the **real** FP32+FP16+INT8
  engines (INT8 from the Model-Optimizer Q/DQ ONNX) → engine-vs-PyTorch drift table. 🔴 OWNER
  sign-off: inspect drift, decide the rel-error metric (max vs percentile), and judge the
  drift by eye. **No coded pass/fail** — `precision_match` reports drift only; the gate is the
  owner's sign-off on the drift table, not a tolerance object (the `PrecisionTolerance`
  dataclass was removed). Engines trusted before the steps below build on them.
  → **OWNER SIGN-OFF (2026-07-10):** both tracks run on real checkpoints; drift judged on
  **abs** only — the `max_rel` column is a near-zero-denominator artifact (explodes with
  element count / batch; FP32-faithful engines show 1e2–1e7 rel) and is **disregarded**, not
  gated. Structural check PASS: FP32 engines faithful (lewm enc_abs ≤7e-3, dino ≤5.6e-2) off
  the shared base ONNX → export/assembly/register-slice/reshape sound, so FP16/INT8 drift is
  genuine quantization loss, **not an export break** (monotone fp32<fp16<int8, bounded/finite).
  Per-precision sign-off: **FP32 trusted**; **FP16 trusted-provisional** (pending SR); **INT8
  recorded-but-flagged degraded** — lewm borderline (enc_abs ~0.6–1.0), dino large (enc_abs
  ~3–4, per-tensor INT8 vs DINOv3 outlier channels) → **INT8 is the FP16-only-fallback
  candidate**, SR-per-precision is the arbiter (Phase-5 benchmark). All rows kept for the
  FP32→FP16→INT8 delta + speed-vs-SR study.
- [ ] 🖥️ **Per-component profiling** — encoder, predictor, and planner (CEM) separately,
  per planning cycle, for both models × precisions (`src/profile.py` or a `benchmark`
  mode). Use the Phase-1 cycle decomposition. Slices must be **mutually exclusive +
  additive** (`planner_ms` = pure CEM/Python overhead, encode/predict time subtracted;
  sum ≈ cycle within the sync barrier). Record the **FP32 baseline per-component time
  shares** + the derived **optimizable fraction** `(enc+pred)/total`, per model (SPEC
  §Speedup study — dilution disclosure).
  → **runtime-weighting fix (plumbing landed, pod-run pending):** `src/profile.py` per-call
  means are now weighted by the CEM per-cycle call counts (predict `(horizon+1)×n_steps`=180
  at the candidate batch, encode ×2, planner ×n_steps) into `*_cycle_ms` that DO sum to the
  cycle, and it derives `optimizable_fraction p=(enc+pred)/cycle` + the Amdahl ceiling
  `1/(1-p)` (unweighted per-call sums under-weighted the predictor and `p` wasn't computed).
  `src/study.py` profiles `encode` at batch 1 / `predict` at `num_samples` so the weights are
  honest. `ComponentProfile` extended; `tests/test_profile.py`.
- [ ] 🖥️ **Fixed-time-budget benchmark** on the L40S: per model × precision, record
  **rollouts completed**, **per-step latency p50/p95**, throughput (rollouts/sec), **peak
  GPU memory** (sample via `cudaMemGetInfo`/nvidia-smi — **not** the torch allocator; TRT
  engine + context device allocations bypass it, SPEC §Interface Contracts), **and SR** —
  the Phase-3 eval driver re-run on the optimized model, which
  slots into `CEMSolver(model=...)` through a thin Python `get_cost`/`get_action` shim over
  the engine's `encode`/`predict` (SPEC §Interface Contracts). The DINO shim reproduces
  `PreJEPA.rollout` (full `404` carry, per-step action-replace, proprio+pixels cost). Same
  env/goal/precision/budget across models; only the model differs.
  → **instrument + labeling fixes (plumbing landed, pod-run + gated SR pending):**
  `src/benchmark.py` `peak_mem_mb` now samples **cudaMemGetInfo** (`torch.cuda.mem_get_info`,
  device-level used) per rollout, not `torch.cuda.max_memory_allocated` — so the TRT
  engine/context arena is counted (was undercounting the optimized path). `rollouts_completed`
  /`throughput` documented as **model-only** (planner-free ceiling; realized = gated eval-shim)
  and `p50/p95` as **predictor-step** latency (encode untimed; LeWM on a launch+sync floor) —
  in the benchmark docstring + `interfaces.BenchResult`. The `get_cost`/`get_action` SR shim
  itself stays 🔴 owner-gated (eval/CEM parity).
  → **owner-gated SR shim (DINO-WM) landed:** `src/sr_shim.py::DINOWMSRShim` subclasses
  `DINOv3PreJEPA` and overrides ONLY `_encode_image` (→ encoder engine) + `predict` (→
  predictor engine); `get_cost`/`rollout`/`criterion`/`split_embedding`/goal-encode are
  inherited byte-unchanged, so cost parity holds by construction (non-`Actionable` → zero-pad
  warm-start, matching the Phase-3 baseline). Parity reference = installed swm 0.1.1
  `PreJEPA.get_cost` (the `~/stable-worldmodel` checkout is 16 commits ahead / diverged).
  `build_engine_fns(engines)` = pod EngineRunner callables; `.from_adapter` = torch path for
  the gate. Gate `src/sr_shim.py::sr_cost_parity` (+ `tests/test_sr_shim.py`): shim.get_cost
  vs `PreJEPA.get_cost` **bit-for-bit** (max_abs 0.0 on dummy `DINOv3PreJEPA`, n_obs∈{1,3});
  pod runs `python -m src.sr_shim` on the real checkpoint. **Static-hist (resolved shim-side,
  no re-export):** the encoder engine traces a **static hist axis** while the inherited
  goal-encode calls it at T=1 vs init at n_obs=hist. `build_engine_fns` wraps the encoder
  callable with `_hist_adapt` (repeat-pad the frame axis up to the traced hist, encode, slice
  back) — exact because the encoder is temporally independent (per-frame), keeping the
  precision-match-gated engine byte-for-byte; `tests/test_sr_shim.py` covers it off-pod.
  → **LeWM encode+predict shim landed (engine-backed, Design A):** `src/sr_shim.py::LeWMSRShim`
  subclasses `LeWM` and routes BOTH `encode` and `predict` through injected engine/adapter
  callables, so the SR reflects the same quantized encoder+predictor engines the benchmark times
  (predictor FP16/INT8 drift enters the cost, as for DINO). `encode` has no `_encode_image` seam
  (`LeWM.encode` fuses backbone + info-dict bookkeeping + the `action_encoder` branch), so the
  override RE-IMPLEMENTS its body. **Predict — Design A (owner-chosen): `action_encoder` lives
  INSIDE the engine** (the boundary the per-frame guard justifies); the exported engine ingests a
  RAW action. Since inherited `LeWM.rollout` pre-encodes the whole action sequence, the shim sets
  its `action_encoder` to an **Identity passthrough** so rollout windows raw actions straight into
  `predict` and the engine's own per-frame `action_encoder` does the encode — bit-for-bit equal to
  the source's whole-sequence pre-encode (per-frame guard). `build_lewm_engine_fns` builds the
  two-input predict engine callable; `from_engines` wires the pod path. Gate `sr_cost_parity_lewm`
  (+ `tests/test_sr_shim.py`, `build_dummy_lewm_model`): shim.get_cost vs `LeWM.get_cost`
  **bit-for-bit** (max_abs 0.0, n_obs∈{1,3}) with encode AND predict via the adapter's torch
  methods. Run at **B=1**: vendored CEM pins `batch_size=1` and `LeWM.criterion` (pinned swm
  0.1.1) only supports one env per solve (broadcasts the single-env goal over candidates, errors
  for B>1 — the checkout removed these methods). Per-frame boundary owner-signed-off 2026-07-11
  (`src/fidelity.py::lewm_action_encoder_per_frame`); the LeWM predict engine + adapter-fidelity
  gate can now build.
- [ ] Headline outputs (tables **and plots**): **LeWM-vs-DINOv3 rollouts-in-budget ratio**
  + **p95 latency ratio**; **per-model FP32→FP16→INT8 delta** in **both speed and SR,
  degradation quoted vs FP32**; **speed-vs-SR plotted**; **per-component
  (encoder/predictor/planner) bottleneck breakdown**. Per model × precision, report
  **both** the *model-only* and the *realized* wall-clock speedup (gap = planner floor,
  ≈ Amdahl from the baseline shares), alongside the optimizable fraction (SPEC §Speedup
  study — dilution disclosure).
  → **table runner (landed, pod-run pending):** `src/study.py` (`uv run python -m src.study
  [track=<lewm|dino>] [wandb=<eval overlay>]`) orchestrates the three boxes above per
  track×precision — profiles the PyTorch adapter (FP32 baseline shares), loads the engines
  `src.export` built (`engines/<track>/{encoder,predictor}.<prec>.plan`), benchmarks each
  built precision, and calls `src.report`. Missing engines → precision skipped (FP16-only
  fallback); SR left NaN for the gated eval-shim join. `tests/test_study.py`.
  → **honesty fixes in `src/report.py` (landed):** rollouts ratio labelled **model-only** and
  the **Amdahl dilution table** (p, ceiling, per-precision model-only vs Amdahl-predicted
  realized speedup; measured-realized marked *gated*) added — so the planner floor that
  dilutes the model-only ratio is visible, not hidden. p95 ratio labelled **predictor-step**.
  Component breakdown table/plot now stack the **runtime-weighted** `*_cycle_ms`. Every
  unpaired-SR row is flagged **SR-PENDING** (a speed number without its SR is not a validated
  win); `sr=<json>` / `report(sr_overrides=)` joins the gated eval-shim SR back in without
  code edits. `tests/test_report.py`.
  → **persist headline artifacts to network storage (pending):** write the tables
    (serialized to `.txt`) **and** plots (`.png`) under `$STABLEWM_HOME/reports/phase5/`
    (not the repo-local `reports/phase5` default) so a completed study survives pod teardown;
    W&B logging stays additive (SPEC §Speedup study — Headline-artifact durability).
    Requires: `src/report.py` serialize each table to a file (currently stdout + W&B HTML
    only); `src/study.py` default `out_dir` under `$STABLEWM_HOME` (env-derived, repo-local
    fallback). Verify: after a study run the three table `.txt` files + four plot `.png`
    files exist under `$STABLEWM_HOME/reports/phase5/`.
  → **canonical per-track results JSON + decoupled render (landed):** `src/study.py`
    (`dump_track_results`) writes each track's raw benchmark + profile numbers plus the run's
    fairness conditions (time_budget_s, num_samples, seed, obs_shape) to `results.<track>.json`
    under `out_dir` **before** rendering — the canonical machine-readable result; tables/plots
    are regenerable views. **Per-track files** so lewm/dino benchmark in separate pod sessions
    without clobbering (CLAUDE.md §8). `src/report.py` gains `load_results` + a `from=<dir|file>`
    entrypoint (`uv run python -m src.report from=$STABLEWM_HOME/reports/phase5 [out=<dir>]
    [sr=<f.json>] [wandb=<ov>]`) that merges the per-track JSONs and re-renders **off-pod** —
    the later, separately-gated SR-per-precision join needs no L40S re-run. Single-track render
    **skips** the two cross-track ratio plots (empty otherwise). `tests/test_study.py` +
    `tests/test_report.py`. Verify: `results.{lewm,dino}.json` round-trip through
    `report.load_results`; a one-track render emits no `*_ratio.png`.
- [ ] ⏱️ **Cap on TensorRT/INT8** (unsupported-op / Model-Optimizer PTQ / Q/DQ); fallback =
  **FP16-only**. 3-attempt debugging cap (CLAUDE.md §6).

**Interface note:** `src/interfaces.py` declares `BenchResult.rollouts_completed`, the
fixed `time_budget_s` on `Benchmark` / `ExportConfig`, and `ComponentProfile` / `Profile`.

**Verify:** engines built on the L40S (gitignored, regenerable); precision-match gate
passed (drift owner-signed-off on the drift table, or documented) **before** benchmarking; fixed-budget
comparison (rollouts + p95 latency **+ SR per precision**, FP32-relative degradation quoted)
and the encoder/predictor/planner profile tables produced and logged to W&B.

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
- `src/adapter.py`, `src/export.py` (incl. the Model-Optimizer INT8 Q/DQ step),
  `src/calibrate.py` (calibration-data construction feeding the Model Optimizer),
  `src/benchmark.py`, `src/profile.py`, `src/qlora.py`, `src/smoke.py` — the owned layer
  (Phases 4–6).
- `src/wandb_log.py` — owned W&B helper for the non-training phases (Phase 3+).
- `src/eval_latency.py` — owned observation-only CEM-solve-latency callback (`CEMSolver.Callback`
  subclass, injected via `cfg.solver.callbacks`; Phase 3).
- `src/eval.py` — owned thin Phase-3 eval driver: runs the byte-unmodified `eval_wm.run`,
  captures `World.evaluate`'s SR, and logs SR + latency to W&B (no monkeypatch/vendored edit).
- `conf/` — owned Hydra overlays (`conf/experiment/{lewm,dinov3}.yaml` train,
  `conf/experiment/eval_{lewm,dino}.yaml` eval — the latter set `policy`/`eval.dataset_name`/
  `wandb:` and inject the latency callback via `cfg.solver.callbacks`).
- `scripts/train/lewm.py`, `scripts/train/prejepa.py` + `scripts/train/config/` —
  vendored platform entrypoints/configs, as used (provenance in `scripts/train/VENDORED.md`).
- `scripts/plan/eval_wm.py` + `scripts/plan/config/{pusht.yaml, solver/cem.yaml}` —
  vendored platform eval entrypoint/config, as used (provenance in `scripts/plan/VENDORED.md`).
- `scripts/verify_encode.py` — Phase-2 encode sanity: latent dims + register-slice
  differential (owned, fails loud; no dataset/checkpoint needed).
- `pyproject.toml`, `uv.lock`, `setup.sh` (pod bootstrap). `Dockerfile` +
  `docker-compose.yml` composed at project end (off-pod).
- `tests/` — pytest for the owned boundaries.

## Cross-cutting rules

- **Owner gates:** anything 🔴 (export/INT8/Model-Optimizer-PTQ debugging, precision
  matching, QLoRA targeting, benchmark methodology, adapter dims, eval/CEM parity) → STOP
  and ask.
- **Git:** never run git. On completing a unit of work, output the files to stage and a
  `type(scope): summary` commit message; the owner runs git.
- **Progress:** each `[x]` records artifact name; tick before advancing.
- **Caps:** TRT/INT8 is time-capped with a fallback; training is epoch-capped; 3-attempt
  debugging cap; log-before-delete.

## End-to-end verification

1. `bash setup.sh` (uv + deps + TensorRT) + import check (Phase 0).
2. Platform API introspection in-container; DINOv3 `config.hidden_size`/
   `last_hidden_state` confirmed (Phase 1).
3. Two checkpoints + W&B runs + `scripts.verify_encode` PASS (Phase 2); both-track
   success-rate + latency under matched CEM config (Phase 3).
4. `python -m src.smoke` + `pytest -v` green on dummy weights (Phase 4).
5. `src.export` builds TRT engines on the L40S; fixed-budget benchmark emits
   rollouts-in-budget + latency comparison and the encoder/predictor/planner profile
   tables (Phase 5).
6. `src.qlora` produces a tuned backbone; task-metric delta vs frozen reported (Phase 6).
