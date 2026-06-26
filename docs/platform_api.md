# Platform API — read from the real pinned source (Phase 1)

> Per CLAUDE.md §10, this records the **actual** `stable-worldmodel` /
> `stable-pretraining` API as installed at our pinned versions — not from memory.
> Values marked **⟨runtime-confirm⟩** need the model instantiated on the L40S pod
> (HF weights download) to be certain; values marked **🔴 OWNER** are the
> sign-off gate before any dim is hard-coded in `src/interfaces.py` (Phase 4).

## Provenance

- `stable-worldmodel==0.1.1`, `stable-pretraining==0.1.7` (from `uv.lock`, PyPI).
- Package source read from the pinned PyPI **sdists**.
- Training/eval entrypoints + Hydra configs are **not shipped in the wheel**; read
  from GitHub `galilai-group/stable-worldmodel` at tag **`0.1.1`** (matches the wheel):
  `scripts/train/{lewm,prejepa}.py`, `scripts/plan/eval_wm.py`,
  `scripts/{train,plan}/config/**`.
- This was read on the **edit-only WSL box** (no GPU pod, platform not installed
  locally). The static API/dim *derivations* below are version-locked and reliable;
  the ⟨runtime-confirm⟩ items still require a one-shot introspection on the pod.

---

## 1. Latent extraction — the two tracks (the core asymmetry)

### LeWM — single-token latent `(B, D)`
`stable_worldmodel/wm/lewm/lewm.py :: LeWM.encode`:
```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]   # CLS token  -> (B*T, D)
emb = self.projector(pixels_emb)              # MLP(D->D)
# info['emb'] : (B, T, D)
```
- Encoder: `stable_pretraining.backbone.utils.vit_hf(size=tiny, patch_size=14,
  image_size=224, pretrained=false)` — scratch **ViT-Tiny**.
- `embed_dim: 192` (config `scripts/train/config/lewm.yaml`); projector/pred_proj are
  `MLP(192->192)`. **LeWM latent = `(B, 192)`, one token.**
- `predict(emb, act_emb)`: `lewm.module.Predictor`, AdaLN-zero conditioned on the
  action embedding; `num_frames=history_size=3`.
- Action encoder: `lewm.module.Embedder(input_dim = frameskip(5) * action_dim(2) = 10,
  emb_dim = 192)`.

### DINO-WM (`prejepa.py`) — full patch grid `(B, N_patches, D)`
`stable_worldmodel/wm/prejepa/prejepa.py :: PreJEPA._encode_image`:
```python
pixels_embed = self.backbone(pixels, interpolate_pos_encoding=True)
if hasattr(pixels_embed, 'last_hidden_state'):
    pixels_embed = pixels_embed.last_hidden_state
    pixels_embed = pixels_embed[:, 1:, :]      # <-- DROPS CLS ONLY
# -> (B, T, P, D)
```
- Backbone is HF `AutoModel.from_pretrained(...)`, config-injected and **frozen** in
  `scripts/train/prejepa.py`: `encoder.eval(); encoder.requires_grad_(False)`. ✓ matches SPEC.
- Default backbone is **`dinov2_small`** — must be overridden to DINOv3 (see §4).
- The encoder patch grid (`pixels_emb`) is `D = encoder.config.hidden_size`.
  Action/proprio embeddings are tiled across patches and **concatenated on the feature
  axis** before the predictor, so the predictor token dim is `D + Σ(extra encoding dims)`.

---

## 2. Dims — runtime-confirmed on the L40S pod (2026-06-26) ✓

DINOv3 alias available at 0.1.1 (`wm/prejepa/module.py :: BACKBONE_ALIASES`):
`dinov3_small -> facebook/dinov3-vits16-pretrain-lvd1689m` (**ViT-S/16, the only v3 alias**).

All values below verified by instantiating the real encoders + PushT env on the pod
(`create_backbone("dinov3_small")`, `vit_hf(size="tiny", ...)`, `gym.make("swm/PushT-v1")`).

| Quantity | Value | Source / confirmation |
|---|---|---|
| `ACTION_DIM` (env) | **2** | `swm/PushT-v1` `action_space == Box(-1.0, 1.0, (2,), float32)` ✓ |
| LeWM `LATENT_DIM` | **192** | `vit_hf(tiny).config.hidden_size == 192`; CLS token `(1, 192)` ✓ |
| DINOv3 `D` (hidden_size) | **384** | `encoder.config.hidden_size == 384` ✓ |
| DINOv3 patch size | **16** | `encoder.config.patch_size == 16` ✓ |
| DINOv3 true patches | **196** | (224/16)² ✓ |
| DINOv3 register tokens | **4** | `encoder.config.num_register_tokens == 4` ✓ |
| `last_hidden_state` length | **201** | `(1, 201, 384)` = 1 CLS + 4 reg + 196 patch ✓ |
| **Resolved `N_patches`** | **196** | OWNER: slice CLS **and** registers (§6) → true 196-patch grid ✓ |
| predictor token dim | **404** | `hidden(384) + proprio(10) + action(10)`; extras from `wm.encoding` |

`scripts/train/prejepa.py` sizes the predictor **from config, not from the encoder
output**:
```python
embed_dim  = encoder.config.hidden_size + sum(cfg.wm.encoding.values())   # 384+20=404
num_patches = (cfg.image_size // cfg.patch_size) ** 2                      # config-derived!
cfg.model.predictor.dim         = embed_dim
cfg.model.predictor.num_patches = num_patches
```
`num_patches` also drives the predictor's positional embedding **and** its causal
time-block attention mask (`CausalPredictor` / `Attention.generate_mask_matrix`), so a
wrong `num_patches` fails **silently** (mis-aligned mask), not loudly.

---

## 3. `World.evaluate` + CEM — signatures & parity numbers

- `swm.World(env_name='swm/PushT-v1', num_envs, image_shape=(224,224), ...)`.
- `World.evaluate(...)` returns `{'success_rate' (percent), 'episode_successes',
  'seeds'}`. Push-T eval is **dataset-driven**: `evaluate(dataset=, episodes_idx=,
  start_steps=, goal_offset=, eval_budget=, callables=, video=)`
  (`scripts/plan/eval_wm.py`).
- Planner stack: `WorldModelPolicy(solver=CEMSolver, config=PlanConfig, transform=,
  process=)` → `solver.solve(info)` → `model.get_cost(infos, candidates)`.

**Parity config (from `scripts/plan/config/{pusht.yaml,solver/cem.yaml}` — load-bearing,
do not vary between tracks):**
- CEM (`CEMSolver`): `num_samples=300`, `topk=30` (elites), `n_steps=30` (iters),
  `var_scale=1.0`, `batch_size=1`, `device=cuda`, `seed=42`.
- `PlanConfig`: `horizon=5`, `receding_horizon=5`, `action_block=5` (frameskip).
- `eval`: `num_eval=50`, `goal_offset_steps=25`, `eval_budget=50`, `img_size=224`.
- Normalization: ImageNet stats applied in `img_transform` (`transforms.Normalize(**spt.data.dataset_stats.ImageNet)`).
- Matches SPEC §Parity ("300 samples, 30 elites, horizon 5, init var 1, 10–30 iters").

---

## 4. DINOv2 → DINOv3 override (Phase-2 `conf/`, flagged here)

`prejepa.yaml` defaults to DINOv2 and assumes patch 14. The DINOv3 track needs, at
minimum:
- `backbone.name=dinov3_small`, `backbone.type=dinov3_small`
- `patch_size=16` (DINOv3 ViT-S/16, **not** the dinov2 default of 14) — otherwise
  `num_patches=(224//14)²=256`, silently wrong.

⚠️ **RESOLVED in PLAN/SPEC (2026-06-26):** the original literal command
`prejepa backbone=dinov3` does **not** map to anything — there is no `backbone=` config
group. Corrected to `backbone.name=dinov3_small backbone.type=dinov3_small patch_size=16`
(or a new `conf/` group we add in Phase 2). The register slice is applied via a `PreJEPA`
subclass in `src/` (§6), not by editing the wheel.

---

## 5. One CEM planning cycle → encoder / predictor / planner (for Phase-5 profiling)

`CEMSolver.solve` (one `policy.get_action` replan):
1. `prepare_init_action` (warm-start).
2. Loop `n_steps=30` CEM iterations:
   - sample `candidates` via `torch.randn` — **PLANNER**
   - `model.get_cost(expanded_infos, candidates)`:
     - encode **goal** — **ENCODER**, computed **once** then cached
       (`_goal_cached_info` / `'goal_emb' in info`)
     - `rollout`: encode **initial obs** — **ENCODER**, once then cached
       (`_init_cached_info` / `'emb' in info`); then `predict` autoregressively over
       `horizon` for **all 300 candidates** — **PREDICTOR**
     - `criterion` MSE-to-goal — cheap
   - `torch.topk` elites + mean/var update — **PLANNER**

**Bottleneck note:** the encoder runs ~once per planning cycle (cached across the 300
candidates × 30 iters); the **predictor dominates call count**. The LeWM↔DINOv3
asymmetry still surfaces because the DINO predictor (and the cost) operate over **~200
patch tokens vs LeWM's 1 token** — i.e. the encoder's token count propagates into
predictor/planner cost. The `encoder/predictor/planner` profile must attribute the gap
accordingly (SPEC §Parity). Export boundary is confirmed: `eval_wm.py` `torch.compile`s
**`encoder` + `predictor` only** — the CEM loop stays in Python, exactly the
`WMStepAdapter` boundary TRT will optimize.

---

## 6. 🔴 OWNER GATE — RESOLVED (2026-06-26)

1. **DINOv3 register tokens → SLICE THEM (match SPEC).** OWNER chose to drop **CLS +
   the N register tokens** so the DINO-WM grid is the true **196 patches** (`D=384`),
   not the platform's default 200. Implementation (Phase 2, now owner-approved as the
   single surgical core-encoder edit): in the vendored/overridden `prejepa` path change
   `_encode_image`'s `last_hidden_state[:, 1:, :]` → `[:, 1+num_reg:, :]`, and set the
   config `num_patches=196`, `patch_size=16`. Must apply to **both** training and eval
   so the predictor is trained on the same 196-token grid it plans over.
   `num_reg == 4` confirmed on the pod → slice is `[:, 5:, :]` (1 CLS + 4 reg).
2. **Dims to hard-code in Phase 4** (after one-shot pod confirmation of `hidden_size`,
   `num_reg`, `vit_hf(tiny)` dim, PushT action space): `LATENT_DIM=192` (LeWM),
   DINO-WM patch grid `(N_patches, D) = (196, 384)`, `ACTION_DIM=2`.
3. **DINOv3 size → stay on `dinov3_small`** (ViT-S/16, 384-d). Asymmetry vs LeWM
   ViT-Tiny is smaller than the paper's ~48×; accepted for now.
