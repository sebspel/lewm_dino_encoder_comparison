# Vendored platform eval entrypoint — provenance

The platform's MPC/CEM evaluation driver, copied so the task baseline runs "as used"
(SPEC §Scope, PLAN Phase 3). The eval entrypoint + its Hydra config group are **not
shipped in the PyPI wheel** (pod-confirmed absent — PLAN Phase 3), so they are fetched
from the tagged GitHub tree, same as the Phase-2 train vendoring
(`scripts/train/VENDORED.md`).

**Deviation from verbatim (one bugfix — owner-approved).** `eval_wm.py`'s episode-index
column detection reads `dataset.column_names` to choose `episode_idx` vs `ep_idx`. That
works for the H5 format but not for `.lance`: `LanceDataset.column_names` excludes the
reserved `episode_idx`/`step_idx` index columns, so the check always falls back to `ep_idx`
— a field that doesn't exist in the lance schema — and `get_col_data('ep_idx')` raises
`Schema error: No field named ep_idx`. Fix (all 3 occurrences): consult the raw schema when
available — `getattr(dataset, '_schema_names', dataset.column_names)` — so lance's
`episode_idx` is seen while H5 behavior is unchanged. Upstream bug, specific to running eval
on a `.lance` dataset (the eval overlays point `eval.dataset_name` at the trained
`pusht_expert_train.lance`). No other lines changed.

Otherwise copied **verbatim.** The DINOv3-WM register-slice does **not** need a change
here: eval reconstructs the model with `swm.wm.utils.load_pretrained(cfg.policy)`, which
rebuilds it from the checkpoint's own saved `config.yaml` — that config already carries
`model._target_: src.dino_patch.DINOv3PreJEPA` (baked in at train time,
`conf/experiment/dinov3.yaml`), so the register-aware encode path flows in through the
checkpoint, not through a runtime overlay.

- **Source:** `galilai-group/stable-worldmodel`
- **Tag:** `0.1.1` (matches the `stable-worldmodel==0.1.1` pin in `uv.lock`)
- **Commit:** `15a5538d492ae524c64cb18cc56a2d70611e877e`

Vendored:

| Repo path @ 0.1.1 | Here |
|---|---|
| `scripts/plan/eval_wm.py` | `scripts/plan/eval_wm.py` |
| `scripts/plan/config/pusht.yaml` | `scripts/plan/config/pusht.yaml` |
| `scripts/plan/config/solver/cem.yaml` | `scripts/plan/config/solver/cem.yaml` |
| `scripts/plan/config/launcher/local.yaml` | `scripts/plan/config/launcher/local.yaml` |

Only the config groups `pusht.yaml` actually pulls are vendored: its `defaults` list is
`[launcher: local, solver: cem, _self_]`, so `launcher/local.yaml` and `solver/cem.yaml`
are required for Hydra composition to resolve — the PLAN names the latter two explicitly
and `launcher/local.yaml` is the transitive third. The other `scripts/plan/*` env configs
(`cube.yaml`, `reacher.yaml`, `tworoom.yaml`) and the `solver/adam.yaml` alternative are
intentionally omitted (SPEC §Simplicity).

**Runtime deps (already pinned transitively — no `pyproject.toml` change).** `eval_wm.py`
imports `sklearn` (`scikit-learn`) and `torchvision`; both already resolve through
`uv.lock` as transitive deps of `stable-pretraining` / `stable-worldmodel`.
