# Vendored platform training entrypoints — provenance

These files are copied from the platform so the foundation trainings run "as used"
(SPEC §Scope, PLAN Phase 2). They are otherwise unmodified — the DINOv3-WM track's only
behavioural change is layered on top via Hydra (`conf/experiment/dinov3.yaml` →
`model._target_: src.dino_patch.DINOv3PreJEPA`), never by editing the wheel or these copies.

**Deliberate divergences from the tag** (kept minimal, recorded here):

- `prejepa.py`: removed `spt.callbacks.CPUOffloadCallback()` from the `pl.Trainer`
  callbacks list. The entrypoint is frozen at swm tag `0.1.1`, but we pin
  `stable-pretraining==0.1.7`, which no longer ships that callback (cross-package
  version skew — see CLAUDE.md §10). 0.1.7 has no equivalent offload callback, and it is
  a GPU-memory optimisation only (no effect on the model, latents, loss, or eval), so the
  line is dropped rather than replaced. DINOv3-WM (frozen ViT-S/16, batch 32) fits the
  L40S without it. `lewm.py` does not reference this callback, so it needs no change.

- `lewm.py`: `SaveCkptCallback` was passed `cfg=cfg` (the whole training config) →
  `save_pretrained` wrote that as `config.json`, which has no top-level `_target_`, so
  `load_pretrained`'s `instantiate(config)` returns a dict and `.load_state_dict` fails at
  eval time. Changed to `cfg=cfg.model` to match `prejepa.py` (which already passes
  `cfg.model`), so `config.json` is the model node and reconstructs the model. Preventive —
  does not fix checkpoints already saved with the whole cfg (repair those config.json files
  to their `["model"]` sub-node; the state_dict is unaffected).

- **Source:** `galilai-group/stable-worldmodel`
- **Tag:** `0.1.1` (matches the `stable-worldmodel==0.1.1` pin in `uv.lock`)
- **Commit:** `15a5538d492ae524c64cb18cc56a2d70611e877e`
- **Note:** entrypoints + their Hydra configs are not shipped in the PyPI wheel, so they
  are fetched from the tagged GitHub tree.

Vendored:

| Repo path @ 0.1.1 | Here |
|---|---|
| `scripts/train/lewm.py` | `scripts/train/lewm.py` |
| `scripts/train/prejepa.py` | `scripts/train/prejepa.py` |
| `scripts/train/config/lewm.yaml` | `scripts/train/config/lewm.yaml` |
| `scripts/train/config/prejepa.yaml` | `scripts/train/config/prejepa.yaml` |
| `scripts/train/config/launcher/local.yaml` | `scripts/train/config/launcher/local.yaml` |
| `scripts/train/config/data/pusht.yaml` | `scripts/train/config/data/pusht.yaml` |

Only the config groups the two entrypoints actually pull are vendored (the other
`scripts/train/*` entrypoints and their configs are intentionally omitted — SPEC §Simplicity).
