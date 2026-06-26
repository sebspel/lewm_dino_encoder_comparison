# Vendored platform training entrypoints — provenance

These files are copied **verbatim** (unmodified) from the platform so the foundation
trainings run "as used" (SPEC §Scope, PLAN Phase 2). Do not edit them — the DINOv3-WM
track's only change is layered on top via Hydra (`conf/experiment/dinov3.yaml` →
`model._target_: src.dino_patch.DINOv3PreJEPA`), never by editing the wheel or these copies.

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
