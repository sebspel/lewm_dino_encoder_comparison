"""Vendored platform entrypoints (`train/`, `plan/`) plus the owned `verify_encode` check.

Importing this package reads `.env`, which is why the hook lives here rather than in the
entrypoints themselves. The vendored scripts never import `src`, so nothing else would load it:
`scripts/train/lewm.py` has no `src` target at all, and `scripts/train/prejepa.py` reaches
`src.dino_patch` only when Hydra instantiates the model — *after* `swm.data.load_dataset` has
already resolved the dataset root off `STABLEWM_HOME`. Both would otherwise fall back silently
to the platform's `~/.stable_worldmodel` default, which is the ephemeral container filesystem
on the pod, and `WandbLogger` would stall on a login prompt.

`python -m scripts.<...>` (how README and both VENDORED.md files invoke them) imports this
package before the entrypoint body runs, so the load is early enough for every path. The
vendored files themselves stay byte-unmodified.
"""

from src.env import load_env

load_env()
