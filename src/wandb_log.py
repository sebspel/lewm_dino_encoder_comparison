"""Owned W&B helper for the non-training phases (eval / benchmark / QLoRA).

Training logs to W&B through the platform's Lightning ``WandbLogger``, driven by the
``wandb:`` block in ``conf/experiment/``. The non-training phases have no Lightning
``Trainer``, so they log through this helper — which reads the project (and entity) from
that **same** ``wandb:`` block, so there is a single source of truth for the shared
project name.

``wandb`` is imported lazily inside the functions so the config-reading (`read_wandb_cfg`)
stays importable without wandb installed.
"""

from pathlib import Path

from omegaconf import OmegaConf

_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "conf" / "experiment"


def read_wandb_cfg(experiment):
    """Return the ``wandb`` block from ``conf/experiment/<experiment>.yaml``.

    ``experiment`` is the overlay name (``"lewm"`` / ``"dinov3"``) — the same file the
    training run for that track logged through, so eval/benchmark land in its project.
    """
    path = _EXPERIMENT_DIR / f"{experiment}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no experiment overlay at {path}")
    block = OmegaConf.load(path).get("wandb")
    if block is None or "config" not in block:
        raise KeyError(f"{path} has no `wandb.config` block")
    return block


def init(experiment, name=None, config=None):
    """`wandb.init` for a non-training phase, project read from the experiment overlay.

    Honours the overlay's ``wandb.enabled`` flag: when false, inits in ``disabled`` mode
    so the calling phase runs (and is testable) without a W&B account.
    """
    import wandb

    block = read_wandb_cfg(experiment)
    return wandb.init(
        project=block.config.project,
        entity=block.config.get("entity", None),
        name=name,
        config=config,
        mode="online" if block.get("enabled", True) else "disabled",
    )
