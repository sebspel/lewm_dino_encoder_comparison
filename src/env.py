"""Runtime configuration, read from an uncommitted `.env` at the repo root.

`STABLEWM_HOME`, `WANDB_API_KEY` and `HF_TOKEN` live in `.env` (gitignored; copy
`.env.example`) rather than being typed into a shell, so a pod session, a `src.pipeline`
subprocess and an off-pod render all resolve the same paths and secrets.

`load_env` runs once on `import src`, covering every `python -m src.<module>` entrypoint, and
once on `import scripts`, covering the vendored train/eval entrypoints — which never import
`src` themselves, so they would otherwise miss it entirely (see `scripts/__init__.py`).

A variable already present in the real environment WINS over the file, so a pod-level export and
`monkeypatch.setenv` both keep working. `setup.sh` reads `.env` under the same rule, so the shell
and Python never resolve the same key differently.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def load_env() -> None:
    """Load `.env` into the process environment without overriding what is already set."""
    load_dotenv(ENV_FILE, override=False)


def stablewm_home() -> Path:
    """The persistent network volume root — datasets, checkpoints, engines and reports.

    Raises rather than falling back to a repo-local path: the platform's own default is the
    ephemeral container filesystem, so a run with the variable unset would write multi-hour
    artifacts somewhere they are lost on pod restart, silently.
    """
    home = os.environ.get("STABLEWM_HOME")
    if not home:
        raise RuntimeError(
            "STABLEWM_HOME is not set. It is the persistent network volume root that datasets, "
            f"checkpoints, engines and reports are written to. Set it in {ENV_FILE} "
            "(copy .env.example), e.g. STABLEWM_HOME=/workspace/.stablewm on the pod."
        )
    return Path(home)
