"""Runtime configuration loading (`.env`) — the hooks and the single parser, not the parsing.

`src/env.py` wraps python-dotenv, so what needs pinning is not the parsing itself but WHERE the
load is hooked in, and that there is only ONE parser. The vendored train/eval entrypoints never
import `src`, so without the `scripts` package hook a training run silently falls back to the
platform's `~/.stable_worldmodel` default (ephemeral on the pod) and loses a multi-hour run's
checkpoints.

Each check runs in its own interpreter — the pytest process has already imported `src`, so an
in-process assertion would pass whether or not the hook exists.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _loads_env(statement: str) -> bool:
    """Does `statement`, in a fresh interpreter, end with `.env` already read?"""
    out = subprocess.run(
        [sys.executable, "-c", f"{statement}\nimport sys; print('src.env' in sys.modules)"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip().splitlines()[-1] == "True"


def test_importing_src_loads_env():
    """Covers every `python -m src.<module>` entrypoint."""
    assert _loads_env("import src")


def test_importing_scripts_loads_env():
    """Covers the vendored `python -m scripts.train.*` / `scripts.plan.*` entrypoints, which
    import `src` NOWHERE: `scripts/train/lewm.py` has no `src` target at all, and
    `scripts/train/prejepa.py` reaches `src.dino_patch` only when Hydra instantiates the model —
    after `swm.data.load_dataset` has already resolved the dataset root off `STABLEWM_HOME`."""
    assert _loads_env("import scripts")


def test_the_vendored_entrypoints_still_do_not_import_src():
    """The premise the `scripts` hook exists for. If an entrypoint ever imports `src` directly the
    hook is redundant, but the reverse — this test failing because a vendored file gained an
    import — would mean the vendoring drifted."""
    for name in ("train/lewm.py", "train/prejepa.py", "plan/eval_wm.py"):
        source = (REPO_ROOT / "scripts" / name).read_text()
        assert "import src" not in source, f"{name} now imports src — revisit scripts/__init__.py"


def test_setup_reads_dotenv_through_the_same_loader():
    """setup.sh must DELEGATE to `src.env`, never re-parse `.env` in shell.

    A hand-rolled shell reader silently disagrees with python-dotenv on inline comments (`a #b`
    is a comment, `a#b` is not), CRLF line endings, quoting, `export ` prefixes, whitespace
    around `=`, and whether an EMPTY-but-set variable is preserved. Each disagreement provisions
    one directory while training reads another — the failure this delegation removes by
    construction rather than by matching cases."""
    setup = (REPO_ROOT / "setup.sh").read_text()
    assert "from src.env import load_env" in setup, "setup.sh no longer delegates to src.env"
    # A `while read ... < .env` loop is the shape of a re-implementation.
    assert 'read -r line' not in setup and '. "$repo_root/.env"' not in setup, (
        "setup.sh looks like it parses .env itself again — use src.env.load_env"
    )
