"""Owned thin eval driver.

Runs the byte-unmodified vendored eval entrypoint (``scripts.plan.eval_wm.run``) under its
own Hydra composition, then logs the Push-T success rate and the per-decision planning-latency
median to one W&B run (owned helper, shared project).

    uv run python -m src.eval --config-dir conf +experiment=eval_<lewm|dino>

No monkeypatch, no class shadow:
  * Per-decision latency rides in via ``cfg.solver.callbacks`` (the eval overlay injects the
    owned :class:`~src.eval_latency.SolveLatencyRecorder`, which brackets one episode's
    decision — NOT a whole solve, which plans every alive episode); the driver reads the
    records back through that module's registry.
  * Success rate is read observation-only from the results file the entrypoint writes —
    the driver points ``output.filename`` at a fresh, driver-owned path so the parse is
    deterministic (the entrypoint appends, so a shared file would accumulate runs).

The driver never edits eval_wm.py and never wraps World/solver — it only sets up the
Hydra overrides, opens the W&B run, and reads the two artifacts the run produces.
"""

import re
import sys
import tempfile
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from scripts.plan import eval_wm
from src import eval_latency, wandb_log

# success_rate in the metrics dict repr eval_wm writes: `'success_rate': 42.0` or, under
# newer numpy, `'success_rate': np.float64(42.0)`. Capture the numeric literal either way.
_SR_RE = re.compile(
    r"success_rate['\"]?\s*[:=]\s*(?:np\.\w+\(\s*)?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLAN_CONFIG_DIR = _REPO_ROOT / "scripts" / "plan" / "config"


def _experiment_from_argv(argv):
    """The ``+experiment=eval_<lewm|dino>`` override names the overlay whose wandb block
    the owned helper reads. Fail loud if absent (the driver has no other project source)."""
    for a in argv:
        if a.startswith("+experiment="):
            return a.split("=", 1)[1]
    raise SystemExit(
        "src.eval requires +experiment=eval_<lewm|dino> "
        "(e.g. --config-dir conf +experiment=eval_lewm)"
    )


def _parse_success_rate(text):
    m = _SR_RE.search(text)
    if not m:
        raise ValueError(
            "could not parse success_rate from the eval results file; tail:\n"
            + text[-500:]
        )
    return float(m.group(1))


def _compose_eval_cfg(argv, out_file):
    """Compose the vendored eval config without invoking eval_wm's Hydra wrapper.

    The vendored decorator uses ``config_path='./config'``. That works when the file is
    executed directly, but when imported as ``scripts.plan.eval_wm`` Hydra resolves it as
    a package path (``pkg://scripts.plan/./config``), producing the bad
    ``scripts.plan...config`` module lookup. Compose from the file config dir here, then
    pass the config through Hydra's supported ``cfg_passthrough`` path.
    """
    config_dirs = []
    overrides = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config-dir":
            i += 1
            if i >= len(argv):
                raise SystemExit("--config-dir requires a path")
            config_dirs.append(Path(argv[i]).resolve())
        elif arg.startswith("--config-dir="):
            config_dirs.append(Path(arg.split("=", 1)[1]).resolve())
        elif arg.startswith("--"):
            raise SystemExit(
                f"unsupported Hydra flag for src.eval: {arg} "
                "(supported: --config-dir plus config overrides)"
            )
        else:
            overrides.append(arg)
        i += 1

    if not config_dirs:
        config_dirs.append(_REPO_ROOT / "conf")

    searchpath = ",".join(f"file://{p}" for p in config_dirs)
    compose_overrides = [
        f"hydra.searchpath=[{searchpath}]",
        *overrides,
        f"output.filename={out_file}",
    ]

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None, config_dir=str(_PLAN_CONFIG_DIR)
    ):
        return compose(config_name="pusht", overrides=compose_overrides)


def main():
    argv = sys.argv[1:]
    experiment = _experiment_from_argv(argv)

    # Driver-owned results file so SR is read from a fresh file, not the appended default.
    out_file = Path(tempfile.mkdtemp(prefix="swm_eval_")) / "results.txt"
    cfg = _compose_eval_cfg(argv, out_file)

    eval_latency.reset_registry()
    run = wandb_log.init(
        experiment, name=experiment, config={"phase": "eval", "experiment": experiment}
    )
    try:
        # Byte-unmodified entrypoint: the composed cfg includes the callback injection
        # from the overlay; eval_wm instantiates it and runs World.evaluate.
        eval_wm.run(cfg)

        latency = eval_latency.pop_records()
        if latency["n_cycles"] == 0:
            raise RuntimeError(
                "the latency callback recorded no decisions — is it injected via "
                "cfg.solver.callbacks in the eval overlay?"
            )
        success_rate = _parse_success_rate(out_file.read_text())

        import wandb

        # Keys renamed from `cem_solve_*`: the callback now brackets per DECISION, not per
        # solve (a solve plans every alive episode), so the number is ~1/n_envs of the old
        # one. A rename keeps the shared W&B project from silently plotting the two together.
        wandb.log(
            {
                "success_rate": success_rate,
                "per_cycle_latency_median_ms": latency["median_ms"],
                "per_cycle_n": latency["n_cycles"],
            }
        )
        print(
            f"[eval:{experiment}] success_rate={success_rate} "
            f"per_cycle_latency_median_ms={latency['median_ms']:.2f} "
            f"(n_cycles={latency['n_cycles']})"
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
