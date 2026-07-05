"""Owned thin Phase-3 eval driver.

Runs the byte-unmodified vendored eval entrypoint (``scripts.plan.eval_wm.run``) under its
own Hydra composition, then logs the Push-T success rate and the CEM-solve latency median
to one W&B run (owned helper, shared project).

    uv run python -m src.eval --config-dir conf +experiment=eval_<lewm|dino>

No monkeypatch, no class shadow:
  * CEM-solve latency rides in via ``cfg.solver.callbacks`` (the eval overlay injects the
    owned :class:`~src.eval_latency.SolveLatencyRecorder`); the driver reads the records
    back through that module's registry.
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

from scripts.plan import eval_wm
from src import eval_latency, wandb_log

# success_rate in the metrics dict repr eval_wm writes: `'success_rate': 42.0` or, under
# newer numpy, `'success_rate': np.float64(42.0)`. Capture the numeric literal either way.
_SR_RE = re.compile(
    r"success_rate['\"]?\s*[:=]\s*(?:np\.\w+\(\s*)?([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


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


def main():
    argv = sys.argv[1:]
    experiment = _experiment_from_argv(argv)

    # Driver-owned results file so SR is read from a fresh file, not the appended default.
    out_file = Path(tempfile.mkdtemp(prefix="swm_eval_")) / "results.txt"
    sys.argv.append(f"output.filename={out_file}")

    eval_latency.reset_registry()
    run = wandb_log.init(
        experiment, name=experiment, config={"phase": "eval", "experiment": experiment}
    )
    try:
        # Byte-unmodified entrypoint: Hydra composes from argv (incl. the callback
        # injection from the overlay), instantiates the recorder, and runs World.evaluate.
        eval_wm.run()

        latency = eval_latency.pop_records()
        if latency["n_solves"] == 0:
            raise RuntimeError(
                "the latency callback recorded no CEM solves — is it injected via "
                "cfg.solver.callbacks in the eval overlay?"
            )
        success_rate = _parse_success_rate(out_file.read_text())

        import wandb

        wandb.log(
            {
                "success_rate": success_rate,
                "cem_solve_latency_median_ms": latency["median_ms"],
                "cem_solve_n": latency["n_solves"],
            }
        )
        print(
            f"[eval:{experiment}] success_rate={success_rate} "
            f"cem_solve_latency_median_ms={latency['median_ms']:.2f} "
            f"(n_solves={latency['n_solves']})"
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
