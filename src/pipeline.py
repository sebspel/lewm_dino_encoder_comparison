"""End-to-end study driver — owned PLUMBING (fails LOUDLY).

Carries both tracks from engines to rendered artifacts in one command: archive → export → SR eval
→ component isolation → per-component benchmark → stats → report → clock-normalized render →
committed figures. Each stage is exactly the per-stage driver's own documented command, so this
module sequences the study rather than reimplementing any part of it.

    uv run python -m src.pipeline                      # both tracks, every default stage
    uv run python -m src.pipeline tracks=lewm          # one track
    uv run python -m src.pipeline stages=stats,report  # resume from a stage
    uv run python -m src.pipeline diagnostics=true     # add the gates + sanity checks
    uv run python -m src.pipeline dry_run=true         # print the stage plan, run nothing
    uv run python -m src.pipeline out=/some/dir        # override the artifact dir

**Every stage runs as a subprocess.** Process isolation is what keeps CUDA contexts, TensorRT
engine arenas and Hydra's global state from leaking between stages — `src.sr_eval` already frees
engine contexts explicitly between precisions because they accumulate, and the benchmark and export
stages allocate on the same scale. It also makes each stage's stdout byte-identical to running the
command by hand, which is how the artifacts were produced before this driver existed.

**Where things land.** `out` defaults to `study.default_out_dir()` —
`$STABLEWM_HOME/reports/phase5/`, the persistent network volume (repo-local `reports/phase5`
fallback off-pod). It is resolved ONCE and passed explicitly to every child, so an override
propagates instead of each stage re-deriving its own. Engine plans keep their own home,
`export.engine_root()` = `$STABLEWM_HOME/engines/<track>/`.

**Not included:** the Phase-2 trainings and the Phase-3 torch baseline (`src.eval`). Neither runs
through an engine, so neither belongs to a pipeline whose subject is the exported model.

Fail-fast: a non-zero exit aborts and names both the failing command and the `stages=` list to
resume from. Every step's command, exit code and duration is appended to
`<out>/pipeline_manifest.json` as it completes, so an interrupted run still records how far it got.
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.interfaces import (
    CALIBRATION_METHODS,
    DEFAULT_CALIBRATION_METHOD,
    QUANTIZED_PRECISIONS,
    ExportConfig,
)

_TRACKS = ("lewm", "dino")

# The method the per-component benchmark is run under. Component latency is
# calibration-method-INVARIANT (SPEC §Parity), so this is provenance — which built engines were
# timed — not a second measurement.
_BENCHMARK_METHOD = "entropy"

# The committed display copies (SPEC §Headline-artifact durability — the one exception to the
# never-in-git rule for artifacts). Source path relative to `out_dir`; they are re-copied from a
# render, never hand-edited.
_DISPLAY_FIGS = (
    "speed_vs_sr.png",
    "speed_vs_sr.titled.png",
    "per_cycle_ratio.png",
    "component_breakdown_fp32.png",
    "gpu_logs/sr_eval_clock_diag.png",
)

# What the archive stage preserves before the run supersedes it (CLAUDE §8 — log before you
# delete). Globs are non-recursive against `out_dir`, so the archive subtree never archives itself.
_ARCHIVE_GLOBS = (
    "results.*.json",
    "latencies.*.json",
    "sr.json",
    "stats.json",
    "derived_clocks.json",
    "*.txt",
    "*.png",
    "gpu_logs/*.dmon.log",
    "gpu_logs/*.png",
)

# Stages that only run with `diagnostics=true`. The gates among them (fidelity, sr_shim,
# precision_match) are owner sign-off surfaces, not coded pass/fail, so they inform rather than
# block; the rest are sanity checks with no artifact of their own.
_DIAGNOSTIC_STAGES = (
    "pytest",
    "verify_encode",
    "smoke",
    "fidelity",
    "sr_shim",
    "probe_ranges",
    "precision_match",
)

# Execution order. Diagnostics sit at the point their subject exists: the pre-export checks before
# `export`, the engine-drift gate after it and before anything measured off an engine.
_STAGE_ORDER = (
    "pytest",
    "verify_encode",
    "smoke",
    "fidelity",
    "sr_shim",
    "probe_ranges",
    "archive",
    "export",
    "precision_match",
    "sr_eval",
    "isolation",
    "benchmark",
    "stats",
    "report",
    "clock_norm",
    "figs",
)


@dataclass(frozen=True)
class Step:
    """One unit of work. `argv` runs as a subprocess; an `fn` instead runs in-process (the archive
    and figure copies are file operations, not drivers, so spawning a python for them would only
    obscure the failure)."""

    stage: str
    label: str
    argv: tuple[str, ...] = ()
    fn: Callable[[], None] | None = field(default=None, compare=False)


def _module(module: str, *args: str) -> tuple[str, ...]:
    """A `python -m <module> …` argv on the interpreter running this pipeline — i.e. the uv
    environment the caller invoked it in, so the child sees the same pinned stack."""
    return (sys.executable, "-m", module, *args)


# --- in-process stages -------------------------------------------------------------------


def archive(out_dir: Path) -> Path | None:
    """Copy every artifact the run is about to supersede into `<out>/archive/<UTC date>/`, verify
    each copy byte-for-byte, and record the pre-run hashes (CLAUDE §8). `src.study` merges into
    `results.<track>.json` and `gpu_clocks.log_gpu` opens each telemetry log with `"w"`, so those
    files are overwritten in place; the renders overwrite their `.txt`/`.png`. Returns the archive
    dir, or None when there is nothing to preserve (a first run)."""
    out_dir = Path(out_dir)
    sources = sorted(
        {p for pattern in _ARCHIVE_GLOBS for p in out_dir.glob(pattern) if p.is_file()}
    )
    if not sources:
        print(f"[pipeline:archive] nothing to archive under {out_dir} — first run")
        return None

    dest = out_dir / "archive" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest.mkdir(parents=True, exist_ok=True)
    digests: list[str] = []
    total = 0
    for src in sources:
        rel = src.relative_to(out_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        # Verified, not assumed: an archive that silently truncated would defeat the whole point of
        # taking one before an irreversible overwrite.
        if not filecmp.cmp(src, target, shallow=False):
            raise SystemExit(f"[pipeline:archive] copy differs from source: {src}")
        digests.append(f"{hashlib.sha256(src.read_bytes()).hexdigest()}  {rel}")
        total += src.stat().st_size
    (dest / "PRE_RUN_SHA256.txt").write_text("\n".join(digests) + "\n")
    print(
        f"[pipeline:archive] {len(sources)} files ({total / 1e6:.1f} MB) -> {dest} "
        f"(all cmp-verified; PRE_RUN_SHA256.txt written)"
    )
    return dest


def refresh_figs(out_dir: Path, repo_root: Path) -> list[Path]:
    """Re-copy the committed display figures from the render into `reports/figs/`. Display-only
    view of the canonical artifacts, never hand-edited (SPEC §Headline-artifact durability).

    A source that a legitimately partial run never produced (the cross-track ratio plot on a
    single-track render) is skipped with a notice rather than failing — but producing NONE of them
    means the render did not happen, which is a real failure."""
    figs = Path(repo_root) / "reports" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for rel in _DISPLAY_FIGS:
        src = Path(out_dir) / rel
        if not src.exists():
            print(f"[pipeline:figs] absent, skipped: {src}")
            continue
        target = figs / Path(rel).name
        shutil.copy2(src, target)
        if not filecmp.cmp(src, target, shallow=False):
            raise SystemExit(f"[pipeline:figs] copy differs from source: {src}")
        copied.append(target)
    if not copied:
        raise SystemExit(
            f"[pipeline:figs] none of {list(_DISPLAY_FIGS)} exist under {out_dir} — "
            "run the `report` and `clock_norm` stages first"
        )
    print(f"[pipeline:figs] refreshed {len(copied)} display copies -> {figs}")
    return copied


# --- the stage plan ----------------------------------------------------------------------


def build_steps(
    tracks: tuple[str, ...] = _TRACKS,
    out_dir: Path | None = None,
    diagnostics: bool = False,
    stages: tuple[str, ...] | None = None,
    repo_root: Path | None = None,
) -> list[Step]:
    """The ordered step list. Pure — it resolves paths and composes argv but runs nothing, so the
    plan is inspectable (`dry_run=true`) and unit-testable off-pod."""
    from src.study import default_out_dir

    out_dir = Path(out_dir) if out_dir is not None else default_out_dir()
    repo_root = Path(repo_root) if repo_root is not None else _repo_root()
    out = str(out_dir)
    sr_json = str(out_dir / "sr.json")

    precisions = ExportConfig().precisions
    unquantized = tuple(p for p in precisions if p not in QUANTIZED_PRECISIONS)

    selected = set(stages) if stages is not None else None

    def wanted(stage: str) -> bool:
        if selected is not None:
            return stage in selected
        return diagnostics or stage not in _DIAGNOSTIC_STAGES

    steps: list[Step] = []

    def add(stage: str, argv: tuple[str, ...]) -> None:
        steps.append(Step(stage, " ".join(argv[1:]), argv=argv))

    for stage in _STAGE_ORDER:
        if not wanted(stage):
            continue

        if stage == "pytest":
            add(stage, _module("pytest", "-q"))

        elif stage == "verify_encode":
            add(stage, _module("scripts.verify_encode"))

        elif stage == "smoke":
            add(stage, _module("src.smoke"))

        elif stage == "fidelity":
            # The DINO adapter-fidelity gate and the LeWM per-frame action-encoder guard.
            add(stage, _module("src.fidelity"))
            add(stage, _module("src.fidelity", "--lewm"))

        elif stage == "sr_shim":
            add(stage, _module("src.sr_shim", "track=both"))

        elif stage == "probe_ranges":
            for track in tracks:
                add(stage, _module("src.probe_ranges", f"track={track}"))

        elif stage == "archive":
            steps.append(
                Step(
                    stage,
                    f"archive {out} -> {out_dir / 'archive'}/<date>",
                    fn=lambda o=out_dir: archive(o),
                )
            )

        elif stage == "export":
            # FP32/FP16 build data-free and are method-invariant, so they are built once; the
            # quantized precisions are built once PER METHOD and land on method-tagged plans, so
            # both methods' engines coexist (`export.engine_filename`).
            for track in tracks:
                for precision in unquantized:
                    add(stage, _module("src.export", f"model={track}", f"precision={precision}"))
                for method in CALIBRATION_METHODS:
                    for precision in QUANTIZED_PRECISIONS:
                        add(
                            stage,
                            _module(
                                "src.export",
                                f"model={track}",
                                f"precision={precision}",
                                f"calibration_method={method}",
                            ),
                        )

        elif stage == "precision_match":
            # Engine-vs-PyTorch drift, per method (the quantized scales differ, so the tables do).
            # No coded pass/fail — the gate is the owner's sign-off on the printed table.
            for track in tracks:
                for method in CALIBRATION_METHODS:
                    add(
                        stage,
                        _module(
                            "src.precision_match",
                            f"track={track}",
                            f"calibration_method={method}",
                        ),
                    )

        elif stage == "sr_eval":
            # The default method's run covers every precision; a further method only re-evaluates
            # the QUANTIZED ones, because FP32/FP16 carry no scales and their SR cannot depend on a
            # PTQ method (`report._select_method` joins them across labels).
            for track in tracks:
                for method in CALIBRATION_METHODS:
                    covered = (
                        precisions if method == DEFAULT_CALIBRATION_METHOD else QUANTIZED_PRECISIONS
                    )
                    add(
                        stage,
                        _module(
                            "src.sr_eval",
                            "--config-dir",
                            "conf",
                            f"+experiment=eval_{track}",
                            f"precision={','.join(covered)}",
                            f"calibration_method={method}",
                            f"out={out}",
                        ),
                    )

        elif stage == "isolation":
            # Component-precision isolation: ONE component quantized, the other held at FP16, two
            # runs per (track, quantized precision, calibration method). Recorded under composite
            # `enc-<A>+pred-<B>` keys that cannot collide with a pure precision, so the headline is
            # unchanged by construction (docs/architecture.md §9).
            # BOTH methods, unlike the benchmark above: the composite keys carry their method and
            # never fall back across methods (`report._select_method`), so a row only explains a
            # headline row rendered at the SAME method — and the headline renders at either. One
            # method per pass keeps each 2x2 inside a single labelled comparison.
            for track in tracks:
                for method in CALIBRATION_METHODS:
                    for precision in QUANTIZED_PRECISIONS:
                        for enc, pred in ((precision, "fp16"), ("fp16", precision)):
                            add(
                                stage,
                                _module(
                                    "src.sr_eval",
                                    "--config-dir",
                                    "conf",
                                    f"+experiment=eval_{track}",
                                    f"encoder_precision={enc}",
                                    f"predictor_precision={pred}",
                                    f"calibration_method={method}",
                                    f"out={out}",
                                ),
                            )

        elif stage == "benchmark":
            # One pass per track, in its own process: DINO's engine contexts alone reserve ~11 GB,
            # so the two tracks' benchmarks do not share an address space.
            for track in tracks:
                add(
                    stage,
                    _module(
                        "src.study",
                        f"track={track}",
                        f"calibration_method={_BENCHMARK_METHOD}",
                        f"out={out}",
                        f"sr={sr_json}",
                    ),
                )

        elif stage == "stats":
            add(stage, _module("src.stats", f"from={out}", f"out={out}"))

        elif stage == "report":
            # The authoritative render: both tracks together (so the cross-track ratio plots exist)
            # and once per method, since the headline tables are method-scoped by filename.
            for method in CALIBRATION_METHODS:
                add(
                    stage,
                    _module(
                        "src.report",
                        f"from={out}",
                        f"out={out}",
                        f"sr={sr_json}",
                        f"calibration_method={method}",
                    ),
                )

        elif stage == "clock_norm":
            for method in CALIBRATION_METHODS:
                add(
                    stage,
                    _module(
                        "src.clock_norm",
                        f"from={out}",
                        f"out={out}",
                        f"calibration_method={method}",
                    ),
                )

        elif stage == "figs":
            steps.append(
                Step(
                    stage,
                    f"refresh {repo_root / 'reports' / 'figs'} from {out}",
                    fn=lambda o=out_dir, r=repo_root: refresh_figs(o, r),
                )
            )

    return steps


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# --- execution ---------------------------------------------------------------------------


def run(steps: list[Step], out_dir: Path, repo_root: Path | None = None) -> Path:
    """Execute the steps in order, appending each outcome to `<out>/pipeline_manifest.json` as it
    completes. Aborts on the first failure, naming the stages left to resume from."""
    repo_root = Path(repo_root) if repo_root is not None else _repo_root()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "pipeline_manifest.json"
    records: list[dict] = []

    for i, step in enumerate(steps):
        started = datetime.now(timezone.utc)
        print(f"\n[pipeline] ({i + 1}/{len(steps)}) {step.stage}: {step.label}", flush=True)
        if step.fn is not None:
            step.fn()
            code = 0
        else:
            code = subprocess.run(step.argv, cwd=repo_root).returncode
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        records.append(
            {
                "stage": step.stage,
                "command": step.label,
                "returncode": code,
                "started": started.isoformat(timespec="seconds"),
                "duration_s": round(elapsed, 1),
            }
        )
        # Written after every step, not at the end: an aborted run must still record how far it got.
        manifest_path.write_text(json.dumps({"steps": records}, indent=2) + "\n")
        if code != 0:
            remaining = sorted({s.stage for s in steps[i:]}, key=_STAGE_ORDER.index)
            raise SystemExit(
                f"[pipeline] FAILED in stage {step.stage!r} (exit {code}):\n"
                f"    {step.label}\n"
                f"[pipeline] fix it, then resume with:\n"
                f"    uv run python -m src.pipeline stages={','.join(remaining)}"
            )
        print(f"[pipeline] {step.stage} ok ({elapsed:.1f}s)", flush=True)

    print(f"\n[pipeline] {len(steps)} steps complete -> {manifest_path}")
    return manifest_path


def main() -> None:
    tracks = _TRACKS
    out_dir = None
    diagnostics = False
    dry_run = False
    stages: tuple[str, ...] | None = None
    for a in sys.argv[1:]:
        if a.startswith("tracks="):
            tracks = tuple(a.split("=", 1)[1].split(","))
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
        elif a.startswith("stages="):
            stages = tuple(a.split("=", 1)[1].split(","))
        elif a.startswith("diagnostics="):
            diagnostics = a.split("=", 1)[1].lower() in ("1", "true", "yes")
        elif a.startswith("dry_run="):
            dry_run = a.split("=", 1)[1].lower() in ("1", "true", "yes")
        else:
            raise SystemExit(f"[pipeline] unknown argument {a!r}")

    unknown = set(tracks) - set(_TRACKS)
    if unknown:
        raise SystemExit(f"[pipeline] unknown track(s) {sorted(unknown)}; expected {list(_TRACKS)}")
    if stages is not None:
        unknown_stages = set(stages) - set(_STAGE_ORDER)
        if unknown_stages:
            raise SystemExit(
                f"[pipeline] unknown stage(s) {sorted(unknown_stages)}; "
                f"expected any of {list(_STAGE_ORDER)}"
            )

    from src.study import default_out_dir

    out_dir = out_dir if out_dir is not None else default_out_dir()
    steps = build_steps(tracks, out_dir, diagnostics, stages)
    if not steps:
        raise SystemExit("[pipeline] no stages selected")

    if dry_run:
        print(f"[pipeline] {len(steps)} steps, tracks={list(tracks)}, out={out_dir}")
        current = None
        for step in steps:
            if step.stage != current:
                current = step.stage
                print(f"\n  {current}:")
            print(f"    {step.label}")
        return

    run(steps, out_dir)


if __name__ == "__main__":
    main()
