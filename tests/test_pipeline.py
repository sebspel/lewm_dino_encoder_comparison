"""Pipeline stage-plan contract (off-pod — `build_steps` is pure, nothing is executed).

The plan IS the thing under test: every stage is a documented per-stage command, so a drifted
argument here would run the wrong study on the L40S for hours before anyone noticed. These pin the
combinations, the ordering, the selection flags, and — the point of resolving it once — that every
child carries the same artifact dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import pipeline
from src.interfaces import CALIBRATION_METHODS, QUANTIZED_PRECISIONS, ExportConfig

OUT = Path("/vol/reports/phase5")


def _labels(stage: str, **kwargs) -> list[str]:
    steps = pipeline.build_steps(out_dir=OUT, **kwargs)
    return [s.label for s in steps if s.stage == stage]


def _stages(**kwargs) -> list[str]:
    """The distinct stages in execution order (a stage may contribute several steps)."""
    seen: list[str] = []
    for step in pipeline.build_steps(out_dir=OUT, **kwargs):
        if step.stage not in seen:
            seen.append(step.stage)
    return seen


def test_export_covers_every_track_precision_method_combination():
    labels = _labels("export")
    precisions = ExportConfig().precisions
    unquantized = [p for p in precisions if p not in QUANTIZED_PRECISIONS]
    # FP32/FP16 build data-free and are method-invariant -> once per track, untagged. The
    # quantized precisions are built per method so both methods' engines coexist.
    expected = len(pipeline._TRACKS) * (
        len(unquantized) + len(CALIBRATION_METHODS) * len(QUANTIZED_PRECISIONS)
    )
    assert len(labels) == expected == 12
    for precision in unquantized:
        assert f"-m src.export model=lewm precision={precision}" in labels
        assert not any(
            f"precision={precision} calibration_method" in x for x in labels
        ), "an unquantized precision must not be built per method"
    assert (
        "-m src.export model=dino precision=int8 calibration_method=entropy" in labels
    )


def test_sr_eval_reruns_only_the_quantized_precisions_for_a_second_method():
    labels = _labels("sr_eval")
    assert len(labels) == 2 * len(pipeline._TRACKS)
    # FP32/FP16 carry no scales, so their SR cannot depend on the PTQ method — evaluating them
    # again under a second label would book the same solves twice.
    assert any("precision=fp32,fp16,int8,fp8 calibration_method=max" in x for x in labels)
    assert any("precision=int8,fp8 calibration_method=entropy" in x for x in labels)
    assert all("+experiment=eval_" in x and "--config-dir conf" in x for x in labels)


def test_isolation_holds_one_component_at_fp16_per_run():
    labels = _labels("isolation")
    per_method = 2 * len(QUANTIZED_PRECISIONS) * len(pipeline._TRACKS)
    assert len(labels) == per_method * len(CALIBRATION_METHODS) == 16
    # Both methods, in full: a composite key never falls back across methods, so an isolation row
    # only explains a headline row rendered at the SAME method (docs/architecture.md §9).
    for method in CALIBRATION_METHODS:
        assert sum(f"calibration_method={method}" in x for x in labels) == per_method
    for label in labels:
        # Exactly one side quantized, the other held at the known-good FP16 baseline.
        assert ("encoder_precision=fp16" in label) != ("predictor_precision=fp16" in label)
    assert "precision=" not in " ".join(
        l.replace("encoder_precision=", "").replace("predictor_precision=", "")
        for l in labels
    ), "mixed mode cannot be combined with precision= (src.sr_eval rejects it)"


def test_benchmark_runs_one_process_per_track():
    labels = _labels("benchmark")
    assert len(labels) == len(pipeline._TRACKS)
    assert all(f"calibration_method={pipeline._BENCHMARK_METHOD}" in x for x in labels)


def test_report_and_clock_norm_render_once_per_method():
    assert len(_labels("report")) == len(CALIBRATION_METHODS)
    assert len(_labels("clock_norm")) == len(CALIBRATION_METHODS)
    # Both tracks in one render, so the cross-track ratio plots exist.
    assert all("track=" not in x for x in _labels("report"))


def test_every_child_carries_the_resolved_out_dir():
    # The whole reason the default is resolved once: a stage that re-derived it would silently
    # write somewhere else under an `out=` override.
    for step in pipeline.build_steps(out_dir=OUT):
        if not step.argv or step.stage == "export":
            continue  # export writes engines to engine_root(), not the artifact dir
        assert str(OUT) in step.label, step.label


def test_out_dir_defaults_to_the_shared_network_storage_default():
    from src.study import default_out_dir

    steps = pipeline.build_steps()
    stats = next(s for s in steps if s.stage == "stats")
    assert f"from={default_out_dir()}" in stats.label


def test_diagnostics_are_off_by_default_and_ordered_before_their_subject():
    default = _stages()
    assert not set(default) & set(pipeline._DIAGNOSTIC_STAGES)

    with_diag = _stages(diagnostics=True)
    assert set(pipeline._DIAGNOSTIC_STAGES) <= set(with_diag)
    # The pre-export checks precede the build; the engine-drift gate follows it and precedes
    # anything measured off an engine.
    assert with_diag.index("smoke") < with_diag.index("export")
    assert with_diag.index("export") < with_diag.index("precision_match")
    assert with_diag.index("precision_match") < with_diag.index("sr_eval")


def test_stages_selection_resumes_in_canonical_order():
    # Selection order must not leak into execution order.
    assert _stages(stages=("figs", "stats", "report")) == ["stats", "report", "figs"]
    # An explicitly selected diagnostic runs without the global flag.
    assert _stages(stages=("precision_match",), diagnostics=False)


def test_single_track_halves_the_per_track_stages():
    both = pipeline.build_steps(out_dir=OUT)
    one = pipeline.build_steps(tracks=("lewm",), out_dir=OUT)
    for stage in ("export", "sr_eval", "isolation", "benchmark"):
        assert len([s for s in one if s.stage == stage]) * 2 == len(
            [s for s in both if s.stage == stage]
        )
    # No dropped track leaks into a command. Checked on the module arguments only: the repo path
    # (hence the interpreter path and every label) contains a track name of its own.
    assert all("dino" not in " ".join(s.argv[3:]) for s in one)


def test_archive_and_figs_run_in_process():
    steps = {s.stage: s for s in pipeline.build_steps(out_dir=OUT)}
    for stage in ("archive", "figs"):
        assert steps[stage].fn is not None and not steps[stage].argv


def test_stage_order_is_the_single_source_of_truth():
    ordered = _stages(diagnostics=True)
    assert ordered == sorted(set(ordered), key=pipeline._STAGE_ORDER.index)


def test_archive_copies_verifies_and_hashes(tmp_path):
    out = tmp_path / "phase5"
    (out / "gpu_logs").mkdir(parents=True)
    (out / "sr.json").write_text('{"lewm": {}}')
    (out / "results.lewm.json").write_text("{}")
    (out / "speed_table.entropy.txt").write_text("table\n")
    (out / "gpu_logs" / "lewm.fp32.max.benchmark.dmon.log").write_text("dmon\n")

    dest = pipeline.archive(out)

    assert dest is not None and dest.parent == out / "archive"
    assert (dest / "sr.json").read_text() == '{"lewm": {}}'
    # Nested layout preserved, so a restored archive drops straight back in place.
    assert (dest / "gpu_logs" / "lewm.fp32.max.benchmark.dmon.log").exists()
    digests = (dest / "PRE_RUN_SHA256.txt").read_text().splitlines()
    assert len(digests) == 4
    assert any(line.endswith("sr.json") for line in digests)


def test_archive_does_not_archive_itself(tmp_path):
    out = tmp_path / "phase5"
    out.mkdir(parents=True)
    (out / "sr.json").write_text("{}")
    first = pipeline.archive(out)
    assert first is not None
    second = pipeline.archive(out)
    # A prior archive must not be swept into the next one (the globs are non-recursive).
    assert second is not None
    assert not (second / "archive").exists()
    assert len((second / "PRE_RUN_SHA256.txt").read_text().splitlines()) == 1


def test_archive_is_a_noop_on_a_first_run(tmp_path):
    assert pipeline.archive(tmp_path / "nothing-here") is None


def test_figs_refresh_skips_absent_plots_but_fails_when_none_exist(tmp_path):
    out, repo = tmp_path / "phase5", tmp_path / "repo"
    (out / "gpu_logs").mkdir(parents=True)
    with pytest.raises(SystemExit):
        pipeline.refresh_figs(out, repo)

    (out / "speed_vs_sr.png").write_bytes(b"png")
    (out / "gpu_logs" / "sr_eval_clock_diag.png").write_bytes(b"diag")
    copied = pipeline.refresh_figs(out, repo)
    # A single-track render has no cross-track ratio plot; that is a partial run, not a failure.
    assert {p.name for p in copied} == {"speed_vs_sr.png", "sr_eval_clock_diag.png"}
    assert (repo / "reports" / "figs" / "speed_vs_sr.png").read_bytes() == b"png"
