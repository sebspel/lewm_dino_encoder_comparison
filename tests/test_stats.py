"""Phase-8 confidence intervals (`src/stats.py`) — the owned re-analysis layer.

Off-pod, CPU-only: the stored `sr.json` samples stand in for the pod artifacts. What these pin is
mostly the OWNER-SET CONSTRUCTION (SPEC §Implementation Boundaries "confidence-interval
construction", architecture.md §12) — a wrong estimator, rank convention, or p-value convention
produces a plausible wrong INTERVAL, which no runtime check would catch. Four tests are explicit
guards against a future "just use scipy" refactor silently changing a number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import binom, false_discovery_control

from src import report, stats
from src.interfaces import PER_CYCLE_WARMUP_DROP


# --- helpers ---------------------------------------------------------------------------
def _sr_json(**vectors) -> dict:
    """A minimal `sr.json` in the real shape: `{track: {label: {method: point}}}`. `vectors` maps
    `<track>__<label>` -> (success_rate, latency vector)."""
    out: dict = {}
    for key, (sr, vec) in vectors.items():
        track, label = key.split("__", 1)
        out.setdefault(track, {})[label] = {
            "entropy": {"success_rate": sr, "per_cycle_latencies_ms": list(vec)}
        }
    return out


def _bench(vec_lewm, vec_dino) -> dict:
    """A bench pair carrying only what `_join_eval`/`_finalize_per_cycle` need."""
    row = lambda: {  # noqa: E731
        "per_cycle_p50_ms": float("nan"), "per_cycle_p95_ms": float("nan"),
        "per_cycle_mean_ms": float("nan"), "encode_p50_ms": 1.0, "encode_p95_ms": 1.0,
        "encode_mean_ms": 1.0, "predict_p50_ms": 1.0, "predict_p95_ms": 1.0,
        "predict_mean_ms": 1.0, "peak_mem_mb": 1.0, "success_rate": float("nan"),
    }
    return {"lewm": {"fp32": row()}, "dino": {"fp32": row()}}


# --- success rate: Clopper-Pearson -----------------------------------------------------
def test_clopper_pearson_matches_the_exact_binomial():
    """The SR interval must be the EXACT binomial one, including at the degenerate ends the study
    actually hits (`dino fp8@entropy` = 0/50, `lewm fp8@entropy` = 50/50), where a normal
    approximation collapses to zero width and asserts a certainty the sample does not contain."""
    lo, hi = stats.clopper_pearson(38, 50)
    assert (round(lo, 4), round(hi, 4)) == (0.6183, 0.8694)

    lo0, hi0 = stats.clopper_pearson(0, 50)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0, "k=0 must give a one-sided interval, not a point"
    lon, hin = stats.clopper_pearson(50, 50)
    assert hin == 1.0 and 0.0 < lon < 1.0, "k=n must give a one-sided interval, not a point"


def test_successes_from_sr_fails_loudly_when_not_integral():
    """The trial count is the one input no artefact records (it lives in `pusht.yaml`). A
    non-integral reconstruction means the assumed n is wrong for this artefact, which would
    mis-scale every SR interval — fail, never round it away."""
    assert stats.successes_from_sr(76.0, 50) == 38
    assert stats.successes_from_sr(28.000000000000004, 50) == 14  # float noise in the real sr.json
    with pytest.raises(SystemExit):
        stats.successes_from_sr(33.3, 50)


# --- per-cycle p50: exact binomial order-statistic interval ----------------------------
def test_order_statistic_ranks_are_conservative_not_naive():
    """GUARD (architecture.md §12): the obvious `binom.ppf(a/2)` / `binom.isf(a/2)` rank choice
    UNDER-covers — 0.9481 at n=59, below the nominal 0.95 — because `ppf` returns the smallest rank
    whose CDF is at LEAST a/2. Achieved coverage must reach 1-alpha at every n this study has."""
    for n in (55, 56, 58, 59, 61, 63, 65, 97, 100):
        ci = stats.order_statistic_ci(list(range(n)))
        assert ci is not None and ci["coverage"] >= 0.95, f"n={n} under-covers"

    naive_lo, naive_hi = int(binom.ppf(0.025, 59, 0.5)), int(binom.isf(0.025, 59, 0.5))
    naive_cov = binom.cdf(naive_hi - 1, 59, 0.5) - binom.cdf(naive_lo - 1, 59, 0.5)
    assert naive_cov < 0.95, "premise of this guard changed"
    assert stats.order_statistic_ci(list(range(59)))["ranks"] != [naive_lo, naive_hi]


def test_order_statistic_ci_returns_none_when_n_too_small():
    """No rank pair reaches 95% at n=5, so NO interval is emitted — the same 'unmeasured, never
    asserted' discipline `src.clock_norm` applies to undersampled clock runs. Inventing one would
    be the silent wrong-number failure the whole gate exists to prevent."""
    assert stats.order_statistic_ci([1.0, 2.0, 3.0, 4.0, 5.0]) is None
    assert stats.order_statistic_ci([]) is None
    assert stats.order_statistic_ci(list(range(6))) is not None


def test_order_statistic_ci_brackets_the_median_and_uses_sample_values():
    """Endpoints are ORDER STATISTICS — actual observed values — and bracket the sample median."""
    sample = [float(v) for v in range(1, 61)]
    ci = stats.order_statistic_ci(sample)
    assert ci["lo"] in sample and ci["hi"] in sample
    assert ci["lo"] < 30.5 < ci["hi"]


# --- independence premise: the Dwass permutation test ----------------------------------
def test_dwass_pvalue_uses_the_add_one_convention():
    """`p = (1 + #{|r*| >= |r_obs|}) / (1 + B)`. A monotone ramp maximizes lag-1 autocorrelation
    over permutations, so no permutation beats it and the count is 0 — the p-value must then be
    exactly 1/(B+1), never 0. That add-one is what keeps a Monte-Carlo test exact."""
    b = 2000
    out = stats.dwass_permutation_test(np.arange(60.0), n_resamples=b, seed=0)
    assert out["lag1_p_permutation"] == pytest.approx(1.0 / (b + 1), rel=1e-12)
    assert out["lag1_p_permutation"] > 0.0


def test_dwass_two_sided_is_absolute_value_not_twice_the_smaller():
    """GUARD: `scipy.stats.permutation_test(alternative="two-sided")` is documented as *twice the
    smaller of the one-sided p-values* — a DIFFERENT convention from Dwass's `|r*| >= |r_obs|`.
    Swapping to it would silently change every reported p-value, so the absolute-value tail is
    pinned here against an independently generated null (different seed, own permutations)."""
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=80))  # strongly autocorrelated
    out = stats.dwass_permutation_test(x, n_resamples=5000, seed=0)

    c = x - x.mean()
    denom = float(c @ c)
    null = np.array([
        (p[:-1] * p[1:]).sum() / denom
        for p in np.random.default_rng(99).permuted(np.broadcast_to(c, (5000, x.size)), axis=1)
    ])
    p_abs = (1 + np.count_nonzero(np.abs(null) >= abs(out["lag1_autocorr"]))) / 5001
    assert out["lag1_p_permutation"] == pytest.approx(p_abs, abs=0.01)


def test_dwass_detects_serial_correlation_and_clears_iid():
    """The test is the PREMISE CHECK on the order-statistic interval (architecture.md §12): serial
    correlation makes that interval too narrow. It must fire on correlated input and stay quiet on
    i.i.d. input, or it licenses nothing."""
    rng = np.random.default_rng(3)
    correlated = stats.dwass_permutation_test(np.cumsum(rng.normal(size=80)), n_resamples=2000)
    iid = stats.dwass_permutation_test(rng.normal(size=80), n_resamples=2000)
    assert correlated["lag1_reject"] is True
    assert iid["lag1_reject"] is False


def test_dwass_records_the_pre_permutation_pvalue_and_the_null_summary():
    """SPEC §Interface Contracts: the artefact carries the observed lag-1 statistic, the
    pre-permutation asymptotic p-value AND the permutation p-value, plus the permutation null's own
    lag-1 summary — so an interval can be audited off the artefact rather than trusted."""
    out = stats.dwass_permutation_test(np.arange(60.0), n_resamples=2000, seed=0)
    for key in ("lag1_autocorr", "lag1_p_asymptotic", "lag1_p_permutation",
                "lag1_null_mean", "lag1_null_sd", "n_resamples", "seed"):
        assert key in out, key
    assert out["seed"] == 0 and out["n_resamples"] == 2000
    # The permutation null is centred near the -1/(n-1) white-noise value, not near the observed.
    assert abs(out["lag1_null_mean"]) < 0.1 < out["lag1_autocorr"]


def test_permutation_test_is_reproducible_at_a_fixed_seed():
    """A Monte-Carlo p-value that moves between renders is not an artefact anyone can audit."""
    x = np.random.default_rng(5).normal(size=70)
    a = stats.dwass_permutation_test(x, n_resamples=2000, seed=0)
    b = stats.dwass_permutation_test(x, n_resamples=2000, seed=0)
    assert a["lag1_p_permutation"] == b["lag1_p_permutation"]


# --- multiplicity: Holm, secondary only ------------------------------------------------
def test_holm_is_step_down_fwer_not_fdr():
    """GUARD: `scipy.stats.false_discovery_control` is Benjamini-Hochberg — FDR, a different
    quantity — and must not be substituted for Holm. Hand-worked m=3 example."""
    adj = stats.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert adj == {"a": pytest.approx(0.03), "b": pytest.approx(0.04), "c": pytest.approx(0.04)}
    bh = false_discovery_control([0.01, 0.02, 0.04])
    assert adj["b"] != pytest.approx(bh[1]), "Holm must not coincide with BH here"


def test_holm_is_monotone_and_capped():
    adj = stats.holm({"a": 0.5, "b": 0.001, "c": 0.4, "d": 0.9})
    ordered = [adj[k] for k in sorted(adj, key=lambda k: {"a": 0.5, "b": 0.001, "c": 0.4, "d": 0.9}[k])]
    assert ordered == sorted(ordered), "adjusted p-values must be non-decreasing in raw order"
    assert all(v <= 1.0 for v in adj.values())


def test_holm_is_secondary_and_drives_no_decision():
    """SPEC §Interface Contracts: the test decision is the UNADJUSTED p-value; Holm is persisted as
    secondary reporting only. Pinned on the real-shaped case where the two disagree — a point that
    rejects raw but not after Holm must still carry `lag1_reject=True`."""
    rng = np.random.default_rng(11)
    sr = _sr_json(
        lewm__fp32=(90.0, np.cumsum(rng.normal(size=40))),
        **{f"lewm__f{i}": (90.0, rng.normal(size=40)) for i in range(12)},
    )
    payload = stats.compute(sr, n_resamples=2000, n_episodes=50)
    entries = [e for lab in payload["points"]["lewm"].values() for e in lab.values()]
    assert all("lag1_p_holm" in e and "lag1_reject" in e for e in entries)
    disagree = [e for e in entries if e["lag1_reject"] and not e["lag1_reject_holm"]]
    assert disagree, "fixture no longer exercises the raw-vs-Holm disagreement"
    assert payload["meta"]["decision_pvalue"].startswith("lag1_p_permutation")


# --- the payload ------------------------------------------------------------------------
def test_ci_sample_is_byte_identical_to_the_p50_sample():
    """The interval must bracket the SAME sample the reported p50 comes from. Both sides go through
    `report.per_cycle_samples`, so this pins the shared rule end-to-end: warm-up drop, then equal-n
    truncation to the common min across tracks."""
    lewm = [500.0] + [100.0 + i for i in range(59)]  # cold head + 59
    dino = [900.0] + [200.0 + i for i in range(70)]  # cold head + 70 -> common n = 59
    bench = _bench(lewm, dino)
    overrides = {
        t: {"fp32": {"entropy": {"success_rate": 90.0, "per_cycle_latencies_ms": v}}}
        for t, v in (("lewm", lewm), ("dino", dino))
    }
    report._join_eval(bench, overrides, "entropy")
    report._finalize_per_cycle(bench, PER_CYCLE_WARMUP_DROP)

    payload = stats.compute(overrides, n_resamples=200)
    for track in ("lewm", "dino"):
        point = payload["points"][track]["fp32"]["entropy"]
        assert point["per_cycle_n"] == bench[track]["fp32"]["_per_cycle_n"] == 59
        assert point["per_cycle_p50_ms"] == bench[track]["fp32"]["per_cycle_p50_ms"]


def test_stats_covers_every_point_including_the_isolation_composites():
    """`stats.json` is method-UNSCOPED and label-complete: the composite `enc-<A>+pred-<B>` points
    report an absolute SR in the isolation table, so they get an interval too — even though they
    deliberately never enter `bench` (architecture.md §9)."""
    vec = list(np.random.default_rng(1).normal(100, 5, size=40))
    point = {"entropy": {"success_rate": 90.0, "per_cycle_latencies_ms": vec}}
    sr = {"lewm": {"fp32": point, "int8": point, "enc-int8+pred-fp16": point}}
    payload = stats.compute(sr, n_resamples=200)
    labels = set(payload["points"]["lewm"])
    assert "enc-int8+pred-fp16" in labels
    for label in labels:
        e = payload["points"]["lewm"][label]["entropy"]
        assert e["sr_ci95_pct"] and e["p50_ci95_ms"]
        assert e["trials"] == 50 and e["per_cycle_n"] == 39


def test_every_point_records_the_n_its_interval_was_computed_over():
    """SPEC §Interface Contracts: an interval that does not state its n cannot be audited."""
    vec = list(np.random.default_rng(2).normal(100, 5, size=40))
    payload = stats.compute(_sr_json(lewm__fp32=(90.0, vec)), n_resamples=200)
    e = payload["points"]["lewm"]["fp32"]["entropy"]
    assert e["trials"] == 50          # episodes, for the SR interval
    assert e["per_cycle_n"] == 39     # cycles, for the p50 interval (40 raw - 1 warm-up)
    assert e["per_cycle_raw_n"] == 40


def test_stats_json_round_trips_and_records_its_construction(tmp_path):
    """Every construction choice a reader would need to re-derive the interval is in `meta` —
    estimators, alpha, episode count, permutation count, the two-sided convention, and the seed."""
    vec = list(np.random.default_rng(4).normal(100, 5, size=40))
    payload = stats.compute(_sr_json(lewm__fp32=(90.0, vec)), n_resamples=200)
    path = stats.write_stats_json(payload, tmp_path)
    loaded = json.loads(path.read_text())

    assert path.name == "stats.json"
    assert loaded["points"] == payload["points"]
    meta = loaded["meta"]
    assert meta["sr_estimator"] == "clopper-pearson-exact"
    assert meta["p50_estimator"] == "exact-binomial-order-statistic"
    assert meta["independence_test"] == "dwass-monte-carlo-permutation"
    assert meta["test_statistic"] == "lag1-autocorrelation"
    assert meta["student_t_adjustment"] is False
    assert "NOT scipy" in meta["two_sided_convention"]
    assert meta["alpha"] == 0.05 and meta["n_episodes"] == 50 and meta["seed"] == 0
    assert "secondary" in meta["holm"]
    assert "ratios" in meta["no_interval_on"]


def test_method_invariant_intervals_join_across_methods(tmp_path):
    """The same label trap `test_method_invariant_precisions_join_across_methods` pins for SR, now
    for the INTERVALS: `src.sr_eval` stamps every precision with the run's method, so fp32/fp16
    points sit under `max`. Without the fallback an `entropy` render shows those rows an SR and a
    p50 but an empty interval — a hole created purely by a label."""
    vec = list(np.random.default_rng(8).normal(100, 5, size=40))
    overrides = {
        "lewm": {
            "fp32": {"max": {"success_rate": 98.0, "per_cycle_latencies_ms": vec}},
            "int8": {"entropy": {"success_rate": 76.0, "per_cycle_latencies_ms": vec}},
        }
    }
    payload = stats.compute(overrides, n_resamples=200)
    assert report._stats_lookup(payload, "lewm", "fp32", "entropy").get("sr_ci95_pct")
    # quantized precisions must NEVER fall back — an entropy render must not show a max interval
    assert report._stats_lookup(payload, "lewm", "int8", "max") == {}


def test_stats_never_rewrites_canonical_results(tmp_path):
    """The same read-only guard as `test_report_never_rewrites_canonical_results` and
    `test_clock_norm_never_rewrites_canonical_results`: intervals are ADDITIVE re-analysis —
    `sr.json`, `latencies.*.json` and `results.*.json` are inputs, never outputs (SPEC §Parity,
    CLAUDE §8). The latency file matters most: it is the ONLY copy of the engine-step samples, so a
    render that rewrote it would put an L40S run at risk."""
    vec = list(np.random.default_rng(6).normal(100, 5, size=40))
    sr_path = tmp_path / "sr.json"
    sr_path.write_text(json.dumps(_sr_json(lewm__fp32=(90.0, vec))))
    results = tmp_path / "results.lewm.json"
    results.write_text(json.dumps({"meta": {"track": "lewm"}, "bench": {}}))
    latencies = tmp_path / "latencies.lewm.json"
    latencies.write_text(json.dumps({"meta": {"track": "lewm"}, "latencies": _components(9)}))
    before = {p: p.read_bytes() for p in (sr_path, results, latencies)}

    stats.write_stats_json(
        stats.compute(
            json.loads(sr_path.read_text()),
            n_resamples=200,
            component_latencies=stats.load_component_latencies([latencies]),
        ),
        tmp_path,
    )

    assert (tmp_path / "stats.json").exists()
    for path, raw in before.items():
        assert path.read_bytes() == raw, f"{path.name} was rewritten by the interval render"


# --- component p50s (Phase 9) ------------------------------------------------------------
def _components(seed: int, n: int = 100) -> dict:
    """One track's `latencies.<track>.json` `latencies` block: a fixed-iteration loop sample per
    component, at one precision."""
    rng = np.random.default_rng(seed)
    return {
        "fp32": {
            "encode_ms": list(rng.normal(12.0, 0.4, size=n)),
            "predict_ms": list(rng.normal(35.0, 1.1, size=n)),
        }
    }


def test_component_interval_uses_the_recorded_vector_as_the_sample():
    """The component sample needs NO truncation and NO warm-up drop — the loop is fixed-iteration
    and drops its warm-up before the first timed call, so the stored vector IS the sample
    (architecture.md §12). n must therefore equal the stored length exactly, and the interval must
    bracket the same p50 the speed table prints (`report._percentile_ms`)."""
    latencies = {"dino": _components(11, n=40)}
    payload = stats.compute({}, n_resamples=200, component_latencies=latencies)

    for component, key in (("encode", "encode_ms"), ("predict", "predict_ms")):
        e = payload["points_components"]["dino"]["fp32"][component]
        sample = latencies["dino"]["fp32"][key]
        assert e["n"] == len(sample) == 40  # nothing dropped, nothing truncated
        assert e["p50_ms"] == report._percentile_ms(sample, 0.50)
        lo, hi = e["p50_ci95_ms"]
        assert lo <= e["p50_ms"] <= hi
        assert e["p50_ci_coverage"] >= 0.95  # the conservative rank convention, same as per-cycle


def test_component_points_carry_p50_only_never_p95_or_mean():
    """SPEC §Interface Contracts: the interval goes on the component p50 ALONE. p95 carries no claim
    and the means are the decomposition basis — an interval on either would assert something the
    owner ruling declines to."""
    payload = stats.compute({}, n_resamples=200, component_latencies={"lewm": _components(12, 30)})
    e = payload["points_components"]["lewm"]["fp32"]["encode"]
    assert "p50_ci95_ms" in e
    assert not [k for k in e if "p95" in k or "mean_ms" in k]


def test_component_holm_family_is_separate_from_the_per_cycle_family():
    """Holm is scoped PER MEASUREMENT SURFACE (owner ruling, architecture.md §12). Pooling would make
    every published per-cycle adjusted p-value a function of which other surfaces happen to exist in
    the file — so adding the component section must leave the per-cycle values byte-identical."""
    vec = list(np.random.default_rng(13).normal(100, 5, size=40))
    sr = _sr_json(lewm__fp32=(90.0, vec), dino__fp32=(70.0, vec))

    without = stats.compute(sr, n_resamples=200)
    with_components = stats.compute(
        sr, n_resamples=200, component_latencies={"lewm": _components(14), "dino": _components(15)}
    )

    assert with_components["points"] == without["points"]  # untouched, Holm values included
    assert with_components["meta"]["holm_family_size"] == without["meta"]["holm_family_size"] == 2
    assert with_components["meta"]["holm_family_size_components"] == 4  # 2 tracks x 2 components
    assert "never pooled" in with_components["meta"]["holm_scope"]


def test_component_section_omitted_without_stored_samples():
    """A `stats.json` from a results dir with no `latencies.*.json` (anything pre-Phase-9) is still a
    valid artefact: the component section is absent rather than empty-but-present, and no component
    meta claims a construction that was never run."""
    payload = stats.compute(_sr_json(lewm__fp32=(90.0, [1.0] * 40)), n_resamples=200)
    assert "points_components" not in payload
    assert "component_sample_rule" not in payload["meta"]


def test_component_latency_paths_finds_the_track_files(tmp_path):
    """`src.stats` discovers the samples beside `sr.json` — the layout `src.study` writes — so the
    off-pod re-analysis needs no path bookkeeping."""
    for track in ("lewm", "dino"):
        (tmp_path / f"latencies.{track}.json").write_text(
            json.dumps({"meta": {"track": track}, "latencies": _components(1)})
        )
    (tmp_path / "results.lewm.json").write_text("{}")  # must not be picked up

    found = stats.component_latency_paths(tmp_path)
    assert [p.name for p in found] == ["latencies.dino.json", "latencies.lewm.json"]
    assert set(stats.load_component_latencies(found)) == {"lewm", "dino"}
