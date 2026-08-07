"""Phase-8 confidence intervals — owned PLUMBING (fails LOUDLY), no GPU, no re-run.

Puts a 95% interval on the two ABSOLUTE quantities this study reports, and tests the assumption
the latency interval rests on:

  - **success rate** → **Clopper-Pearson** exact binomial interval over the 50 eval episodes.
    Exact at the 0/50 and 50/50 boundaries the data actually hits, where a normal approximation
    collapses to zero width and reports a certainty the sample does not contain.
  - **per-cycle p50 latency** → the **exact binomial order-statistic** interval, computed from the
    SAME warm-up-dropped, equal-n-truncated sample the reported p50 is computed from
    (`report.per_cycle_samples` — shared, never reimplemented here).
  - **encode-step / predictor-step p50 latency** → that same order-statistic interval over the
    engine-step loop sample stored in `latencies.<track>.json`. That sample needs neither truncation
    nor a warm-up drop: the loop is fixed-iteration (n equal across tracks by construction) and drops
    its warm-up before the first timed call, so the recorded vector IS the sample. p50 only — no
    interval on a p95 or a mean.
  - **the i.i.d. premise** that interval rests on → a two-sided **Dwass Monte-Carlo permutation
    test** on the sample's **lag-1 autocorrelation** (50,000 permutations, statistic used raw with
    NO Student-t transform). Serial correlation would make the interval too NARROW — a stronger
    result than the sample supports — so it is measured, flagged, and never silently corrected.

**No interval on any difference or ratio** — not ΔSR, not the FP32-relative p50 speedup, not the
DINOv3÷LeWM per-cycle ratio, not Δ(entropy−max). Owner ruling; rationale in architecture.md §12.

**Holm is scoped per measurement surface**: the per-cycle tests form one family, the component tests
another, never pooled. Pooling would make every published adjusted p-value a function of which other
surfaces happen to exist in the file. The decision is the unadjusted p-value either way.

**Pure re-analysis of stored samples.** Reads `sr.json` and `latencies.<track>.json`, nothing else.
It requires no
`src.study`, no `src.benchmark`, no `src.sr_eval`, no engines and no L40S — the samples it needs
were persisted by the completed runs. `sr.json`, `latencies.*.json` and `results.*.json` are
**read-only** here and are
never rewritten (SPEC §Parity, CLAUDE §8), the same discipline `src.clock_norm` obeys.

Writes ONE artifact, `stats.json`, defaulting to `$STABLEWM_HOME/reports/phase5/` — the persistent
network volume, same durability contract as the other headline artifacts — with `from=`/`out=` to
point it elsewhere off-pod. It is **method-unscoped**: it covers every (track, precision, method)
point in `sr.json`, including the composite `enc-<A>+pred-<B>` isolation labels, so one file answers
for the whole study rather than one render's method.

**Deviations from stock scipy, flagged** (the brief was to use scipy where it matches and say where
it cannot — each is pinned by a test in `tests/test_stats.py`):
  1. `scipy.stats.permutation_test` is NOT the Dwass test: its `two-sided` is "twice the smaller of
     the one-sided p-values", not `|r*| >= |r_obs|`. The permutation loop is written out below.
  2. No scipy function computes the order-statistic quantile interval. `mstats.median_cihs`
     (Hettmansperger-Sheather) and `mstats.mquantiles_cimj` (Maritz-Jarrett) are DIFFERENT
     estimators and must not be substituted; only `binom.cdf` is used.
  3. scipy has no Holm. `scipy.stats.false_discovery_control` is Benjamini-Hochberg/Yekutieli —
     FDR, not FWER. Holm is written out below.
  4. The naive rank choice `binom.ppf(a/2)` / `binom.isf(a/2)` UNDER-covers (0.9481 at n=59); the
     conservative recipe in `order_statistic_ci` is used instead.
Clopper-Pearson is the one construction taken wholesale from scipy, because it is exact there.

**Writes no interpretation.** What a rejected independence test licenses about the headline is
owner-authored (SPEC §Implementation Boundaries), like the Phase-7 disclosure prose.

Usage:
  uv run python -m src.stats
  uv run python -m src.stats from=<dir> [sr=<sr.json>] [latencies=<dir|file>] [out=<dir>]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import binom, binomtest, norm

from src import report
from src.interfaces import (
    CI_ALPHA,
    DEFAULT_CALIBRATION_METHOD,
    EVAL_NUM_EPISODES,
    PER_CYCLE_WARMUP_DROP,
    PERMUTATION_RESAMPLES,
    PERMUTATION_SEED,
)

_MEDIAN_Q = 0.50
# Permutations are generated in chunks so a 50k x n matrix never materializes whole.
_CHUNK = 10_000


# --- success rate: Clopper-Pearson over the eval episodes -----------------------------
def successes_from_sr(sr_pct: float, n_episodes: int = EVAL_NUM_EPISODES) -> int:
    """Recover the success COUNT from the stored success RATE.

    `sr.json` records the percentage, never k/n, and the trial count lives only in
    `scripts/plan/config/pusht.yaml` (`eval.num_eval: 50`). **Every SR in this study is over exactly
    50 episodes**, so the reconstruction is always exact — SRs are multiples of 2.0.

    A non-integral result therefore means the artefact is not what it claims: a different episode
    count, or an SR that did not come from an eval run at all. Fail loudly rather than round, because
    rounding would emit a **silently mis-scaled interval** — the exact failure mode the whole
    construction is owner-gated to prevent (SPEC §Implementation Boundaries)."""
    exact = sr_pct / 100.0 * n_episodes
    k = round(exact)
    if abs(exact - k) > 1e-6:
        raise SystemExit(
            f"[stats] SR {sr_pct} over n={n_episodes} implies {exact} successes, not an integer — "
            f"this SR did not come from {n_episodes} episodes (pusht.yaml eval.num_eval). "
            "Refusing to round: the interval would be silently mis-scaled."
        )
    return int(k)


def clopper_pearson(k: int, n: int, alpha: float = CI_ALPHA) -> tuple[float, float]:
    """Exact binomial (Clopper-Pearson) interval for a proportion, as a (lo, hi) FRACTION.

    Taken wholesale from scipy — `binomtest(k, n).proportion_ci(method="exact")` IS
    Clopper-Pearson, including the one-sided degenerate ends at k=0 and k=n, which is exactly where
    this study's data sits (`dino fp8@entropy` = 0/50, `lewm fp8@entropy` = 50/50)."""
    ci = binomtest(k, n).proportion_ci(confidence_level=1.0 - alpha, method="exact")
    return float(ci.low), float(ci.high)


# --- per-cycle p50: exact binomial order-statistic interval ---------------------------
def order_statistic_ci(sample, q: float = _MEDIAN_Q, alpha: float = CI_ALPHA) -> dict | None:
    """Distribution-free exact interval for the `q`-quantile: `[x_(j), x_(k)]` on the sorted sample,
    with coverage a binomial tail sum. Returns the endpoints, the 1-indexed ranks, and the ACHIEVED
    coverage — or **None** when no rank pair reaches `1-alpha` (n <= 5 at q=0.5).

    Never invent an interval for a sample too small to support one: the same "unmeasured, never
    asserted" discipline `src.clock_norm` applies to undersampled clock runs.

    **The rank convention is load-bearing** (architecture.md §12). The obvious
    `j = binom.ppf(alpha/2)` / `k = binom.isf(alpha/2)` UNDER-covers — 0.9481 at n=59 — because
    `ppf` returns the smallest rank whose CDF is at LEAST alpha/2. The conservative recipe is:

        j-1 = largest r with cdf(r) <= alpha/2
        k-1 = smallest r with cdf(r) >= 1 - alpha/2
        coverage = cdf(k-1) - cdf(j-1) >= 1 - alpha

    Chosen over a bootstrap because it needs no resampling assumption and no smoothness: with
    n ~ 60 of a right-skewed, clock-drifting latency distribution, an exact binomial statement is
    worth more than a percentile bootstrap's asymptotics."""
    x = sorted(float(v) for v in sample)
    n = len(x)
    if n == 0:
        return None
    cdf = binom.cdf(np.arange(n + 1), n, q)
    lower = np.flatnonzero(cdf <= alpha / 2.0)
    upper = np.flatnonzero(cdf >= 1.0 - alpha / 2.0)
    if lower.size == 0 or upper.size == 0:
        return None  # n too small for ANY rank pair to reach the confidence level
    j, k = int(lower[-1]) + 1, int(upper[0]) + 1  # 1-indexed order statistics
    if k > n:
        return None
    return {
        "lo": x[j - 1],
        "hi": x[k - 1],
        "ranks": [j, k],
        "coverage": float(cdf[k - 1] - cdf[j - 1]),
    }


# --- independence premise: Dwass Monte-Carlo permutation test on lag-1 autocorrelation -
def lag1_autocorr(x) -> float:
    """Lag-1 autocorrelation of a sequence, in RECORDING order (the order carries the whole
    signal — this is meaningless on a sorted sample)."""
    a = np.asarray(x, dtype=np.float64)
    c = a - a.mean()
    denom = float(c @ c)
    if denom == 0.0:
        return 0.0  # a constant sequence has no structure to detect
    return float(c[:-1] @ c[1:] / denom)


def asymptotic_lag1_p(r1: float, n: int) -> float:
    """The PRE-permutation p-value: the classical large-sample white-noise result `r1 ~ N(0, 1/n)`,
    two-sided. Recorded for comparison against the permutation p-value — it is NOT the decision, and
    it is exactly the approximation the permutation test exists to avoid trusting at n ~ 60. Uses
    the normal, not a Student-t transform (owner ruling: no t adjustment)."""
    if n < 2:
        return float("nan")
    return float(2.0 * norm.sf(abs(r1) * np.sqrt(n)))


def dwass_permutation_test(
    x,
    n_resamples: int = PERMUTATION_RESAMPLES,
    seed: int = PERMUTATION_SEED,
    alpha: float = CI_ALPHA,
) -> dict:
    """Two-sided Dwass Monte-Carlo permutation test for lag-1 autocorrelation.

    Permuting the vector destroys temporal order while preserving the marginal distribution
    exactly, so it is the right null for "is there structure in the ordering". The p-value is the
    **add-one (Dwass) form**

        p = (1 + #{|r1*| >= |r1_obs|}) / (1 + B)

    which keeps the test exact: a zero count reports `1/(B+1)`, never a p of 0.

    NOT `scipy.stats.permutation_test`: its `alternative="two-sided"` is "twice the smaller of the
    one-sided p-values", a different convention that would report a different number here.

    Centering once outside the loop is exact, not an approximation: permutation preserves the mean
    and the sum of squares, so the denominator is invariant and only the lag-1 cross-product moves.

    Returns the observed statistic, the permutation p-value (**the decision**), the asymptotic
    p-value (reference), and the permutation null's own lag-1 summary."""
    a = np.asarray(x, dtype=np.float64)
    n = a.size
    r1 = lag1_autocorr(a)
    out = {
        "lag1_autocorr": r1,
        "lag1_p_asymptotic": asymptotic_lag1_p(r1, n),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }
    c = a - a.mean()
    denom = float(c @ c)
    if n < 3 or denom == 0.0:
        # Nothing to permute meaningfully; report the absence rather than a fabricated p-value.
        out.update(lag1_p_permutation=float("nan"), lag1_reject=False,
                   lag1_null_mean=float("nan"), lag1_null_sd=float("nan"))
        return out
    rng = np.random.default_rng(seed)
    obs = abs(r1)
    tol = 1e-12 * max(obs, 1.0)  # float slack: a permutation equal to the observed must still count
    count, null_sum, null_sq, done = 0, 0.0, 0.0, 0
    while done < n_resamples:
        m = min(_CHUNK, n_resamples - done)
        perm = rng.permuted(np.broadcast_to(c, (m, n)), axis=1)
        r = (perm[:, :-1] * perm[:, 1:]).sum(axis=1) / denom
        count += int(np.count_nonzero(np.abs(r) >= obs - tol))
        null_sum += float(r.sum())
        null_sq += float((r * r).sum())
        done += m
    p = (1.0 + count) / (1.0 + n_resamples)
    mean = null_sum / n_resamples
    out.update(
        lag1_p_permutation=p,
        lag1_reject=bool(p < alpha),
        lag1_null_mean=mean,
        lag1_null_sd=float(np.sqrt(max(null_sq / n_resamples - mean * mean, 0.0))),
    )
    return out


def holm(pvalues: dict) -> dict:
    """Holm step-down FWER-adjusted p-values, keyed as given. **Secondary reporting only** — the
    test decision is the unadjusted p-value (SPEC §Interface Contracts), because each run's interval
    is read on its own rather than as one of 18 simultaneous claims. Computed unconditionally so
    "did multiplicity matter here" is answerable off the artefact instead of argued.

    `p_adj_(i) = min(1, max_{j<=i} (m-j+1) * p_(j))` — the running max enforces the monotonicity
    Holm requires. NOT `scipy.stats.false_discovery_control`, which controls FDR, a different
    quantity; Holm over Bonferroni because it is uniformly more powerful at the same FWER."""
    items = [(k, v) for k, v in pvalues.items() if v == v]  # drop NaN (untested runs)
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(sorted(items, key=lambda kv: kv[1])):
        running = max(running, (m - i) * p)
        adjusted[key] = min(1.0, running)
    return {k: adjusted.get(k, float("nan")) for k in pvalues}


# --- the payload ----------------------------------------------------------------------
def _method_map(raw) -> dict:
    """One sr.json precision entry -> `{method: point}`. Legacy flat entries (pre-labelling) are
    `max`-calibrated by definition — the same folding `src.sr_eval._as_method_map` does."""
    if not isinstance(raw, dict):
        return {}
    if "success_rate" in raw:
        return {DEFAULT_CALIBRATION_METHOD: raw}
    return {m: pt for m, pt in raw.items() if isinstance(pt, dict)}


def compute(
    sr_json: dict,
    warmup_drop: int = PER_CYCLE_WARMUP_DROP,
    alpha: float = CI_ALPHA,
    n_episodes: int = EVAL_NUM_EPISODES,
    n_resamples: int = PERMUTATION_RESAMPLES,
    seed: int = PERMUTATION_SEED,
    component_latencies: dict | None = None,
) -> dict:
    """Every (track, precision-label, method) point in `sr.json` -> its intervals + independence
    test. Method-unscoped and label-complete: the composite `enc-<A>+pred-<B>` isolation points are
    covered too, since the isolation table reports their absolute SR.

    The per-cycle sample comes from `report.per_cycle_samples`, so it is byte-identical to the one
    `_finalize_per_cycle` reduces to the reported p50 — including the equal-n truncation across the
    tracks present at that (label, method).

    `component_latencies` (the merged `latencies.<track>.json` blocks) adds the **component**
    surface — encode-/predict-step p50 intervals under `points_components` — with its OWN Holm
    family. Absent, that section is simply omitted: a `stats.json` without it is still valid, and the
    per-cycle values are byte-identical either way (docs/architecture.md §12)."""
    # Invert to (label, method) -> {track: raw vector}: the equal-n truncation is defined ACROSS
    # tracks at one label, so the grouping has to happen before any sample is cut.
    vectors: dict = {}
    for track, by_label in sr_json.items():
        for label, raw in by_label.items():
            for method, point in _method_map(raw).items():
                vec = point.get("per_cycle_latencies_ms") or []
                if vec:
                    vectors.setdefault((label, method), {})[track] = list(vec)

    samples = {
        key: report.per_cycle_samples(by_track, warmup_drop)
        for key, by_track in vectors.items()
    }

    points: dict = {}
    raw_p: dict = {}
    for track, by_label in sr_json.items():
        for label, raw in by_label.items():
            for method, point in _method_map(raw).items():
                entry: dict = {}
                sr = point.get("success_rate")
                if sr is not None:
                    k = successes_from_sr(sr, n_episodes)
                    lo, hi = clopper_pearson(k, n_episodes, alpha)
                    entry.update(
                        success_rate=sr,
                        successes=k,
                        trials=n_episodes,  # the n the SR interval is computed over
                        sr_ci95_pct=[100.0 * lo, 100.0 * hi],
                    )
                sample = samples.get((label, method), {}).get(track)
                if sample:
                    ci = order_statistic_ci(sample, _MEDIAN_Q, alpha)
                    entry.update(
                        per_cycle_p50_ms=report._percentile_ms(sample, _MEDIAN_Q),
                        per_cycle_n=len(sample),  # cycles the p50 interval is computed over
                        per_cycle_raw_n=len(point.get("per_cycle_latencies_ms") or []),
                        p50_ci95_ms=None if ci is None else [ci["lo"], ci["hi"]],
                        p50_ci_ranks=None if ci is None else ci["ranks"],
                        p50_ci_coverage=None if ci is None else ci["coverage"],
                    )
                    entry.update(dwass_permutation_test(sample, n_resamples, seed, alpha))
                    raw_p[(track, label, method)] = entry["lag1_p_permutation"]
                points.setdefault(track, {}).setdefault(label, {})[method] = entry

    # Holm across the per-cycle family of independence tests — SECONDARY, flags nothing. The
    # component tests form their OWN family below and are never pooled into this one, so these
    # values do not move when a component section is added (docs/architecture.md §12).
    for key, adj in holm(raw_p).items():
        track, label, method = key
        entry = points[track][label][method]
        entry["lag1_p_holm"] = adj
        entry["lag1_reject_holm"] = bool(adj == adj and adj < alpha)

    components = (
        compute_components(component_latencies, alpha, n_resamples, seed)
        if component_latencies
        else None
    )

    return {
        "meta": {
            "sr_estimator": "clopper-pearson-exact",
            "p50_estimator": "exact-binomial-order-statistic",
            "rank_convention": (
                "j-1 = largest r with cdf(r) <= a/2; k-1 = smallest r with cdf(r) >= 1-a/2; "
                "coverage = cdf(k-1) - cdf(j-1) >= 1-a"
            ),
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "n_episodes": n_episodes,
            "independence_test": "dwass-monte-carlo-permutation",
            "test_statistic": "lag1-autocorrelation",
            "two_sided": True,
            "two_sided_convention": "|r*| >= |r_obs|  (NOT scipy's twice-the-smaller)",
            "n_resamples": n_resamples,
            "seed": seed,
            "student_t_adjustment": False,
            "decision_pvalue": "lag1_p_permutation (UNADJUSTED)",
            "holm": "secondary reporting only — adjusts no flag and no table",
            "holm_scope": "per measurement surface — the per-cycle and component families are never pooled",
            "holm_family_size": len(raw_p),  # the per-cycle family
            "per_cycle_warmup_drop": warmup_drop,
            "no_interval_on": (
                "differences and ratios (dSR, fp32-relative speedup, cross-model ratio); "
                "any p95; the means and everything derived from them"
            ),
            **(
                {}
                if components is None
                else {
                    "component_estimator": "exact-binomial-order-statistic (p50 only)",
                    "component_sample_rule": (
                        "the fixed-iteration engine-step loop sample as recorded — warm-up dropped "
                        "at record time, no truncation, no report-time drop"
                    ),
                    "holm_family_size_components": components["holm_family_size"],
                }
            ),
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "points": points,
        **({} if components is None else {"points_components": components["points"]}),
    }


# --- component p50s: the same construction over the engine-step loop samples ----------
_COMPONENTS = {"encode": "encode_ms", "predict": "predict_ms"}


def compute_components(
    latencies_by_track: dict,
    alpha: float = CI_ALPHA,
    n_resamples: int = PERMUTATION_RESAMPLES,
    seed: int = PERMUTATION_SEED,
) -> dict:
    """Every (track, precision, component) engine-step sample -> its p50 interval + independence
    test. Same estimator and same test as the per-cycle p50 — only the SAMPLE differs, and it differs
    by being simpler: the loop is fixed-iteration and drops its warm-up at RECORD time, so the stored
    vector is already the sample (no equal-n truncation, no report-time drop —
    docs/architecture.md §12).

    `latencies_by_track` is `{track: {precision: {encode_ms: [...], predict_ms: [...]}}}` — the
    `latencies` block of `latencies.<track>.json`. Method-free by design: component latency is
    calibration-method-invariant (SPEC §Parity).

    **p50 only.** No interval on the p95 (it carries no claim — SPEC §Interface Contracts), on the
    means (the decomposition basis, an algebraic identity rather than an inference), or on anything
    derived from them.

    Holm is applied over THIS family alone, never pooled with the per-cycle tests: pooling would make
    every published adjusted p-value a function of which other surfaces happen to exist in the file
    (docs/architecture.md §12). It is secondary reporting either way — the decision is the unadjusted
    p-value."""
    points: dict = {}
    raw_p: dict = {}
    for track, by_precision in latencies_by_track.items():
        for precision, vectors in by_precision.items():
            for component, key in _COMPONENTS.items():
                sample = list(vectors.get(key) or [])
                if not sample:
                    continue
                ci = order_statistic_ci(sample, _MEDIAN_Q, alpha)
                entry = {
                    "p50_ms": report._percentile_ms(sample, _MEDIAN_Q),
                    "n": len(sample),  # timed iterations the interval is computed over
                    "p50_ci95_ms": None if ci is None else [ci["lo"], ci["hi"]],
                    "p50_ci_ranks": None if ci is None else ci["ranks"],
                    "p50_ci_coverage": None if ci is None else ci["coverage"],
                }
                entry.update(dwass_permutation_test(sample, n_resamples, seed, alpha))
                points.setdefault(track, {}).setdefault(precision, {})[component] = entry
                raw_p[(track, precision, component)] = entry["lag1_p_permutation"]

    for key, adj in holm(raw_p).items():
        track, precision, component = key
        entry = points[track][precision][component]
        entry["lag1_p_holm"] = adj
        entry["lag1_reject_holm"] = bool(adj == adj and adj < alpha)
    return {"points": points, "holm_family_size": len(raw_p)}


def load_component_latencies(paths) -> dict:
    """Merge the per-track `latencies.<track>.json` files (written by `src.study`) into the
    `{track: {precision: {encode_ms, predict_ms}}}` shape `compute_components` consumes. Read-only,
    like every input to this module."""
    out: dict = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        out[data["meta"]["track"]] = data["latencies"]
    return out


def write_stats_json(payload: dict, out_dir: Path) -> Path:
    """Persist to `out_dir/stats.json`. The ONE file this module writes: `sr.json` and
    `results.*.json` are read-only to it (SPEC §Parity, CLAUDE §8)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stats.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


# --- driver ---------------------------------------------------------------------------
def component_latency_paths(base: Path, explicit=None) -> list[Path]:
    """The `latencies.<track>.json` files to read: an explicit `latencies=` path (a file or a dir to
    glob), else whatever sits beside `sr.json`. Empty is fine — the component section is then omitted
    rather than faked."""
    src = Path(explicit) if explicit else base
    if src.is_file():
        return [src]
    return sorted(src.glob("latencies.*.json"))


def main() -> None:
    args = sys.argv[1:]
    from src.study import default_out_dir  # shared default; lazy to avoid an import cycle

    base = default_out_dir()
    sr_path, out_dir, latencies_arg = None, None, None
    for a in args:
        if a.startswith("from="):
            base = Path(a.split("=", 1)[1])
        elif a.startswith("sr="):
            sr_path = Path(a.split("=", 1)[1])
        elif a.startswith("latencies="):
            latencies_arg = a.split("=", 1)[1]
        elif a.startswith("out="):
            out_dir = Path(a.split("=", 1)[1])
    base = base.parent if base.is_file() else base
    sr_path = sr_path or base / "sr.json"
    out_dir = out_dir or base
    if not sr_path.exists():
        raise SystemExit(f"[stats] no sr.json at {sr_path} — nothing to build intervals from")

    latency_paths = component_latency_paths(base, latencies_arg)
    payload = compute(
        json.loads(sr_path.read_text()),
        component_latencies=load_component_latencies(latency_paths) or None,
    )
    path = write_stats_json(payload, out_dir)
    meta = payload["meta"]
    n_sr = sum(1 for t in payload["points"].values() for p in t.values()
               for e in p.values() if "sr_ci95_pct" in e)
    n_p50 = sum(1 for t in payload["points"].values() for p in t.values()
                for e in p.values() if e.get("p50_ci95_ms"))
    n_comp = sum(1 for t in payload.get("points_components", {}).values() for p in t.values()
                 for e in p.values() if e.get("p50_ci95_ms"))
    print(
        f"[stats] {n_sr} SR intervals ({meta['sr_estimator']}, n={meta['n_episodes']}) + "
        f"{n_p50} per-cycle p50 intervals ({meta['p50_estimator']})\n"
        f"[stats] {n_comp} component (encode/predict) p50 intervals from "
        f"{len(latency_paths)} latencies.*.json"
        + ("" if latency_paths else " — none found, component section omitted")
        + "\n"
        f"[stats] independence: {meta['independence_test']} on {meta['test_statistic']}, "
        f"B={meta['n_resamples']}, seed={meta['seed']}, decision on the UNADJUSTED p-value; "
        f"Holm {meta['holm_scope']}\n"
        f"[stats] wrote {path}  (sr.json + latencies.*.json read-only, untouched)"
    )


if __name__ == "__main__":
    main()
