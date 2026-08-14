from typing import Protocol, Literal, TypedDict, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

from torch import Tensor
from jaxtyping import Float

if TYPE_CHECKING:
    # The INT8 calibration set, drawn through the platform (src.calibrate); export turns it
    # into the numpy dict the Model-Optimizer PTQ pass (explicit Q/DQ) consumes. TYPE_CHECKING
    # -only so this foundation module stays import-light and free of a runtime cycle.
    from src.calibrate import CalibrationData

Precision = Literal["fp32", "fp16", "int8", "fp8"]

# Precisions that require an explicit Q/DQ calibration pass before the TRT build (a separately
# quantized ONNX per format, scales baked in — SPEC §Export shape). FP32/FP16 build data-free
# off the shared base graph; INT8 and FP8 each draw the SAME calibration streams (format-
# independent) and route through the Model-Optimizer PTQ. Single source of truth so the export
# gates + the precision-match calib-loader branch never re-enumerate this set per module.
QUANTIZED_PRECISIONS: tuple[str, ...] = ("int8", "fp8")

# PTQ calibration method — a BUILD OPTION available to BOTH tracks (`max` | `entropy`), not a
# hidden per-track setting (architecture.md §7). `max` (ORT MinMax + symmetric)
# sets each per-tensor scale to the largest abs activation seen — zero outlier rejection; `entropy`
# (ORT Entropy) picks a KL-optimal threshold that clips the outlier tail. The ONNX int8/fp8 flow
# supports exactly these two. Which one wins is an SR question, MEASURED per (track, precision,
# method) — NOT asserted per track: LeWM's action signal (widened by architecture.md §7) may prefer `max`;
# DINO's outlier-heavy frozen-DINOv3 activations may prefer `entropy`'s tail-clip. Held CONSTANT
# across a track's INT8 and FP8 within a labelled comparison so the INT8->FP8 step isolates the
# format (SPEC §Parity). Surfaced as a report LABEL (results.<track>.json meta + method-scoped
# artefact filenames), so `max`- and `entropy`-calibrated points COEXIST and existing artefacts are
# never rewritten (CLAUDE §8). The cross-track LATENCY headline is method-invariant; per-model SR
# is a per-(track, precision, method) quality-retention measure. A wrong value degrades SR SILENTLY,
# so the method set + default are owner-gated like the dims above.
CALIBRATION_METHODS: tuple[str, ...] = ("max", "entropy")
# The method every EXISTING artefact was built with; a new method's runs are additive and never
# overwrite the `max` points (method-scoped filenames — src.study / src.sr_eval).
DEFAULT_CALIBRATION_METHOD = "max"


def check_calibration_method(method: str) -> str:
    """Validate a CLI-supplied calibration method against the supported set, failing loudly on a
    typo (an unknown method would otherwise reach modelopt and fail deep in the quant pass, or
    silently mislabel an artefact). Returns the method unchanged."""
    if method not in CALIBRATION_METHODS:
        raise SystemExit(
            f"unknown calibration_method {method!r}; expected one of {list(CALIBRATION_METHODS)}"
        )
    return method

# --- Dims (owner-confirmed; the 🔴 adapter-dims gate). Defined ONCE here; the platform's
# own dims are read from its config, never re-guessed. Read from the pinned installed
# source (docs/platform_api.md §2).
LATENT_DIM = 192  # LeWM single-token latent width (ViT-Tiny hidden_size)
DINO_N_PATCHES = 196  # DINOv3 patch grid after CLS + 4 register slice ((224/16)²)
DINO_LATENT_DIM = 384  # DINOv3 hidden_size
# DINO-WM predictor-input token width: 384 latent + 20 extras (proprio 10 + action 10),
# tiled and concatenated on the feature axis before `predict`. Distinct from the 384
# latent; the CausalPredictor is dim-preserving so its output width is 404 too. A wrong
# value mis-shapes the predictor engine SILENTLY — owner-confirmed against the source.
DINO_PREDICTOR_DIM = 404
ACTION_DIM = 2  # swm/PushT-v1 action_space Box(-1, 1, (2,)) — env/planner-facing action
HISTORY_SIZE = 3  # wm.history_size for both tracks
# Model-facing action width: the CEM plans env ACTION_DIM (2) actions, but the world
# models' action_encoder ingests the frameskip pack (5 × 2 = 10) — action_encoder is a
# Conv1d(10, …) in BOTH tracks (owner-confirmed on the pod, 2026-07-08). Distinct from the
# env ACTION_DIM; a wrong width mis-shapes the predictor engine SILENTLY.
MODEL_ACTION_DIM = 10
# DINO-WM proprio extra: extra_encoders['proprio'] is a Conv1d(4, 10) — proprio input is
# 4-wide, embedded to 10 and concatenated (with action's 10) onto the 384 latent -> 404.
DINO_PROPRIO_DIM = 4

# CEM planning-cycle call counts (docs/platform_api.md §5). The measured per-cycle latency is
# decomposed by weighting the isolated engine-step MEANS by these counts and subtracting from
# the mean cycle (overhead = cycle − enc·ENC_CALLS − pred·PRED_CALLS; SPEC §Interface
# Contracts). MEANS, not percentiles: linearity of expectation makes that identity exact for
# any distribution, while p50(a+b) ≠ p50(a)+p50(b) — a percentile decomposition would silently
# book the non-additivity error as planner overhead.
# Confirmed against the installed `CEMSolver.solve` → `get_cost` → `rollout`
# (`solver/cem.py:191-199`, `wm/prejepa/prejepa.py:218-348`, `wm/lewm/lewm.py:58-108`): the
# `candidates` tensor CEMSolver samples has time-length `horizon` ONLY (not `n_obs + horizon`);
# `rollout` splits it into an `n_obs`-length prefix (tags the current state, no predict call)
# and a `horizon − n_obs`-length remainder that drives `n_steps = horizon − n_obs` autoregressive
# predict calls, plus one final call → `(horizon − n_obs) + 1` predict calls per solve, identical
# in both tracks. NOT `horizon + 1` (that assumes the candidates include the n_obs prefix, which
# they don't).
CEM_NUM_SAMPLES = 300  # candidate fan-out — the batch `predict` is timed at
ENCODER_CALLS_PER_CYCLE = 2  # goal encode + initial-obs encode (both cached, batch 1)
PREDICTOR_CALLS_PER_CYCLE = 150  # (horizon 5 − n_obs 1 + 1) × n_steps 30, batched over the candidates

# Per-cycle warm-up: decisions dropped from the HEAD of each per-cycle latency vector before the
# equal-n truncation (architecture.md §8). The engine-step loops
# already drop `ExportConfig.warmup` iters; the per-cycle callback records from the first decision of
# the first solve, so without this the cold first execute_v2 / kernel autotune / clock ramp sits in
# the cycle mean and NOT in the component means — and `overhead = cycle − enc − pred` books all of it
# as planner overhead, deflating `p` and the Amdahl ceiling. Applied at REPORT time, never at record
# time: `sr.json` keeps the complete raw vector, so the architecture.md §8 span-sum reconciliation still holds
# and both views re-render off-pod. Costs 1 of ~50-100 samples; the p50 headline is unmoved either
# way (median robustness) — this is for the mean-based decomposition.
PER_CYCLE_WARMUP_DROP = 1

# --- Confidence intervals (owner-gated; SPEC §Implementation Boundaries "confidence-interval
# construction", signed off 2026-08-05; rationale `docs/architecture.md` §12). A wrong value here is
# a plausible wrong INTERVAL — silent — so these are set by the owner, not tuned.
#
# Trial count for the SR binomial. It is the ONE input to the intervals that no artifact records:
# `sr.json` stores the success RATE, not k/n. Source: `scripts/plan/config/pusht.yaml`
# `eval.num_eval: 50` (== `world.num_envs`), unchanged by the eval overlays. Successes are recovered
# as `round(SR/100 * n)` under a loud integrality guard — never a silent round.
EVAL_NUM_EPISODES = 50
# Two-sided 95% intervals throughout: Clopper-Pearson for SR, the exact binomial order-statistic
# interval for the per-cycle p50. Also the significance level of the independence test.
CI_ALPHA = 0.05
# Dwass Monte-Carlo permutation test on the sample's lag-1 autocorrelation — the check on the
# order-statistic interval's i.i.d. premise. The per-cycle vector is consecutive decisions across
# still-alive episodes on a thermally drifting GPU, so independence is a claim, not a given; serial
# correlation would make the interval too NARROW (a stronger result than the sample supports).
PERMUTATION_RESAMPLES = 50_000
# Fixed and recorded, and set to the owned layer's seed convention (`ExportConfig.seed`,
# `calibrate.predictor_batches`) rather than a fresh value: a Monte-Carlo p-value that moves between
# renders is not an artefact anyone can audit.
PERMUTATION_SEED = 0
# Non-parametric percentile bootstrap on the five MEAN per-cycle latency quantities (enc_cyc,
# pred_cyc, t_comp, cycle, overhead — SPEC §Interface Contracts). A percentile interval only needs
# its 2.5/97.5 resample points to settle, where the lag-1 test above needs 50,000 to resolve a TAIL
# p-value — different jobs, different budgets. Seeded like `PERMUTATION_SEED` and for the same
# reason: a resampled interval that moves between renders cannot be audited.
BOOTSTRAP_RESAMPLES = 3_000
BOOTSTRAP_SEED = 0

# --- Clock normalization (owner-gated; SPEC §Implementation Boundaries "clock-normalization
# construction", signed off 2026-07-25). GPU clocks are unlockable on this platform, and the
# observed throttle is DIFFERENTIAL — the heavier track power-throttles while the lighter one holds
# the boost ceiling — so it does not cancel in the cross-model ratio. These fix the derived-bound
# construction; a wrong value is a plausible wrong CORRECTED number (silent), hence owner-set.
#
# Scaling model: `T_ref = T × f_measured / f_ref` (time ∝ 1/f_sm). It knowingly OVER-corrects —
# memory-bound and host/Python time do not scale with SM clock — which is exactly what makes the
# normalized figure a BOUND (the maximum plausible correction) rather than a point estimate
# (SPEC §Parity). ALL measured latency is treated as clock-bound: per-cycle, encode-step and
# predict-step alike, so the overhead decomposition subtracts terms taken at a matched clock.
# `f_ref` cancels in every ratio (R′ and the within-model deltas); it only scales the absolute
# derived ms values, which read as "as if unthrottled" at the boost ceiling.
CLOCK_F_REF_MHZ = 2520  # L40S boost ceiling — the clock the lighter track (LeWM) actually held
# `f_measured` is the UTIL-CONDITIONED median SM clock: the median `pclk` over the run's dmon
# samples with SM utilization ≥ the threshold. Conditioning is load-bearing, not cosmetic — a
# 1 Hz dmon over a short run catches idle/ramp samples (one log medians to 1260 MHz at 7% util),
# and an unconditioned median would halve that run's normalized time from a sample where nothing
# was running.
CLOCK_BUSY_UTIL_PCT = 50
# A run with fewer than this many busy samples is recorded as UNMEASURED (null) and excluded from
# normalization — its measured latency is reported without a derived counterpart and the gap is
# disclosed. Never invent a clock: asserting one from an idle sample is the silent-failure mode
# this whole gate exists to prevent.
CLOCK_MIN_BUSY_SAMPLES = 3

# CEM action-proposal shape — the distribution `predict` is ACTUALLY driven by at eval, and
# what the INT8 predictor calibration stream reproduces (SPEC §Interface Contracts —
# calibration distribution). Read from the vendored configs + `CEMSolver` source, not assumed:
#   `solver/cem.py:191-204`  candidates = randn(B, num_samples, horizon, action_dim) * var + mean
#                            -> an UNCLAMPED N(0, var_scale) about `mean` (0 at the zero-pad
#                            warm start); there is NO clamp to the action space.
#   `solver/cem.py:80`       action_dim = env_action_dim (2) * action_block (5) = 10, i.e. CEM
#                            samples the MODEL-facing 10-wide frameskip pack DIRECTLY — no
#                            env->model packing sits between the proposal and `predict`.
# Expert dataset actions are bounded by Box(-1, 1); the proposal reaches ~4 sigma. Calibrating
# on expert actions therefore under-scales the action tensor ~4x and saturates INT8 — invisible
# in FP16 (no fixed clip). 🔴 confirm against the installed solver on the pod, not assumed.
CEM_VAR_SCALE = 1.0  # scripts/plan/config/solver/cem.yaml `var_scale` — initial proposal std
CEM_HORIZON = 5  # scripts/plan/config/pusht.yaml `plan_config.horizon`
# n_obs at eval: the rollout starts from ONE encoded frame and fills the rest of the window
# with its OWN predictions, so a steady-state `predict` window holds ZERO encoder latents
# (`LeWM.rollout` lo=max(0, H+t-HS) with H=1 -> windows 1,2,3,3,…; same for `PreJEPA.rollout`).
EVAL_N_OBS = 1


class WMStepAdapter(Protocol):
    """Common two-method boundary the export/benchmark/profile plumbing binds to, so it
    treats both tracks identically and never branches per-model. `encode` and `predict`
    are separately callable (and separately exported), because the CEM rollout encodes
    the obs ONCE and calls `predict` autoregressively over the horizon for all candidates.

    The latent *shape* is model-specific; the two concrete implementations (adapter.py)
    carry the precise annotations. The variadic `*latent` below admits both without the
    plumbing knowing which:
        LeWMAdapter   -> Float[Tensor, "batch hist latent"]        (single token, D=192)
        DINOWMAdapter -> Float[Tensor, "batch hist patch latent"]  (patch grid, 196x384)
    Conditioning enters `predict` per-track and with per-track *arity* (hence the variadic
    `*conditioning`): LeWM takes `(latent, action)` — a single AdaLN-conditioning tensor
    after the latent. DINO-WM's `predict` is a faithful, dim-preserving 404->404 step over a
    *pre-assembled* embedding, so it takes a SINGLE 404-wide tensor (no separate conditioning
    arg): the 384->404 proprio/action assembly (`DINOWMAdapter.assemble_embedding`) and the
    per-step action-replacement live in the Python rollout/shim, not this call. The plumbing
    stays arity-agnostic by driving `predict(*inputs)`. The CEM planner / rollout loop stays
    in Python outside the adapter — the adapter is the unit TensorRT optimizes.
    """

    def encode(
        self,
        obs: Float[Tensor, "batch hist channel height width"],
    ) -> Float[Tensor, "batch hist *latent"]: ...

    def predict(
        self,
        latent: Float[Tensor, "batch hist *latent"],
        *conditioning: Float[Tensor, "batch hist cond"],
    ) -> Float[Tensor, "batch hist *latent"]: ...


class EnginePaths(TypedDict):
    # export traces `encode` and `predict` separately -> one engine each (two per model).
    encoder: Path
    predictor: Path


class Export(Protocol):
    def __call__(
        self,
        adapter: WMStepAdapter,
        precision: Precision,
        encode_inputs: tuple[Tensor, ...],  # example obs for the encoder graph
        # example predict inputs: LeWM (cached latent, action); DINO (assembled 404 embedding)
        predict_inputs: tuple[Tensor, ...],
        engine_dir: Path,
        # required iff precision is quantized (int8/fp8): the Model-Optimizer PTQ input (export builds a
        # per-method numpy dict keyed by ONNX input name from it — explicit Q/DQ, not a TRT
        # build-time calibrator).
        calib_loader: "CalibrationData | None" = None,
        # PTQ calibration method (`max` | `entropy`) — a build option for both tracks, held
        # constant across a track's int8/fp8 within a labelled comparison (SPEC §Export shape).
        # Ignored for FP32/FP16.
        calibration_method: str = "max",
    ) -> EnginePaths: ...


class BenchResult(TypedDict):
    # HEADLINE latency — the full per-decision PLANNING CYCLE (encode + predict + overhead),
    # measured on the REAL CEM solve via the eval-latency callback over the SR eval-shim run.
    # NOT produced by `benchmark` (no planner in the harness); left NaN here and JOINED per
    # precision from that gated run, so per-cycle latency and SR come from the same solves
    # (SPEC §Interface Contracts). Equal-n across tracks (report truncates to the common
    # min-n before reducing).
    # `p50` is the COMPARISON basis (the LeWM-vs-DINOv3 headline ratio + the FP32-relative
    # degradation): robust to the tail at this n. `p95` is reported as the descriptive tail.
    per_cycle_p50_ms: float
    per_cycle_p95_ms: float
    # `mean` is the DECOMPOSITION basis only (never the headline) — see the call counts above.
    per_cycle_mean_ms: float
    # COMPONENT latency — isolated per-precision engine-step stats from fixed-iteration loops
    # (warm-up dropped, equal-n). `encode_*` exposes the encoder token-count asymmetry
    # (LeWM 1 token vs DINOv3 196); `predict_*` is quantization's kernel target. Each engine
    # call syncs its stream, so for LeWM's tiny ops these sit on a launch+sync floor.
    # p50/p95 are reported; `*_mean_ms` feeds the decomposition ONLY.
    encode_p50_ms: float
    encode_p95_ms: float
    encode_mean_ms: float
    predict_p50_ms: float
    predict_p95_ms: float
    predict_mean_ms: float
    # Sampled from cudaMemGetInfo (`torch.cuda.mem_get_info`), NOT `torch.cuda.max_memory_
    # allocated`: TensorRT's engine + execution-context device allocations bypass torch's
    # caching allocator, so the allocator would undercount exactly the optimized path
    # (SPEC §Interface Contracts). Device-level → whole-GPU used memory (benchmark pod is
    # dedicated).
    peak_mem_mb: float
    success_rate: float  # Push-T SR paired with this engine config (gated eval shim; NaN until then)


class ComponentSamples(TypedDict):
    """The engine-step loops' RAW per-call latencies (ms), one list per component — the sample
    every component statistic is computed over.

    Returned ALONGSIDE `BenchResult` rather than inside it: `BenchResult` is what
    `results.<track>.json` serializes, and that file is the summary-shaped canonical artifact every
    table, plot and derived-clock render parses. The samples are persisted beside it
    (`latencies.<track>.json`, `src.study.dump_track_latencies`) so a later statistic over the
    component distributions — the p50 confidence interval + its independence test (`src.stats`), or
    any future quantile — is an off-pod re-analysis instead of an L40S booking
    (SPEC §Interface Contracts, docs/architecture.md §12).

    Warm-up is already excluded: `benchmark` runs `warmup` untimed iters before the timed loop, so
    the recorded vector IS the sample — no truncation and no report-time drop, unlike the per-cycle
    vector (which needs both). Length is `n_iters` per component, equal across tracks by
    construction."""

    encode_ms: list[float]
    predict_ms: list[float]


class Benchmark(Protocol):
    def __call__(
        self,
        engines: EnginePaths,  # encoder + predictor engines, timed as isolated step loops
        encode_inputs: tuple[Tensor, ...],  # obs at batch 1 (the cached per-cycle encode)
        predict_inputs: tuple[Tensor, ...],  # predictor state at the candidate fan-out batch
        n_iters: int,  # fixed-iteration count per step loop (equal-n percentiles)
        warmup: int,
        # The summary numbers AND the raw samples they were reduced from — the samples are
        # persisted so the reduction is auditable and re-analysable off the artefact.
    ) -> tuple[BenchResult, ComponentSamples]: ...


@dataclass(frozen=True)
class ExportConfig:
    hist: int = HISTORY_SIZE
    obs_shape: tuple[int, int, int] = (3, 224, 224)
    action_dim: int = MODEL_ACTION_DIM  # model-facing action fed to predict (frameskip pack)
    proprio_dim: int = DINO_PROPRIO_DIM  # DINO-WM proprio extra fed to predict
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8", "fp8")
    warmup: int = 10  # warm-up iters dropped before timing each engine-step loop
    n_latency_iters: int = 100  # fixed timed iters per engine-step loop (equal-n p50/p95)
    seed: int = 0
