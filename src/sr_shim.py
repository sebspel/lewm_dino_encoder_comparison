"""Owner-gated SR shim — re-enter the platform CEM eval on the OPTIMIZED (engine) model.

Phase-5 pairs every speed number with a Push-T success rate. The CEM solver calls the world
model through ``get_cost`` (NOT ``encode`` / ``predict`` directly — see
``stable_worldmodel.solver.cem.CEMSolver.solve``: ``self.model.get_cost(expanded_infos,
candidates)``), so to produce the SR that goes with each precision the exported/quantized
engines are re-wrapped in an object exposing ``get_cost`` and slotted into
``CEMSolver(model=...)``, letting the Phase-3 eval run re-use the same solver/criterion on the
optimized model.

**Parity is the load-bearing, silently-failing part** (SPEC §Parity, §Interface Contracts):
the predicted-proprio channels must survive, the ``404`` carry + per-step action-replace must
mirror ``PreJEPA.rollout``, and the cost must be MSE of predicted proprio AND pixels vs goal.
A plausible-but-wrong assembly passes the engine precision-match (which only compares
engine-vs-adapter) yet corrupts every SR with no error. So this shim does NOT re-implement
``get_cost``. It **subclasses the platform model** (``DINOv3PreJEPA``) and overrides ONLY the
two engine-boundary methods:

    ``_encode_image``  -> the encoder engine   (register-sliced patch grid, ``(B, T, 196, 384)``)
    ``predict``        -> the predictor engine  (dim-preserving ``(B, T, P, 404) -> …404``)

``encode``, ``rollout``, ``replace_action_in_embedding``, ``criterion``, ``split_embedding``,
the goal encoding, and ``get_cost`` are inherited **byte-unchanged**, so cost parity holds by
construction. The ``extra_encoders`` (proprio/action ``Embedder``s), the ``384 -> 404``
assembly, and the action-carry stay in PyTorch on the shim — they are the Python-side ops the
SPEC keeps out of the engine.

The shim is **non-``Actionable``** (no ``get_action``, inherited from ``PreJEPA``), so the CEM
warm-start zero-pads exactly as the Phase-3 baseline (LeWM / DINO-WM are ``get_cost``-only,
docs/platform_api.md), keeping the SR comparable.

Parity reference: **stable_worldmodel 0.1.1** (the installed/pinned version that actually runs
here) — ``wm/prejepa/prejepa.py``. The ``/home/sebastian/stable-worldmodel`` checkout is 16
commits ahead of the pin (``0.1.1-16-g24515aa``, "Refactor plan #282") and has diverged; its
``prejepa.py`` is byte-identical to the pin, but the pin is what this mirrors.

The parity claim is proven the same way as the adapter-fidelity gate (``src.fidelity``):
routing the two overrides through the adapter's pure-torch ``encode`` / ``predict`` (the exact
functions the engines reconstruct) must reproduce ``model.get_cost`` **bit-for-bit** on the
real checkpoint — see ``sr_cost_parity`` / ``python -m src.sr_shim``.
"""

from __future__ import annotations

import sys
from typing import Callable

import torch
from torch import Tensor, nn

from stable_worldmodel.wm.lewm.lewm import LeWM

from src.dino_patch import DINOv3PreJEPA
from src.interfaces import DINO_PREDICTOR_DIM, MODEL_ACTION_DIM, EnginePaths, ExportConfig

# (B, T, C, H, W) -> (B, T, 196, 384): the register-sliced patch grid, same contract as
# DINOv3PreJEPA._encode_image. (B, T, P, 404) -> (B, T, P, 404): the dim-preserving predictor.
EncodeFn = Callable[[Tensor], Tensor]
PredictFn = Callable[[Tensor], Tensor]
# LeWM predict engine boundary: (emb (B,T,192), RAW action (B,T,10)) -> (B,T,192). The engine
# runs the per-frame action_encoder + predictor + pred_proj internally (Design A), so it ingests
# the raw action, NOT a pre-encoded act_emb — distinct from DINO's single-tensor PredictFn.
LeWMPredictFn = Callable[[Tensor, Tensor], Tensor]


class DINOWMSRShim(DINOv3PreJEPA):
    """A ``DINOv3PreJEPA`` whose ``_encode_image`` / ``predict`` route through injected
    callables (the exported engines on the pod, or the adapter's torch methods in tests).
    Every other method — crucially ``get_cost`` / ``rollout`` / ``criterion`` / goal-encode —
    is inherited unchanged, so the cost is identical to the platform's up to the engines'
    quantization drift."""

    def __init__(self, model: DINOv3PreJEPA, encode_fn: EncodeFn, predict_fn: PredictFn):
        # Bypass PreJEPA.__init__ (it builds modules from ctor args); wire the shim from the
        # already-loaded model instead. backbone/predictor are kept only so config-derived
        # attributes still resolve — the engines replace their compute via the overrides below.
        nn.Module.__init__(self)
        self.backbone = model.backbone
        self.predictor = model.predictor
        self.extra_encoders = model.extra_encoders
        self.decoder = getattr(model, "decoder", None)
        self.history_size = model.history_size
        self.num_pred = getattr(model, "num_pred", 1)
        self.interpolate_pos_encoding = getattr(model, "interpolate_pos_encoding", True)
        self._encode_fn = encode_fn
        self._predict_fn = predict_fn

    def _encode_image(self, pixels: Tensor) -> Tensor:
        # Inherited `encode` calls this with (B, T, C, H, W) (already .float()); return the
        # register-sliced grid (B, T, 196, 384) the engine produces. `.detach()` matches the
        # platform's own `_encode_image` (the encoder is frozen; the latent never carries grad).
        return self._encode_fn(pixels).detach()

    def predict(self, embedding: Tensor) -> Tensor:
        # Inherited `rollout` calls this with (B, T, P, 404); the engine is dim-preserving.
        # The 404 width is the silently-failing dim — assert it so a mis-assembled carry is
        # a loud error here, not a wrong SR downstream.
        assert embedding.shape[-1] == DINO_PREDICTOR_DIM, (
            f"predict input width {embedding.shape[-1]} != {DINO_PREDICTOR_DIM}"
        )
        return self._predict_fn(embedding)

    @classmethod
    def from_engines(cls, model: DINOv3PreJEPA, engines: EnginePaths) -> "DINOWMSRShim":
        """Pod path: build the shim over the two TensorRT engines `src.export` produced."""
        encode_fn, predict_fn = build_engine_fns(engines)
        return cls(model, encode_fn, predict_fn)

    @classmethod
    def from_adapter(cls, model: DINOv3PreJEPA, adapter) -> "DINOWMSRShim":
        """Off-engine path (parity test / a PyTorch-reference SR run): route through the
        adapter's pure-torch `encode` / `predict` — the exact functions the engines
        reconstruct — so the shim's cost equals `model.get_cost` bit-for-bit."""
        return cls(model, adapter.encode, adapter.predict)


class LeWMSRShim(LeWM):
    """LeWM whose pixel-encode AND predict paths route through injected callables (the exported
    encoder + predictor engines on the pod, or the adapter's torch ``encode`` / ``predict`` in
    the parity check), so the SR reflects the SAME quantized engines the benchmark times — the
    predictor's FP16/INT8 drift enters the cost, exactly as it does for DINO.

    **Why ``encode`` needs its own check, unlike DINO.** DINO's shim overrides the narrow
    ``_encode_image`` seam and inherits ``encode`` untouched, so parity holds *by construction*.
    ``LeWM.encode`` has **no such seam** — it fuses the backbone call (``encoder -> cls ->
    projector``) with the info-dict bookkeeping *and* the ``action_encoder`` branch in one
    method that returns the mutated dict. So this override **re-implements ``encode``'s body**;
    it is NOT inherited, and a wrong key/dtype/shape would silently corrupt every LeWM SR (the
    inherited ``rollout`` / ``get_cost`` consume ``info['emb']``). That is exactly what
    ``sr_cost_parity_lewm`` validates: the override must reproduce ``LeWM.get_cost`` bit-for-bit.

    **Predict routes through the engine (Design A — ``action_encoder`` lives INSIDE the engine).**
    The exported LeWM predict engine (``LeWMAdapter.predict``) ingests a **raw** action and runs
    ``action_encoder -> predictor -> pred_proj`` internally — the boundary the per-frame guard
    (``src.fidelity.lewm_action_encoder_per_frame``, owner-signed-off) exists to justify. But the
    inherited ``LeWM.rollout`` pre-encodes the *whole* action sequence once
    (``all_act_emb = self.action_encoder(...)``) and hands ``predict`` a pre-encoded ``act_emb``.
    To feed the engine RAW actions while inheriting ``rollout`` byte-unchanged, the shim replaces
    its ``action_encoder`` with an **Identity passthrough**: ``rollout`` then windows the raw
    actions straight into ``predict``, and the engine's own per-frame ``action_encoder`` does the
    encode. Because the encoder is per-frame, the windowed per-step encode equals the
    whole-sequence encode bit-for-bit (same guard), so this reproduces ``LeWM.predict``'s cost
    contribution exactly on the adapter path and carries the predictor engine's quantization drift
    on the pod. (This mirrors DINO, where the small ``extra_encoders`` stay native in ``rollout``
    and only the core predictor is compiled — here the small ``action_encoder`` sits in the
    engine and ``rollout`` is neutralized to a passthrough instead.)

    Non-``Actionable`` (``LeWM`` has no ``get_action``) -> the CEM warm-start zero-pads, matching
    the Phase-3 baseline. NOTE: ``LeWM.get_cost`` (pinned swm 0.1.1) only supports a **single
    env per solve** (``batch_size=1`` -> ``current_bs=1``); its ``criterion`` broadcasts the
    single-env goal over the candidate axis but errors for ``B>1``. The vendored CEM config
    pins ``batch_size=1``, so this is the real contract — the check runs at ``B=1``.
    """

    def __init__(self, model: LeWM, encode_fn: EncodeFn, predict_fn: LeWMPredictFn):
        nn.Module.__init__(self)
        self.encoder = model.encoder
        self.projector = model.projector
        # action_encoder lives INSIDE the predict engine (Design A / per-frame guard), so the
        # inherited rollout must NOT pre-encode: an Identity passthrough makes rollout window the
        # RAW actions straight into `predict`, whose engine does the per-frame-exact encode.
        self.action_encoder = nn.Identity()
        self.predictor = model.predictor  # kept for rollout's `getattr(predictor, 'num_frames')`
        self.pred_proj = model.pred_proj
        self._encode_fn = encode_fn
        self._predict_fn = predict_fn

    def encode(self, info: dict) -> dict:
        # Wholesale override of LeWM.encode (no _encode_image seam). Mirror the source EXACTLY
        # except the engine replaces encoder->cls->projector: set info['emb'] from encode_fn,
        # run the (now passthrough) action-encode branch, return the mutated dict. `encode_fn`
        # (adapter/engine) returns (b, t, D) — the same shape LeWM.encode's
        # `rearrange('(b t) d -> b t d')` produces — so the info-dict contract the inherited
        # rollout consumes is preserved. `info['act_emb']` (raw action here, since action_encoder
        # is Identity) is dead: rollout recomputes act_emb from action_sequence and the goal
        # encode pops 'action' — kept only to mirror the source's branch faithfully.
        info["emb"] = self._encode_fn(info["pixels"])
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb: Tensor, act: Tensor) -> Tensor:
        # Inherited `rollout` calls this with (emb (BS,HS,192), act) where — because
        # action_encoder is an Identity passthrough — `act` is the RAW windowed action (width
        # MODEL_ACTION_DIM), exactly what the engine ingests. Assert the raw width so a forgotten
        # passthrough (pre-encoded 192-wide act_emb) is a loud error, not a silent wrong SR.
        assert act.shape[-1] == MODEL_ACTION_DIM, (
            f"predict action width {act.shape[-1]} != {MODEL_ACTION_DIM}: the inherited rollout "
            "must pass RAW actions to the engine (action_encoder is an Identity passthrough)"
        )
        return self._predict_fn(emb, act)

    @classmethod
    def from_adapter(cls, model: LeWM, adapter) -> "LeWMSRShim":
        """Check path: route encode/predict through the adapter's pure-torch methods — the exact
        functions the engines reconstruct — so the shim's cost equals `LeWM.get_cost` bit-for-bit
        (proving the encode override + predict-engine boundary preserve the inherited cost path)."""
        return cls(model, adapter.encode, adapter.predict)

    @classmethod
    def from_engines(cls, model: LeWM, engines: EnginePaths) -> "LeWMSRShim":
        """Pod path: build the shim over the two TensorRT engines `src.export` produced."""
        encode_fn, predict_fn = build_lewm_engine_fns(engines)
        return cls(model, encode_fn, predict_fn)


def _hist_adapt(encode: EncodeFn, enc_hist: int) -> EncodeFn:
    """Make a fixed-hist encoder callable ``(B, T, C, H, W) -> (B, T, …)`` T-agnostic.

    The exported encoder engine traces a **static hist axis** (``ExportConfig.hist``) — only
    the batch axis is dynamic (``src.export._batch_dynamic``). But the inherited ``get_cost``
    encodes the initial state at ``n_obs`` (= ``history_size`` = the traced hist) AND the goal
    at a single frame (``info['goal'][:, 0]`` -> T=1; ``_encode_image`` does NOT pad — only
    ``_encode_video`` does), so the engine is also asked for T=1 and would raise a TensorRT
    shape error.

    The encoder is **temporally independent**: ``_encode_image`` / the adapter fold
    ``(b t) -> (b·t)`` and run the backbone per-frame, so a frame's embedding does not depend
    on any other frame (temporal mixing happens only in the predictor). A ``T < enc_hist`` call
    is therefore served EXACTLY by repeat-padding the frame axis up to ``enc_hist`` (the
    platform's own ``_encode_video`` padding idiom), encoding, and slicing the first ``T``
    frames back — the padded frames are copies that leave the real frames' embeddings
    unchanged. ``T > enc_hist`` cannot occur in the CEM cost path (init = hist, goal = 1) and
    is a loud error. This keeps the precision-match-gated engine byte-for-byte (no re-export)."""

    def adapted(pixels: Tensor) -> Tensor:
        t = pixels.shape[1]
        if t == enc_hist:
            return encode(pixels)
        if t > enc_hist:
            raise ValueError(
                f"encode hist {t} > engine hist {enc_hist}: the static-hist encoder engine "
                "cannot serve more frames than it was traced for (re-export with a larger hist)"
            )
        pad = pixels[:, -1:].repeat(1, enc_hist - t, *([1] * (pixels.ndim - 2)))
        return encode(torch.cat([pixels, pad], dim=1))[:, :t]

    return adapted


def _predict_hist_adapt(predict: Callable[..., Tensor], pred_hist: int) -> Callable[..., Tensor]:
    """Make a fixed-hist PREDICTOR callable ``(B, T, …) -> (B, T, …)`` T-agnostic — the
    predictor analogue of ``_hist_adapt`` (and general over 1+ inputs sharing the frame axis:
    DINO's single ``404`` embedding; LeWM's ``(emb, RAW action)``).

    The exported predictor engine traces a **static hist (frame) axis** (``ExportConfig.hist``
    = ``HS``); only the batch axis is dynamic. But the inherited ``rollout`` feeds ``predict`` a
    window that GROWS ``min(n_obs, HS) -> HS`` (``predict(z[:, -HS:])``; at eval ``n_obs = 1`` the
    windows are ``1, 2, 3, 3, …``), so the first steps hand the engine a ``T < HS`` window it
    cannot bind (a negative-dim TensorRT output). A ``T < HS`` call is served by **right-padding
    the frame axis up to ``HS``, running the engine, and slicing the first ``T`` frames back**.

    Unlike the encoder (temporally independent), the predictor **does** mix across the frame
    axis, so this is exact only under a **model-specific mask-free-padding exception**: it holds
    iff the predictor is **causal** (frame ``i`` reads only frames ``<= i``) with **prefix
    positional embeddings** and the padded (tail) frames' outputs are discarded — then no real
    read frame ever attends a pad frame and every real frame keeps its positional embedding, so
    frames ``0..T-1`` are byte-identical with or without the pad. (The general case — padding a
    causal transformer — needs an attention mask; this one does not, because the pad sits AFTER
    every frame we read.) **Both** predictors are owner-confirmed causal with prefix positional
    embeddings, so the identical fix applies to LeWM and DINO alike — no per-track gating. The
    pad content is therefore immaterial; a repeat-pad of the last frame (the encoder idiom) is
    used to stay NaN-free. Keeps the precision-match-gated engine byte-for-byte (no re-export).

    ``T > HS`` cannot occur (the rollout caps the window at ``HS``) and is a loud error. Because
    the exactness is a silent-failure assumption, it is proven at the off-nominal windows
    (``T in {1, 2}``) by ``src.precision_match`` (engine-vs-torch) — the fixed-``HS`` gates never
    exercised ``T < HS``, which is why the mismatch passed every gate yet crashed the SR run."""

    def adapted(*inputs: Tensor) -> Tensor:
        t = inputs[0].shape[1]
        if t == pred_hist:
            return predict(*inputs)
        if t > pred_hist:
            raise ValueError(
                f"predict hist {t} > engine hist {pred_hist}: the static-hist predictor engine "
                "cannot serve more frames than it was traced for (re-export with a larger hist)"
            )

        def pad(x: Tensor) -> Tensor:
            tail = x[:, -1:].repeat(1, pred_hist - t, *([1] * (x.ndim - 2)))
            return torch.cat([x, tail], dim=1)

        return predict(*(pad(x) for x in inputs))[:, :t]

    return adapted


def build_engine_fns(engines: EnginePaths) -> tuple[EncodeFn, PredictFn]:
    """Wrap the encoder + predictor engines as ``encode`` / ``predict`` callables.

    Pod-only: ``EngineRunner`` lazy-imports ``tensorrt`` and allocates CUDA buffers, so this
    is imported lazily to keep ``src.sr_shim`` importable off-pod (tests use the adapter path).

    Both engines trace a dynamic batch axis but a **static hist axis**, and the inherited
    ``get_cost`` / ``rollout`` drive both at ``T != HS``. The encoder is called at ``n_obs``
    frames (init) AND ``1`` frame (goal) -> wrapped with ``_hist_adapt`` (repeat-pad/slice, exact
    because the encoder is temporally independent). The predictor is fed a GROWING window
    (``predict(z[:, -HS:])``; ``n_obs = 1`` -> ``1, 2, 3, …``) -> wrapped with
    ``_predict_hist_adapt`` (right-pad the frame axis to ``HS``, run, slice ``[:, :T]``, exact
    because the predictor is causal with prefix positional embeddings). Both leave the
    precision-match-gated engines byte-for-byte. Each traced hist is read from that engine's own
    input binding (the single source of truth for what ``T`` it accepts)."""
    from src.trt_runtime import EngineRunner

    encoder = EngineRunner(engines["encoder"])
    predictor = EngineRunner(engines["predictor"])

    # Read the static hist off each engine's input binding (axis 1; axis 0 is the dynamic batch
    # -> -1, hist is concrete). Wrap so the goal encode (T=1) and the sub-HS predict windows reuse it.
    enc_hist = int(encoder.engine.get_tensor_shape(encoder.input_names[0])[1])
    encode_fn = _hist_adapt(lambda pixels: encoder.run((pixels,)), enc_hist)

    pred_hist = int(predictor.engine.get_tensor_shape(predictor.input_names[0])[1])
    predict_fn = _predict_hist_adapt(lambda embedding: predictor.run((embedding,)), pred_hist)

    return encode_fn, predict_fn


def build_lewm_engine_fns(engines: EnginePaths) -> tuple[EncodeFn, LeWMPredictFn]:
    """Wrap the LeWM encoder + predictor engines as ``encode`` / ``predict`` callables.

    Same static-hist handling as ``build_engine_fns`` (encoder ``_hist_adapt``; predictor
    ``_predict_hist_adapt`` right-pad-to-``HS``/slice, exact because the predictor is causal with
    prefix positional embeddings — identical to DINO, no per-track gating). The predictor callable
    differs from DINO's only in arity: the LeWM predict engine is **two-input** — ``(emb, RAW
    action)`` — because ``action_encoder`` lives inside it (Design A), so both tensors are padded
    on the shared frame axis and bound."""
    from src.trt_runtime import EngineRunner

    encoder = EngineRunner(engines["encoder"])
    predictor = EngineRunner(engines["predictor"])

    enc_hist = int(encoder.engine.get_tensor_shape(encoder.input_names[0])[1])
    encode_fn = _hist_adapt(lambda pixels: encoder.run((pixels,)), enc_hist)

    pred_hist = int(predictor.engine.get_tensor_shape(predictor.input_names[0])[1])
    predict_fn = _predict_hist_adapt(lambda emb, act: predictor.run((emb, act)), pred_hist)

    return encode_fn, predict_fn


def _make_info(
    model: DINOv3PreJEPA,
    cfg: ExportConfig,
    batch: int,
    candidates: int,
    n_obs: int,
    pred_steps: int,
    device: str | torch.device,
) -> tuple[dict, Tensor]:
    """Build a synthetic Push-T-shaped ``info_dict`` + action candidates that ``get_cost``
    accepts: initial ``pixels``/``proprio`` at ``n_obs`` frames, a single-frame
    ``goal``/``goal_proprio``, an ``action`` placeholder (overwritten in ``rollout``), and a
    candidate action sequence spanning the horizon. Extra-encoder input widths are read off the
    loaded model so the same call works on the dummy and the real checkpoint. Random inputs are
    valid — the parity check compares two implementations of the SAME cost, not task quality."""
    proprio_dim = model.extra_encoders["proprio"].in_chans
    action_dim = model.extra_encoders["action"].in_chans
    horizon = n_obs + pred_steps

    def r(*shape):
        return torch.randn(*shape, device=device)

    info = {
        "pixels": r(batch, candidates, n_obs, *cfg.obs_shape),
        "proprio": r(batch, candidates, n_obs, proprio_dim),
        "goal": r(batch, candidates, 1, *cfg.obs_shape),
        "goal_proprio": r(batch, candidates, 1, proprio_dim),
        "action": r(batch, candidates, n_obs, action_dim),  # placeholder; rollout overwrites
    }
    candidates_act = r(batch, candidates, horizon, action_dim)
    return info, candidates_act


def _clone(info: dict) -> dict:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()}


def sr_cost_parity(
    model: DINOv3PreJEPA,
    encode_fn: EncodeFn,
    predict_fn: PredictFn,
    cfg: ExportConfig,
    batch: int = 2,
    candidates: int = 4,
    n_obs: int | None = None,
    pred_steps: int = 2,
) -> dict:
    """Compare the shim's ``get_cost`` (encode/predict via ``encode_fn`` / ``predict_fn``)
    against the platform model's native ``get_cost`` on identical inputs, returning the max
    abs/rel drift on the ``(batch, candidates)`` cost tensor. Bit-for-bit (drift 0) when the
    fns are the adapter's pure-torch encode/predict — proving the subclass inherits the cost
    path unchanged; the drift row is the quantization signal when they are engines."""
    device = next(model.parameters()).device
    n_obs = model.history_size if n_obs is None else n_obs

    model.eval()
    # Fresh instances hold their own goal/init caches, so a single call needs no cache reset;
    # drop any stale cache defensively (the shim shares no state with the model).
    for m in (model,):
        for attr in ("_init_cached_info", "_goal_cached_info"):
            if hasattr(m, attr):
                delattr(m, attr)

    shim = DINOWMSRShim(model, encode_fn, predict_fn)
    shim.eval()

    info, cand = _make_info(model, cfg, batch, candidates, n_obs, pred_steps, device)

    with torch.no_grad():
        ref = model.get_cost(_clone(info), cand.clone())
        mine = shim.get_cost(_clone(info), cand.clone())

    if ref.shape != mine.shape:
        raise AssertionError(
            f"SR-parity shape mismatch: model cost {tuple(ref.shape)} vs shim {tuple(mine.shape)}"
        )
    diff = (ref.float() - mine.float()).abs()
    return {
        "shape": tuple(ref.shape),
        "max_abs": diff.max().item(),
        "max_rel": (diff / ref.float().abs().clamp_min(1e-12)).max().item(),
    }


def sr_cost_parity_lewm(
    model: LeWM,
    encode_fn: EncodeFn,
    predict_fn: LeWMPredictFn,
    cfg: ExportConfig,
    candidates: int = 4,
    n_obs: int | None = None,
    pred_steps: int = 2,
) -> dict:
    """Compare ``LeWMSRShim.get_cost`` (pixel-encode via ``encode_fn``, predict via ``predict_fn``)
    against ``LeWM.get_cost`` on identical inputs, returning the max abs/rel drift on the
    ``(1, candidates)`` cost. Bit-for-bit (drift 0) when the fns are the adapter's pure-torch
    ``encode`` / ``predict`` — proving the wholesale ``encode`` override AND the raw-action
    predict-engine boundary reproduce the source's cost path; the drift row is the quantization
    signal when they are engines.

    Runs at ``B=1`` (single env): the vendored CEM pins ``batch_size=1`` and ``LeWM.criterion``
    only supports ``B=1`` (it broadcasts the single-env goal over the candidate axis)."""
    device = next(model.parameters()).device
    n_obs = getattr(model.predictor, "num_frames", 3) if n_obs is None else n_obs
    horizon = n_obs + pred_steps
    action_dim = MODEL_ACTION_DIM

    model.eval()
    for attr in ("_init_cached_info", "_goal_cached_info"):
        if hasattr(model, attr):
            delattr(model, attr)

    shim = LeWMSRShim(model, encode_fn, predict_fn)
    shim.eval()

    def r(*shape):
        return torch.randn(*shape, device=device)

    # B=1 (single env per solve). LeWM cost needs pixels/goal/action only (no proprio/extras).
    info = {
        "pixels": r(1, candidates, n_obs, *cfg.obs_shape),
        "goal": r(1, candidates, 1, *cfg.obs_shape),
        "action": r(1, candidates, n_obs, action_dim),  # placeholder; rollout overwrites
    }
    cand = r(1, candidates, horizon, action_dim)

    with torch.no_grad():
        ref = model.get_cost(_clone(info), cand.clone())
        mine = shim.get_cost(_clone(info), cand.clone())

    if ref.shape != mine.shape:
        raise AssertionError(
            f"LeWM SR-parity shape mismatch: model {tuple(ref.shape)} vs shim {tuple(mine.shape)}"
        )
    diff = (ref.float() - mine.float()).abs()
    return {
        "shape": tuple(ref.shape),
        "max_abs": diff.max().item(),
        "max_rel": (diff / ref.float().abs().clamp_min(1e-12)).max().item(),
    }


def build_dummy_lewm_model() -> LeWM:
    """A REAL ``LeWM`` (so it exposes the native ``encode`` / ``rollout`` / ``get_cost`` the
    check compares against) wired from the shared ``src.smoke`` stand-in submodules — no
    ViT-Tiny download. Mirrors ``src.fidelity.build_dummy_dino_model``."""
    from src.smoke import build_dummy_lewm

    d = build_dummy_lewm()
    return LeWM(
        encoder=d.encoder,
        predictor=d.predictor,
        action_encoder=d.action_encoder,
        projector=d.projector,
        pred_proj=d.pred_proj,
    )


def _track_arg(argv, default: str = "both") -> str:
    for a in argv:
        if a.startswith("track="):
            return a.split("=", 1)[1]
    return default


def _run_dino_parity() -> None:
    import stable_worldmodel as swm

    from src.adapter import DINOWMAdapter

    model = swm.wm.utils.load_pretrained("dino/weights_epoch_10.pt")
    adapter = DINOWMAdapter(model)
    # n_obs = history_size drives the traced-HS predict window; n_obs = 1 drives the GROWING
    # sub-HS windows the eval actually feeds (1, 2, 3) — the variable-window coverage the
    # fixed-HS gate missed. Bit-for-bit on the torch path either way (the shim reproduces the
    # platform get_cost); on the pod `from_engines` path the sub-HS run also exercises the
    # predictor hist-adapt end-to-end.
    for n_obs in (model.history_size, 1):
        torch.manual_seed(0)
        result = sr_cost_parity(
            model, adapter.encode, adapter.predict, ExportConfig(), n_obs=n_obs
        )
        print(
            f"[dino] SR-cost parity n_obs={n_obs} (shim.get_cost vs PreJEPA.get_cost) "
            f"{result['shape']}: max_abs={result['max_abs']:.3e} max_rel={result['max_rel']:.3e}"
        )
        if result["max_abs"] > 1e-4:
            raise SystemExit(
                f"DINO SR-COST PARITY FAILED (n_obs={n_obs}): shim.get_cost diverges from "
                f"PreJEPA.get_cost (max_abs={result['max_abs']:.3e}) — the subclass overrides "
                "changed the cost path."
            )
    print("[dino] sr-cost parity: PASS")


def _run_lewm_parity() -> None:
    import stable_worldmodel as swm

    from src.adapter import LeWMAdapter

    model = swm.wm.utils.load_pretrained("lewm/weights_epoch_10.pt")
    adapter = LeWMAdapter(model)
    # n_obs = num_frames drives the traced-HS predict window; n_obs = 1 drives the GROWING sub-HS
    # windows the eval feeds (1, 2, 3) — the variable-window coverage the fixed-HS gate missed.
    n_frames = getattr(model.predictor, "num_frames", 3)
    for n_obs in (n_frames, 1):
        torch.manual_seed(0)
        result = sr_cost_parity_lewm(
            model, adapter.encode, adapter.predict, ExportConfig(), n_obs=n_obs
        )
        print(
            f"[lewm] SR-cost parity n_obs={n_obs} (shim.get_cost vs LeWM.get_cost) "
            f"{result['shape']}: max_abs={result['max_abs']:.3e} max_rel={result['max_rel']:.3e}"
        )
        # The encode override re-implements LeWM.encode's body and predict routes through the
        # engine boundary (raw action -> per-frame action_encoder); nonzero drift means one of
        # those diverges (a silent-SR bug) and must be fixed before any LeWM engine run.
        if result["max_abs"] > 1e-4:
            raise SystemExit(
                f"LeWM SR-COST PARITY FAILED (n_obs={n_obs}): shim.get_cost diverges from "
                f"LeWM.get_cost (max_abs={result['max_abs']:.3e}) — the encode override or "
                "predict boundary changed the cost path."
            )
    print("[lewm] sr-cost parity: PASS")


def main() -> None:
    """Run the SR-cost parity check(s) on the REAL checkpoint(s) (L40S) through the adapter's
    pure-torch encode/predict — the pre-engine gate that each shim's ``get_cost`` reproduces the
    platform's exactly before it is trusted to carry the engines' SR.

        uv run python -m src.sr_shim [track=dino|lewm|both]   (default: both)
    """
    track = _track_arg(sys.argv[1:])
    if track in ("dino", "both"):
        _run_dino_parity()
    if track in ("lewm", "both"):
        _run_lewm_parity()
    if track not in ("dino", "lewm", "both"):
        raise SystemExit(f"unknown track {track!r}; expected dino | lewm | both")


if __name__ == "__main__":
    main()
