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

from src.dino_patch import DINOv3PreJEPA
from src.interfaces import DINO_PREDICTOR_DIM, EnginePaths, ExportConfig

# (B, T, C, H, W) -> (B, T, 196, 384): the register-sliced patch grid, same contract as
# DINOv3PreJEPA._encode_image. (B, T, P, 404) -> (B, T, P, 404): the dim-preserving predictor.
EncodeFn = Callable[[Tensor], Tensor]
PredictFn = Callable[[Tensor], Tensor]


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


def build_engine_fns(engines: EnginePaths) -> tuple[EncodeFn, PredictFn]:
    """Wrap the encoder + predictor engines as ``encode`` / ``predict`` callables.

    Pod-only: ``EngineRunner`` lazy-imports ``tensorrt`` and allocates CUDA buffers, so this
    is imported lazily to keep ``src.sr_shim`` importable off-pod (tests use the adapter path).

    NOTE (encoder hist axis — pod/owner export dependency): the engines are traced with a
    dynamic batch axis but a **static hist axis** (``ExportConfig.hist``). The inherited
    ``get_cost`` calls the encoder twice with different frame counts — the initial state at
    ``n_obs`` frames and the goal at ``1`` frame — so the encoder engine must accept both.
    These callables feed the engine the ``(B, T, …)`` tensor as-is; a frame count that differs
    from the traced hist raises a loud TensorRT shape error (never a silent wrong SR). Exporting
    the encoder with a dynamic hist axis is the owner-gated follow-up if the eval's real
    ``n_obs`` / goal frame counts differ from the traced value.
    """
    from src.trt_runtime import EngineRunner

    encoder = EngineRunner(engines["encoder"])
    predictor = EngineRunner(engines["predictor"])

    def encode_fn(pixels: Tensor) -> Tensor:
        return encoder.run((pixels,))

    def predict_fn(embedding: Tensor) -> Tensor:
        return predictor.run((embedding,))

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


def main() -> None:
    """Run the SR-cost parity check on the REAL DINO-WM checkpoint (L40S) through the adapter's
    pure-torch encode/predict — the pre-engine gate that the shim's inherited ``get_cost``
    reproduces the platform's exactly before it is trusted to carry the engines' SR."""
    import stable_worldmodel as swm

    from src.adapter import DINOWMAdapter

    model = swm.wm.utils.load_pretrained("dino/weights_epoch_10.pt")
    adapter = DINOWMAdapter(model)

    torch.manual_seed(0)
    result = sr_cost_parity(model, adapter.encode, adapter.predict, ExportConfig())
    print(
        f"[dino] SR-cost parity (shim.get_cost vs PreJEPA.get_cost) {result['shape']}: "
        f"max_abs={result['max_abs']:.3e} max_rel={result['max_rel']:.3e}"
    )
    # Same submodules on both sides -> bit-for-bit; a nonzero drift means the subclass wiring
    # altered the inherited cost path (a silent-SR bug) and must be fixed before any engine run.
    if result["max_abs"] > 1e-4:
        raise SystemExit(
            f"SR-COST PARITY FAILED: shim.get_cost diverges from PreJEPA.get_cost "
            f"(max_abs={result['max_abs']:.3e}) — the subclass overrides changed the cost path."
        )
    print("sr-cost parity: PASS")


if __name__ == "__main__":
    main()
