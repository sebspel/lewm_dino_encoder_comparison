"""Adapter-fidelity gate for DINO-WM (before export).

The DINO-WM adapter's `predict` *reconstructs* `PreJEPA.predict` rather than calling it, and
the Python shim (`src.shim.dino_rollout`) reconstructs `PreJEPA.rollout` around it. A wrong
404 assembly, a wrong concat order, a dropped proprio channel, or a mis-wired action-carry
would pass the engine precision-match (which only compares engine-vs-adapter) yet silently
corrupt every SR. This gate closes that hole BEFORE any engine is built: it asserts the
adapter `encode` + `assemble_embedding` + `predict` + shim rollout reproduce
`DINOv3PreJEPA.rollout`'s predicted-embedding trajectory within tolerance.

Because both the reference (`model.rollout`) and the shim drive the SAME submodules
(same weights), a faithful shim matches bit-for-bit; the tolerance only absorbs float
reordering. This is float32-vs-float32 orchestration (NOT quantization), so the tolerance is
CLAUDE-owned and fails LOUDLY — distinct from the OWNER-gated FP16/INT8 precision tolerances.

Runs anywhere on dummy weights (a real `DINOv3PreJEPA` with a tiny random backbone — no
DINOv3 download); `main()` runs it on the real checkpoint on the L40S via `load_pretrained`.

    uv run python -m src.fidelity            # DINO-WM shim vs PreJEPA.rollout (real checkpoint)
    uv run python -m src.fidelity --lewm     # LeWM action_encoder per-frame guard (real checkpoint)

This module also carries the **LeWM action-encoder per-frame guard** (owner-signed-off
2026-07-11, `lewm_action_encoder_per_frame`): the silent-failure boundary that lets LeWM's
`action_encoder` live inside the compiled per-step `predict` engine (SPEC §Interface Contracts).
"""

from __future__ import annotations

import sys

import torch
from torch import Tensor, nn

from src.adapter import DINOWMAdapter
from src.interfaces import (
    DINO_LATENT_DIM,
    DINO_PREDICTOR_DIM,
    DINO_PROPRIO_DIM,
    HISTORY_SIZE,
    MODEL_ACTION_DIM,
    ExportConfig,
)
from src.shim import dino_rollout

# fp32-vs-fp32 orchestration: a faithful shim matches to float roundoff. Tight, CLAUDE-owned.
_RTOL = 1e-4
_ATOL = 1e-5

# Real checkpoints (epoch-10 .pt), same addresses the precision-match / eval paths use.
_CHECKPOINT = "dino/weights_epoch_10.pt"
_LEWM_CHECKPOINT = "lewm/weights_epoch_10.pt"


def dino_fidelity(
    model: nn.Module,
    adapter: DINOWMAdapter,
    cfg: ExportConfig,
    batch: int = 2,
    candidates: int = 4,
    pred_steps: int = 2,
) -> dict:
    """Roll out a synthetic Push-T-shaped input through both `model.rollout` (reference) and
    the adapter shim, and return the max abs/rel drift on the predicted-embedding trajectory.
    Dims are read from the loaded model (extra-encoder `in_chans`, `history_size`) so the same
    call works on the dummy and the real checkpoint. Random inputs are valid — the gate
    compares two implementations of the SAME function, not task quality."""
    device = next(model.parameters()).device
    n_obs = model.history_size
    proprio_dim = adapter.extra_encoders["proprio"].in_chans
    action_dim = adapter.extra_encoders["action"].in_chans

    pixels = torch.randn(batch, candidates, n_obs, *cfg.obs_shape, device=device)
    proprio = torch.randn(batch, candidates, n_obs, proprio_dim, device=device)
    horizon = n_obs + pred_steps
    actions = torch.randn(batch, candidates, horizon, action_dim, device=device)

    model.eval()
    adapter.eval()
    # Skip the rollout cache fast-path so the reference recomputes from scratch (we call once).
    if hasattr(model, "_init_cached_info"):
        del model._init_cached_info

    with torch.no_grad():
        ref: Tensor = model.rollout(
            {"pixels": pixels.clone(), "proprio": proprio.clone()}, actions.clone()
        )["predicted_embedding"]
        mine = dino_rollout(
            adapter,
            {"pixels": pixels.clone(), "proprio": proprio.clone()},
            actions.clone(),
            history_size=n_obs,
        )

    if ref.shape != mine.shape:
        raise AssertionError(
            f"fidelity shape mismatch: reference {tuple(ref.shape)} vs shim {tuple(mine.shape)}"
        )
    diff = (ref.float() - mine.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / ref.float().abs().clamp_min(1e-12)).max().item()
    passed = torch.allclose(mine.float(), ref.float(), rtol=_RTOL, atol=_ATOL)
    return {
        "shape": tuple(ref.shape),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "passed": passed,
    }


# LeWM per-frame action-encoder guard (owner sign-off 2026-07-11).
# LeWM.rollout pre-encodes the WHOLE action sequence once; LeWMAdapter.predict re-encodes each
# per-step window inside the engine. These agree — so the action_encoder may live inside the
# compiled per-step predict engine — iff the encoder is per-frame: output at step t depends only
# on the action at t, with no receptive field along the macro-step (T) axis. LeWM's Embedder is a
# Conv1d(kernel_size=1) + per-position MLP, so it is per-frame; a kernel_size>1 (temporal-
# smoothing) config would leak neighbouring/future actions into act_emb[:, t] and silently break
# per-step faithfulness. This asserts the property directly on the real weights (SPEC §Interface
# Contracts): action_encoder(seq)[:, t] must equal action_encoder(seq[:, :t+1])[:, -1] at every t.
_ACT_RTOL = 1e-4
_ACT_ATOL = 1e-5


def lewm_action_encoder_per_frame(
    action_encoder: nn.Module,
    seq_len: int = 5,
    batch: int = 2,
    action_dim: int = MODEL_ACTION_DIM,
) -> dict:
    """Self-guarding check: encoding step t within the full sequence must equal encoding it as
    the last step of the prefix ending at t. Exact for a per-frame encoder (float roundoff);
    a temporal kernel makes the two disagree because the full-sequence step sees future actions
    the prefix cannot. Random actions are valid — it compares the encoder against itself."""
    device = next(action_encoder.parameters()).device
    seq = torch.randn(batch, seq_len, action_dim, device=device)
    action_encoder.eval()
    passed = True
    max_abs = 0.0
    with torch.no_grad():
        full: Tensor = action_encoder(seq)  # (B, T, D)
        for t in range(seq_len):
            step = action_encoder(seq[:, : t + 1])[:, -1]  # (B, D) — last step of the prefix
            max_abs = max(max_abs, (full[:, t] - step).abs().max().item())
            passed = passed and torch.allclose(full[:, t], step, rtol=_ACT_RTOL, atol=_ACT_ATOL)
    return {"seq_len": seq_len, "max_abs": max_abs, "passed": passed}


def build_dummy_dino_model() -> nn.Module:
    """A REAL `DINOv3PreJEPA` (so it exposes the native `rollout`/`replace_action_in_embedding`
    the gate compares against) wired with a tiny random backbone + the real `Embedder` extras
    (which carry `emb_dim`, needed by `replace_action_in_embedding`) + a dim-preserving linear
    predictor stand-in. No DINOv3 download — exercises the full orchestration on CPU."""
    from stable_worldmodel.wm.prejepa.module import Embedder

    from src.dino_patch import DINOv3PreJEPA
    from src.smoke import _PatchEncoder

    return DINOv3PreJEPA(
        encoder=_PatchEncoder(DINO_LATENT_DIM, num_register_tokens=4),
        predictor=nn.Linear(DINO_PREDICTOR_DIM, DINO_PREDICTOR_DIM),
        extra_encoders=nn.ModuleDict(
            {
                "proprio": Embedder(in_chans=DINO_PROPRIO_DIM, emb_dim=10),
                "action": Embedder(in_chans=MODEL_ACTION_DIM, emb_dim=10),
            }
        ),
        history_size=HISTORY_SIZE,
        num_pred=1,
        interpolate_pos_encoding=True,
    )


def _run_dino(dummy: bool) -> None:
    """Run the DINO-WM adapter-fidelity gate on the REAL checkpoint (L40S) via load_pretrained."""
    if dummy:
        model = build_dummy_dino_model()
    else:
        import stable_worldmodel as swm

        model = swm.wm.utils.load_pretrained(_CHECKPOINT)

    torch.manual_seed(0)
    adapter = DINOWMAdapter(model)
    result = dino_fidelity(model, adapter, ExportConfig())
    print(
        f"[dino] adapter-fidelity vs PreJEPA.rollout {result['shape']}: "
        f"max_abs={result['max_abs']:.3e} max_rel={result['max_rel']:.3e} "
        f"(rtol={_RTOL}, atol={_ATOL})"
    )
    if not result["passed"]:
        raise SystemExit(
            f"ADAPTER-FIDELITY GATE FAILED: shim diverges from PreJEPA.rollout "
            f"(max_abs={result['max_abs']:.3e}) — the 404 assembly/carry is wrong; do NOT "
            f"export until this passes."
        )
    print("adapter-fidelity: PASS")


def _run_lewm(dummy: bool) -> None:
    """Run the LeWM action-encoder per-frame guard on the REAL checkpoint (L40S)."""
    if dummy:
        from src.smoke import build_dummy_lewm

        model = build_dummy_lewm()
    else:
        import stable_worldmodel as swm

        model = swm.wm.utils.load_pretrained(_LEWM_CHECKPOINT)

    torch.manual_seed(0)
    result = lewm_action_encoder_per_frame(model.action_encoder)
    print(
        f"[lewm] action_encoder per-frame (seq_len={result['seq_len']}): "
        f"max_abs={result['max_abs']:.3e} (rtol={_ACT_RTOL}, atol={_ACT_ATOL})"
    )
    if not result["passed"]:
        raise SystemExit(
            f"LEWM PER-FRAME GUARD FAILED: action_encoder mixes across the macro-step axis "
            f"(max_abs={result['max_abs']:.3e}) — per-step predict is NOT faithful to "
            f"LeWM.rollout's whole-sequence act-encode; do NOT put the action encoder inside "
            f"the predict engine (kernel_size>1?)."
        )
    print("lewm action-encoder per-frame: PASS")


def main() -> None:
    """`--lewm` runs the LeWM per-frame guard; otherwise the DINO-WM fidelity gate. `--dummy`
    exercises either path on random weights off-pod."""
    args = sys.argv[1:]
    dummy = "--dummy" in args
    if "--lewm" in args:
        _run_lewm(dummy)
    else:
        _run_dino(dummy)


if __name__ == "__main__":
    main()
