"""Python rollout/shim reproducing `PreJEPA.rollout` via the owned DINO-WM adapter.

The DINO-WM adapter's `predict` *reconstructs* the platform forward (it is not the platform's
own method), so the autoregressive orchestration around it — the 384->404 assembly, the
per-step action-replacement, the proprio carry, and the history windowing — has to be
reproduced in Python, NOT compiled into the engine. This module
is that reproduction, expressed against the adapter's `encode` / `assemble_embedding` /
`predict` so the SAME orchestration drives the PyTorch adapter (fidelity gate, `src.fidelity`)
and, later, the exported engines (the benchmark SR shim).

It is a faithful line-map of `stable_worldmodel.wm.prejepa.prejepa.PreJEPA.rollout` /
`.replace_action_in_embedding`; the fidelity gate asserts it matches `DINOv3PreJEPA.rollout`
bit-for-bit on the real checkpoint (a wrong assembly/carry passes engine precision-match yet
silently corrupts every SR — the gate is what catches it).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from einops import rearrange, repeat


def _replace_action(extra_encoders: nn.ModuleDict, embedding: Tensor, act: Tensor) -> Tensor:
    """Verbatim port of `PreJEPA.replace_action_in_embedding` (5-D `(B, N, T, P, d)`): re-embed
    the action, tile it across patches, and splice it back over ONLY the action channels —
    keeping the pixel and (predicted) proprio channels. The action block is located by summing
    the `emb_dim`s of the extras preceding 'action' in key order (the SILENTLY-failing concat
    order)."""
    n_patches = embedding.shape[3]
    B, N = act.shape[:2]
    act_flat = rearrange(act, "b n ... -> (b n) ...")
    z_act = extra_encoders["action"](act_flat)
    action_dim = z_act.shape[-1]
    act_tiled = repeat(
        z_act.unsqueeze(2), "(b n) t 1 a -> b n t p a", b=B, n=N, p=n_patches
    )
    extra_dim = sum(enc.emb_dim for enc in extra_encoders.values())
    pixel_dim = embedding.shape[-1] - extra_dim
    start = pixel_dim
    for key, encoder in extra_encoders.items():
        if key == "action":
            break
        start += encoder.emb_dim
    prefix = embedding[..., :start]
    suffix = embedding[..., start + action_dim :]
    return torch.cat([prefix, act_tiled, suffix], dim=-1)


def dino_rollout(adapter, info: dict, action_sequence: Tensor, history_size: int) -> Tensor:
    """Faithful port of `PreJEPA.rollout` driven by the DINO-WM adapter (encode-once /
    predict-many). Returns the predicted-embedding trajectory `(B, N, n_obs+n_steps+1, P, 404)`
    — the platform's `info['predicted_embedding']`. The caching fast-path is intentionally
    omitted (the shim encodes once per call); every other step mirrors the source.

    Args:
        adapter: a `DINOWMAdapter` (exposes `encode`, `assemble_embedding`, `predict`,
            `extra_encoders`).
        info: dict with `pixels` `(B, N, n_obs, C, H, W)` and `proprio` `(B, N, n_obs, dp)`;
            `action` is set here from `action_sequence` (mirroring the source).
        action_sequence: `(B, N, n_obs+n_steps, action_in_chans)`.
        history_size: the predictor window (`PreJEPA.history_size`).
    """
    extra_encoders = adapter.extra_encoders
    n_obs = info["pixels"].shape[2]
    N = action_sequence.shape[1]

    act_0 = action_sequence[:, :, :n_obs]
    info["action"] = act_0

    # --- initial embedding: encode candidate 0, assemble 384->404, expand over candidates.
    init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
    pixels_emb = adapter.encode(init["pixels"])  # (B, n_obs, P, 384)
    emb = adapter.assemble_embedding(
        pixels_emb, init["proprio"], init["action"]
    )  # (B, n_obs, P, 404)
    emb = emb.unsqueeze(1).expand(-1, N, *([-1] * (emb.ndim - 1))).clone()  # (B, N, n_obs, P, 404)

    # replace the (candidate-0) action channels with each candidate's own act_0.
    emb = _replace_action(extra_encoders, emb, act_0)

    act_pred = action_sequence[:, :, n_obs:]
    n_steps = act_pred.shape[2]

    B = emb.shape[0]
    z_flat = rearrange(emb, "b n ... -> (b n) ...").clone()  # (B*N, n_obs, P, 404)
    act_pred_flat = rearrange(act_pred, "b n ... -> (b n) ...")  # (B*N, n_steps, in_chans)

    for t in range(n_steps):
        pred_embed = adapter.predict(z_flat[:, -history_size:])[:, -1:]  # (B*N, 1, P, 404)
        new_action = act_pred_flat[None, :, t : t + 1, :]  # (1, B*N, 1, in_chans)
        new_embed = _replace_action(extra_encoders, pred_embed.unsqueeze(0), new_action)[0]
        z_flat = torch.cat([z_flat, new_embed], dim=1)

    # final predicted state (n+t+1), no action to replace.
    pred_embed = adapter.predict(z_flat[:, -history_size:])[:, -1:]
    z_flat = torch.cat([z_flat, pred_embed], dim=1)

    return rearrange(z_flat, "(b n) ... -> b n ...", b=B, n=N)
