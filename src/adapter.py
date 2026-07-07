"""Owned two-method adapters over the platform world models (Phase 4).

One shared boundary — `encode(obs) -> latent`, `predict(latent, action) -> latent` —
with two concrete implementations so the export/benchmark plumbing never branches:

    LeWMAdapter   single-token latent  (B, T, 192)        — action via AdaLN conditioning
    DINOWMAdapter patch-grid latent    (B, T, 196, 384)   — action concatenated to 404

Each adapter *wraps* the model's encoder + predictor (it calls those submodules; it does
not reimplement the predictor/encoder internals — CLAUDE.md §8). The CEM rollout loop
stays in Python outside the adapter; `predict` is a single autoregressive predictor step
over the history window, exactly the unit TensorRT will optimize (SPEC §Interface
Contracts). Every boundary is jaxtyping+beartype checked so a shape violation raises.

The action boundary is the env `ACTION_DIM` (SPEC constant); the models' frameskip action
packing (5×2) and the proprio conditioning are folded into each adapter's action-embedding
submodule here and wired to the real modules in Phase 5.
"""

import torch
from torch import Tensor, nn
from jaxtyping import Float, jaxtyped
from beartype import beartype

from src.interfaces import DINO_PREDICTOR_DIM

typed = jaxtyped(typechecker=beartype)


class LeWMAdapter(nn.Module):
    """Wraps a LeWM model: CLS-token encoder + AdaLN-conditioned predictor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.encoder = model.encoder
        self.projector = model.projector
        self.action_encoder = model.action_encoder
        self.predictor = model.predictor
        self.pred_proj = model.pred_proj

    @typed
    def encode(
        self,
        obs: Float[Tensor, "batch hist channel height width"],
    ) -> Float[Tensor, "batch hist latent"]:
        # CLS token -> projector, per LeWM.encode.
        b, t = obs.shape[:2]
        flat = obs.reshape(b * t, *obs.shape[2:])
        cls = self.encoder(flat).last_hidden_state[:, 0]  # (b*t, D)
        emb = self.projector(cls)
        return emb.reshape(b, t, -1)

    @typed
    def predict(
        self,
        latent: Float[Tensor, "batch hist latent"],
        action: Float[Tensor, "batch hist action_dim"],
    ) -> Float[Tensor, "batch hist latent"]:
        # Action enters as the AdaLN-conditioning arg, per LeWM.predict.
        b, t = latent.shape[:2]
        act_emb = self.action_encoder(action)  # (b, t, D)
        preds = self.predictor(latent, act_emb)  # (b, t, D)
        preds = self.pred_proj(preds.reshape(b * t, -1))
        return preds.reshape(b, t, -1)


class DINOWMAdapter(nn.Module):
    """Wraps a DINOv3-WM model: register-slicing patch encoder + concat-action predictor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.predictor = model.predictor
        # Embeds the action (+ proprio, in Phase 5) into the 20-wide extras concatenated
        # onto the 384 latent to reach the 404 predictor-input width.
        self.action_encoder = model.action_encoder
        self.num_register_tokens = getattr(
            self.backbone.config, "num_register_tokens", 0
        )

    @typed
    def encode(
        self,
        obs: Float[Tensor, "batch hist channel height width"],
    ) -> Float[Tensor, "batch hist patch latent"]:
        # Drop CLS + register tokens -> true patch grid, per DINOv3PreJEPA._encode_image.
        b, t = obs.shape[:2]
        flat = obs.reshape(b * t, *obs.shape[2:])
        tokens = self.backbone(flat).last_hidden_state[:, 1 + self.num_register_tokens :]
        return tokens.reshape(b, t, tokens.shape[1], tokens.shape[2])

    @typed
    def predict(
        self,
        latent: Float[Tensor, "batch hist patch latent"],
        action: Float[Tensor, "batch hist action_dim"],
    ) -> Float[Tensor, "batch hist patch latent"]:
        # Tile the action embedding across patches and concatenate on the feature axis,
        # widening tokens to DINO_PREDICTOR_DIM (404); the predictor is dim-preserving so
        # we slice the pixel latent (384) back out, per PreJEPA.rollout/predict.
        b, t, p, d = latent.shape
        extras = self.action_encoder(action)  # (b, t, extra_dim)
        extras_tiled = extras.unsqueeze(2).expand(b, t, p, extras.shape[-1])
        tokens = torch.cat([latent, extras_tiled], dim=-1)  # (b, t, p, 404)
        assert tokens.shape[-1] == DINO_PREDICTOR_DIM, (
            f"predictor input width {tokens.shape[-1]} != {DINO_PREDICTOR_DIM}"
        )
        preds = self.predictor(tokens.reshape(b, t * p, tokens.shape[-1]))
        return preds.reshape(b, t, p, -1)[..., :d]
