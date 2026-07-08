"""Owned two-method adapters over the platform world models (Phase 4).

One shared boundary — `encode(obs) -> latent`, `predict(latent, *conditioning) -> latent` —
with two concrete implementations so the export/benchmark plumbing never branches:

    LeWMAdapter   single-token latent  (B, T, 192)        — action via AdaLN conditioning
    DINOWMAdapter patch-grid latent    (B, T, 196, 384)   — dim-preserving 404->404 predict

Each adapter *wraps* the model's encoder + predictor (it calls those submodules; it does
not reimplement the predictor/encoder internals — CLAUDE.md §8). The CEM rollout loop
stays in Python outside the adapter; `predict` is a single autoregressive predictor step,
exactly the unit TensorRT will optimize (SPEC §Interface Contracts). Every boundary is
jaxtyping+beartype checked so a shape violation raises.

The two tracks feed the action differently. LeWM `predict` ingests the *model-facing*
frameskip action pack (`MODEL_ACTION_DIM=10`, not the env `ACTION_DIM=2` the CEM plans over)
as an AdaLN-conditioning arg. DINO-WM `predict` mirrors `PreJEPA.predict` — a faithful,
dim-preserving 404->404 step over the pre-assembled `(pixels 384 | proprio 10 | action 10)`
embedding; the 384->404 assembly (`assemble_embedding`) and the per-step action-replacement
live in the Python rollout/shim, NOT the compiled `predict` (SPEC §Interface Contracts). The
env->model action packing lives in the CEM shim outside the adapter.
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
        # CLS token -> projector, per LeWM.encode (which passes interpolate_pos_encoding=True
        # so the from-scratch ViT-Tiny pos-embeddings match the 224px grid — omitting it
        # shifts the CLS embedding and shows up as precision-match drift).
        b, t = obs.shape[:2]
        flat = obs.reshape(b * t, *obs.shape[2:])
        cls = self.encoder(flat, interpolate_pos_encoding=True).last_hidden_state[:, 0]
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
    """Wraps a DINOv3-WM model: register-slicing patch encoder + dim-preserving 404 predictor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.predictor = model.predictor
        # Real PreJEPA keeps the proprio + action embedders in a ModuleDict (NOT a bare
        # `action_encoder`); each Embedder maps its extra to a 10-wide code that is tiled
        # across patches and concatenated onto the 384 latent to reach the 404 predictor
        # input. Order of the concat is the SILENTLY-failing boundary (SPEC §Impl Boundaries).
        self.extra_encoders = model.extra_encoders  # ModuleDict{proprio: 4->10, action: 10->10}
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
        embedding: Float[Tensor, "batch hist patch pred_dim"],
    ) -> Float[Tensor, "batch hist patch pred_dim"]:
        # Faithful mirror of PreJEPA.predict: run the dim-preserving causal predictor over
        # the flattened (hist*patch) token axis, then reshape back. Output width == input
        # width == DINO_PREDICTOR_DIM (404); the predicted proprio channels are KEPT (not
        # sliced to 384) — they are load-bearing for the CEM criterion and the autoregressive
        # carry (SPEC §Interface Contracts). The 384->404 assembly and the per-step
        # action-replacement live in the Python rollout/shim (`assemble_embedding` /
        # `replace_action_in_embedding`), never in this compiled step.
        b, t, p, d = embedding.shape
        assert d == DINO_PREDICTOR_DIM, (
            f"predict input width {d} != {DINO_PREDICTOR_DIM}"
        )
        preds = self.predictor(embedding.reshape(b, t * p, d))
        return preds.reshape(b, t, p, d)

    @typed
    def assemble_embedding(
        self,
        latent: Float[Tensor, "batch hist patch latent"],
        proprio: Float[Tensor, "batch hist proprio"],
        action: Float[Tensor, "batch hist action"],
    ) -> Float[Tensor, "batch hist patch pred_dim"]:
        # Python-side (NOT compiled): mirror PreJEPA.encode's extra assembly. Each extra is
        # embedded by its Embedder, tiled across the patch axis, and concatenated onto the
        # 384 pixel latent on the feature axis to reach DINO_PREDICTOR_DIM (404). Extras are
        # concatenated in `extra_encoders` key order (the order the trained predictor
        # learned); a wrong order is a plausible-but-wrong SR with NO error (SPEC §Impl
        # Boundaries), so each input is matched to its encoder by name, not position.
        b, t, p, _ = latent.shape
        extras = {"proprio": proprio, "action": action}
        embedding = latent
        for key in self.extra_encoders:
            extra_embed = self.extra_encoders[key](extras[key])  # (b, t, emb_dim)
            extra_tiled = extra_embed.unsqueeze(2).expand(b, t, p, extra_embed.shape[-1])
            embedding = torch.cat([embedding, extra_tiled], dim=-1)  # (b, t, p, 404)
        assert embedding.shape[-1] == DINO_PREDICTOR_DIM, (
            f"assembled width {embedding.shape[-1]} != {DINO_PREDICTOR_DIM}"
        )
        return embedding
