"""Register-aware encode override for the DINOv3-WM track (Phase 2).

The platform's `PreJEPA._encode_image` drops only the CLS token
(`last_hidden_state[:, 1:, :]`). DINOv3 prepends **CLS + N register tokens**, so that
slice leaves `1 + num_register_tokens` extra tokens in the grid. The predictor is sized
from the config-derived `num_patches = (image_size // patch_size) ** 2` (196 at
patch_size=16), so the extra tokens *silently* misalign the predictor's positional
embedding and causal time-block mask (docs/platform_api.md §2, §6).

This subclass overrides **only** `_encode_image` to also slice off the register tokens,
yielding the true 196-patch grid. Everything else (predictor, prejepa losses, rollout,
planning) is inherited unchanged. It is injected via Hydra `model._target_`
(conf/experiment/dinov3.yaml) so the platform wheel is never edited, and it is imported
by both the train and eval entrypoints so the predictor plans over the same grid it was
trained on (PLAN Phase 2).
"""

import torch
from einops import rearrange

from stable_worldmodel.wm.prejepa.prejepa import PreJEPA


class DINOv3PreJEPA(PreJEPA):
    def _encode_image(self, pixels):
        # Faithful copy of PreJEPA._encode_image; the only change is the register slice.
        B = pixels.shape[0]
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')

        kwargs = (
            {'interpolate_pos_encoding': True}
            if self.interpolate_pos_encoding
            else {}
        )
        pixels_embed = self.backbone(pixels, **kwargs)

        if hasattr(pixels_embed, 'last_hidden_state'):
            pixels_embed = pixels_embed.last_hidden_state
            # Layout: [CLS] + [register] * num_reg + [patch] * N. Platform drops CLS
            # only ([:, 1:, :]); we also drop the registers (docs §6). Read num_reg
            # from the backbone config so a non-register backbone degrades to CLS-only.
            num_reg = getattr(self.backbone.config, 'num_register_tokens', 0)
            pixels_embed = pixels_embed[:, 1 + num_reg:, :]
        else:
            pixels_embed = pixels_embed.logits.unsqueeze(1)

        pixels_embed = rearrange(
            pixels_embed.detach(), '(b t) p d -> b t p d', b=B
        )

        return pixels_embed
