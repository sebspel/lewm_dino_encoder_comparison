"""Phase-2 encode sanity check (owned, fails loud).

Confirms both tracks' latent dims match Phase-1 (docs/platform_api.md §1–2) BEFORE
trusting any trained checkpoint:

  LeWM     -> single-token latent (B, 192)              [platform code, unchanged]
  DINO-WM  -> full patch grid     (B, T, 196, 384)      [our register-slice override]

The load-bearing assertion is the DINO-WM grid being **196**, not 200: if the
`DINOv3PreJEPA._encode_image` override is not active, the base `PreJEPA` slice
(`[:, 1:, :]`) leaves the 4 register tokens in, yielding 200 tokens that silently
misalign the predictor. We prove the override does real work with a differential:
the same backbone, the two encode methods, 196 vs 200.

The encode path depends only on the encoder, so we build just the real backbone (as
the entrypoints do) and call the encode methods directly — no predictor, no dataset,
no checkpoint. Run on the pod after `setup.sh` (downloads the DINOv3 weights):

    uv run python -m scripts.verify_encode
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def check_dino():
    from stable_worldmodel.wm.prejepa.module import create_backbone
    from stable_worldmodel.wm.prejepa.prejepa import PreJEPA
    from src.dino_patch import DINOv3PreJEPA

    # --- config wiring: the overlay must select the register-slice subclass + /16 ---
    overlay = OmegaConf.load(ROOT / "conf/experiment/dinov3.yaml")
    base = OmegaConf.load(ROOT / "scripts/train/config/prejepa.yaml")
    assert overlay.model._target_ == "src.dino_patch.DINOv3PreJEPA", overlay.model._target_
    assert overlay.patch_size == 16, overlay.patch_size
    image_size = base.image_size
    expected_patches = (image_size // overlay.patch_size) ** 2  # (224//16)**2 = 196

    # --- real backbone, built exactly as the entrypoint does (create_backbone(name=)) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = create_backbone(name=overlay.backbone.name).to(device).eval()
    D = encoder.config.hidden_size
    num_reg = getattr(encoder.config, "num_register_tokens", 0)
    assert D == 384, D
    assert num_reg == 4, num_reg  # docs §2: 1 CLS + 4 reg + 196 patch = 201

    # `_encode_image` reads only `self.backbone` / `self.interpolate_pos_encoding`, so a
    # stub carrying those two lets us call the real (unbound) methods without building the
    # whole world model (which would need the dataset for extra-encoder dims).
    stub = type("Stub", (), {"backbone": encoder, "interpolate_pos_encoding": True})()
    pixels = torch.randn(2, 3, 3, image_size, image_size, device=device)  # (B, T, C, H, W)

    with torch.no_grad():
        ours = DINOv3PreJEPA._encode_image(stub, pixels)   # override -> 196
        theirs = PreJEPA._encode_image(stub, pixels)       # base     -> 200

    assert ours.shape == (2, 3, expected_patches, D), ours.shape
    # differential: dropping the registers removes exactly num_reg tokens
    assert theirs.shape[-2] - ours.shape[-2] == num_reg, (theirs.shape, ours.shape)
    print(f"[DINO-WM] override grid {tuple(ours.shape)} vs base {tuple(theirs.shape)} "
          f"(D={D}, num_reg={num_reg}) -> {expected_patches} patches OK")


def check_lewm():
    from stable_pretraining.backbone.utils import vit_hf

    cfg = OmegaConf.load(ROOT / "scripts/train/config/lewm.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = vit_hf(
        size=cfg.encoder_scale, patch_size=cfg.patch_size,
        image_size=cfg.img_size, pretrained=False, use_mask_token=False,
    ).to(device).eval()

    pixels = torch.randn(2, 3, cfg.img_size, cfg.img_size, device=device)  # (B, C, H, W)
    with torch.no_grad():
        # LeWM.encode takes the CLS token: last_hidden_state[:, 0] -> (B, D) (docs §1)
        cls = encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]

    assert cls.shape == (2, cfg.embed_dim), cls.shape  # single token, D=192
    print(f"[LeWM] CLS latent {tuple(cls.shape)} -> single token, D={cfg.embed_dim} OK")


if __name__ == "__main__":
    check_dino()
    check_lewm()
    print("encode sanity: PASS")
