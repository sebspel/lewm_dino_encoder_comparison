"""Tracer bullet (Phase 4) — the sole pre-optimization integration check.

Flows a dummy checkpoint through the owned layer end-to-end on random weights:
    dummy model -> adapter.encode -> adapter.predict (on the cached latent)
                -> export stub (encoder + predictor engines) -> benchmark stub
asserting the typed latent shapes at every owned boundary for both tracks. Run on CPU
with tiny random modules — dims come from the encoder architecture (not training), so this
is valid pre-checkpoint (SPEC §Requirements: tracer bullet on dummy weights, typed checks
at every boundary).

The dummy builders are shared with tests/. They are *stand-ins* with the platform module
interfaces (`.encoder(x).last_hidden_state`, `.predictor(...)`, etc.) — not the real
backbones — so the boundary shapes are exercised without downloading DINOv3.
"""

from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch
from torch import nn

from src.interfaces import (
    LATENT_DIM,
    DINO_N_PATCHES,
    DINO_LATENT_DIM,
    DINO_PREDICTOR_DIM,
    MODEL_ACTION_DIM,
    DINO_PROPRIO_DIM,
    HISTORY_SIZE,
    ExportConfig,
)
from src.adapter import LeWMAdapter, DINOWMAdapter
from src.export import export
from src.benchmark import benchmark

_EXTRA_DIM = DINO_PREDICTOR_DIM - DINO_LATENT_DIM  # 20 = proprio 10 + action 10


class _PatchEncoder(nn.Module):
    """Conv patchifier that mimics a ViT: returns `.last_hidden_state` of prepended
    special tokens (CLS + registers) followed by a 14x14 patch grid."""

    def __init__(self, dim: int, num_register_tokens: int):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=16, stride=16)  # 224/16 -> 14
        self.specials = nn.Parameter(torch.randn(1, 1 + num_register_tokens, dim))
        self.config = SimpleNamespace(
            hidden_size=dim, num_register_tokens=num_register_tokens
        )

    def forward(self, pixels, **kwargs):
        grid = self.patch(pixels).flatten(2).transpose(1, 2)  # (n, 196, dim)
        specials = self.specials.expand(grid.shape[0], -1, -1)
        tokens = torch.cat([specials, grid], dim=1)
        return SimpleNamespace(last_hidden_state=tokens)


def build_dummy_lewm() -> nn.Module:
    """LeWM stand-in: CLS-token ViT-Tiny encoder + AdaLN-conditioned predictor."""

    class DummyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(LATENT_DIM, LATENT_DIM)

        def forward(self, x, c):  # x, c: (b, t, D)
            return self.net(x + c)

    return SimpleNamespace(
        encoder=_PatchEncoder(LATENT_DIM, num_register_tokens=0),
        projector=nn.Linear(LATENT_DIM, LATENT_DIM),
        action_encoder=nn.Linear(MODEL_ACTION_DIM, LATENT_DIM),  # 10-wide frameskip action
        predictor=DummyPredictor(),
        pred_proj=nn.Linear(LATENT_DIM, LATENT_DIM),
    )


def build_dummy_dino() -> nn.Module:
    """DINOv3-WM stand-in: register-slicing patch encoder + 404-wide concat predictor, with
    the real ModuleDict of proprio/action extra-encoders (insertion order proprio→action, the
    order the concat must follow to reach 404)."""

    return SimpleNamespace(
        backbone=_PatchEncoder(DINO_LATENT_DIM, num_register_tokens=4),
        predictor=nn.Linear(DINO_PREDICTOR_DIM, DINO_PREDICTOR_DIM),
        extra_encoders=nn.ModuleDict(
            {
                "proprio": nn.Linear(DINO_PROPRIO_DIM, _EXTRA_DIM // 2),  # 4 -> 10
                "action": nn.Linear(MODEL_ACTION_DIM, _EXTRA_DIM // 2),  # 10 -> 10
            }
        ),
    )


def _run_track(
    name: str, adapter, latent_shape_no_batch: tuple[int, ...], conditioning: tuple
) -> None:
    b, t = 2, HISTORY_SIZE
    obs = torch.randn(b, t, 3, 224, 224)

    latent = adapter.encode(obs)  # typed boundary
    assert latent.shape == (b, t, *latent_shape_no_batch), latent.shape

    # predict on the cached latent + the per-track conditioning (LeWM: action; DINO:
    # proprio, action) — the exact tuple export/benchmark trace and drive.
    predict_inputs = (latent, *conditioning)
    nxt = adapter.predict(*predict_inputs)  # typed boundary
    assert nxt.shape == latent.shape, nxt.shape

    cfg = ExportConfig()
    with tempfile.TemporaryDirectory() as d:
        engines = export(
            adapter,
            precision="fp32",
            encode_inputs=(obs,),
            predict_inputs=predict_inputs,
            engine_dir=Path(d) / name,
        )
        result = benchmark(
            engines,
            encode_inputs=(obs,),
            predict_inputs=predict_inputs,
            time_budget_s=cfg.time_budget_s,
            warmup=cfg.warmup,
        )
    assert set(result) == {
        "latency_p50_ms",
        "latency_p95_ms",
        "rollouts_completed",
        "throughput",
        "peak_mem_mb",
        "success_rate",
    }
    print(f"[{name}] encode {tuple(latent.shape)} -> predict {tuple(nxt.shape)}; "
          f"engines={ {k: v.name for k, v in engines.items()} } OK")


def main() -> None:
    torch.manual_seed(0)
    b, t = 2, HISTORY_SIZE
    action = torch.randn(b, t, MODEL_ACTION_DIM)
    proprio = torch.randn(b, t, DINO_PROPRIO_DIM)
    _run_track("lewm", LeWMAdapter(build_dummy_lewm()), (LATENT_DIM,), (action,))
    _run_track(
        "dino",
        DINOWMAdapter(build_dummy_dino()),
        (DINO_N_PATCHES, DINO_LATENT_DIM),
        (proprio, action),
    )
    print("smoke: PASS")


if __name__ == "__main__":
    main()
