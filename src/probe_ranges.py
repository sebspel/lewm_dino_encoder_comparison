"""Range probe: does the INT8 calibration set actually cover the eval-time input range?

READ-ONLY diagnostic. It changes no calibration behaviour, runs no PTQ, builds no engine and
alters no scale — it only MEASURES, so it stays clear of the OWNER-ONLY calibration/PTQ
boundary and instead informs it (SPEC §Implementation Boundaries).

Why it exists: `max` calibration sets each tensor's INT8 scale to the largest absolute value
it observed during the calibration pass, so "is calibration the culprit?" reduces to a
measurable question — **is `max|x|` at eval larger than `max|x|` during calibration?** Every
value above the calibration max saturates. FP16 has no fixed clip, which is why a gap here is
invisible in FP16 and catastrophic in INT8 (lewm: FP32 94% / FP16 96% / INT8 48%).

It compares, per track, the two predictor-input axes the fix targets (SPEC §Interface
Contracts — calibration distribution):

  * action  — the clips' EXPERT actions (bounded by `Box(-1, 1)`) vs the CEM proposal
    `predict` is actually driven by (`randn * var_scale`, unclamped — `solver/cem.py:191-204`).
    The mechanism is settled from the solver source; what this measures is the one assumption
    left, that the DRAWN expert actions really do sit in the box.
  * latent  — the encoder latents the old single-step draw showed the quantizer vs the
    predictor's OWN autoregressive latents, which is what a steady-state rollout window holds
    (at eval `n_obs=1`, so by the third step the window contains no encoder latent at all).

Reading it: `ratio > 1` means eval values exceed the calibration scale and are being clipped.
A mismatch CONFIRMS the diagnosis. A clean result does NOT fully exonerate calibration — this
samples the engine's INPUT tensors, while the graph quantizes every tensor inside it, so
internal activations could still mismatch.

Also the pass criterion for the fix: re-run it after, and the calibration column should be
>= the eval column on every row.

Pod-only (needs the real dataset + checkpoints):
    uv run python -m src.probe_ranges [track=<lewm|dino>] [n_clips=<int>]
"""

from __future__ import annotations

import sys

import torch
from torch import Tensor

from src.calibrate import (
    _ROLL_FRAMES,
    _dino_predictor_stream,
    _lewm_predictor_stream,
    _sample_cem_actions,
    build_calibration_data,
)
from src.interfaces import EVAL_N_OBS

_PROBE_BATCH = 8
_PROBE_CLIPS = 64  # enough for a stable max; the probe is a diagnostic, not the calib set


def _max_abs(t: Tensor) -> float:
    return t.abs().max().item()


def _stream_max(batches: list[tuple[Tensor, ...]], idx: int) -> float:
    """max|x| over input `idx` of a captured predict stream."""
    return max(_max_abs(b[idx]) for b in batches)


def probe(track: str, n_clips: int = _PROBE_CLIPS) -> list[tuple[str, float, float]]:
    """Return (tensor, calib max|x|, eval max|x|) rows for `track`, on the real checkpoint."""
    from src.adapter import DINOWMAdapter
    from src.precision_match import _build_adapter

    adapter, _ = _build_adapter(track)
    adapter.eval()
    data = build_calibration_data(batch=_PROBE_BATCH, n_clips=n_clips)
    gen = torch.Generator().manual_seed(0)

    rows: list[tuple[str, float, float]] = []
    with torch.no_grad():
        # --- action axis: expert clips (what calibration showed the quantizer) vs the
        # unclamped CEM proposal (what the rollout actually feeds predict).
        expert_action = _max_abs(data.action)
        proposal = _sample_cem_actions(len(data.obs), _ROLL_FRAMES, gen)
        rows.append(("action", expert_action, _max_abs(proposal)))

        # --- latent axis: the old single-step encoder latent vs the predictor's own
        # autoregressive latents, captured off the real roll.
        obs, proprio = data.obs, data.proprio
        encoder_latent = _max_abs(adapter.encode(obs[:, :EVAL_N_OBS]))
        actions = _sample_cem_actions(len(obs), _ROLL_FRAMES, gen)
        if isinstance(adapter, DINOWMAdapter):
            stream = _dino_predictor_stream(adapter, obs, proprio, actions)
            # DINO's predict input is the assembled 404 embedding (latent + extras fused), so
            # the rolled tensor is not directly comparable to the bare 384 encoder latent —
            # report it on its own row rather than as a misleading ratio.
            rows.append(("embedding(404)", float("nan"), _stream_max(stream, 0)))
        else:
            stream = _lewm_predictor_stream(adapter, obs, actions)
            rows.append(("latent", encoder_latent, _stream_max(stream, 0)))
    return rows


def main() -> None:
    track = "lewm"
    n_clips = _PROBE_CLIPS
    for a in sys.argv[1:]:
        if a.startswith("track="):
            track = a.split("=", 1)[1]
        elif a.startswith("n_clips="):
            n_clips = int(a.split("=", 1)[1])

    rows = probe(track, n_clips)
    print(f"\nrange probe — {track} ({n_clips} clips)")
    print(f"{'tensor':<16}{'calib max|x|':>14}{'eval max|x|':>14}{'ratio':>10}  status")
    for name, calib, ev in rows:
        ratio = ev / calib if calib else float("nan")
        status = "SATURATING" if ratio > 1 else "covered" if ratio == ratio else "n/a"
        c = "     n/a" if calib != calib else f"{calib:14.3f}"
        r = "       n/a" if ratio != ratio else f"{ratio:9.2f}x"
        print(f"{name:<16}{c}{ev:14.3f}{r}  {status}")
    print(
        "\nratio > 1 -> eval values exceed the INT8 scale and saturate. Measures the engine's "
        "INPUT tensors only:\na mismatch confirms the diagnosis; a clean result does not fully "
        "exonerate calibration (internals unsampled)."
    )


if __name__ == "__main__":
    main()
