"""INT8 calibration set + procedure (owner-signed-off knobs).

Builds the calibration dataset the NVIDIA TensorRT **Model Optimizer** observes to derive
per-tensor INT8 scales (explicit Q/DQ — it inserts QuantizeLinear/DequantizeLinear into the
base ONNX and bakes the scales in), drawn **through the platform** so the activations match
inference exactly:

  * source — Push-T expert data (`pusht_expert_train.lance`, the same set the Phase-3 eval
    overlays replay — `conf/experiment/eval_*.yaml` override pusht.yaml's `.h5` default),
    loaded via `swm.data.load_dataset` with the SAME ImageNet `img_transform` the vendored
    `eval_wm` applies (single source of truth for normalization).
  * clips  — history-windows (`num_steps=HISTORY_SIZE`, `frameskip=CALIB_FRAMESKIP`), so each
    clip yields pixels `(hist, 3, 224, 224)`, proprio `(hist, 4)`, action `(hist, 10)`.
  * two streams — the encoder sees obs; the predictor sees the per-track predict input,
    produced by running the clips through the REAL adapter (`encode` [+ `assemble_embedding`
    for DINO]) so it observes true predict activations, not synthetic ones.

**The predictor stream reproduces the EVAL-TIME distribution, not the expert one** (SPEC
§Interface Contracts — calibration distribution). `max` calibration bakes fixed per-tensor
scales from the largest activation it observes, so anything wider at inference SATURATES —
invisible in FP16 (no fixed clip), catastrophic in INT8. A single-step draw off expert clips
mismatches the rollout on two axes, and did: lewm scored FP32 94% / FP16 96% / **INT8 48%**.

  * actions  — `CEMSolver.solve` drives `predict` with `randn(...) * var + mean`, **unclamped**
    (`solver/cem.py:191-204`; `var_scale=1.0`, mean 0 at the zero-pad warm start), and its
    `action_dim` is already the model-facing 10-wide pack (`cem.py:80` = env 2 × action_block
    5), so the proposal reaches ~4 sigma while expert actions are bounded by `Box(-1, 1)` —
    a ~4x under-scale. Under Design A LeWM's `action_encoder` lives INSIDE the predict engine,
    so the raw action and every action-encoder activation quantize on that range. Reproduced by
    `_sample_cem_actions` (same formula, fixed seed -> deterministic).
  * latents  — `predict` runs autoregressively: at eval `n_obs=1`, so a steady-state window
    holds ZERO encoder latents (`LeWM.rollout` lo=max(0, H+t-HS) -> windows 1,2,3,3,…; ditto
    `PreJEPA.rollout`). Reproduced by rolling `predict` over `CEM_HORIZON` and capturing its
    own inputs — via the REAL rollout, never a re-implementation (see `_dino_predictor_stream`).

Only the `T == HISTORY_SIZE` windows are captured: the engine's frame axis is static at `HS`, so
the rollout's `T < HS` transients reach it right-padded by `sr_shim._predict_hist_adapt` — and
that pad REPEATS the last real frame, adding no value outside the `T == HS` windows' range.

Owner sign-off (OWNER-ONLY silent-failure boundary — a bad calib set degrades every INT8
number with NO error): 512 clips; strided evenly across all episodes; the predictor roll yields
3 windows per clip -> ~1536 predictor samples (owner-confirmed 2026-07-15: clip coverage held at
512, sample count allowed to grow — coverage is the point of the fix). This module owns only the
calibration *streams* (format- AND method-independent); the calibration *method* is applied
downstream in `export.quantize_onnx` and is a BUILD OPTION for both tracks (`max` | `entropy`,
`src.export calibration_method=…`, docs/adr/0002). The remaining Model-Optimizer quant knobs
(Q/DQ format, per-channel-vs-per-tensor, op-type exclusions) stay at the tool's INT8 defaults
pending owner confirmation at the pod precision-match gate.

Accepted residual (SPEC): `max` on a Gaussian grows with draw count, so the calibration max
(~4.4 sigma over the whole set) sits just under the eval max (~5.5 sigma over 50 episodes ×
30 CEM iters). That clips ~1e-5 of action values, against the ~32% clipped when the scale was
fit to expert actions at 1.0 — benign, and the same tail suppression a percentile calibrator
does deliberately.

The clip draw (`build_calibration_data`) needs the real dataset -> pod-only. The batching /
adapter-streaming logic (`CalibrationData`) is pure torch and unit-tested off-pod.
`make_calibration_dict` turns the per-method streams into the numpy dict (keyed by ONNX
input name) the Model Optimizer consumes; it reads the ONNX input names via `onnx` (a uv
dep, available off-pod), so it is unit-tested off-pod too. The Model-Optimizer PTQ call
itself lives in `src.export.quantize_onnx` (imports `modelopt` lazily -> pod-only).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from src.interfaces import (
    CEM_HORIZON,
    CEM_VAR_SCALE,
    EVAL_N_OBS,
    HISTORY_SIZE,
    MODEL_ACTION_DIM,
)

# Owner-set: the calibration set is the eval dataset (representative Push-T), history-windows
# spaced by the frameskip/action_block (5) so obs history + the 10-wide action pack match the
# rollout, and 512 clips strided across all episodes for even trajectory coverage.
CALIB_DATASET = "pusht_expert_train.lance"
CALIB_FRAMESKIP = 5
DEFAULT_N_CLIPS = 512
_IMG_SIZE = 224
# Frames of action the roll needs: CEMSolver's real `candidates` tensor (`solver/cem.py:191-199`)
# has time-length `horizon` ONLY — NOT `n_obs + horizon` — and `rollout` splits that into the
# n_obs prefix (tags the current state, no predict call) and a `horizon − n_obs` remainder that
# drives `n_steps = horizon − n_obs` predict calls, plus one final call (= `(horizon − n_obs) + 1`
# predict calls, matching PREDICTOR_CALLS_PER_CYCLE = ((horizon − n_obs) + 1) × n_steps). So the
# roll's action-sequence length must be CEM_HORIZON itself, not EVAL_N_OBS + CEM_HORIZON.
_ROLL_FRAMES = CEM_HORIZON


def _sample_cem_actions(n: int, frames: int, generator: torch.Generator) -> Tensor:
    """The CEM proposal `predict` is actually driven by, reproduced from the solver source:
    `candidates = randn(...) * var + mean` with `var_scale` and mean 0 (`solver/cem.py:191-204`)
    — **unclamped**; there is no projection back into the action space. `CEMSolver.action_dim`
    is already the model-facing 10-wide pack (`cem.py:80`), so this samples `predict`'s input
    width directly — no env->model packing sits in between. Seeded -> the draw stays
    deterministic, the property the strided clip draw was built for."""
    return torch.randn(n, frames, MODEL_ACTION_DIM, generator=generator) * CEM_VAR_SCALE


class _CaptureAdapter:
    """Records every `predict` input, then delegates to the real adapter.

    Lets the REAL rollout drive the capture, so this module never re-implements the
    orchestration around `predict` (for DINO that is the 404 assembly / action-replace /
    proprio-carry — precisely the logic whose duplication the adapter-fidelity gate exists to
    catch). Everything except `predict` falls through to the wrapped adapter."""

    def __init__(self, adapter):
        self._adapter = adapter
        self.captured: list[tuple[Tensor, ...]] = []

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def predict(self, *inputs: Tensor) -> Tensor:
        self.captured.append(tuple(i.detach() for i in inputs))
        return self._adapter.predict(*inputs)


def _steady_windows(captured: list[tuple[Tensor, ...]]) -> list[tuple[Tensor, ...]]:
    """Keep only the `T == HISTORY_SIZE` windows — the shape the static-hist engine binds. The
    rollout's `T < HS` transients reach it repeat-padded (`sr_shim._predict_hist_adapt`), which
    adds no value outside these windows' range, so they contribute nothing to a `max` scale.
    """
    return [c for c in captured if c[0].shape[1] == HISTORY_SIZE]


def _pixel_transform():
    """The vendored eval `img_transform` (ImageNet Normalize) — reused verbatim so the
    calibration pixels are normalized identically to eval-time obs (matched normalization is
    load-bearing; a second copy could silently drift)."""
    from scripts.plan.eval_wm import img_transform

    cfg = SimpleNamespace(eval=SimpleNamespace(img_size=_IMG_SIZE))
    return img_transform(cfg)


def draw_calibration_clips(
    n_clips: int,
    hist: int,
    frameskip: int,
    dataset_name: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw `n_clips` history-window clips strided evenly across every episode, through the
    platform. Returns (obs, proprio, action) CPU tensors with a leading clip axis. Pod-only
    (needs the real dataset). Fails loud."""
    import numpy as np
    import stable_worldmodel as swm

    ds = swm.data.load_dataset(
        dataset_name,
        num_steps=hist,
        frameskip=frameskip,
        keys_to_load=["pixels", "proprio", "action"],
    )
    total = len(ds)
    if total == 0:
        raise RuntimeError(f"calibration dataset {dataset_name!r} yielded no clips")

    # Strided across all episodes: clip_indices is grouped by episode then start, so evenly
    # spaced indices cover the whole trajectory distribution. Deterministic (no RNG).
    idxs = np.unique(np.linspace(0, total - 1, min(n_clips, total)).round().astype(int))
    transform = _pixel_transform()

    obs, proprio, action = [], [], []
    for i in idxs:
        clip = ds[int(i)]
        obs.append(transform(clip["pixels"]))
        proprio.append(clip["proprio"].float())
        action.append(clip["action"].float())
    return torch.stack(obs), torch.stack(proprio), torch.stack(action)


class CalibrationData:
    """Holds the drawn clips and produces the per-method calibration streams. `batch` is only
    an internal chunk that bounds the adapter forward-pass memory when building the predictor
    stream (the Model Optimizer batches internally, so this is no longer tied to any engine
    profile); the drawn clips are trimmed to a whole multiple of it. Pure torch — unit-tested
    off-pod."""

    def __init__(self, obs: Tensor, proprio: Tensor, action: Tensor, batch: int):
        n = (len(obs) // batch) * batch
        if n == 0:
            raise ValueError(
                f"need >= batch ({batch}) calibration clips, got {len(obs)}"
            )
        self.obs, self.proprio, self.action = obs[:n], proprio[:n], action[:n]
        self.batch = batch

    def _chunks(self, t: Tensor) -> list[Tensor]:
        return [t[i : i + self.batch] for i in range(0, len(t), self.batch)]

    def encoder_batches(self) -> list[tuple[Tensor, ...]]:
        """Encoder calibration stream: obs batches, one input per batch."""
        return [(o,) for o in self._chunks(self.obs)]

    def predictor_batches(self, adapter, seed: int = 0) -> list[tuple[Tensor, ...]]:
        """Predictor calibration stream: the per-track predict input as the CEM rollout ACTUALLY
        drives it — CEM-proposal actions + the predictor's own autoregressive latents (module
        docstring; SPEC §Interface Contracts — calibration distribution). LeWM -> (latent, RAW
        action); DINO -> the assembled 404 embedding. Each clip contributes the roll's
        `T == HISTORY_SIZE` windows, so the stream is ~3x the clip count.

        The clips' EXPERT actions (`self.action`) are deliberately unused here — feeding them is
        the ~4x under-scale this fix removes. They stay on `CalibrationData` as the reference the
        range probe (`src.probe_ranges`) compares the proposal against."""
        from src.adapter import DINOWMAdapter

        adapter.eval()
        gen = torch.Generator().manual_seed(seed)
        batches: list[tuple[Tensor, ...]] = []
        with torch.no_grad():
            for o, p in zip(self._chunks(self.obs), self._chunks(self.proprio)):
                actions = _sample_cem_actions(len(o), _ROLL_FRAMES, gen)
                if isinstance(adapter, DINOWMAdapter):
                    batches.extend(_dino_predictor_stream(adapter, o, p, actions))
                else:
                    batches.extend(_lewm_predictor_stream(adapter, o, actions))
        return batches


def _lewm_predictor_stream(
    adapter, obs: Tensor, actions: Tensor
) -> list[tuple[Tensor, ...]]:
    """Roll LeWM's predictor and capture its own inputs.

    A line-map of `LeWM.rollout`'s window loop (installed swm 0.1.1 `wm/lewm/lewm.py:94-100`):
    `lo = max(0, H + t - HS)` over an emb_list that grows with PREDICTED frames. Mirrored rather
    than driven, because `rollout` is a method on the LeWM *model* while this boundary only has
    the adapter; the loop is plain windowing (no per-track carry), and only the resulting input
    *distribution* — not bit-exactness — feeds a `max` scale.

    `actions` are RAW: Design A puts LeWM's `action_encoder` inside the engine, so `rollout`
    windows raw actions through an Identity passthrough (`sr_shim.LeWMSRShim`)."""
    z = adapter.encode(
        obs[:, :EVAL_N_OBS]
    )  # eval encodes ONE frame; the rest are predicted
    emb_list = list(z.unbind(dim=1))
    n_steps = actions.shape[1] - EVAL_N_OBS
    captured: list[tuple[Tensor, ...]] = []
    for t in range(n_steps + 1):
        lo = max(0, EVAL_N_OBS + t - HISTORY_SIZE)
        emb_trunc = torch.stack(emb_list[lo:], dim=1)
        act_trunc = actions[:, lo : EVAL_N_OBS + t]
        captured.append((emb_trunc.detach(), act_trunc.detach()))
        emb_list.append(adapter.predict(emb_trunc, act_trunc)[:, -1])
    return _steady_windows(captured)


def _dino_predictor_stream(
    adapter, obs: Tensor, proprio: Tensor, actions: Tensor
) -> list[tuple[Tensor, ...]]:
    """Roll DINO-WM's predictor and capture its own inputs, by driving the REAL rollout
    (`src.shim.dino_rollout` — the fidelity-gated port of `PreJEPA.rollout`) through
    `_CaptureAdapter`. Nothing about the 404 assembly / action-replace / proprio-carry is
    re-implemented here; the rollout does it and the proxy just records."""
    from src.shim import dino_rollout

    proxy = _CaptureAdapter(adapter)
    info = {
        "pixels": obs[:, :EVAL_N_OBS].unsqueeze(1),  # (B, N=1, n_obs, C, H, W)
        "proprio": proprio[:, :EVAL_N_OBS].unsqueeze(1),  # (B, N=1, n_obs, dp)
    }
    dino_rollout(proxy, info, actions.unsqueeze(1), HISTORY_SIZE)
    return _steady_windows(proxy.captured)


def build_calibration_data(
    batch: int,
    n_clips: int = DEFAULT_N_CLIPS,
    hist: int = HISTORY_SIZE,
    frameskip: int = CALIB_FRAMESKIP,
    dataset_name: str = CALIB_DATASET,
) -> CalibrationData:
    """Draw the calibration clips and wrap them at the engine's calibration batch size
    (the export optimization-profile opt point). Pod-only (the draw needs the dataset).
    """
    obs, proprio, action = draw_calibration_clips(
        n_clips, hist, frameskip, dataset_name
    )
    return CalibrationData(obs, proprio, action, batch)


def make_calibration_dict(onnx_path: Path, batches: list[tuple[Tensor, ...]]) -> dict:
    """Turn a per-method stream into the numpy dict the Model Optimizer consumes: concatenate
    the chunked batches into one array per method input, then key those arrays by the base
    ONNX graph's real input names (read from the graph, initializers excluded) so the tool
    binds each calibration array to the right activation. `batches` positional order matches
    the traced `forward` signature (encoder `(obs,)`; LeWM predictor `(latent, action)`; DINO
    predictor `(embedding,)`), which is also the graph's input order — so a positional zip is
    correct. Uses `onnx` (a uv dep) -> off-pod. A miskeyed dict silently mis-scales, so the
    input-count mismatch fails loud."""
    import numpy as np
    import onnx

    n_inputs = len(batches[0])
    arrays = [
        torch.cat([b[i] for b in batches]).cpu().numpy().astype(np.float32)
        for i in range(n_inputs)
    ]

    model = onnx.load(str(onnx_path), load_external_data=False)
    init = {t.name for t in model.graph.initializer}
    names = [i.name for i in model.graph.input if i.name not in init]
    if len(names) != n_inputs:
        raise ValueError(
            f"ONNX {onnx_path} declares {len(names)} inputs {names} but the calibration "
            f"stream has {n_inputs} arrays"
        )
    return dict(zip(names, arrays))
