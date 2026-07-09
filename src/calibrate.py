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
    clip yields pixels `(hist, 3, 224, 224)`, proprio `(hist, 4)`, action `(hist, 10)` —
    exactly the encode / predict inputs the CEM rollout drives.
  * two streams — the encoder sees obs; the predictor sees the per-track predict input,
    produced by running the clips through the REAL adapter (`encode` [+ `assemble_embedding`
    for DINO]) so it observes true predict activations, not synthetic ones.

Owner sign-off (OWNER-ONLY silent-failure boundary — a bad calib set degrades every INT8
number with NO error): calibration method = `max` (the Model-Optimizer analogue of the old
MinMax calibrator, suited to the ViT activations); 512 clips; strided evenly across all
episodes. The remaining Model-Optimizer quant knobs (Q/DQ format, per-channel-vs-per-tensor,
op-type exclusions) stay at the tool's INT8 defaults pending owner confirmation at the
pod precision-match gate.

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

from src.interfaces import HISTORY_SIZE

# Owner-set: the calibration set is the eval dataset (representative Push-T), history-windows
# spaced by the frameskip/action_block (5) so obs history + the 10-wide action pack match the
# rollout, and 512 clips strided across all episodes for even trajectory coverage.
CALIB_DATASET = "pusht_expert_train.lance"
CALIB_FRAMESKIP = 5
DEFAULT_N_CLIPS = 512
_IMG_SIZE = 224


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
            raise ValueError(f"need >= batch ({batch}) calibration clips, got {len(obs)}")
        self.obs, self.proprio, self.action = obs[:n], proprio[:n], action[:n]
        self.batch = batch

    def _chunks(self, t: Tensor) -> list[Tensor]:
        return [t[i : i + self.batch] for i in range(0, len(t), self.batch)]

    def encoder_batches(self) -> list[tuple[Tensor, ...]]:
        """Encoder calibration stream: obs batches, one input per batch."""
        return [(o,) for o in self._chunks(self.obs)]

    def predictor_batches(self, adapter) -> list[tuple[Tensor, ...]]:
        """Predictor calibration stream: the per-track predict input, produced by running the
        clips through the REAL adapter so TensorRT observes true predict activations. LeWM ->
        (cached latent, action); DINO -> the assembled 404 embedding."""
        from src.adapter import DINOWMAdapter

        adapter.eval()
        batches: list[tuple[Tensor, ...]] = []
        with torch.no_grad():
            for o, p, a in zip(
                self._chunks(self.obs),
                self._chunks(self.proprio),
                self._chunks(self.action),
            ):
                latent = adapter.encode(o)
                if isinstance(adapter, DINOWMAdapter):
                    batches.append((adapter.assemble_embedding(latent, p, a),))
                else:
                    batches.append((latent, a))
        return batches


def build_calibration_data(
    batch: int,
    n_clips: int = DEFAULT_N_CLIPS,
    hist: int = HISTORY_SIZE,
    frameskip: int = CALIB_FRAMESKIP,
    dataset_name: str = CALIB_DATASET,
) -> CalibrationData:
    """Draw the calibration clips and wrap them at the engine's calibration batch size
    (the export optimization-profile opt point). Pod-only (the draw needs the dataset)."""
    obs, proprio, action = draw_calibration_clips(n_clips, hist, frameskip, dataset_name)
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
