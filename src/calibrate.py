"""INT8 calibration set + procedure (owner-signed-off knobs).

Builds the calibration dataset TensorRT observes to derive per-tensor INT8 scales, drawn
**through the platform** so the activations match inference exactly:

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
number with NO error): calibrator = `IInt8MinMaxCalibrator` (min/max, suited to the ViT
activations); 512 clips; strided evenly across all episodes.

The clip draw (`build_calibration_data`) needs the real dataset -> pod-only. The batching /
adapter-streaming logic (`CalibrationData`) is pure torch and unit-tested off-pod. The
`IInt8MinMaxCalibrator` subclass (`make_calibrator`) imports `tensorrt` lazily -> pod-only.
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
    """Holds the drawn clips and produces the per-method calibration batches. Trims to a whole
    multiple of `batch` so every calibration batch matches the calibrator's `get_batch_size`
    (and the engine's optimization-profile opt point). Pure torch — unit-tested off-pod."""

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


def make_calibrator(batches: list[tuple[Tensor, ...]], cache_path: Path):
    """Owner-chosen `IInt8MinMaxCalibrator` over pre-built calibration batches. Each
    `get_batch` moves one batch to CUDA (holding it alive) and hands TensorRT the device
    pointers, in the engine's input order; scales are cached to `cache_path` so a rebuild is
    data-free. Imports `tensorrt` lazily -> pod-only. Fails loud."""
    import tensorrt as trt

    cache_path = Path(cache_path)

    class _MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self):
            super().__init__()
            self._batches = batches
            self._i = 0
            self._held: list[Tensor] = []  # keep CUDA buffers alive during the build step
            self._batch_size = int(batches[0][0].shape[0])

        def get_batch_size(self):
            return self._batch_size

        def get_batch(self, names):
            if self._i >= len(self._batches):
                return None
            batch = self._batches[self._i]
            self._i += 1
            # Engine bindings are FP32 (build sets no tensor dtypes); feed float on CUDA.
            self._held = [t.to("cuda").contiguous().float() for t in batch]
            return [int(t.data_ptr()) for t in self._held]

        def read_calibration_cache(self):
            return cache_path.read_bytes() if cache_path.exists() else None

        def write_calibration_cache(self, cache):
            cache_path.write_bytes(cache)

    return _MinMaxCalibrator()
