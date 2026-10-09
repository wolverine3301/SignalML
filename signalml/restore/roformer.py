"""Run a community MSST RoFormer checkpoint (de-reverb, de-echo...) over a whole vocal.

The model code is vendored in ``third_party/msst_roformer`` (MIT); checkpoints live under
``DATA_ROOT/models/<name>/`` next to the training config they shipped with. Torch and the
vendored model are imported lazily, so the stage and its tests run without them.

Long audio is processed in overlapping chunks of the model's training length and
re-assembled by a weighted overlap-add (``chunked_apply``, numpy only): each chunk is
faded in and out over a tenth of its length, except at the very start and end of the
signal, and the sum is divided by the summed weights, so a model that returns its input
returns the signal unchanged.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from ..config import CONFIGS_DIR

THIRD_PARTY = CONFIGS_DIR.parent / "third_party"

# fn(batch (B, C, chunk)) -> (B, C, chunk)
ChunkFn = Callable[[np.ndarray], np.ndarray]
ARCHES = ("bs_roformer", "mel_band_roformer")


def target_stem(y: np.ndarray, stem_index: int, length: int) -> np.ndarray:
    """A model's output -> (B, C, length) of the wanted stem.

    BS-RoFormer returns (B, stems, C, T); Mel-band RoFormer squeezes a single stem to
    (B, C, T). The iSTFT can come back a few samples short of the input: zero-pad."""
    y = np.asarray(y)
    if y.ndim == 4:
        y = y[:, stem_index]
    if y.shape[-1] < length:
        y = np.pad(y, ((0, 0), (0, 0), (0, length - y.shape[-1])))
    return y[..., :length]


class _MsstLoader(yaml.SafeLoader):
    """Safe YAML plus ``!!python/tuple`` (MSST configs write band layouts as tuples).
    Every other python/* tag still fails, as with ``yaml.safe_load``."""


_MsstLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)))


def load_msst_config(path: str | Path) -> dict:
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_MsstLoader) or {}


def chunked_apply(audio: np.ndarray, fn: ChunkFn, chunk: int, overlap: int = 2,
                  batch_size: int = 1) -> np.ndarray:
    """Apply ``fn`` to ``audio`` (C, T) in overlapping chunks; returns (C, T)."""
    audio = np.asarray(audio, dtype=np.float32)
    channels, length = audio.shape
    if length <= chunk:  # one pass: overlapping a single chunk only doubles the work
        mode = "reflect" if chunk - length < length else "constant"
        x = np.pad(audio, ((0, 0), (0, chunk - length)), mode=mode)
        return np.asarray(fn(x[None]), dtype=np.float32)[0, :, :length]
    step = max(1, chunk // max(1, overlap))
    border = chunk - step
    padded = length > 2 * border and border > 0
    x = np.pad(audio, ((0, 0), (border, border)), mode="reflect") if padded else audio
    total = x.shape[1]
    fade = max(1, chunk // 10)
    window = np.ones(chunk, dtype=np.float32)
    window[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    window[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)

    result = np.zeros_like(x)
    weight = np.zeros(total, dtype=np.float32)
    starts = list(range(0, total, step))
    for b in range(0, len(starts), max(1, batch_size)):
        group = starts[b:b + max(1, batch_size)]
        parts, lengths = [], []
        for i in group:
            part = x[:, i:i + chunk]
            n = part.shape[1]
            if n < chunk:
                mode = "reflect" if n > chunk // 2 else "constant"
                part = np.pad(part, ((0, 0), (0, chunk - n)), mode=mode)
            parts.append(part)
            lengths.append(n)
        out = np.asarray(fn(np.stack(parts)), dtype=np.float32)
        for i, n, y in zip(group, lengths, out):
            w = window.copy()
            if i == 0:
                w[:fade] = 1.0
            if i + step >= total:
                w[-fade:] = 1.0
            result[:, i:i + n] += y[:, :n] * w[:n]
            weight[i:i + n] += w[:n]
    result /= np.maximum(weight, 1e-8)
    return result[:, border:border + length] if padded else result


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RoformerRunner:
    """A loaded MSST RoFormer (``arch``: BS or Mel-band). ``runner(audio, sr)`` returns
    the target stem mixed to mono; ``runner.separate(audio, sr)`` keeps its channels.

    ``audio`` is mono (T,) or (C, T) at the model's sample rate (no silent resampling:
    the caller passes prod-profile audio, and a mismatch is an error). Mono input is
    duplicated for a stereo model.
    """

    def __init__(self, checkpoint: str | Path, model_config: str | Path, *,
                 arch: str = "bs_roformer", device: str = "auto", batch_size: int = 2,
                 num_overlap: int | None = None, use_amp: bool = True):
        import torch

        if arch not in ARCHES:
            raise ValueError(f"arch {arch!r} not in {ARCHES}")
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        if arch == "bs_roformer":
            from msst_roformer.bs_roformer import BSRoformer as Model
        else:
            from msst_roformer.mel_band_roformer import MelBandRoformer as Model

        cfg = load_msst_config(model_config)
        audio_cfg, model_cfg = cfg["audio"], cfg["model"]
        training = cfg.get("training") or {}
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.chunk = int(audio_cfg["chunk_size"])
        self.stereo = bool(model_cfg.get("stereo", False))
        self.overlap = int(num_overlap or (cfg.get("inference") or {}).get("num_overlap", 2))
        self.batch_size = batch_size
        instruments = list(training.get("instruments") or [])
        target = training.get("target_instrument")
        # a single-stem model outputs only its target; a multi-stem one, all instruments
        self.stem_index = 0 if int(model_cfg.get("num_stems", 1)) == 1 or not target \
            else instruments.index(target)
        self.target = target or (instruments[0] if instruments else "stem0")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"

        model = Model(**model_cfg)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)
        self._torch = torch
        self.info = {
            "arch": arch,
            "checkpoint": Path(checkpoint).name,
            "checkpoint_sha256": file_sha256(checkpoint),
            "target": self.target,
            "sample_rate": self.sample_rate,
            "chunk": self.chunk,
            "overlap": self.overlap,
            "device": str(self.device),
            "amp": self.use_amp,
            "torch": torch.__version__,
        }

    def _run(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode(), torch.autocast(
                self.device.type, dtype=torch.float16, enabled=self.use_amp):
            y = self.model(torch.from_numpy(batch).to(self.device))
        return target_stem(y.float().cpu().numpy(), self.stem_index, batch.shape[-1])

    def separate(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """The target stem, (C, T) in the model's channel layout."""
        if sr != self.sample_rate:
            raise ValueError(f"model runs at {self.sample_rate} Hz, got {sr} Hz audio "
                             f"(pass audio at the model's rate)")
        x = np.asarray(audio, dtype=np.float32)
        x = x[None, :] if x.ndim == 1 else x
        if self.stereo and x.shape[0] == 1:
            x = np.repeat(x, 2, axis=0)
        elif not self.stereo and x.shape[0] > 1:
            x = x.mean(axis=0, keepdims=True)
        return chunked_apply(x, self._run, self.chunk, self.overlap, self.batch_size)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return self.separate(audio, sr).mean(axis=0).astype(np.float32)
