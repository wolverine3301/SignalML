"""Audio file I/O — the single choke point for reading/writing audio.

Merged from the former ``ingest/audio_loader.py`` and ``ingest/io.py``.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..config import AudioConfig


def load_audio(
    path: str | Path,
    cfg: AudioConfig,
    *,
    offset_sec: float = 0.0,
    duration_sec: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio from disk with consistent sample rate + mono handling."""
    y, sr = librosa.load(
        str(path),
        sr=cfg.sample_rate,
        mono=cfg.mono,
        offset=offset_sec,
        duration=duration_sec,
    )
    return y, sr


def audio_duration_sec(y: np.ndarray, sr: int) -> float:
    return float(librosa.get_duration(y=y, sr=sr))


def save_wav(path_no_ext: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path_no_ext).with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def save_npz_arrays(path_no_ext: str | Path, arrays: list[np.ndarray]) -> Path:
    """Save a list of arrays into an NPZ: arr_0, arr_1, ...

    Legacy format for the masking track only — production features use named keys
    (docs/PIPELINE_AND_CONTRACTS.md §S6, lands in Migration P4).
    """
    path = Path(path_no_ext).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), *arrays)
    return path


def load_npz_arrays(path: str | Path) -> list[np.ndarray]:
    data = np.load(str(path))
    return [data[k] for k in data.files]
