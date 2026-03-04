# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:45:15 2026

@author: Owner
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import librosa
import numpy as np

from .config import AudioConfig


def load_audio(
    path: str | Path,
    cfg: AudioConfig,
    *,
    offset_sec: float = 0.0,
    duration_sec: float | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from disk using librosa with consistent sample rate + mono handling.
    """
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