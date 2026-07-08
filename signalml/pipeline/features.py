# -*- coding: utf-8 -*-
"""Mel/STFT feature extraction from chunk lists (merges into stages/features.py in P4)."""
from __future__ import annotations

from collections.abc import Sequence

import librosa
import numpy as np

from ..config import SpectrogramConfig


def mels_from_chunks(chunks: Sequence[np.ndarray], cfg: SpectrogramConfig) -> list[np.ndarray]:
    return [
        librosa.feature.melspectrogram(
            y=x,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
        )
        for x in chunks
    ]


def stft_from_chunks(chunks: Sequence[np.ndarray], cfg: SpectrogramConfig) -> list[np.ndarray]:
    return [
        librosa.stft(
            x,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
        )
        for x in chunks
    ]
