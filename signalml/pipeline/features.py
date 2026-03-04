# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:50:19 2026

@author: Owner
"""
from __future__ import annotations
from typing import Sequence, List

import librosa
import numpy as np

from ..ingest.config import SpectrogramConfig


def mels_from_chunks(chunks: Sequence[np.ndarray], cfg: SpectrogramConfig) -> List[np.ndarray]:
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


def stft_from_chunks(chunks: Sequence[np.ndarray], cfg: SpectrogramConfig) -> List[np.ndarray]:
    return [
        librosa.stft(
            x,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
        )
        for x in chunks
    ]