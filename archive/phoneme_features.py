# -*- coding: utf-8 -*-
"""Featurize per-phoneme audio segments (merges into stages/features.py in P4).

Sample rate is now an explicit parameter — the former hardcoded 22050 in the MFCC
branch is gone (Q11).
"""
from __future__ import annotations

from typing import Literal

import librosa
import numpy as np

from ..config import SpectrogramConfig

FeatureKind = Literal["stft", "mfcc", "chroma", "mel"]


def featurize_phonemes(
    phoneme_segments: dict[str, list[np.ndarray]],
    *,
    kind: FeatureKind,
    spec: SpectrogramConfig,
    sr: int,
) -> dict[str, list[np.ndarray]]:
    """Compute one feature kind for every segment of every phoneme."""
    out: dict[str, list[np.ndarray]] = {}

    for ph, segs in phoneme_segments.items():
        feats: list[np.ndarray] = []
        for y in segs:
            if kind == "stft":
                feats.append(
                    librosa.stft(
                        y, n_fft=spec.n_fft, hop_length=spec.hop_length, win_length=spec.win_length
                    )
                )
            elif kind == "mfcc":
                feats.append(librosa.feature.mfcc(y=y, sr=sr))
            elif kind == "chroma":
                feats.append(
                    librosa.feature.chroma_stft(
                        y=y,
                        n_fft=spec.n_fft,
                        hop_length=spec.hop_length,
                        win_length=spec.win_length,
                        n_chroma=48,
                    )
                )
            elif kind == "mel":
                feats.append(
                    librosa.feature.melspectrogram(
                        y=y,
                        n_fft=spec.n_fft,
                        hop_length=spec.hop_length,
                        win_length=spec.win_length,
                        n_mels=spec.n_mels,
                    )
                )
            else:
                raise ValueError(f"Unknown kind: {kind}")

        out[ph] = feats

    return out
