# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:19:57 2026

@author: Owner
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal

import librosa
import numpy as np

from signalml.ingest.config import SpectrogramConfig


FeatureKind = Literal["stft", "mfcc", "chroma", "mel"]


def featurize_phonemes(
    phoneme_segments: Dict[str, List[np.ndarray]],
    *,
    kind: FeatureKind,
    spec: SpectrogramConfig,
) -> Dict[str, List[np.ndarray]]:
    """
    Replaces synthesis2/3/4/5 pattern with one function.
    """
    out: Dict[str, List[np.ndarray]] = {}

    for ph, segs in phoneme_segments.items():
        feats: List[np.ndarray] = []
        for y in segs:
            if kind == "stft":
                feats.append(librosa.stft(y, n_fft=spec.n_fft, hop_length=spec.hop_length, win_length=spec.win_length))
            elif kind == "mfcc":
                feats.append(librosa.feature.mfcc(y=y, sr=22050))  # you can parameterize sr if needed
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