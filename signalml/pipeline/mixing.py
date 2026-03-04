# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:48:46 2026

@author: Owner
"""
from __future__ import annotations
from typing import Sequence, List
import random

import librosa
import numpy as np


def pitch_shift_random(
    wav: np.ndarray,
    sr: int,
    *,
    min_steps: int = -6,
    max_steps: int = 6,
    rng: random.Random | None = None,
) -> np.ndarray:
    rng = rng or random.Random()
    steps = rng.randint(min_steps, max_steps)
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=steps)


def augment_with_pitch(
    chunks: Sequence[np.ndarray],
    sr: int,
    *,
    enabled: bool = True,
    rng: random.Random | None = None,
) -> List[np.ndarray]:
    """
    Matches your original behavior: for each chunk, optionally add a pitch-shifted copy.
    """
    if not enabled:
        return list(chunks)
    rng = rng or random.Random()
    out: List[np.ndarray] = []
    for x in chunks:
        out.append(x)
        out.append(pitch_shift_random(x, sr, rng=rng))
    return out


def mix_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def mix_aligned_subset(
    a: Sequence[np.ndarray],
    b: Sequence[np.ndarray],
    percent: float,
    *,
    rng: random.Random | None = None,
) -> List[np.ndarray]:
    """
    Equivalent to your augmentMixedSignals_init subset behavior (but operates on time-domain chunks).
    """
    rng = rng or random.Random()
    n = min(len(a), len(b))
    k = int(n * percent)
    if k <= 0:
        return []
    idxs = rng.sample(range(n), k)
    return [mix_average(a[i], b[i]) for i in idxs]


def mix_random_background(
    foreground: Sequence[np.ndarray],
    background: Sequence[np.ndarray],
    *,
    rng: random.Random | None = None,
) -> List[np.ndarray]:
    """
    Equivalent to your generateMaskingElements mixing strategy:
    each foreground sample mixed with a random background sample.
    """
    rng = rng or random.Random()
    if not background:
        raise ValueError("background is empty")
    out: List[np.ndarray] = []
    for fg in foreground:
        bg = background[rng.randint(0, len(background) - 1)]
        out.append(mix_average(fg, bg))
    return out


def mix_mel_npz_roundtrip(
    mel_npz_a: str,
    mel_npz_b: str,
    *,
    percent: float,
    rng: random.Random | None = None,
) -> List[np.ndarray]:
    """
    Mirrors your old augmentMixedSignals() intent:
    - load mel specs from npz
    - invert to audio
    - mix
    - return mixed audio list (caller decides how to re-extract/save)
    """
    rng = rng or random.Random()
    data_a = np.load(mel_npz_a)
    data_b = np.load(mel_npz_b)
    a = [data_a[k] for k in data_a.files]
    b = [data_b[k] for k in data_b.files]

    n = min(len(a), len(b))
    k = int(n * percent)
    if k <= 0:
        return []

    idxs = rng.sample(range(n), k)
    out: List[np.ndarray] = []
    for i in idxs:
        t1 = librosa.feature.inverse.mel_to_audio(a[i])
        t2 = librosa.feature.inverse.mel_to_audio(b[i])
        out.append(mix_average(t1, t2))
    return out