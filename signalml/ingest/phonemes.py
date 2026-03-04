# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:19:15 2026

@author: Owner
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Union
import random

import librosa
import numpy as np

from .config import AudioConfig
from .audio_loader import load_audio
from .textgrid import PhoneAlignment


@dataclass
class PhonemeDataset:
    """
    phoneme -> list of wave segments
    counts is redundant but handy.
    """
    segments: Dict[str, List[np.ndarray]]
    counts: Dict[str, int]


def time_stretch_to_factor(y: np.ndarray, factor: float) -> np.ndarray:
    # factor > 1 stretches (slower/longer), < 1 compresses
    return librosa.effects.time_stretch(y.astype("float64"), rate=factor)


def scale_to_constant_timeframe(y: np.ndarray) -> np.ndarray:
    """
    Your scaleToConstantTimeFrame was effectively picking a factor based on duration.
    It was also printing. We remove printing and keep the behavior shape.
    """
    dur = librosa.get_duration(y=y)
    factor = (dur / 2.0) * 1.5  # matches your math: phase=dur/2; phase += phase/2
    return time_stretch_to_factor(y, factor)


def resample_series_to_length(series: Dict[int, np.ndarray], target_len: int) -> Dict[int, np.ndarray]:
    """
    Generalized version of scaleToMinTimeFrame / scaleToMaxTimeFrame using interpolation.
    Keys are original lengths.
    """
    out: Dict[int, np.ndarray] = {}
    for length, data in series.items():
        out[length] = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, length),
            data,
        )
    return out


def scale_to_min_timeframe(series: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    target = min(series.keys())
    return resample_series_to_length(series, target)


def scale_to_max_timeframe(series: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    target = max(series.keys())
    return resample_series_to_length(series, target)


def extract_phoneme_segments(
    alignment: PhoneAlignment,
    wav_path: str | Path,
    cfg: AudioConfig,
    *,
    min_duration_sec: float = 0.001,
    constant_timeframe: bool = False,
) -> PhonemeDataset:
    """
    Replacement for generatePhoneElements / generatePhoneElements2 without globals.
    """
    segments: Dict[str, List[np.ndarray]] = {}
    counts: Dict[str, int] = {}

    for ph, t0, t1 in zip(alignment.phones, alignment.starts, alignment.ends):
        dur = float(t1 - t0)
        if dur <= min_duration_sec:
            continue

        y, _sr = load_audio(wav_path, cfg, offset_sec=t0, duration_sec=dur)

        if constant_timeframe:
            y = scale_to_constant_timeframe(y)

        segments.setdefault(ph, []).append(y)
        counts[ph] = counts.get(ph, 0) + 1

    return PhonemeDataset(segments=segments, counts=counts)


def phoneme_safe_name(ph: str) -> str:
    # mirrors your special cases
    if ph == "<p:>":
        return "pause"
    if ph == "?":
        return "unknown"
    if ph == "h\\":
        return "hhh"
    return ph