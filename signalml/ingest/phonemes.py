# -*- coding: utf-8 -*-
"""Phoneme segment extraction from TextGrid alignments (legacy, dies in Migration P5)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from ..audio.io import load_audio
from ..config import AudioConfig
from .textgrid import PhoneAlignment


@dataclass
class PhonemeDataset:
    """phoneme -> list of wave segments; counts is redundant but handy."""

    segments: dict[str, list[np.ndarray]]
    counts: dict[str, int]


def time_stretch_to_factor(y: np.ndarray, factor: float) -> np.ndarray:
    # factor > 1 stretches (slower/longer), < 1 compresses
    return librosa.effects.time_stretch(y.astype("float64"), rate=factor)


def scale_to_constant_timeframe(y: np.ndarray) -> np.ndarray:
    """Legacy-parity time stretch; arbitrary inherited formula — do not use for the
    singing path (docs/CODE_SURVEY.md)."""
    dur = librosa.get_duration(y=y)
    factor = (dur / 2.0) * 1.5
    return time_stretch_to_factor(y, factor)


def resample_series_to_length(
    series: dict[int, np.ndarray], target_len: int
) -> dict[int, np.ndarray]:
    """Interpolate each series to target_len. Keys are original lengths."""
    out: dict[int, np.ndarray] = {}
    for length, data in series.items():
        out[length] = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, length),
            data,
        )
    return out


def scale_to_min_timeframe(series: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    target = min(series.keys())
    return resample_series_to_length(series, target)


def scale_to_max_timeframe(series: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
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
    """Slice per-phoneme audio segments out of a wav using a parsed alignment."""
    segments: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}

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
    # MAUS/SAMPA-era special cases; replaced by the IPA phone-set mapping in P5
    if ph == "<p:>":
        return "pause"
    if ph == "?":
        return "unknown"
    if ph == "h\\":
        return "hhh"
    return ph
