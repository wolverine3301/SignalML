"""Audio segmentation: in-memory fixed chunking + silence-aware phrase segmentation.

P4 rewrite of the legacy ``ingest/chunking.py``: the file is decoded **once** and
sliced in memory (the inherited decode-per-chunk O(n^2) pattern is gone; see
docs/CODE_SURVEY.md). The public ``chunk_audio_file`` API is unchanged for callers
(masking track).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from ..config import AudioConfig
from .io import load_audio


@dataclass(frozen=True)
class ChunkResult:
    chunks: list[np.ndarray]
    num_chunks: int
    total_duration_sec: float


def slice_fixed(y: np.ndarray, sr: int, chunk_sec: float) -> list[np.ndarray]:
    """Slice a loaded signal into fixed-duration chunks (remainder clipped)."""
    step = int(round(chunk_sec * sr))
    if step <= 0:
        raise ValueError(f"chunk_sec too small: {chunk_sec}")
    num = len(y) // step
    return [y[i * step:(i + 1) * step] for i in range(num)]


def chunk_audio_file(
    path: str | Path,
    chunk_size_sec: float,
    cfg: AudioConfig,
    *,
    show_progress: bool = False,  # kept for API compat; slicing is instant now
) -> ChunkResult:
    """Load an audio file once and slice into fixed-duration chunks."""
    y, sr = load_audio(path, cfg)
    chunks = slice_fixed(y, sr, chunk_size_sec)
    return ChunkResult(
        chunks=chunks,
        num_chunks=len(chunks),
        total_duration_sec=len(y) / sr,
    )


def segment_by_silence(
    y: np.ndarray,
    sr: int,
    *,
    top_db: float = 40.0,
    min_phrase_sec: float = 1.0,
    max_phrase_sec: float = 15.0,
    max_gap_sec: float = 0.5,
    pad_sec: float = 0.1,
) -> list[tuple[float, float]]:
    """Silence-aware phrase segmentation for singing (start, end) in seconds.

    Voiced intervals from ``librosa.effects.split`` are merged when the silence
    *gap* between them is at most ``max_gap_sec`` and the merged span stays within
    ``max_phrase_sec``. Phrases still shorter than ``min_phrase_sec`` after merging
    (typically breaths/noise on separated vocals) are dropped from the map — the
    map is metadata, the audio itself is untouched. Oversized intervals are split
    evenly. ``pad_sec`` of context is kept per side, clamped at the midpoint to a
    neighbor so phrases never overlap.
    """
    mono = y if y.ndim == 1 else y.mean(axis=0)
    if not np.any(np.abs(mono) > 1e-6):  # librosa.split calls all-zero "non-silent"
        return []
    intervals = librosa.effects.split(mono, top_db=top_db)
    if len(intervals) == 0:
        return []

    spans = [(s / sr, e / sr) for s, e in intervals]

    # merge across short gaps, capped by max phrase length
    merged: list[list[float]] = [list(spans[0])]
    for s, e in spans[1:]:
        gap = s - merged[-1][1]
        if gap <= max_gap_sec and e - merged[-1][0] <= max_phrase_sec:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # drop sub-minimum fragments; if nothing qualifies keep the longest one so a
    # short-but-real recording still yields a phrase
    kept = [m for m in merged if m[1] - m[0] >= min_phrase_sec]
    merged = kept or [max(merged, key=lambda m: m[1] - m[0])]

    # split oversized phrases evenly
    phrases: list[tuple[float, float]] = []
    for s, e in merged:
        length = e - s
        if length <= max_phrase_sec:
            phrases.append((s, e))
        else:
            parts = int(np.ceil(length / max_phrase_sec))
            step = length / parts
            phrases.extend((s + i * step, s + (i + 1) * step) for i in range(parts))

    # pad with context, clamped against neighbors and signal bounds
    total = len(mono) / sr
    padded: list[tuple[float, float]] = []
    for i, (s, e) in enumerate(phrases):
        left = 0.0 if i == 0 else phrases[i - 1][1]
        right = total if i == len(phrases) - 1 else phrases[i + 1][0]
        s2 = max(left if i == 0 else (s + left) / 2, s - pad_sec, 0.0)
        e2 = min(right if i == len(phrases) - 1 else (e + right) / 2, e + pad_sec, total)
        padded.append((round(s2, 3), round(e2, 3)))
    return padded
