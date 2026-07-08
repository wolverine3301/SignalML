"""Fixed-duration chunking of audio files.

Moved from ``ingest/chunking.py``. NOTE: the per-chunk re-decode inherited from the
original is a known O(n^2) I/O problem; Migration P4 replaces this with single-decode,
in-memory slicing plus silence-aware phrase segmentation. Behavior is preserved until
then (docs/CODE_SURVEY.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..config import AudioConfig
from .io import audio_duration_sec, load_audio


@dataclass(frozen=True)
class ChunkResult:
    chunks: list[np.ndarray]
    num_chunks: int
    total_duration_sec: float


def chunk_audio_file(
    path: str | Path,
    chunk_size_sec: float,
    cfg: AudioConfig,
    *,
    show_progress: bool = True,
) -> ChunkResult:
    """Loads an audio file and slices into fixed-duration chunks (clips remainder)."""
    y_full, sr = load_audio(path, cfg)
    total_duration = audio_duration_sec(y_full, sr)
    num_chunks = math.floor(total_duration / chunk_size_sec)

    chunks: list[np.ndarray] = []
    iterator = range(num_chunks)
    if show_progress:
        iterator = tqdm(iterator, total=num_chunks, ncols=100, desc="processing chunks", leave=True)

    offset = 0.0
    for _ in iterator:
        y, _sr = load_audio(path, cfg, offset_sec=offset, duration_sec=chunk_size_sec)
        chunks.append(y)
        offset += chunk_size_sec

    return ChunkResult(chunks=chunks, num_chunks=num_chunks, total_duration_sec=total_duration)
