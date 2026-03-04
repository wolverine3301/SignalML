# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:46:37 2026

@author: Owner
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import math
import numpy as np
from tqdm import tqdm

from .audio_loader import load_audio, audio_duration_sec
from .config import AudioConfig


@dataclass(frozen=True)
class ChunkResult:
    chunks: List[np.ndarray]
    num_chunks: int
    total_duration_sec: float


def chunk_audio_file(
    path: str | Path,
    chunk_size_sec: float,
    cfg: AudioConfig,
    *,
    show_progress: bool = True,
) -> ChunkResult:
    """
    Loads an audio file and slices into fixed-duration chunks (clips remainder).
    Mirrors your original genElements behavior (floor(duration / chunk_size)).
    """
    y_full, sr = load_audio(path, cfg)
    total_duration = audio_duration_sec(y_full, sr)
    num_chunks = math.floor(total_duration / chunk_size_sec)

    chunks: List[np.ndarray] = []
    iterator = range(num_chunks)
    if show_progress:
        iterator = tqdm(iterator, total=num_chunks, ncols=100, desc="processing chunks", leave=True)

    offset = 0.0
    for _ in iterator:
        y, _sr = load_audio(path, cfg, offset_sec=offset, duration_sec=chunk_size_sec)
        chunks.append(y)
        offset += chunk_size_sec

    return ChunkResult(chunks=chunks, num_chunks=num_chunks, total_duration_sec=total_duration)