# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:47:52 2026

@author: Owner
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Iterable
import random

import numpy as np
from tqdm import tqdm

from .chunking import chunk_audio_file
from .config import AudioConfig
from .io import save_wav
from ..pipeline.mixing import augment_with_pitch  # intentional reuse


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def iter_audio_files(folder: str | Path) -> List[Path]:
    p = Path(folder)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _AUDIO_EXTS]
    return sorted(files)


def build_class_chunks(
    raw_folder: str | Path,
    sample_duration_sec: float,
    cfg: AudioConfig,
    *,
    pitch_augment: bool = True,
    save_wavs: bool = False,
    rng: random.Random | None = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Folder -> {filename: [chunks...]}.
    Equivalent intent to your makeClassSamples().
    """
    rng = rng or random.Random()
    files = iter_audio_files(raw_folder)

    out: Dict[str, List[np.ndarray]] = {}
    total_chunks = 0
    total_duration = 0.0

    pbar = tqdm(files, ncols=100, desc="Progress")
    for path in pbar:
        res = chunk_audio_file(path, sample_duration_sec, cfg, show_progress=False)
        chunks = augment_with_pitch(res.chunks, cfg.sample_rate, enabled=pitch_augment, rng=rng)

        if save_wavs:
            # mirrors your saveWavs option; names include original file stem + chunk index
            stem = path.stem
            for idx, ch in enumerate(chunks):
                save_wav(Path(raw_folder) / f"{stem}_{idx}", ch, cfg.sample_rate)

        out[path.name] = chunks
        total_chunks += res.num_chunks
        total_duration += res.total_duration_sec

    print(f"GENERATED {total_chunks} base chunks")
    print(f"APPROX {total_duration:.2f} seconds of audio scanned for class")
    return out


def flatten_class_dict(class_dict: Dict[str, List[np.ndarray]]) -> List[np.ndarray]:
    flat: List[np.ndarray] = []
    for chunks in class_dict.values():
        flat.extend(chunks)
    return flat