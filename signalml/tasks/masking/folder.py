"""Folder-of-raw-audio -> per-class chunk dictionaries (masking track).

Moved from ``ingest/folder.py`` — this is masking-dataset tooling, not core ingest.
Known design smells preserved for now (augmentation at chunk time, save_wavs writing
into the source folder); see docs/CODE_SURVEY.md.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...audio.io import save_wav
from ...audio.segment import chunk_audio_file
from ...config import AudioConfig
from .mixing import augment_with_pitch

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


def iter_audio_files(folder: str | Path) -> list[Path]:
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
) -> dict[str, list[np.ndarray]]:
    """Folder -> {filename: [chunks...]}."""
    rng = rng or random.Random()
    files = iter_audio_files(raw_folder)

    out: dict[str, list[np.ndarray]] = {}
    total_chunks = 0
    total_duration = 0.0

    pbar = tqdm(files, ncols=100, desc="Progress")
    for path in pbar:
        res = chunk_audio_file(path, sample_duration_sec, cfg, show_progress=False)
        chunks = augment_with_pitch(res.chunks, cfg.sample_rate, enabled=pitch_augment, rng=rng)

        if save_wavs:
            stem = path.stem
            for idx, ch in enumerate(chunks):
                save_wav(Path(raw_folder) / f"{stem}_{idx}", ch, cfg.sample_rate)

        out[path.name] = chunks
        total_chunks += res.num_chunks
        total_duration += res.total_duration_sec

    print(f"GENERATED {total_chunks} base chunks")
    print(f"APPROX {total_duration:.2f} seconds of audio scanned for class")
    return out


def flatten_class_dict(class_dict: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    flat: list[np.ndarray] = []
    for chunks in class_dict.values():
        flat.extend(chunks)
    return flat
