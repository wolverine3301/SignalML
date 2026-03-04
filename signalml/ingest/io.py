# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:47:17 2026

@author: Owner
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf


def save_wav(path_no_ext: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path_no_ext).with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def save_npz_arrays(path_no_ext: str | Path, arrays: List[np.ndarray]) -> Path:
    """
    Save a list of arrays into an NPZ: arr_0, arr_1, ...
    This is the most reliable format for later loading.
    """
    path = Path(path_no_ext).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), *arrays)
    return path


def load_npz_arrays(path: str | Path) -> List[np.ndarray]:
    data = np.load(str(path))
    return [data[k] for k in data.files]