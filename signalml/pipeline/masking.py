# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:49:12 2026

@author: Owner
"""
from __future__ import annotations
from typing import Dict, List
import random

import numpy as np
from tqdm import tqdm

from .mixing import mix_random_background
from ..ingest.io import save_wav


def generate_masking_dataset(
    class_dict_foreground: Dict[str, List[np.ndarray]],
    class_dict_background: Dict[str, List[np.ndarray]],
    *,
    label: str,
    save_wavs: bool = False,
    wav_out_dir: str | None = None,
    sr: int = 22050,
    rng: random.Random | None = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Equivalent to your generateMaskingElements():
      - foreground = "ground truth"
      - background = noise/music/etc
      - mix each foreground sample with a random background sample
    Returns: {label: [mixed...]}
    """
    rng = rng or random.Random()
    fg = [x for xs in class_dict_foreground.values() for x in xs]
    bg = [x for xs in class_dict_background.values() for x in xs]

    mixed = []
    pbar = tqdm(total=len(fg), ncols=100, desc="Masking mix")  # fixed bug: total is len(fg)
    for i, fg_i in enumerate(fg):
        m = mix_random_background([fg_i], bg, rng=rng)[0]
        mixed.append(m)

        if save_wavs and wav_out_dir:
            save_wav(f"{wav_out_dir}/fg_{i}", fg_i, sr)
            save_wav(f"{wav_out_dir}/mix_{i}", m, sr)

        pbar.update(1)
    pbar.close()

    return {label: mixed}