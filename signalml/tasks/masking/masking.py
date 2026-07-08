"""Masking-dataset generation (foreground mixed with random backgrounds).

Moved from ``pipeline/masking.py``. The sample rate is now a required argument —
callers pass the active profile's rate (no hardcoded 22050, per Q11).
"""

from __future__ import annotations

import random

import numpy as np
from tqdm import tqdm

from ...audio.io import save_wav
from .mixing import mix_random_background


def generate_masking_dataset(
    class_dict_foreground: dict[str, list[np.ndarray]],
    class_dict_background: dict[str, list[np.ndarray]],
    *,
    label: str,
    sr: int,
    save_wavs: bool = False,
    wav_out_dir: str | None = None,
    rng: random.Random | None = None,
) -> dict[str, list[np.ndarray]]:
    """Mix each foreground sample with a random background sample.

    Returns ``{label: [mixed...]}``.
    """
    rng = rng or random.Random()
    fg = [x for xs in class_dict_foreground.values() for x in xs]
    bg = [x for xs in class_dict_background.values() for x in xs]

    mixed = []
    pbar = tqdm(total=len(fg), ncols=100, desc="Masking mix")
    for i, fg_i in enumerate(fg):
        m = mix_random_background([fg_i], bg, rng=rng)[0]
        mixed.append(m)

        if save_wavs and wav_out_dir:
            save_wav(f"{wav_out_dir}/fg_{i}", fg_i, sr)
            save_wav(f"{wav_out_dir}/mix_{i}", m, sr)

        pbar.update(1)
    pbar.close()

    return {label: mixed}
