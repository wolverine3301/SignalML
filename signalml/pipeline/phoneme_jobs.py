# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:20:40 2026

@author: Owner
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Iterable, Literal

import numpy as np

from signalml.ingest.config import AudioConfig, SpectrogramConfig
from signalml.ingest.textgrid import find_textgrid_pair, parse_textgrid_phones
from signalml.ingest.phonemes import extract_phoneme_segments, phoneme_safe_name
from signalml.ingest.io import save_npz_arrays
from .phoneme_features import featurize_phonemes, FeatureKind


def build_phoneme_feature_npz(
    dataset_root: str | Path,
    *,
    sample_ids: Iterable[int],
    feature: FeatureKind,
    out_dir: str | Path,
    audio_cfg: AudioConfig,
    spec_cfg: SpectrogramConfig,
    constant_timeframe: bool = False,
    min_duration_sec: float = 0.001,
) -> Dict[str, int]:
    """
    Generalized replacement for synthesis2/3/4/5.
    - Reads dataset/{id}/(wav+TextGrid)
    - Extracts phoneme segments
    - Featurizes
    - Saves one NPZ per phoneme into out_dir
    Returns counts per phoneme.
    """
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_segments: Dict[str, List[np.ndarray]] = {}
    counts: Dict[str, int] = {}

    for sid in sample_ids:
        sample_dir = dataset_root / str(sid)
        wav, tg = find_textgrid_pair(sample_dir)
        alignment = parse_textgrid_phones(tg)
        ds = extract_phoneme_segments(
            alignment,
            wav,
            audio_cfg,
            min_duration_sec=min_duration_sec,
            constant_timeframe=constant_timeframe,
        )

        # merge into one big set
        for ph, segs in ds.segments.items():
            merged_segments.setdefault(ph, []).extend(segs)
        for ph, c in ds.counts.items():
            counts[ph] = counts.get(ph, 0) + c

    features = featurize_phonemes(merged_segments, kind=feature, spec=spec_cfg)

    # Save one file per phoneme, like your originals
    for ph, arrs in features.items():
        fname = phoneme_safe_name(ph)
        save_npz_arrays(out_dir / fname, arrs)

    return counts