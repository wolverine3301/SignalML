# -*- coding: utf-8 -*-
"""Batch job: TextGrid-aligned samples -> per-phoneme feature NPZs.

This module's *shape* (select work -> process -> write artifacts -> return stats) is
the template every production stage follows (docs/PIPELINE_AND_CONTRACTS.md §3).
The numeric-folder dataset layout it reads is replaced by the manifest in P1.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..audio.io import save_npz_arrays
from ..config import AudioConfig, SpectrogramConfig
from ..ingest.phonemes import extract_phoneme_segments, phoneme_safe_name
from ..ingest.textgrid import find_textgrid_pair, parse_textgrid_phones
from .phoneme_features import FeatureKind, featurize_phonemes


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
) -> dict[str, int]:
    """Reads dataset/{id}/(wav+TextGrid), extracts + featurizes phoneme segments,
    saves one NPZ per phoneme into out_dir. Returns counts per phoneme."""
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_segments: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}

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

        for ph, segs in ds.segments.items():
            merged_segments.setdefault(ph, []).extend(segs)
        for ph, c in ds.counts.items():
            counts[ph] = counts.get(ph, 0) + c

    features = featurize_phonemes(
        merged_segments, kind=feature, spec=spec_cfg, sr=audio_cfg.sample_rate
    )

    for ph, arrs in features.items():
        fname = phoneme_safe_name(ph)
        save_npz_arrays(out_dir / fname, arrs)

    return counts
