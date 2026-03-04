# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:21:18 2026

@author: Owner
"""
from __future__ import annotations

from signalml.ingest.config import AudioConfig, SpectrogramConfig
from signalml.pipeline.phoneme_jobs import build_phoneme_feature_npz


def main():
    audio_cfg = AudioConfig(sample_rate=22050, mono=True)
    spec_cfg = SpectrogramConfig(n_fft=2048, hop_length=256, win_length=2048, n_mels=256)

    counts = build_phoneme_feature_npz(
        dataset_root="dataset",
        sample_ids=range(1, 107),
        feature="stft",              # "stft"|"mfcc"|"chroma"|"mel"
        out_dir="preprocessed",      # change to preprocessed_MFCC, etc
        audio_cfg=audio_cfg,
        spec_cfg=spec_cfg,
        constant_timeframe=False,
        min_duration_sec=0.001,
    )
    print("Counts:", counts)


if __name__ == "__main__":
    main()