# -*- coding: utf-8 -*-
"""Example driver: TextGrid-aligned dataset -> per-phoneme feature NPZs (legacy path).

The hardcoded sample-id range illustrates exactly why the manifest (Migration P1)
replaces directory conventions. Audio params come from the active profile (Q11).
"""
from __future__ import annotations

from signalml.config import active_profile
from signalml.pipeline.phoneme_jobs import build_phoneme_feature_npz


def main():
    profile = active_profile()  # dev by default; override via SIGNALML_AUDIO_PROFILE

    counts = build_phoneme_feature_npz(
        dataset_root="dataset",
        sample_ids=range(1, 107),
        feature="stft",              # "stft"|"mfcc"|"chroma"|"mel"
        out_dir="preprocessed",      # change to preprocessed_MFCC, etc
        audio_cfg=profile.audio_config(),
        spec_cfg=profile.spectrogram_config(),
        constant_timeframe=False,
        min_duration_sec=0.001,
    )
    print("Counts:", counts)


if __name__ == "__main__":
    main()
