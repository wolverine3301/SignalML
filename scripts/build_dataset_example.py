# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 14:52:58 2026

@author: Owner
"""
from __future__ import annotations
from pathlib import Path
import random

from signalml.ingest import AudioConfig, SpectrogramConfig
from signalml.ingest.folder import build_class_chunks, flatten_class_dict
from signalml.pipeline.features import mels_from_chunks
from signalml.pipeline.masking import generate_masking_dataset
from signalml.ingest.io import save_npz_arrays


def main():
    rng = random.Random(123)  # reproducible runs

    audio_cfg = AudioConfig(sample_rate=22050, mono=True)
    spec_cfg = SpectrogramConfig(n_fft=2048, hop_length=256, win_length=2048, n_mels=256)

    raw = Path("data/raw")
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    voice = build_class_chunks(raw / "voice", 1.0, audio_cfg, pitch_augment=True, rng=rng)
    singing = build_class_chunks(raw / "singing", 1.0, audio_cfg, pitch_augment=True, rng=rng)
    vehicle = build_class_chunks(raw / "vehicle", 1.0, audio_cfg, pitch_augment=True, rng=rng)
    music = build_class_chunks(raw / "instramentalMusic", 1.0, audio_cfg, pitch_augment=True, rng=rng)

    # Save MELs for base classes
    voice_mels = mels_from_chunks(flatten_class_dict(voice), spec_cfg)
    save_npz_arrays(out / "voice_mels", voice_mels)

    singing_mels = mels_from_chunks(flatten_class_dict(singing), spec_cfg)
    save_npz_arrays(out / "singing_mels", singing_mels)

    # Masking datasets
    voice_vehicle = generate_masking_dataset(voice, vehicle, label="voiceVehicle", rng=rng)
    voice_vehicle_mels = mels_from_chunks(flatten_class_dict(voice_vehicle), spec_cfg)
    save_npz_arrays(out / "voiceVehicle_mels", voice_vehicle_mels)

    voice_music = generate_masking_dataset(voice, music, label="voiceMusic", rng=rng)
    voice_music_mels = mels_from_chunks(flatten_class_dict(voice_music), spec_cfg)
    save_npz_arrays(out / "voiceMusic_mels", voice_music_mels)

    sing_vehicle = generate_masking_dataset(singing, vehicle, label="singVehicle", rng=rng)
    sing_vehicle_mels = mels_from_chunks(flatten_class_dict(sing_vehicle), spec_cfg)
    save_npz_arrays(out / "singVehicle_mels", sing_vehicle_mels)

    sing_music = generate_masking_dataset(singing, music, label="singMusic", rng=rng)
    sing_music_mels = mels_from_chunks(flatten_class_dict(sing_music), spec_cfg)
    save_npz_arrays(out / "singMusic_mels", sing_music_mels)


if __name__ == "__main__":
    main()