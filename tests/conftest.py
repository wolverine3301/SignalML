"""Shared fixtures. Audio fixtures are generated on the fly (seconds-long, CPU-only,
offline — per the testing convention in docs/PIPELINE_AND_CONTRACTS.md §6).
``tests/fixtures/`` is reserved for future golden files (TextGrid, score JSON)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from signalml.config import AudioConfig, active_profile

TONE_HZ = 220.0
TONE_SEC = 2.0


@pytest.fixture(scope="session")
def dev_profile():
    return active_profile("dev")


@pytest.fixture(scope="session")
def dev_audio_cfg(dev_profile) -> AudioConfig:
    return dev_profile.audio_config()


@pytest.fixture()
def sine_wav(tmp_path: Path, dev_audio_cfg: AudioConfig) -> Path:
    """A 2-second 220 Hz sine WAV at the dev profile's sample rate."""
    sr = dev_audio_cfg.sample_rate
    t = np.linspace(0, TONE_SEC, int(TONE_SEC * sr), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
    path = tmp_path / "tone_220.wav"
    sf.write(str(path), y, sr)
    return path
