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
def make_wav(dev_audio_cfg: AudioConfig):
    """Factory: write a sine WAV to an arbitrary path (parents auto-created)."""

    def _make(path: Path, *, seconds: float = TONE_SEC, hz: float = TONE_HZ) -> Path:
        sr = dev_audio_cfg.sample_rate
        t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), y, sr)
        return path

    return _make


@pytest.fixture()
def sine_wav(tmp_path: Path, make_wav) -> Path:
    """A 2-second 220 Hz sine WAV at the dev profile's sample rate."""
    return make_wav(tmp_path / "tone_220.wav")
