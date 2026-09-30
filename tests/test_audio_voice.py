"""harshness() contract tests - synthetic singing vs synthetic scream, offline."""

from __future__ import annotations

import numpy as np

from signalml.audio.voice import harshness

SR = 16000


def _phrased(y: np.ndarray) -> np.ndarray:
    t = np.arange(len(y)) / SR
    return (y * ((t % 4.0) < 3.0)).astype(np.float32)


def test_singing_is_not_harsh():
    t = np.arange(20 * SR) / SR
    f = 220 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    sung = _phrased(sum(0.2 / k * np.sin(k * phase) for k in range(1, 6)))
    h = harshness(sung, SR)
    assert not h["likely"] and h["pitchless"] < 0.2 and h["hnr_median"] > 10


def test_pitchless_noise_is_harsh():
    scream = _phrased(np.random.default_rng(1).normal(0, 0.2, 20 * SR))
    h = harshness(scream, SR)
    assert h["likely"] and h["pitchless"] > 0.35 and h["noisy"] > 0.6


def test_too_little_audio_is_not_judged():
    assert harshness(np.zeros(5 * SR, np.float32), SR) is None
    assert harshness(np.zeros(10, np.float32), SR) is None
