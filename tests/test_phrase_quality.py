"""Per-phrase audio quality (signalml quality) - CPU only, synthetic audio."""

from __future__ import annotations

import numpy as np
import pytest

from signalml.config import active_profile
from signalml.stages.phrase_quality import (
    QUALITY_VERSION,
    is_current,
    measure_phrases,
    quality_name,
)

SR = active_profile("dev").sample_rate


def _voice(seconds: float, rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    """A harmonic 'sung' tone (220 Hz + overtones), optionally buried in noise."""
    t = np.arange(int(seconds * SR)) / SR
    y = sum(0.3 / k * np.sin(2 * np.pi * 220 * k * t) for k in range(1, 6))
    return y + noise * rng.standard_normal(len(t))


def _song() -> tuple[np.ndarray, list[dict], list[dict]]:
    """4 s: a clean phrase at 0.5-1.7 s, silence, a noisy one at 2.3-3.5 s."""
    rng = np.random.default_rng(0)
    y = np.zeros(int(4.0 * SR))
    for start, noise in ((0.5, 0.0), (2.3, 0.4)):
        seg = _voice(1.2, rng, noise=noise)
        y[int(start * SR):int(start * SR) + len(seg)] = seg
    utts = [{"start": 0.4, "end": 2.0}, {"start": 2.0, "end": 3.8}]
    phones = ([{"ph": "a", "start": 0.5 + 0.3 * i, "end": 0.8 + 0.3 * i} for i in range(4)]
              + [{"ph": "a", "start": 2.3 + 0.3 * i, "end": 2.6 + 0.3 * i} for i in range(4)])
    return y, utts, phones


def test_clean_phrase_scores_high_and_noisy_low():
    y, utts, phones = _song()
    clean, noisy = measure_phrases(y, SR, utts, phones)
    assert clean["hnr_db"] > 20 > 13 > noisy["hnr_db"]
    assert clean["voiced"] > 0.9
    # the sung span, not the window, is what gets measured
    assert (clean["sung_start"], clean["sung_end"]) == (0.5, 1.7)
    assert (clean["start"], clean["end"]) == (0.4, 2.0)
    # silence follows the clean phrase: its tail is far below the singing
    assert clean["tail_db"] < -40


def test_phrase_without_phones_is_not_scored():
    y, utts, phones = _song()
    utts = utts + [{"start": 3.8, "end": 4.0}]
    *_, empty = measure_phrases(y, SR, utts, phones)
    assert empty["hnr_db"] is None and empty["sung_start"] is None


def test_quality_tracks_its_utterances():
    utts = [{"start": 0.4, "end": 2.0}, {"start": 2.0, "end": 3.8}]
    q = {"version": QUALITY_VERSION, "phrases": [dict(u, hnr_db=20.0) for u in utts]}
    assert is_current(q, utts)
    assert not is_current(q, [{"start": 0.4, "end": 2.1}, utts[1]])  # re-aligned
    assert not is_current(q, utts[:1])
    assert not is_current(dict(q, version=QUALITY_VERSION + 1), utts)


@pytest.mark.parametrize("utts, quality", [
    ("utterances.json", "quality.json"),
    ("utterances.phrase.json", "quality.phrase.json"),
])
def test_quality_file_sits_beside_its_utterances(utts, quality):
    assert quality_name(utts) == quality
