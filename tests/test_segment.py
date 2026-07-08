from __future__ import annotations

import numpy as np

import signalml.audio.segment as segment_mod
from signalml.audio.segment import chunk_audio_file, segment_by_silence, slice_fixed

SR = 22050


def _sine(seconds: float, hz: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_chunking_counts_and_lengths(sine_wav, dev_audio_cfg):
    res = chunk_audio_file(sine_wav, 0.5, dev_audio_cfg)
    assert res.num_chunks == 4  # 2.0 s / 0.5 s
    expected_len = int(0.5 * dev_audio_cfg.sample_rate)
    for chunk in res.chunks:
        assert len(chunk) == expected_len
    assert abs(res.total_duration_sec - 2.0) < 0.05


def test_chunking_clips_remainder(sine_wav, dev_audio_cfg):
    res = chunk_audio_file(sine_wav, 0.75, dev_audio_cfg)
    assert res.num_chunks == 2  # floor(2.0 / 0.75)


def test_chunking_decodes_file_exactly_once(sine_wav, dev_audio_cfg, monkeypatch):
    """The legacy decode-per-chunk O(n^2) pattern must not come back."""
    calls = {"n": 0}
    real = segment_mod.load_audio

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(segment_mod, "load_audio", counting)
    res = chunk_audio_file(sine_wav, 0.25, dev_audio_cfg)
    assert res.num_chunks == 8
    assert calls["n"] == 1


def test_slice_fixed():
    y = np.arange(10, dtype=np.float32)
    chunks = slice_fixed(y, sr=2, chunk_sec=2.0)  # step = 4 samples
    assert [list(c) for c in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7]]


class TestSegmentBySilence:
    def test_finds_two_phrases(self):
        y = np.concatenate([_sine(1.5), np.zeros(2 * SR, dtype=np.float32), _sine(1.5)])
        phrases = segment_by_silence(y, SR)
        assert len(phrases) == 2
        (s1, e1), (s2, e2) = phrases
        assert s1 < 0.2 and 1.3 < e1 < 2.0
        assert 3.3 < s2 < 3.7 and e2 > 4.7

    def test_short_gaps_merge_into_one_phrase(self):
        gap = np.zeros(int(0.2 * SR), dtype=np.float32)
        y = np.concatenate([_sine(1.0), gap, _sine(1.0), gap, _sine(1.0)])
        phrases = segment_by_silence(y, SR, max_phrase_sec=15.0)
        assert len(phrases) == 1

    def test_oversized_phrase_is_split(self):
        y = _sine(10.0)
        phrases = segment_by_silence(y, SR, max_phrase_sec=4.0)
        assert len(phrases) == 3  # ceil(10/4)
        assert all(e - s <= 4.0 + 0.25 for s, e in phrases)

    def test_silence_only_returns_empty(self):
        assert segment_by_silence(np.zeros(SR, dtype=np.float32), SR) == []

    def test_phrases_never_overlap(self):
        y = np.concatenate([_sine(2.0), np.zeros(SR, dtype=np.float32), _sine(2.0)])
        phrases = segment_by_silence(y, SR)
        for (s1, e1), (s2, e2) in zip(phrases, phrases[1:]):
            assert e1 <= s2
