from __future__ import annotations

import numpy as np

from signalml.audio.io import (
    audio_duration_sec,
    load_audio,
    load_npz_arrays,
    save_npz_arrays,
    save_wav,
)


def test_load_audio_roundtrip(sine_wav, dev_audio_cfg):
    y, sr = load_audio(sine_wav, dev_audio_cfg)
    assert sr == dev_audio_cfg.sample_rate
    assert y.ndim == 1  # mono
    assert abs(audio_duration_sec(y, sr) - 2.0) < 0.05


def test_load_audio_offset_duration(sine_wav, dev_audio_cfg):
    y, sr = load_audio(sine_wav, dev_audio_cfg, offset_sec=0.5, duration_sec=1.0)
    assert abs(len(y) / sr - 1.0) < 0.01


def test_save_wav_appends_extension(tmp_path, dev_audio_cfg):
    sr = dev_audio_cfg.sample_rate
    y = np.zeros(sr // 10, dtype=np.float32)
    out = save_wav(tmp_path / "sub" / "clip", y, sr)
    assert out.suffix == ".wav"
    assert out.exists()  # parent dir auto-created


def test_npz_roundtrip(tmp_path):
    arrays = [np.arange(5, dtype=np.float32), np.ones((2, 3), dtype=np.float32)]
    path = save_npz_arrays(tmp_path / "arrays", arrays)
    loaded = load_npz_arrays(path)
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0], arrays[0])
    np.testing.assert_array_equal(loaded[1], arrays[1])
