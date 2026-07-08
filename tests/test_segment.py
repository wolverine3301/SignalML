from __future__ import annotations

from signalml.audio.segment import chunk_audio_file


def test_chunking_counts_and_lengths(sine_wav, dev_audio_cfg):
    res = chunk_audio_file(sine_wav, 0.5, dev_audio_cfg, show_progress=False)
    assert res.num_chunks == 4  # 2.0 s / 0.5 s
    assert len(res.chunks) == 4
    expected_len = int(0.5 * dev_audio_cfg.sample_rate)
    for chunk in res.chunks:
        assert abs(len(chunk) - expected_len) <= 1
    assert abs(res.total_duration_sec - 2.0) < 0.05


def test_chunking_clips_remainder(sine_wav, dev_audio_cfg):
    res = chunk_audio_file(sine_wav, 0.75, dev_audio_cfg, show_progress=False)
    assert res.num_chunks == 2  # floor(2.0 / 0.75)
