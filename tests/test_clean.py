"""S4 clean contract tests — offline, CPU-only.

Acceptance criterion from the migration plan: output loudness within +/-1 LU of the
configured target on fixtures.
"""

from __future__ import annotations

import json

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

from signalml.config import active_profile
from signalml.manifest import Manifest, scan_directory
from signalml.stages.clean import CleanConfig, clean, load_clean_config, process_audio

DEV = active_profile("dev")
SR = DEV.sample_rate
STEM_SR = 44100  # stems arrive at demucs's native rate


def _sine(sr: int, seconds: float, hz: float, amp: float = 0.1) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.fixture()
def data_root(tmp_path, make_wav):
    """One scanned song with fake separated stems (stereo, 44.1 kHz)."""
    root = tmp_path / "dr"
    make_wav(root / "raw" / "song.wav", seconds=0.5)
    manifest, _ = scan_directory(root)
    rec = manifest.records[0]
    rec.status.separated = True
    manifest.upsert(rec)
    manifest.save()

    stems_dir = root / "songs" / rec.id / "stems"
    stems_dir.mkdir(parents=True)
    y = _sine(STEM_SR, 2.0, 440.0)
    stereo = np.stack([y, 0.8 * y], axis=1)  # (T, C)
    for name in ("vocals", "drums", "bass", "other"):
        sf.write(str(stems_dir / f"{name}.wav"), stereo, STEM_SR)
    return root


def test_clean_end_to_end(data_root):
    summary = clean(data_root, profile=DEV)
    assert summary.cleaned == ["sng_0001"] and not summary.failed

    out = data_root / "songs" / "sng_0001" / "clean" / "vocals.wav"
    info = sf.info(str(out))
    assert info.samplerate == SR  # resampled to dev profile
    assert info.channels == 1  # mono per profile

    # acceptance: within +/-1 LU of target
    y, sr = sf.read(str(out))
    measured = pyln.Meter(sr).integrated_loudness(y)
    assert abs(measured - (-23.0)) < 1.0

    analysis = json.loads((data_root / "songs" / "sng_0001" / "analysis.json").read_text())
    section = analysis["clean"]
    assert section["profile"] == "dev"
    assert section["stems"]["vocals"]["loudness_after"] == pytest.approx(-23.0, abs=1.0)
    assert "loudness_normalize" in section["stems"]["vocals"]["ops"]
    assert section["stems"]["vocals"]["silence"]["voiced_sec"] > 0

    assert Manifest.for_data_root(data_root).get("sng_0001").status.cleaned is True


def test_requires_separated_and_skips_cleaned(data_root):
    manifest = Manifest.for_data_root(data_root)
    rec = manifest.records[0]
    rec.status.separated = False
    manifest.upsert(rec)
    manifest.save()

    summary = clean(data_root, profile=DEV)
    assert summary.cleaned == [] and summary.skipped == ["sng_0001"]

    rec.status.separated = True
    manifest.upsert(rec)
    manifest.save()
    clean(data_root, profile=DEV)
    summary3 = clean(data_root, profile=DEV)  # second pass: already cleaned
    assert summary3.skipped == ["sng_0001"]
    summary4 = clean(data_root, profile=DEV, force=True)
    assert summary4.cleaned == ["sng_0001"]


def test_missing_stem_fails_song(data_root):
    (data_root / "songs" / "sng_0001" / "stems" / "vocals.wav").unlink()
    summary = clean(data_root, profile=DEV)
    assert "sng_0001" in summary.failed
    assert Manifest.for_data_root(data_root).get("sng_0001").status.cleaned is False


class TestProcessAudio:
    def test_ops_all_off_is_identity(self):
        y = _sine(SR, 1.0, 220.0)
        cfg = CleanConfig(loudness_normalize=False, silence_map=False, highpass=False)
        out, stats = process_audio(y, SR, cfg)
        np.testing.assert_array_equal(out, y)
        assert stats["ops"] == []
        assert "silence" not in stats

    def test_loudness_hits_target(self):
        y = _sine(SR, 2.0, 440.0, amp=0.02)  # quiet input
        out, stats = process_audio(y, SR, CleanConfig(silence_map=False))
        assert stats["loudness_after"] == pytest.approx(-23.0, abs=1.0)
        assert stats["gain_db"] > 0

    def test_silent_input_skips_gain(self):
        y = np.zeros(SR, dtype=np.float32)
        out, stats = process_audio(y, SR, CleanConfig(silence_map=False))
        assert stats["loudness_before"] is None
        np.testing.assert_array_equal(out, y)

    def test_peak_ceiling_clamps(self):
        y = _sine(SR, 2.0, 440.0, amp=0.001)  # huge gain needed
        cfg = CleanConfig(target_lufs=0.0, peak_ceiling_dbfs=-1.0, silence_map=False)
        out, stats = process_audio(y, SR, cfg)
        assert stats["peak_clamped"] is True
        assert float(np.max(np.abs(out))) <= 10 ** (-1.0 / 20) + 1e-4

    def test_highpass_kills_rumble(self):
        rumble = _sine(SR, 2.0, 25.0, amp=0.3)
        tone = _sine(SR, 2.0, 440.0, amp=0.1)
        cfg = CleanConfig(highpass=True, highpass_hz=50.0,
                          loudness_normalize=False, silence_map=False)
        out_rumble, _ = process_audio(rumble, SR, cfg)
        out_tone, _ = process_audio(tone, SR, cfg)
        # 2nd-order Butterworth via sosfiltfilt ~ 24 dB/oct: expect >=20 dB down at 25 Hz
        assert np.sqrt(np.mean(out_rumble**2)) < 0.1 * np.sqrt(np.mean(rumble**2))
        assert np.sqrt(np.mean(out_tone**2)) > 0.9 * np.sqrt(np.mean(tone**2))

    def test_silence_map_finds_gap(self):
        seg = _sine(SR, 1.0, 440.0)
        y = np.concatenate([seg, np.zeros(SR, dtype=np.float32), seg])
        _, stats = process_audio(y, SR, CleanConfig(loudness_normalize=False))
        sil = stats["silence"]
        assert len(sil["intervals_sec"]) == 2
        assert sil["silence_sec"] == pytest.approx(1.0, abs=0.1)


def test_load_clean_config_defaults_and_file(tmp_path):
    assert load_clean_config(tmp_path / "missing.yaml") == CleanConfig()
    f = tmp_path / "clean.yaml"
    f.write_text("target_lufs: -18\nhighpass: true\n", encoding="utf-8")
    cfg = load_clean_config(f)
    assert cfg.target_lufs == -18
    assert cfg.highpass is True
    assert cfg.silence_map is True  # default preserved
