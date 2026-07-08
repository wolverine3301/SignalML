"""S6 features contract tests — offline; pyin path is torch-free, torchcrepe optional."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from signalml.config import active_profile
from signalml.manifest import Manifest, scan_directory
from signalml.stages.features import (
    FeaturesConfig,
    estimate_key,
    extract_features,
    features,
    load_features_config,
)

DEV = active_profile("dev")
SR = DEV.sample_rate


def _sine(seconds: float, hz: float, amp: float = 0.3) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * SR), endpoint=False)
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.fixture()
def data_root(tmp_path, make_wav):
    """One song, already separated+cleaned, clean vocal = 220 Hz sine at dev rate."""
    root = tmp_path / "dr"
    make_wav(root / "raw" / "song.wav", seconds=0.5)
    manifest, _ = scan_directory(root)
    rec = manifest.records[0]
    rec.status.separated = True
    rec.status.cleaned = True
    manifest.upsert(rec)
    manifest.save()

    clean_dir = root / "songs" / rec.id / "clean"
    clean_dir.mkdir(parents=True)
    sf.write(str(clean_dir / "vocals.wav"), _sine(2.0, 220.0), SR)
    return root


class TestExtract:
    def test_frame_alignment_and_shapes(self):
        y = _sine(2.0, 220.0)
        feats = extract_features(y, SR, DEV, FeaturesConfig())
        n = feats["mel"].shape[0]
        assert feats["mel"].shape[1] == DEV.n_mels == 128
        assert len(feats["f0"]) == len(feats["voicing"]) == len(feats["energy"]) == n
        assert feats["voicing"].dtype == bool

    def test_pyin_finds_220hz(self):
        y = _sine(2.0, 220.0)
        feats = extract_features(y, SR, DEV, FeaturesConfig())
        voiced_f0 = feats["f0"][feats["voicing"]]
        assert len(voiced_f0) > 0.5 * len(feats["f0"])  # mostly voiced
        assert np.median(voiced_f0) == pytest.approx(220.0, abs=5.0)

    def test_silence_is_unvoiced(self):
        y = np.zeros(2 * SR, dtype=np.float32)
        feats = extract_features(y, SR, DEV, FeaturesConfig())
        assert feats["voicing"].sum() < 0.05 * len(feats["voicing"])
        assert np.all(feats["f0"][~feats["voicing"]] == 0.0)

    def test_rmvpe_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="P7"):
            extract_features(_sine(1.0, 220.0), SR, DEV, FeaturesConfig(f0_method="rmvpe"))

    def test_torchcrepe_backend(self):
        pytest.importorskip("torchcrepe")
        y = _sine(2.0, 220.0)
        feats = extract_features(y, SR, DEV, FeaturesConfig(f0_method="torchcrepe"))
        voiced_f0 = feats["f0"][feats["voicing"]]
        assert len(voiced_f0) > 0
        assert np.median(voiced_f0) == pytest.approx(220.0, abs=10.0)


def test_estimate_key_c_major_triad():
    y = _sine(3.0, 261.63, 0.2) + _sine(3.0, 329.63, 0.2) + _sine(3.0, 392.0, 0.2)
    assert estimate_key(y.astype(np.float32), SR) == "C:major"


def test_stage_end_to_end(data_root):
    summary = features(data_root, profile=DEV)
    assert summary.featurized == ["sng_0001"] and not summary.failed

    npz = np.load(str(data_root / "songs" / "sng_0001" / "features" / "vocals.npz"))
    assert set(npz.files) >= {
        "mel", "f0", "voicing", "energy",
        "profile", "sr", "frame_hop", "n_mels", "f0_method", "mel_log_base",
    }
    assert not any(k.startswith("arr_") for k in npz.files)  # named keys only
    assert str(npz["profile"]) == "dev"
    assert int(npz["sr"]) == SR
    assert int(npz["frame_hop"]) == DEV.hop_length
    assert npz["mel"].shape[0] == len(npz["f0"])

    analysis = json.loads((data_root / "songs" / "sng_0001" / "analysis.json").read_text())
    section = analysis["features"]
    assert section["profile"] == "dev"
    assert "bpm" in section and "key" in section and "beats_sec" in section
    assert section["phrases_sec"]  # sine is one voiced phrase
    assert section["n_frames"] == npz["mel"].shape[0]

    assert Manifest.for_data_root(data_root).get("sng_0001").status.featurized is True


def test_requires_cleaned_and_idempotent(data_root):
    manifest = Manifest.for_data_root(data_root)
    rec = manifest.records[0]
    rec.status.cleaned = False
    manifest.upsert(rec)
    manifest.save()
    assert features(data_root, profile=DEV).featurized == []

    rec.status.cleaned = True
    manifest.upsert(rec)
    manifest.save()
    features(data_root, profile=DEV)
    assert features(data_root, profile=DEV).skipped == ["sng_0001"]
    assert features(data_root, profile=DEV, force=True).featurized == ["sng_0001"]


def test_profile_mismatch_rejected(data_root):
    """Clean audio at 44.1k but active profile dev (22.05k) must fail loudly."""
    clean_wav = data_root / "songs" / "sng_0001" / "clean" / "vocals.wav"
    y, _ = sf.read(str(clean_wav))
    sf.write(str(clean_wav), y, 44100)

    summary = features(data_root, profile=DEV)
    assert "sng_0001" in summary.failed
    assert "profile mismatch" in summary.failed["sng_0001"]


def test_load_features_config_defaults_and_file(tmp_path):
    assert load_features_config(tmp_path / "missing.yaml") == FeaturesConfig()
    f = tmp_path / "features.yaml"
    f.write_text("f0_method: torchcrepe\nphrase_max_sec: 10\n", encoding="utf-8")
    cfg = load_features_config(f)
    assert cfg.f0_method == "torchcrepe"
    assert cfg.phrase_max_sec == 10
    assert cfg.stem == "vocals"
