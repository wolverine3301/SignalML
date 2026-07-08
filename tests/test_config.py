from __future__ import annotations

import pytest

from signalml.config import ENV_PROFILE, active_profile, load_audio_profiles


def test_profiles_load_and_validate():
    cfg = load_audio_profiles()
    assert cfg.default_profile == "dev"
    assert set(cfg.profiles) >= {"dev", "prod"}


def test_dev_and_prod_values_match_contract():
    cfg = load_audio_profiles()
    dev, prod = cfg.profiles["dev"], cfg.profiles["prod"]

    assert dev.sample_rate == 22050
    assert dev.hop_length == 256
    assert dev.n_mels == 128

    # prod matches the NSF-HiFiGAN vocoder contract (ARCHITECTURE.md §3.3)
    assert prod.sample_rate == 44100
    assert prod.hop_length == 512
    assert prod.win_length == 2048
    assert prod.n_mels == 128
    assert prod.fmax == 16000


def test_active_profile_precedence(monkeypatch):
    # default: YAML default_profile
    monkeypatch.delenv(ENV_PROFILE, raising=False)
    assert active_profile().name == "dev"

    # env var overrides YAML default
    monkeypatch.setenv(ENV_PROFILE, "prod")
    assert active_profile().name == "prod"

    # explicit argument overrides env var
    assert active_profile("dev").name == "dev"


def test_unknown_profile_raises(monkeypatch):
    monkeypatch.delenv(ENV_PROFILE, raising=False)
    with pytest.raises(KeyError):
        active_profile("nope")


def test_typed_carriers():
    p = active_profile("prod")
    assert p.audio_config().sample_rate == 44100
    assert p.spectrogram_config().hop_length == 512
