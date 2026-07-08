"""Audio-profile configuration (OPEN_QUESTIONS Q11).

All audio/spectrogram parameters live in ``configs/audio.yaml`` as named profiles
(``dev`` = 22.05 kHz fast testing, ``prod`` = 44.1 kHz training). The active profile is
chosen per run: explicit argument > ``SIGNALML_AUDIO_PROFILE`` env var > YAML default.
Artifacts must record the profile that produced them; profiles never mix within a
dataset or checkpoint (docs/PIPELINE_AND_CONTRACTS.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = _REPO_ROOT / "configs"
DEFAULT_AUDIO_YAML = CONFIGS_DIR / "audio.yaml"
ENV_PROFILE = "SIGNALML_AUDIO_PROFILE"


@dataclass(frozen=True)
class AudioConfig:
    """Typed carrier for load/save parameters (passed into audio functions)."""

    sample_rate: int
    mono: bool = True


@dataclass(frozen=True)
class SpectrogramConfig:
    """Typed carrier for STFT/mel parameters (passed into feature functions)."""

    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    fmax: float | None = None


class AudioProfile(BaseModel):
    """One named audio parameterization from ``configs/audio.yaml``."""

    name: str = ""
    sample_rate: int
    mono: bool = True
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    fmax: float | None = None

    def audio_config(self) -> AudioConfig:
        return AudioConfig(sample_rate=self.sample_rate, mono=self.mono)

    def spectrogram_config(self) -> SpectrogramConfig:
        return SpectrogramConfig(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )


class AudioProfiles(BaseModel):
    default_profile: str
    profiles: dict[str, AudioProfile]


def load_audio_profiles(path: str | Path = DEFAULT_AUDIO_YAML) -> AudioProfiles:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = AudioProfiles.model_validate(raw)
    for name, profile in cfg.profiles.items():
        profile.name = name
    if cfg.default_profile not in cfg.profiles:
        raise ValueError(
            f"default_profile {cfg.default_profile!r} not among profiles "
            f"{sorted(cfg.profiles)} in {path}"
        )
    return cfg


def active_profile(
    name: str | None = None, path: str | Path = DEFAULT_AUDIO_YAML
) -> AudioProfile:
    """Resolve the active profile: explicit arg > env var > YAML default."""
    cfg = load_audio_profiles(path)
    resolved = name or os.environ.get(ENV_PROFILE) or cfg.default_profile
    if resolved not in cfg.profiles:
        raise KeyError(
            f"Unknown audio profile {resolved!r}; available: {sorted(cfg.profiles)}"
        )
    return cfg.profiles[resolved]
