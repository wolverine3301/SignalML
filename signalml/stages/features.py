"""S6 features — mel/F0/voicing/energy into named-key NPZ + BPM/key/phrase analysis.

Contract: docs/PIPELINE_AND_CONTRACTS.md §S6. Every per-frame array shares one frame
grid (hop from the active audio profile); the NPZ stores the profile name, sample
rate, hop, and F0 method so downstream stages can refuse mixed-profile datasets.
No anonymous ``arr_0`` keys — that legacy pattern ends here.

F0 backends: ``pyin`` (librosa, default — no torch), ``torchcrepe`` (train extra),
``rmvpe`` (reserved: no standalone PyPI package exists, so it gets vendored with
third_party/ in P7 — until then selecting it raises with instructions).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel

from ..audio.segment import segment_by_silence
from ..config import CONFIGS_DIR, AudioProfile, active_profile
from ..manifest import Manifest
from .common import song_dir, update_analysis

_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class FeaturesConfig(BaseModel):
    stem: str = "vocals"
    f0_method: Literal["pyin", "torchcrepe", "rmvpe"] = "pyin"
    f0_min_hz: float = 65.0
    f0_max_hz: float = 1400.0
    analysis: bool = True
    phrases: bool = True
    phrase_min_sec: float = 1.0
    phrase_max_sec: float = 15.0


def load_features_config(path: str | Path | None = None) -> FeaturesConfig:
    path = Path(path) if path else CONFIGS_DIR / "features.yaml"
    if path.exists():
        return FeaturesConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        )
    return FeaturesConfig()


@dataclass
class FeaturesSummary:
    featurized: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def _align(arr: np.ndarray, n_frames: int, fill: float = 0.0) -> np.ndarray:
    """Trim/pad a per-frame array to exactly n_frames (defensive off-by-one guard)."""
    if len(arr) >= n_frames:
        return arr[:n_frames]
    return np.pad(arr, (0, n_frames - len(arr)), constant_values=fill)


def _f0_pyin(y, sr, profile: AudioProfile, cfg: FeaturesConfig):
    f0, voiced, _prob = librosa.pyin(
        y,
        fmin=cfg.f0_min_hz,
        fmax=cfg.f0_max_hz,
        sr=sr,
        frame_length=profile.win_length,
        hop_length=profile.hop_length,
    )
    return np.nan_to_num(f0, nan=0.0), voiced.astype(bool)


def _f0_torchcrepe(y, sr, profile: AudioProfile, cfg: FeaturesConfig):
    try:
        import torch
        import torchcrepe
    except ImportError as exc:
        raise RuntimeError(
            "torchcrepe not installed - install the train extra: "
            "python -m uv sync --extra train"
        ) from exc

    crepe_sr = 16000
    y16 = librosa.resample(y, orig_sr=sr, target_sr=crepe_sr)
    hop16 = max(1, int(round(profile.hop_length / sr * crepe_sr)))
    audio = torch.from_numpy(y16.astype(np.float32))[None]
    f0, periodicity = torchcrepe.predict(
        audio,
        crepe_sr,
        hop_length=hop16,
        fmin=cfg.f0_min_hz,
        fmax=min(cfg.f0_max_hz, 1500.0),  # crepe's own ceiling
        model="full",
        return_periodicity=True,
        device="cpu",
        batch_size=512,
    )
    f0 = f0[0].numpy().astype(np.float32)
    voiced = periodicity[0].numpy() > 0.5
    f0[~voiced] = 0.0
    return f0, voiced


def _compute_f0(y, sr, profile, cfg) -> tuple[np.ndarray, np.ndarray]:
    if cfg.f0_method == "pyin":
        return _f0_pyin(y, sr, profile, cfg)
    if cfg.f0_method == "torchcrepe":
        return _f0_torchcrepe(y, sr, profile, cfg)
    raise NotImplementedError(
        "rmvpe has no standalone PyPI package; it is vendored with third_party/ in "
        "Migration P7. Use f0_method: pyin (default) or torchcrepe until then."
    )


def estimate_key(y: np.ndarray, sr: int) -> str:
    """Krumhansl-Schmuckler correlation over mean chroma -> e.g. 'G:major'."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best, best_corr = "C:major", -np.inf
    for profile_vec, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
        for shift in range(12):
            corr = np.corrcoef(np.roll(profile_vec, shift), chroma)[0, 1]
            if corr > best_corr:
                best, best_corr = f"{_NOTES[shift]}:{mode}", float(corr)
    return best


def extract_features(
    y: np.ndarray, sr: int, profile: AudioProfile, cfg: FeaturesConfig
) -> dict[str, np.ndarray]:
    """Frame-aligned features on one shared hop grid. Mel is natural-log, floored."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=profile.n_fft,
        hop_length=profile.hop_length,
        win_length=profile.win_length,
        n_mels=profile.n_mels,
        fmax=profile.fmax,
    )
    log_mel = np.log(np.clip(mel, 1e-5, None)).T.astype(np.float32)  # (T, n_mels)
    n_frames = log_mel.shape[0]

    f0, voiced = _compute_f0(y, sr, profile, cfg)
    energy = librosa.feature.rms(
        y=y, frame_length=profile.win_length, hop_length=profile.hop_length
    )[0]

    return {
        "mel": log_mel,
        "f0": _align(f0, n_frames).astype(np.float32),
        "voicing": _align(voiced.astype(np.float32), n_frames).astype(bool),
        "energy": _align(energy, n_frames).astype(np.float32),
    }


def save_features_npz(
    path: Path,
    feats: dict[str, np.ndarray],
    *,
    profile: AudioProfile,
    cfg: FeaturesConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        **feats,
        profile=profile.name,
        sr=profile.sample_rate,
        frame_hop=profile.hop_length,
        win_length=profile.win_length,
        n_mels=profile.n_mels,
        mel_log_base="e",
        f0_method=cfg.f0_method,
        stem=cfg.stem,
    )


def features(
    data_root: str | Path,
    *,
    cfg: FeaturesConfig | None = None,
    profile: AudioProfile | None = None,
    force: bool = False,
    limit: int | None = None,
) -> FeaturesSummary:
    data_root = Path(data_root)
    cfg = cfg or load_features_config()
    profile = profile or active_profile()
    manifest = Manifest.for_data_root(data_root)
    summary = FeaturesSummary()

    work = []
    for rec in manifest.records:
        if not rec.status.cleaned or (rec.status.featurized and not force):
            summary.skipped.append(rec.id)
        else:
            work.append(rec)
    if limit is not None:
        work = work[:limit]

    for rec in work:
        try:
            sdir = song_dir(data_root, rec.id)
            src = sdir / "clean" / f"{cfg.stem}.wav"
            if not src.exists():
                raise FileNotFoundError(f"clean stem missing: {src}")

            y, sr = sf.read(str(src), dtype="float32")
            if y.ndim == 2:
                y = y.mean(axis=1)
            if int(sr) != profile.sample_rate:
                raise ValueError(
                    f"profile mismatch: {src} is {sr} Hz but active profile "
                    f"{profile.name!r} expects {profile.sample_rate} Hz - "
                    f"re-run clean under this profile or switch profiles"
                )

            feats = extract_features(y, int(sr), profile, cfg)
            save_features_npz(
                sdir / "features" / f"{cfg.stem}.npz", feats, profile=profile, cfg=cfg
            )

            info: dict = {
                "profile": profile.name,
                "f0_method": cfg.f0_method,
                "n_frames": int(feats["mel"].shape[0]),
                "date": _dt.date.today().isoformat(),
            }
            if cfg.analysis:
                tempo, beats = librosa.beat.beat_track(
                    y=y, sr=sr, hop_length=profile.hop_length
                )
                info["bpm"] = round(float(np.atleast_1d(tempo)[0]), 1)
                info["beats_sec"] = [
                    round(float(t), 3)
                    for t in librosa.frames_to_time(beats, sr=sr, hop_length=profile.hop_length)
                ]
                info["key"] = estimate_key(y, int(sr))
            if cfg.phrases:
                info["phrases_sec"] = [
                    list(p)
                    for p in segment_by_silence(
                        y, int(sr),
                        min_phrase_sec=cfg.phrase_min_sec,
                        max_phrase_sec=cfg.phrase_max_sec,
                    )
                ]

            update_analysis(sdir, "features", info)
            rec.status.featurized = True
            manifest.upsert(rec)
            manifest.save()
            summary.featurized.append(rec.id)
        except Exception as exc:  # one bad song must not kill the batch
            summary.failed[rec.id] = str(exc)

    return summary
