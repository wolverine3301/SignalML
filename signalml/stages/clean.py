"""S4 clean — profile resample, loudness normalization, optional filters, silence map.

Contract: docs/PIPELINE_AND_CONTRACTS.md §S4. Manifest-driven over separated songs;
writes ``songs/<id>/clean/<stem>.wav`` at the active audio profile's rate (the profile
name is recorded in analysis.json — artifacts always know which profile made them, Q11).
Raw files and stems are never modified; the silence map is stored, not applied.

Every op is an independent config flag. De-click is deliberately not implemented yet
(a real de-clicker is nontrivial; faking one with a median filter would quietly damage
vocals) — it can join as another optional op when there's evidence it's needed.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml
from pydantic import BaseModel
from scipy import signal as _sig

from ..config import CONFIGS_DIR, AudioProfile, active_profile
from ..manifest import Manifest
from .common import song_dir, update_analysis


class CleanConfig(BaseModel):
    stems: list[str] = ["vocals"]
    resample: bool = True  # to active profile rate + mono
    loudness_normalize: bool = True
    target_lufs: float = -23.0
    peak_ceiling_dbfs: float = -1.0
    highpass: bool = False
    highpass_hz: float = 50.0
    silence_map: bool = True
    silence_top_db: float = 40.0


def load_clean_config(path: str | Path | None = None) -> CleanConfig:
    path = Path(path) if path else CONFIGS_DIR / "clean.yaml"
    if path.exists():
        return CleanConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return CleanConfig()


@dataclass
class CleanSummary:
    cleaned: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already cleaned / not yet separated
    failed: dict[str, str] = field(default_factory=dict)


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 1 else y.mean(axis=0)


def _apply_highpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = _sig.butter(2, hz, btype="highpass", fs=sr, output="sos")
    return _sig.sosfiltfilt(sos, y, axis=-1).astype(np.float32)


def _normalize_loudness(
    y: np.ndarray, sr: int, target_lufs: float, ceiling_dbfs: float
) -> tuple[np.ndarray, dict]:
    meter = pyln.Meter(sr)
    measured = y.T if y.ndim == 2 else y  # pyloudnorm expects (T,) or (T, C)
    before = float(meter.integrated_loudness(measured.astype(np.float64)))
    if not np.isfinite(before):  # silence: nothing to normalize against
        return y, {"loudness_before": None, "note": "silent input, gain skipped"}

    gain_db = target_lufs - before
    out = (y * 10 ** (gain_db / 20)).astype(np.float32)

    ceiling = 10 ** (ceiling_dbfs / 20)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    clamped = peak > ceiling
    if clamped:
        out = (out * (ceiling / peak)).astype(np.float32)

    measured_out = out.T if out.ndim == 2 else out
    after = float(meter.integrated_loudness(measured_out.astype(np.float64)))
    return out, {
        "loudness_before": round(before, 2),
        "loudness_after": round(after, 2),
        "gain_db": round(gain_db, 2),
        "peak_clamped": clamped,
    }


def _silence_map(y: np.ndarray, sr: int, top_db: float) -> dict:
    mono = _to_mono(y)
    intervals = librosa.effects.split(mono, top_db=top_db)
    total = len(mono) / sr
    voiced = float(sum(e - s for s, e in intervals)) / sr
    return {
        "top_db": top_db,
        "intervals_sec": [[round(s / sr, 3), round(e / sr, 3)] for s, e in intervals],
        "voiced_sec": round(voiced, 3),
        "silence_sec": round(total - voiced, 3),
    }


def process_audio(y: np.ndarray, sr: int, cfg: CleanConfig) -> tuple[np.ndarray, dict]:
    """Pure DSP part of the stage (resampling happens at load). Returns (audio, stats)."""
    stats: dict = {"ops": []}
    if cfg.highpass:
        y = _apply_highpass(y, sr, cfg.highpass_hz)
        stats["ops"].append("highpass")
        stats["highpass_hz"] = cfg.highpass_hz
    if cfg.loudness_normalize:
        y, loudness_stats = _normalize_loudness(y, sr, cfg.target_lufs, cfg.peak_ceiling_dbfs)
        stats["ops"].append("loudness_normalize")
        stats.update(loudness_stats)
    if cfg.silence_map:
        stats["silence"] = _silence_map(y, sr, cfg.silence_top_db)
    return y, stats


def clean(
    data_root: str | Path,
    *,
    cfg: CleanConfig | None = None,
    profile: AudioProfile | None = None,
    force: bool = False,
    limit: int | None = None,
) -> CleanSummary:
    data_root = Path(data_root)
    cfg = cfg or load_clean_config()
    profile = profile or active_profile()
    manifest = Manifest.for_data_root(data_root)
    summary = CleanSummary()

    work = []
    for rec in manifest.records:
        if not rec.status.separated or (rec.status.cleaned and not force):
            summary.skipped.append(rec.id)
        else:
            work.append(rec)
    if limit is not None:
        work = work[:limit]

    for rec in work:
        try:
            sdir = song_dir(data_root, rec.id)
            clean_dir = sdir / "clean"
            per_stem: dict[str, dict] = {}
            for stem in cfg.stems:
                src = sdir / "stems" / f"{stem}.wav"
                if not src.exists():
                    raise FileNotFoundError(f"stem missing: {src}")
                y, sr = librosa.load(
                    str(src),
                    sr=profile.sample_rate if cfg.resample else None,
                    mono=profile.mono if cfg.resample else False,
                )
                y, stats = process_audio(y, int(sr), cfg)

                clean_dir.mkdir(parents=True, exist_ok=True)
                out = y.T if y.ndim == 2 else y  # soundfile wants (T,) / (T, C)
                sf.write(str(clean_dir / f"{stem}.wav"), out.astype(np.float32), int(sr))
                per_stem[stem] = {"sample_rate": int(sr), **stats}

            update_analysis(
                sdir,
                "clean",
                {
                    "profile": profile.name,
                    "resampled": cfg.resample,
                    "stems": per_stem,
                    "date": _dt.date.today().isoformat(),
                },
            )
            rec.status.cleaned = True
            manifest.upsert(rec)
            manifest.save()
            summary.cleaned.append(rec.id)
        except Exception as exc:  # one bad song must not kill the batch
            summary.failed[rec.id] = str(exc)

    return summary
