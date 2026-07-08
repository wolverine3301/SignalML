"""S3 separate — Demucs stem separation into uniform per-song stem directories.

Contract: docs/PIPELINE_AND_CONTRACTS.md §S3. Manifest-driven (never directory-scans),
idempotent (skips ``status.separated`` records unless ``force``), records the exact
model/version/device into ``analysis.json`` for reproducibility, and saves the manifest
after every song so long batches are resumable.

The separator is injectable (like S1's downloader) so the core test suite runs offline
without torch/demucs installed; the real backend lazy-imports demucs from the ``train``
extra. Torch/demucs stay out of module import scope on purpose.

Stems are written at Demucs's native sample rate (44.1 kHz) in the source's channel
layout; per-profile resampling/mono-ing is S4's job. Spleeter has no successor here —
retired (Q6).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from pydantic import BaseModel

from ..config import CONFIGS_DIR
from ..manifest import Manifest
from .common import song_dir, update_analysis

STEM_NAMES = ("vocals", "drums", "bass", "other")

# separator(source path) -> (stem name -> array (C,T) or (T,), sample rate, info dict)
SeparatorFn = Callable[[Path], tuple[dict[str, np.ndarray], int, dict]]


class SeparateConfig(BaseModel):
    model: str = "htdemucs_ft"
    device: str = "auto"  # auto | cuda | cpu
    shifts: int = 1
    overlap: float = 0.25


def load_separate_config(path: str | Path | None = None) -> SeparateConfig:
    path = Path(path) if path else CONFIGS_DIR / "separate.yaml"
    if path.exists():
        return SeparateConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return SeparateConfig()


@dataclass
class SeparateSummary:
    separated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already separated
    failed: dict[str, str] = field(default_factory=dict)  # id -> error


def build_demucs_separator(cfg: SeparateConfig) -> SeparatorFn:
    """Real backend: Demucs via its Python API (lazy imports from the train extra)."""
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError(
            "demucs/torch not installed - install the train extra: "
            "python -m uv sync --extra train  (GPU rig: see README 'GPU install')"
        ) from exc

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(cfg.model)
    model.to(device)
    model.eval()

    def _separate(path: Path) -> tuple[dict[str, np.ndarray], int, dict]:
        import librosa

        y, _sr = librosa.load(str(path), sr=model.samplerate, mono=False)
        if y.ndim == 1:  # model expects stereo; duplicate mono
            y = np.stack([y, y])
        wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

        # Demucs-CLI-style normalization, undone on the way out.
        ref = wav.mean(0)
        mean, std = float(ref.mean()), float(ref.std()) or 1.0
        with torch.no_grad():
            sources = apply_model(
                model,
                ((wav - mean) / std)[None],
                device=device,
                shifts=cfg.shifts,
                split=True,
                overlap=cfg.overlap,
                progress=False,
            )[0]
        sources = sources * std + mean

        stems = {name: sources[i].cpu().numpy() for i, name in enumerate(model.sources)}
        info = {
            "model": cfg.model,
            "demucs_version": _pkg_version("demucs"),
            "device": device,
            "sample_rate": int(model.samplerate),
            "shifts": cfg.shifts,
            "overlap": cfg.overlap,
        }
        return stems, int(model.samplerate), info

    return _separate


def _write_stem(path: Path, arr: np.ndarray, sr: int) -> None:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] <= 8:  # channels-first (C,T) -> (T,C)
        a = a.T
    sf.write(str(path), a, sr)


def separate(
    data_root: str | Path,
    *,
    cfg: SeparateConfig | None = None,
    separator_factory: Callable[[], SeparatorFn] | None = None,
    force: bool = False,
    limit: int | None = None,
) -> SeparateSummary:
    data_root = Path(data_root)
    cfg = cfg or load_separate_config()
    manifest = Manifest.for_data_root(data_root)
    summary = SeparateSummary()

    work = []
    for rec in manifest.records:
        if rec.status.separated and not force:
            summary.skipped.append(rec.id)
        else:
            work.append(rec)
    if limit is not None:
        work = work[:limit]
    if not work:
        return summary  # never builds the (heavy) separator when idle

    separator = (separator_factory or (lambda: build_demucs_separator(cfg)))()

    for rec in work:
        src = data_root / rec.file.path
        try:
            if not src.exists():
                raise FileNotFoundError(f"source audio missing: {src}")
            stems, sr, info = separator(src)
            missing = set(STEM_NAMES) - stems.keys()
            if missing:
                raise ValueError(f"separator returned no {sorted(missing)} stem(s)")

            sdir = song_dir(data_root, rec.id) / "stems"
            sdir.mkdir(parents=True, exist_ok=True)
            for name, arr in stems.items():
                _write_stem(sdir / f"{name}.wav", arr, sr)

            update_analysis(
                song_dir(data_root, rec.id),
                "separate",
                {**info, "date": _dt.date.today().isoformat()},
            )
            rec.status.separated = True
            manifest.upsert(rec)
            manifest.save()  # per-song save: long batches are resumable
            summary.separated.append(rec.id)
        except Exception as exc:  # one bad song must not kill the batch
            summary.failed[rec.id] = str(exc)

    return summary
