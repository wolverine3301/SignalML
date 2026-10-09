"""Shared importer for a cappella corpora that ship lyrics but no phone alignment (S2).

A corpus adapter (``ravdess.py``, ``vocadito.py``, ...) reads its own layout into
:class:`Take` objects; :func:`import_takes` does the rest the same way for every corpus:
the take's WAV(s) are copied (one segment) or joined end to end (several) into
``RAW/<corpus>/...`` at their native rate, the lyrics become a ``lyrics.txt`` sidecar,
and the record lands ``separated=true`` (nothing to separate) with the stem in place.
From there the take follows the normal path: clean -> Whisper -> ``lyrics --resolve``
-> ``align --phrases``. Corpora that DO ship phone alignments convert them instead
(``gtsinger.py``).

Idempotent by checksum of the copied / joined take.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from ..manifest import (
    FileInfo,
    Manifest,
    ManifestRecord,
    MetaInfo,
    SourceInfo,
    _normalize_singer,
    probe_audio,
    sha256_file,
)
from ..stages.common import song_dir, update_analysis


@dataclass
class Take:
    key: str                          # unique label, used in reports / skipped
    segments: list[Path]              # source WAV(s), joined end to end in this order
    rel_dir: Path                     # destination under the data root (RAW/<corpus>/...)
    singer: str                       # namespaced singer id, e.g. "ravdess-actor01"
    gender: Literal["F", "M"] | None
    language: str | None
    song: str
    lyrics: list[str]                 # lyric lines (one per phrase where known)
    source_quality: Literal["studio", "separated"] | None = "studio"
    processing: Literal["dry", "produced", "heavy"] | None = "dry"
    sidecars: list[Path] = field(default_factory=list)  # annotation files kept beside the take
    extra: dict = field(default_factory=dict)            # corpus labels -> analysis.json

    @property
    def filename(self) -> str:
        return self.segments[0].name if len(self.segments) == 1 else "take.wav"


def join_wavs(segments: Iterable[Path]) -> tuple[np.ndarray, int, list[float]]:
    """WAVs joined end to end, downmixed to mono -> (audio, sr, start offset of each)."""
    parts, offsets, sr, t = [], [], None, 0.0
    for w in segments:
        y, r = sf.read(str(w), dtype="float32", always_2d=True)
        if sr is None:
            sr = r
        elif r != sr:
            raise ValueError(f"{w}: {r} Hz, the rest of the take is {sr} Hz")
        parts.append(y.mean(axis=1))
        offsets.append(t)
        t += len(parts[-1]) / r
    if sr is None:
        raise ValueError("no segments to join")
    return np.concatenate(parts), int(sr), offsets


def _place(take: Take, out_dir: Path) -> tuple[Path, list[float]]:
    """Write the take's audio into ``out_dir`` (once) -> (path, segment offsets)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / take.filename
    if len(take.segments) == 1:
        if not dest.exists():
            shutil.copyfile(take.segments[0], dest)
        return dest, [0.0]
    audio, sr, offsets = join_wavs(take.segments)
    if not dest.exists():
        sf.write(str(dest), audio, sr, subtype="PCM_24")
    return dest, offsets


def import_takes(
    data_root: str | Path,
    takes: list[Take],
    *,
    corpus: str,
    license_note: str,
    source_url: str,
    lyrics_source: str,
    dry_run: bool = False,
) -> tuple[Manifest, list[ManifestRecord], dict[str, str]]:
    """Import prepared takes. Caller saves the manifest. -> (manifest, new, skipped)."""
    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    new_records: list[ManifestRecord] = []
    skipped: dict[str, str] = {}
    if dry_run:
        return manifest, new_records, skipped
    today = date.today().isoformat()
    for take in takes:
        try:
            lines = [ln.strip() for ln in take.lyrics if ln.strip()]
            if not lines:
                skipped[take.key] = "no lyrics"
                continue
            out_dir = data_root / take.rel_dir
            wav, offsets = _place(take, out_dir)
            digest = sha256_file(wav)
            if manifest.by_sha256(digest):
                skipped[take.key] = "checksum already in manifest"
                continue
            (out_dir / "lyrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            for extra in take.sidecars:
                shutil.copyfile(extra, out_dir / extra.name)
            duration, rate, channels = probe_audio(wav)
            rec = ManifestRecord(
                id=manifest.next_id(),
                source=SourceInfo(kind="other", url=source_url, retrieved=today),
                file=FileInfo(path=(take.rel_dir / wav.name).as_posix(), sha256=digest,
                              duration_sec=duration, sample_rate=rate, channels=channels),
                meta=MetaInfo(
                    singer=_normalize_singer(take.singer),
                    gender=take.gender,
                    song=take.song,
                    language=take.language,
                    license_note=license_note,
                    has_lyrics=True,
                    lyrics_path=(take.rel_dir / "lyrics.txt").as_posix(),
                    lyrics_source=lyrics_source,
                    source_quality=take.source_quality,
                    processing=take.processing,
                    domain="sung",
                    corpus=corpus,
                ),
            )
            rec.status.separated = True  # a cappella: nothing to separate
            sdir = song_dir(data_root, rec.id)
            (sdir / "stems").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wav, sdir / "stems" / "vocals.wav")
            update_analysis(sdir, "separate", {
                "model": f"{corpus}-acapella",
                "imported_from": [str(s) for s in take.segments],
                "segment_offsets_sec": [round(o, 4) for o in offsets],
                "license": license_note, "date": today})
            if take.extra:
                update_analysis(sdir, "import", json.loads(json.dumps(take.extra)))
            manifest.add(rec)
            new_records.append(rec)
        except Exception as exc:  # one bad take must not stop the corpus
            skipped[take.key] = f"failed: {exc}"
    return manifest, new_records, skipped
