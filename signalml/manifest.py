"""Manifest — the JSONL spine of the pipeline (docs/PIPELINE_AND_CONTRACTS.md §2).

One record per source recording, keyed by an ``sng_NNNN`` id assigned exactly once.
Stages never directory-scan for work: they query the manifest (``select``) and update
their own status flags. Saves are atomic (write-temp -> replace).

``scan_directory`` is the S2 backfill command: it proposes records for audio already
sitting in ``raw/`` (checksum, duration, lyrics-sidecar detection per Q14) so the
existing 78 h corpus can be onboarded without re-downloading anything.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import soundfile as sf
from pydantic import BaseModel, Field

# Formats libsndfile may not read (duration probing falls back to None for these).
AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".opus", ".webm", ".mka",
}

ENV_DATA_ROOT = "SIGNALML_DATA_ROOT"
MANIFEST_NAME = "manifest.jsonl"


class SourceInfo(BaseModel):
    kind: Literal["youtube", "local", "other"] = "local"
    url: str | None = None
    retrieved: str | None = None  # ISO date


class FileInfo(BaseModel):
    path: str  # relative to DATA_ROOT, posix-style
    sha256: str
    duration_sec: float | None = None
    sample_rate: int | None = None
    channels: int | None = None


class MetaInfo(BaseModel):
    singer: str | None = None
    gender: Literal["F", "M"] | None = None  # required non-null before dataset build (S6b)
    song: str | None = None
    language: str | None = None  # "en", "ga", "gd", ... (Q13)
    license_note: str | None = None
    has_lyrics: bool = False  # Q14: .txt sidecar next to the audio
    lyrics_path: str | None = None


class StatusFlags(BaseModel):
    separated: bool = False
    cleaned: bool = False
    aligned: bool = False
    featurized: bool = False


class QualityInfo(BaseModel):
    separation_snr_est: float | None = None
    align_score: float | None = None
    notes: str = ""


class ManifestRecord(BaseModel):
    id: str
    source: SourceInfo = Field(default_factory=SourceInfo)
    file: FileInfo
    meta: MetaInfo = Field(default_factory=MetaInfo)
    status: StatusFlags = Field(default_factory=StatusFlags)
    quality: QualityInfo = Field(default_factory=QualityInfo)


def resolve_data_root(arg: str | Path | None = None) -> Path:
    """Precedence: explicit arg > SIGNALML_DATA_ROOT env > ./data."""
    if arg:
        return Path(arg)
    env = os.environ.get(ENV_DATA_ROOT)
    if env:
        return Path(env)
    return Path("data")


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def probe_audio(path: str | Path) -> tuple[float | None, int | None, int | None]:
    """(duration_sec, sample_rate, channels), or Nones for formats libsndfile can't read."""
    try:
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate), int(info.channels)
    except Exception:
        return None, None, None


class Manifest:
    """In-memory view of manifest.jsonl with atomic persistence."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: dict[str, ManifestRecord] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = ManifestRecord.model_validate(json.loads(line))
                self._records[rec.id] = rec

    @classmethod
    def for_data_root(cls, data_root: str | Path) -> Manifest:
        return cls(Path(data_root) / MANIFEST_NAME)

    # -- queries ---------------------------------------------------------

    @property
    def records(self) -> list[ManifestRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def get(self, record_id: str) -> ManifestRecord | None:
        return self._records.get(record_id)

    def by_url(self, url: str) -> ManifestRecord | None:
        for rec in self._records.values():
            if rec.source.url == url:
                return rec
        return None

    def by_sha256(self, digest: str) -> ManifestRecord | None:
        for rec in self._records.values():
            if rec.file.sha256 == digest:
                return rec
        return None

    def select(self, **status_flags: bool) -> list[ManifestRecord]:
        """Filter by status flags, e.g. select(separated=True, aligned=False)."""
        valid = set(StatusFlags.model_fields)
        unknown = set(status_flags) - valid
        if unknown:
            raise KeyError(f"Unknown status flags {sorted(unknown)}; valid: {sorted(valid)}")
        out = []
        for rec in self._records.values():
            if all(getattr(rec.status, k) == v for k, v in status_flags.items()):
                out.append(rec)
        return out

    def next_id(self, prefix: str = "sng") -> str:
        top = 0
        for rid in self._records:
            head, _, num = rid.rpartition("_")
            if head == prefix and num.isdigit():
                top = max(top, int(num))
        return f"{prefix}_{top + 1:04d}"

    # -- mutation --------------------------------------------------------

    def add(self, record: ManifestRecord) -> None:
        if record.id in self._records:
            raise ValueError(f"Duplicate manifest id {record.id!r}")
        self._records[record.id] = record

    def upsert(self, record: ManifestRecord) -> None:
        self._records[record.id] = record

    def save(self) -> None:
        """Atomic write: temp file in the same directory, then os.replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        lines = [rec.model_dump_json(exclude_none=False) for rec in self._records.values()]
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, self.path)


def _find_lyrics_sidecar(audio_path: Path) -> Path | None:
    candidate = audio_path.with_suffix(".txt")
    return candidate if candidate.exists() else None


def scan_directory(
    data_root: str | Path,
    *,
    subpath: str = "raw",
    language: str | None = None,
    gender: Literal["F", "M"] | None = None,
    singer: str | None = None,
) -> tuple[Manifest, list[ManifestRecord]]:
    """Backfill manifest records for audio files already under ``data_root/subpath``.

    Recursive; skips files whose checksum is already recorded. ``language``/``gender``/
    ``singer`` apply to every *new* record — run once per corpus folder to tag the
    en/ga/gd splits (Q13). Returns (manifest, newly added records); caller saves.
    """
    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    scan_root = data_root / subpath
    if not scan_root.exists():
        raise FileNotFoundError(f"Scan path does not exist: {scan_root}")

    new_records: list[ManifestRecord] = []
    for path in sorted(scan_root.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in AUDIO_EXTS):
            continue
        digest = sha256_file(path)
        if manifest.by_sha256(digest):
            continue

        duration, sr, channels = probe_audio(path)
        lyrics = _find_lyrics_sidecar(path)
        rec = ManifestRecord(
            id=manifest.next_id(),
            source=SourceInfo(kind="local"),
            file=FileInfo(
                path=path.relative_to(data_root).as_posix(),
                sha256=digest,
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
            ),
            meta=MetaInfo(
                song=path.stem,
                language=language,
                gender=gender,
                singer=singer,
                has_lyrics=lyrics is not None,
                lyrics_path=lyrics.relative_to(data_root).as_posix() if lyrics else None,
            ),
        )
        manifest.add(rec)
        new_records.append(rec)

    return manifest, new_records
