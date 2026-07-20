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
import re
import shutil
import time
from contextlib import contextmanager
from datetime import date
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
    # "studio" = real dry stems (gold for vocoder training); "separated" = Demucs output
    # with the mix's production baked in. Dataset recipes filter/weight on this.
    source_quality: Literal["studio", "separated"] | None = None
    # how much production is baked into the VOICE itself (orthogonal to source_quality):
    # dry = natural voice, produced = normal mix polish, heavy = audible autotune/FX.
    # Human-tagged via META.txt PROCESSING:; recipes exclude/include per experiment.
    processing: Literal["dry", "produced", "heavy"] | None = None
    # wave-3 (DECISION_POINTS D10): speech corpora join with domain=spoken;
    # everything sung so far defaults accordingly
    domain: Literal["sung", "spoken"] = "sung"
    genre: str | None = None  # from META.txt GENRE: (covers-experiment analysis, style tags)


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
        """Atomic write: temp file in the same directory, then os.replace.

        Dumps this instance's full in-memory view — correct for single-writer flows
        (scan, acquire). Stage loops that may run concurrently with other stages must
        use :meth:`commit` instead, or a parallel stage's save will silently clobber
        their updates with its stale snapshot.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        lines = [rec.model_dump_json(exclude_none=False) for rec in self._records.values()]
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, self.path)

    def commit(self, record: ManifestRecord) -> None:
        """Persist ONE record: lock, re-read the file, merge just this record, write.

        This is the concurrency-safe save for stage loops. Cross-song: only this
        record is touched, so a parallel stage can't wipe other songs (the bug the
        full-snapshot ``save`` had). Same-song: status flags OR-merge (stages only
        ever set their flag true; nothing legitimately unsets one via commit) and
        quality fields prefer non-None, so two stages committing the same song keep
        both stages' updates. Unsetting flags deliberately = edit the manifest, not
        a stage commit. The in-memory view refreshes to the merged state.
        """
        with _manifest_lock(self.path):
            on_disk = Manifest(self.path)
            existing = on_disk._records.get(record.id)
            if existing is not None:
                for flag in StatusFlags.model_fields:
                    if getattr(existing.status, flag):
                        setattr(record.status, flag, True)
                for field in QualityInfo.model_fields:
                    if getattr(record.quality, field) in (None, "") and \
                            getattr(existing.quality, field) not in (None, ""):
                        setattr(record.quality, field, getattr(existing.quality, field))
            on_disk._records[record.id] = record
            on_disk.save()
            self._records = on_disk._records


@contextmanager
def _manifest_lock(manifest_path: Path, timeout_sec: float = 15.0):
    """Cross-process mutex via an O_EXCL lock file next to the manifest. A lock older
    than ``timeout_sec`` is treated as abandoned (crashed writer) and stolen."""
    lock = manifest_path.with_suffix(".jsonl.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > timeout_sec:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                continue  # holder released it between our checks
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not acquire manifest lock {lock}") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def _find_lyrics_sidecar(audio_path: Path) -> Path | None:
    """``<audio-stem>.txt`` next to the audio, or ``lyrics.txt`` in its folder (the
    corpus convention: one song per folder with lyrics.txt + META.txt siblings)."""
    candidate = audio_path.with_suffix(".txt")
    if candidate.exists():
        return candidate
    folder_lyrics = audio_path.parent / "lyrics.txt"
    return folder_lyrics if folder_lyrics.exists() else None


def _normalize_singer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split()).lower()
    return cleaned or None


_PROCESSING_VALUES = ("dry", "produced", "heavy")
_DOMAIN_VALUES = ("sung", "spoken")

# Logan's letter grades (META.txt QUALITY:): A = clean/natural recording, lower =
# progressively heavier production (autotune etc.). Mapped onto the processing tag.
_QUALITY_LETTERS = {"a": "dry", "b": "produced", "c": "heavy", "d": "heavy",
                    "e": "heavy", "f": "heavy"}


def _valid_tag(value: str | None, allowed: tuple[str, ...]) -> str | None:
    """Normalize a META.txt tag value; unknown values are dropped (None), not errors —
    the retag report surfaces them for the human to fix."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned if cleaned in allowed else None


def _processing_from_sidecar(sidecar: dict[str, str]) -> str | None:
    """PROCESSING: dry|produced|heavy wins; QUALITY: letter grade maps as fallback."""
    explicit = _valid_tag(sidecar.get("PROCESSING"), _PROCESSING_VALUES)
    if explicit:
        return explicit
    letter = (sidecar.get("QUALITY") or "").strip().lower()
    return _QUALITY_LETTERS.get(letter)


def _read_meta_sidecar(audio_path: Path) -> dict[str, str]:
    """Parse a ``META.txt`` next to the audio (corpus convention): ``KEY:value`` lines
    (SONG/SINGER/ARTIST/GENRE/TYPE/QUALITY). Empty values are dropped."""
    meta_path = audio_path.parent / "META.txt"
    if not meta_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, sep, val = line.partition(":")
        if sep and val.strip():
            out[key.strip().upper()] = val.strip()
    return out


def scan_directory(
    data_root: str | Path,
    *,
    subpath: str = "raw",
    language: str | None = None,
    gender: Literal["F", "M"] | None = None,
    singer: str | None = None,
    source_quality: Literal["studio", "separated"] | None = None,
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
        sidecar = _read_meta_sidecar(path)  # per-song META.txt beats the blanket CLI tags
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
                song=sidecar.get("SONG") or path.stem,
                language=language,
                gender=gender,
                # singer is the timbre-space label key (ARCHITECTURE §4): normalize
                # case/whitespace so "RUNN" and "runn" are one singer, not two
                singer=_normalize_singer(sidecar.get("SINGER") or singer),
                has_lyrics=lyrics is not None,
                lyrics_path=lyrics.relative_to(data_root).as_posix() if lyrics else None,
                source_quality=source_quality,
                processing=_processing_from_sidecar(sidecar),
                domain=_valid_tag(sidecar.get("DOMAIN"), _DOMAIN_VALUES) or "sung",
                genre=(sidecar.get("GENRE") or "").strip().lower() or None,
            ),
        )
        manifest.add(rec)
        new_records.append(rec)

    return manifest, new_records


_YT_ID_RE = re.compile(r"[-_]([A-Za-z0-9_-]{11})$")


def _yt_id(name: str) -> str | None:
    """Trailing 11-char YouTube id in a folder/file name — the cross-corpus dedupe
    key (the same performance can exist as a full mix and as legacy stems)."""
    m = _YT_ID_RE.search(name)
    return m.group(1) if m else None


def import_stem_folders(
    data_root: str | Path,
    *,
    subpath: str,
    language: str | None = None,
    gender: Literal["F", "M"] | None = None,
    stem_filename: str = "vocals.wav",
) -> tuple[Manifest, list[ManifestRecord], dict[str, str]]:
    """Onboard pre-separated one-song-per-folder stems (vocals.wav [+ accompaniment])
    without running Demucs: the vocal stem is copied into ``songs/<id>/stems/`` and
    the record starts life ``separated=true`` (provenance recorded in analysis.json).

    Dedupe: a folder whose YouTube id matches an existing record is skipped — the
    existing (htdemucs-separated) version beats legacy Spleeter stems. Returns
    (manifest, new_records, skipped {folder: reason}); caller saves.
    """
    from .stages.common import song_dir, update_analysis  # local: avoid import cycle

    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    root = data_root / subpath
    if not root.exists():
        raise FileNotFoundError(f"Import path does not exist: {root}")

    known_ids: dict[str, str] = {}
    for rec in manifest.records:
        p = Path(rec.file.path)
        yid = _yt_id(p.parent.name) or _yt_id(p.stem)
        if yid:
            known_ids[yid] = rec.id

    new_records: list[ManifestRecord] = []
    skipped: dict[str, str] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        vocal = folder / stem_filename
        if not vocal.exists():
            skipped[folder.name] = f"no {stem_filename}"
            continue
        yid = _yt_id(folder.name)
        if yid and yid in known_ids:
            skipped[folder.name] = (f"duplicate of {known_ids[yid]} "
                                    f"(existing separation preferred)")
            continue
        digest = sha256_file(vocal)
        if manifest.by_sha256(digest):
            skipped[folder.name] = "checksum already in manifest"
            continue

        duration, sr, channels = probe_audio(vocal)
        lyrics = _find_lyrics_sidecar(vocal)
        sidecar = _read_meta_sidecar(vocal)
        rec = ManifestRecord(
            id=manifest.next_id(),
            source=SourceInfo(kind="local"),
            file=FileInfo(
                path=vocal.relative_to(data_root).as_posix(),
                sha256=digest,
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
            ),
            meta=MetaInfo(
                song=sidecar.get("SONG") or folder.name,
                language=language,
                gender=gender,
                singer=_normalize_singer(sidecar.get("SINGER")),
                has_lyrics=lyrics is not None,
                lyrics_path=lyrics.relative_to(data_root).as_posix() if lyrics else None,
                source_quality="separated",
                processing=_processing_from_sidecar(sidecar),
                domain=_valid_tag(sidecar.get("DOMAIN"), _DOMAIN_VALUES) or "sung",
                genre=(sidecar.get("GENRE") or "").strip().lower() or None,
            ),
        )
        rec.status.separated = True

        sdir = song_dir(data_root, rec.id)
        stems_dir = sdir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vocal, stems_dir / "vocals.wav")
        update_analysis(sdir, "separate", {
            "model": "imported-legacy-stems",
            "imported_from": vocal.relative_to(data_root).as_posix(),
            "date": date.today().isoformat(),
        })

        if yid:
            known_ids[yid] = rec.id
        manifest.add(rec)
        new_records.append(rec)

    return manifest, new_records, skipped


RETAG_SAFE_FIELDS = ("processing", "domain", "genre")


def retag_from_sidecars(
    data_root: str | Path,
    *,
    fields: tuple[str, ...] = RETAG_SAFE_FIELDS,
) -> tuple[Manifest, list[tuple[str, str, str | None, str | None]], list[tuple[str, str]]]:
    """Refresh chosen meta fields on EXISTING records from their META.txt sidecars.

    Default fields are the additive tags only — deliberately NOT singer/song, which
    were hand-repaired in the manifest and must not be clobbered by source-file typos
    (fix META.txt first, then opt in via fields=). Returns (manifest, changes as
    (id, field, old, new), warnings as (id, message)); caller saves.
    """
    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    changes: list[tuple[str, str, str | None, str | None]] = []
    warnings: list[tuple[str, str]] = []

    for rec in manifest.records:
        sidecar = _read_meta_sidecar(data_root / rec.file.path)
        if not sidecar:
            continue
        new_values: dict[str, str | None] = {}
        if "processing" in fields and ("PROCESSING" in sidecar or "QUALITY" in sidecar):
            value = _processing_from_sidecar(sidecar)
            if value is None:
                raw = sidecar.get("PROCESSING") or sidecar.get("QUALITY")
                warnings.append((rec.id, f"PROCESSING/QUALITY {raw!r} not in "
                                         f"{_PROCESSING_VALUES} or letter grades "
                                         f"{sorted(_QUALITY_LETTERS)} — ignored"))
            else:
                new_values["processing"] = value
        if "domain" in fields and "DOMAIN" in sidecar:
            value = _valid_tag(sidecar["DOMAIN"], _DOMAIN_VALUES)
            if value is None:
                warnings.append((rec.id, f"DOMAIN:{sidecar['DOMAIN']!r} not in "
                                         f"{_DOMAIN_VALUES} — ignored"))
            else:
                new_values["domain"] = value
        if "genre" in fields and sidecar.get("GENRE"):
            new_values["genre"] = sidecar["GENRE"].strip().lower()
        if "singer" in fields and sidecar.get("SINGER"):
            new_values["singer"] = _normalize_singer(sidecar["SINGER"])
        if "song" in fields and sidecar.get("SONG"):
            new_values["song"] = sidecar["SONG"]

        for field, new in new_values.items():
            old = getattr(rec.meta, field)
            if old != new:
                setattr(rec.meta, field, new)
                changes.append((rec.id, field, old, new))
        manifest.upsert(rec)

    return manifest, changes, warnings


def _hours(records: list[ManifestRecord]) -> float:
    return round(sum(r.file.duration_sec or 0.0 for r in records) / 3600, 2)


def manifest_report(manifest: Manifest) -> str:
    """Corpus census: the numbers that shape training decisions (ARCHITECTURE §4/§7).

    Singer *count* is the load-bearing one — it decides whether the voice-bank
    sampling space produces novel voices or blends. Also reports lyrics coverage
    (Q14 verification pass), language/gender/source-quality splits, and stage status.
    """
    recs = manifest.records
    lines = [f"records: {len(recs)}   hours: {_hours(recs)}"]

    def group(title: str, key) -> None:
        buckets: dict[str, list[ManifestRecord]] = {}
        for r in recs:
            buckets.setdefault(key(r) or "(untagged)", []).append(r)
        lines.append(f"\n{title}:")
        for name in sorted(buckets):
            b = buckets[name]
            lines.append(f"  {name}: {len(b)} song(s), {_hours(b)} h")

    group("by language", lambda r: r.meta.language)
    group("by gender", lambda r: r.meta.gender)
    group("by domain", lambda r: r.meta.domain)
    group("by source quality", lambda r: r.meta.source_quality)
    group("by processing (voice naturalness)", lambda r: r.meta.processing)
    group("by singer (voice-bank census)", lambda r: r.meta.singer)

    n_singers = len({r.meta.singer for r in recs if r.meta.singer})
    untagged = sum(1 for r in recs if not r.meta.singer)
    lines.append(f"\ndistinct singers: {n_singers}"
                 + (f"   (WARNING: {untagged} record(s) missing singer — "
                    f"the timbre space needs singer labels)" if untagged else ""))

    missing_lyrics = [r.id for r in recs if not r.meta.has_lyrics]
    lines.append(f"lyrics coverage: {len(recs) - len(missing_lyrics)}/{len(recs)}")
    if missing_lyrics:
        lines.append(f"  missing: {', '.join(missing_lyrics[:20])}"
                     + (" ..." if len(missing_lyrics) > 20 else ""))

    lines.append("\nstage status:")
    for flag in StatusFlags.model_fields:
        done = sum(1 for r in recs if getattr(r.status, flag))
        lines.append(f"  {flag}: {done}/{len(recs)}")
    return "\n".join(lines)
