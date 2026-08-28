"""MedleyDB corpus adapter — metadata-driven vocal-stem ingest (S2).

MedleyDB ships one ``<Track>_METADATA.yaml`` per song (artist/title/genre/bleed plus
a per-stem instrument taxonomy) alongside the audio; the audio tarball itself carries
no labels. The taxonomy is what makes this corpus usable here: ``female singer`` /
``male singer`` are *annotated per stem*, so gender and singer identity land in the
manifest as data rather than as a blanket CLI flag — and ``dataset build``'s
``filters.gender`` becomes a real, auditable guard instead of a hopeful one.

Per matching stem the adapter records the stem WAV in the manifest with
``source_quality=studio`` and ``gender`` from the instrument label, copies it to
``songs/<id>/stems/vocals.wav`` and marks ``separated=true`` (a MedleyDB stem is
already an isolated source — Demucs would only degrade it), then writes the MedleyDB
provenance into ``analysis.json``.

Two things this corpus does NOT give us:

* **Lyrics.** MedleyDB has no transcripts, so these records reach S5 with
  ``has_lyrics=false`` and are refused by ``align``. They are immediately useful for
  the vocoder / ECAPA reference set; the acoustic model needs transcripts first (drop
  a ``<stem>.txt`` next to the source WAV and re-run — the standard sidecar rule).
* **Singer identity.** The finest label is ``artist``, so two different singers in one
  band collapse to one key, and classical entries name the *composer*. Curate those in
  the overrides file (``configs/medleydb_overrides.yaml``).

Licence: MedleyDB is CC BY-NC-SA 4.0 — non-commercial. It is recorded on every record
and rolls up into the dataset card, same discipline as the NC vocoder checkpoint (Q4).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Sequence

import yaml

from ..manifest import (
    FileInfo,
    Manifest,
    ManifestRecord,
    MetaInfo,
    SourceInfo,
    _find_lyrics_sidecar,
    _normalize_singer,
    probe_audio,
    sha256_file,
)
from ..stages.common import song_dir, update_analysis

METADATA_SUFFIX = "_METADATA.yaml"
LICENSE_NOTE = "MedleyDB — CC BY-NC-SA 4.0 (non-commercial)"

# MedleyDB's vocal instrument labels -> our gender/domain tags. ``vocalists`` is a
# group label (mixed/unknown gender) and is deliberately unmapped: it must never be
# guessed into a gendered record.
VOCAL_INSTRUMENTS: dict[str, tuple[str, str]] = {  # instrument -> (gender, domain)
    "female singer": ("F", "sung"),
    "male singer": ("M", "sung"),
    "male rapper": ("M", "sung"),
    "male speaker": ("M", "spoken"),
}
UNGENDERED_VOCAL_INSTRUMENTS = frozenset({"vocalists", "choir"})

DEFAULT_INSTRUMENTS: tuple[str, ...] = ("female singer",)

# Stem-level audio is the mix engineer's processed submix; RAW is the untouched
# mic/DI feed. Dataset recipes filter on `processing`, so tag the difference.
LEVEL_PROCESSING = {"stem": "produced", "raw": "dry"}

OVERRIDE_FIELDS = ("singer", "gender", "language", "processing", "song", "exclude")


@dataclass(frozen=True)
class VocalStem:
    """One importable vocal file plus the metadata that explains it."""

    track: str
    stem_key: str
    raw_key: str | None
    instrument: str
    component: str
    gender: str
    domain: str
    level: Literal["stem", "raw"]
    audio_path: Path
    artist: str
    title: str
    genre: str | None
    has_bleed: bool
    # other instruments bounced into the same stem (only ever non-empty with
    # include_mixed; the vocal is not isolated in that case)
    mixed_with: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Stable human/override key: ``Track:S07`` or ``Track:S07/R01``."""
        tail = f"{self.stem_key}/{self.raw_key}" if self.raw_key else self.stem_key
        return f"{self.track}:{tail}"


def load_metadata(metadata_dir: str | Path) -> dict[str, dict]:
    """``{track_name: metadata dict}`` for every ``*_METADATA.yaml`` in the folder."""
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.is_dir():
        raise FileNotFoundError(
            f"MedleyDB metadata folder not found: {metadata_dir}\n"
            f"Fetch it with:\n"
            f"  git clone --filter=blob:none --sparse --depth 1 "
            f"https://github.com/marl/medleydb.git\n"
            f"  cd medleydb && git sparse-checkout set medleydb/data/Metadata\n"
            f"then copy medleydb/data/Metadata into {metadata_dir}"
        )
    out: dict[str, dict] = {}
    for path in sorted(metadata_dir.glob(f"*{METADATA_SUFFIX}")):
        out[path.name[: -len(METADATA_SUFFIX)]] = (
            yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        )
    return out


def discover_audio_roots(
    data_root: str | Path, tracks: Sequence[str], *, max_depth: int = 4
) -> list[Path]:
    """Find folders under ``data_root`` that hold MedleyDB track directories.

    The tarball extracts to an arbitrary nesting (``MedleyDB_V2.tar/MedleyDB_V2/V2``
    here), so rather than hardcode a layout we look for directories whose children are
    named like tracks we have metadata for. Our own output trees are skipped.
    """
    data_root = Path(data_root)
    known = set(tracks)
    roots: list[Path] = []

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = [p for p in directory.iterdir() if p.is_dir()]
        except OSError:
            return
        if any(p.name in known for p in children):
            roots.append(directory)
            return  # track folders are leaves for our purposes
        for child in children:
            if child.name in {"songs", "datasets", "voices", "logs"}:
                continue
            walk(child, depth + 1)

    walk(data_root, 0)
    return roots


def _yes(value) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1"}


def _instrument_labels(value) -> list[str]:
    """``instrument`` is a string for single-source stems and a *list* for stems that
    blend sources (``[male singer, vocalists]`` — a lead with the group bounced in).
    Normalised to a lowercase list either way."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(v).strip().lower() for v in values if str(v or "").strip()]


def _resolve_track_dir(track: str, audio_roots: Sequence[Path]) -> Path | None:
    for root in audio_roots:
        candidate = Path(root) / track
        if candidate.is_dir():
            return candidate
    return None


def find_vocal_stems(
    metadata: dict[str, dict],
    audio_roots: Sequence[Path],
    *,
    instruments: Sequence[str] = DEFAULT_INSTRUMENTS,
    level: Literal["stem", "raw"] = "stem",
    melody_only: bool = False,
    include_bleed: bool = True,
    include_mixed: bool = False,
) -> tuple[list[VocalStem], dict[str, str]]:
    """Select importable vocal files from parsed metadata + on-disk audio.

    Returns ``(stems, skipped)`` where ``skipped`` maps a track/stem label to the
    reason it was left out — metadata covers all 330 MedleyDB tracks while a given
    machine holds a subset, so "audio not on this machine" is normal, not an error.
    """
    wanted = {i.strip().lower() for i in instruments}
    unknown = wanted - set(VOCAL_INSTRUMENTS) - UNGENDERED_VOCAL_INSTRUMENTS
    if unknown:
        raise ValueError(
            f"unknown MedleyDB vocal instrument(s) {sorted(unknown)}; "
            f"known: {sorted(VOCAL_INSTRUMENTS)}"
        )
    ungendered = wanted & UNGENDERED_VOCAL_INSTRUMENTS
    if ungendered:
        raise ValueError(
            f"{sorted(ungendered)} is a group label with no annotated gender — "
            f"importing it would put null-gender records in the manifest, which "
            f"`dataset build` refuses anyway. Split and tag those stems by hand."
        )

    stems: list[VocalStem] = []
    skipped: dict[str, str] = {}
    for track in sorted(metadata):
        meta = metadata[track]
        matches = {
            key: stem
            for key, stem in (meta.get("stems") or {}).items()
            if wanted & set(_instrument_labels(stem.get("instrument")))
        }
        if not matches:
            continue
        track_dir = _resolve_track_dir(track, audio_roots)
        if track_dir is None:
            skipped[track] = "audio not on this machine"
            continue
        if not include_bleed and _yes(meta.get("has_bleed")):
            skipped[track] = "has_bleed: yes (other sources leak into the stem)"
            continue

        for stem_key in sorted(matches):
            stem = matches[stem_key]
            labels = _instrument_labels(stem.get("instrument"))
            if len(labels) > 1 and not include_mixed:
                skipped[f"{track}:{stem_key}"] = (
                    f"multi-source stem {labels} — more than one instrument bounced "
                    f"into it (pass --allow-mixed to take it anyway)"
                )
                continue
            genders = {VOCAL_INSTRUMENTS[label][0]
                       for label in labels if label in VOCAL_INSTRUMENTS}
            if len(genders) > 1:
                skipped[f"{track}:{stem_key}"] = (
                    f"stem mixes genders {sorted(genders)} — no single gender label"
                )
                continue
            instrument = next(label for label in labels if label in wanted)
            gender, domain = VOCAL_INSTRUMENTS[instrument]
            component = (stem.get("component") or "").strip()
            if melody_only and component != "melody":
                skipped[f"{track}:{stem_key}"] = (
                    f"component {component or 'none'!r} != 'melody' (--melody-only)"
                )
                continue

            if level == "stem":
                files = [(None, stem.get("filename"), meta.get("stem_dir"))]
            else:
                files = [
                    (raw_key, raw.get("filename"), meta.get("raw_dir"))
                    for raw_key, raw in sorted((stem.get("raw") or {}).items())
                    if wanted & set(_instrument_labels(raw.get("instrument")))
                    and (include_mixed
                         or len(_instrument_labels(raw.get("instrument"))) == 1)
                ]
                if not files:
                    skipped[f"{track}:{stem_key}"] = "no matching raw files"
                    continue

            for raw_key, filename, subdir in files:
                if not filename or not subdir:
                    skipped[f"{track}:{stem_key}"] = "metadata has no filename/dir"
                    continue
                audio_path = track_dir / subdir / filename
                if not audio_path.exists():
                    skipped[f"{track}:{stem_key}"] = f"missing audio {audio_path.name}"
                    continue
                stems.append(
                    VocalStem(
                        track=track,
                        stem_key=stem_key,
                        raw_key=raw_key,
                        instrument=instrument,
                        component=component,
                        gender=gender,
                        domain=domain,
                        level=level,
                        audio_path=audio_path,
                        artist=(meta.get("artist") or "").strip(),
                        title=(meta.get("title") or track).strip(),
                        genre=(meta.get("genre") or "").strip().lower() or None,
                        has_bleed=_yes(meta.get("has_bleed")),
                        mixed_with=tuple(x for x in labels if x != instrument),
                    )
                )
    return stems, skipped


def load_overrides(path: str | Path | None) -> dict[str, dict]:
    """Curation file: per-track or per-stem corrections the metadata cannot express.

    ``{"Track": {...}, "Track:S05": {...}}`` with keys from :data:`OVERRIDE_FIELDS`.
    A stem-level entry wins over its track-level entry.
    """
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = data.get("tracks", data) or {}
    for key, value in entries.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"override {key!r} must be a mapping, got {type(value).__name__}")
        unknown = set(value) - set(OVERRIDE_FIELDS)
        if unknown:
            raise ValueError(
                f"override {key!r} has unknown field(s) {sorted(unknown)}; "
                f"allowed: {list(OVERRIDE_FIELDS)}"
            )
    return entries


@dataclass(frozen=True)
class PlannedImport:
    """A stem plus the tags that WILL be written — overrides already applied.

    Returned (also for ``dry_run``) so a gender-filtered import can be audited before
    a single byte is copied: what you see listed is what lands in the manifest.
    """

    stem: VocalStem
    singer: str
    gender: str
    language: str | None
    processing: str
    song: str


def _override_for(stem: VocalStem, overrides: dict[str, dict]) -> dict:
    merged = dict(overrides.get(stem.track, {}))
    merged.update(overrides.get(f"{stem.track}:{stem.stem_key}", {}))
    merged.update(overrides.get(stem.label, {}))
    return merged


def import_medleydb(
    data_root: str | Path,
    *,
    audio_roots: Sequence[str | Path] | None = None,
    metadata_dir: str | Path | None = None,
    instruments: Sequence[str] = DEFAULT_INSTRUMENTS,
    level: Literal["stem", "raw"] = "stem",
    melody_only: bool = False,
    include_bleed: bool = True,
    include_mixed: bool = False,
    language: str | None = "en",
    overrides_path: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[Manifest, list[ManifestRecord], dict[str, str], list[PlannedImport]]:
    """Import MedleyDB vocal stems into the manifest. Caller saves the manifest.

    Idempotent: a stem whose checksum is already recorded is skipped, so re-running
    after adding audio or editing overrides only adds what is new. Returns
    ``(manifest, new_records, skipped, planned)``; with ``dry_run`` nothing is copied
    or added and ``new_records`` is empty.
    """
    data_root = Path(data_root)
    metadata_dir = Path(metadata_dir) if metadata_dir \
        else data_root / "medleydb" / "Metadata"
    metadata = load_metadata(metadata_dir)
    roots = [Path(r) for r in audio_roots] if audio_roots \
        else discover_audio_roots(data_root, list(metadata))
    if not roots:
        raise FileNotFoundError(
            f"no MedleyDB audio found under {data_root} — pass --audio-root explicitly"
        )
    overrides = load_overrides(overrides_path)

    stems, skipped = find_vocal_stems(
        metadata,
        roots,
        instruments=instruments,
        level=level,
        melody_only=melody_only,
        include_bleed=include_bleed,
        include_mixed=include_mixed,
    )

    manifest = Manifest.for_data_root(data_root)
    new_records: list[ManifestRecord] = []
    planned: list[PlannedImport] = []
    for stem in stems:
        over = _override_for(stem, overrides)
        if over.get("exclude"):
            skipped[stem.label] = "excluded by overrides file"
            continue
        singer = _normalize_singer(over.get("singer") or stem.artist)
        if not singer:
            skipped[stem.label] = "no artist in metadata and no singer override"
            continue
        gender = over.get("gender") or stem.gender
        if gender not in ("F", "M"):
            skipped[stem.label] = f"invalid gender override {gender!r}"
            continue
        try:
            rel_path = stem.audio_path.relative_to(data_root).as_posix()
        except ValueError:
            skipped[stem.label] = (
                f"audio lives outside DATA_ROOT ({stem.audio_path}); manifest paths "
                f"are DATA_ROOT-relative"
            )
            continue
        # multiple vocal stems per track (lead + doubles/harmonies) share a title:
        # keep the stem key so the manifest stays readable
        plan = PlannedImport(
            stem=stem,
            singer=singer,
            gender=gender,
            language=over.get("language", language),
            processing=over.get("processing") or LEVEL_PROCESSING[stem.level],
            song=over.get("song") or f"{stem.title} [{stem.stem_key}]",
        )
        planned.append(plan)
        if dry_run:
            continue

        digest = sha256_file(stem.audio_path)
        if manifest.by_sha256(digest):
            skipped[stem.label] = "checksum already in manifest"
            planned.pop()
            continue

        duration, sr, channels = probe_audio(stem.audio_path)
        lyrics = _find_lyrics_sidecar(stem.audio_path)
        rec = ManifestRecord(
            id=manifest.next_id(),
            source=SourceInfo(kind="other", url="https://medleydb.weebly.com",
                              retrieved=date.today().isoformat()),
            file=FileInfo(
                path=rel_path,
                sha256=digest,
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
            ),
            meta=MetaInfo(
                singer=plan.singer,
                gender=plan.gender,
                song=plan.song,
                language=plan.language,
                license_note=LICENSE_NOTE,
                has_lyrics=lyrics is not None,
                lyrics_path=lyrics.relative_to(data_root).as_posix() if lyrics else None,
                source_quality="studio",
                processing=plan.processing,
                domain=stem.domain,
                genre=stem.genre,
            ),
        )
        rec.status.separated = True  # a MedleyDB stem IS the isolated source
        rec.quality.notes = (
            f"medleydb {stem.label} instrument={stem.instrument} "
            f"component={stem.component or 'none'} "
            f"bleed={'yes' if stem.has_bleed else 'no'}"
            + (f" mixed_with={list(stem.mixed_with)}" if stem.mixed_with else "")
        )

        sdir = song_dir(data_root, rec.id)
        stems_dir = sdir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stem.audio_path, stems_dir / "vocals.wav")
        update_analysis(sdir, "separate", {
            "model": "medleydb-stem" if stem.level == "stem" else "medleydb-raw",
            "imported_from": rel_path,
            "medleydb": {
                "track": stem.track,
                "stem": stem.stem_key,
                "raw": stem.raw_key,
                "instrument": stem.instrument,
                "mixed_with": list(stem.mixed_with) or None,
                "component": stem.component or None,
                "has_bleed": stem.has_bleed,
                "license": LICENSE_NOTE,
            },
            "date": date.today().isoformat(),
        })

        manifest.add(rec)
        new_records.append(rec)

    return manifest, new_records, skipped, planned
