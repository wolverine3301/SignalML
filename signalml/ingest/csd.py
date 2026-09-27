"""CSD (Children's Song Dataset) adapter — studio a cappella children's songs (S2).

50 English (and 50 Korean) children's songs by one professional female singer, each
recorded in two keys (``…a`` / ``…b``), 44.1 kHz 16-bit, with plain-text lyrics per
recording. **CC BY-NC-SA 4.0** — research / development only.

Layout read: ``<root>/english/{wav,lyric}/<stem>.{wav,txt}`` (Zenodo 4785016, CSD.zip).
Records land ``separated=true`` (a cappella: nothing to separate) with the lyrics as a
sidecar, so they take the normal path from here: clean -> Whisper -> ``lyrics
--resolve`` -> ``align --phrases``. CSD's own annotations are note-level (phonemes tied
per syllable, no phone boundaries), so they are not used as alignments.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from ..manifest import (
    FileInfo,
    Manifest,
    ManifestRecord,
    MetaInfo,
    SourceInfo,
    probe_audio,
    sha256_file,
)
from ..stages.common import song_dir, update_analysis

CORPUS = "csd"
SINGER = "csd-en"
LICENSE_NOTE = "CSD — CC BY-NC-SA 4.0 (non-commercial: research / dev only)"
SOURCE_URL = "https://zenodo.org/records/4785016"


def _lang_dir(root: Path, language: str) -> Path:
    for d in root.iterdir() if root.is_dir() else []:
        if d.is_dir() and d.name.lower() == language.lower():
            return d
    raise FileNotFoundError(f"no {language}/ folder under {root} (extract CSD.zip there)")


def import_csd(
    data_root: str | Path,
    *,
    root: str | Path,
    language: str = "english",
    limit: int | None = None,
) -> tuple[Manifest, list[ManifestRecord], dict[str, str]]:
    """Import CSD's recordings for one language. Caller saves the manifest.

    Idempotent by checksum. Returns ``(manifest, new_records, skipped)``."""
    data_root = Path(data_root)
    lang = _lang_dir(Path(root), language)
    wavs = sorted((lang / "wav").glob("*.wav"))
    if limit is not None:
        wavs = wavs[:limit]
    manifest = Manifest.for_data_root(data_root)
    new_records: list[ManifestRecord] = []
    skipped: dict[str, str] = {}
    today = date.today().isoformat()
    for wav in wavs:
        lyrics_src = lang / "lyric" / f"{wav.stem}.txt"
        if not lyrics_src.exists():
            skipped[wav.stem] = f"no lyrics at {lyrics_src}"
            continue
        text = lyrics_src.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            skipped[wav.stem] = "empty lyrics file"
            continue
        digest = sha256_file(wav)
        if manifest.by_sha256(digest):
            skipped[wav.stem] = "checksum already in manifest"
            continue
        rel_dir = Path("RAW") / CORPUS / language.lower() / wav.stem
        out_dir = data_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav, out_dir / wav.name)
        (out_dir / "lyrics.txt").write_text(text + "\n", encoding="utf-8")
        duration, sr, channels = probe_audio(out_dir / wav.name)
        rec = ManifestRecord(
            id=manifest.next_id(),
            source=SourceInfo(kind="other", url=SOURCE_URL, retrieved=today),
            file=FileInfo(path=(rel_dir / wav.name).as_posix(), sha256=digest,
                          duration_sec=duration, sample_rate=sr, channels=channels),
            meta=MetaInfo(
                singer=SINGER,
                gender="F",
                song=wav.stem,
                language="en" if language.lower().startswith("en") else language.lower(),
                license_note=LICENSE_NOTE,
                has_lyrics=True,
                lyrics_path=(rel_dir / "lyrics.txt").as_posix(),
                lyrics_source="dataset:csd",
                source_quality="studio",
                processing="dry",
                domain="sung",
                corpus=CORPUS,
            ),
        )
        rec.status.separated = True  # studio a cappella: nothing to separate
        sdir = song_dir(data_root, rec.id)
        (sdir / "stems").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(wav, sdir / "stems" / "vocals.wav")
        update_analysis(sdir, "separate", {"model": "csd-acapella",
                                           "imported_from": str(wav),
                                           "license": LICENSE_NOTE, "date": today})
        manifest.add(rec)
        new_records.append(rec)
    return manifest, new_records, skipped
