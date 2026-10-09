"""vocadito adapter — 40 short solo-singing excerpts with lyrics, notes and F0 (S2).

vocadito (Bittner et al. 2021, Zenodo 5578807): 29 singers of mixed training, seven
languages, recorded on assorted consumer devices; 44.1 kHz mono. Annotations: lyrics
per track, two independent note annotations (A1 / A2: onset, Hz, duration) and an F0
curve. ``vocadito_metadata.csv`` gives singer id and language; no gender.
Licence string: CC BY 4.0.

Not studio audio, so records get ``source_quality=None`` (a recipe filtering on
``studio`` leaves them out) — evaluation and note-label material more than acoustic
training data. The note and F0 CSVs are kept beside the take for the variance / SOME
check. Gender is left unset for review (the metadata has none).
"""

from __future__ import annotations

import csv
from pathlib import Path

from .acapella import Take, import_takes

CORPUS = "vocadito"
LICENSE_NOTE = "vocadito (Bittner et al. 2021) — CC BY 4.0"
SOURCE_URL = "https://zenodo.org/records/5578807"
LANGUAGES = {"english": "en", "french": "fr", "spanish": "es", "tagalog": "tl",
             "catalan/valencian": "ca", "mandarin": "zh", "hawaiian": "haw"}


def _language(label: str) -> str | None:
    """'English' -> 'en'; mixed ('French+English') and 'none' -> None."""
    return LANGUAGES.get(label.strip().lower())


def find_takes(root: str | Path, *, languages=("en",)) -> tuple[list[Take], dict[str, str]]:
    root = Path(root)
    meta = root / "vocadito_metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"no vocadito_metadata.csv under {root} (extract vocadito.zip)")
    ann = root / "Annotations"
    takes, skipped = [], {}
    with open(meta, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        tid, singer, lang_label = row["track_id"].strip(), row["singer_id"].strip(), row["language"]
        key = f"vocadito_{tid}"
        lang = _language(lang_label)
        if lang is None or lang not in languages:
            skipped[key] = f"language {lang_label!r} not in {list(languages)}"
            continue
        wav = root / "Audio" / f"{key}.wav"
        lyrics = ann / "Lyrics" / f"{key}_lyrics.txt"
        if not wav.exists() or not lyrics.exists():
            skipped[key] = "missing audio or lyrics file"
            continue
        lines = lyrics.read_text(encoding="utf-8", errors="replace").splitlines()
        sidecars = [p for p in (ann / "Notes" / f"{key}_notesA1.csv",
                                ann / "Notes" / f"{key}_notesA2.csv",
                                ann / "F0" / f"{key}_f0.csv") if p.exists()]
        takes.append(Take(
            key=key,
            segments=[wav],
            rel_dir=Path("RAW") / CORPUS / key,
            singer=f"{CORPUS}-{singer}",
            gender=None,
            language=lang,
            song=key,
            lyrics=lines,
            source_quality=None,
            sidecars=sidecars,
            extra={"vocadito": {"track_id": int(tid), "singer_id": singer,
                                "language": lang_label,
                                "average_pitch_midi": float(row["average_pitch"]),
                                "annotations": [p.name for p in sidecars]}},
        ))
    return takes, skipped


def import_vocadito(data_root, *, root, languages=("en",), limit=None, dry_run=False):
    """-> (manifest, new_records, skipped, takes). Caller saves the manifest."""
    takes, skipped = find_takes(root, languages=languages)
    if limit is not None:
        takes = takes[:limit]
    manifest, new, more = import_takes(
        data_root, takes, corpus=CORPUS, license_note=LICENSE_NOTE, source_url=SOURCE_URL,
        lyrics_source="dataset:vocadito", dry_run=dry_run)
    skipped.update(more)
    return manifest, new, skipped, takes
