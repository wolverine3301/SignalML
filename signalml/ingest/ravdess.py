"""RAVDESS song adapter — short sung statements by 23 actors (S2).

The Ryerson Audio-Visual Database of Emotional Speech and Song (Zenodo 1188976,
``Audio_Song_Actors_01-24.zip``): professional actors (odd ids male, even ids female;
actor 18 has no song recordings) sing two fixed statements to a set melody in neutral,
calm, happy, sad, angry and fearful renditions, two intensities, two repetitions each.
Studio, 48 kHz 16-bit. Licence string: CC BY-NC-SA 4.0.

File names are the label: ``03-02-EE-II-SS-RR-AA.wav`` = audio-only, song, emotion,
intensity, statement, repetition, actor. One record per (actor, emotion, intensity):
its four clips (2 statements x 2 repetitions) are joined into one take whose lyrics are
the four statements, so each record is a short phrase sequence the normal phrase
alignment handles. The emotion / intensity labels go to ``analysis.json``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .acapella import Take, import_takes

CORPUS = "ravdess"
LICENSE_NOTE = "RAVDESS (Livingstone & Russo 2018) — CC BY-NC-SA 4.0"
SOURCE_URL = "https://zenodo.org/records/1188976"
STATEMENTS = {"01": "kids are talking by the door", "02": "dogs are sitting by the door"}
EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry",
            "06": "fearful", "07": "disgust", "08": "surprised"}
INTENSITIES = {"01": "normal", "02": "strong"}
_NAME = re.compile(r"(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)")


def gender_of(actor: int) -> str:
    return "M" if actor % 2 else "F"


def find_takes(root: str | Path, *, genders=("F", "M")) -> tuple[list[Take], dict[str, str]]:
    root = Path(root)
    wavs = sorted(root.glob("Actor_*/*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no Actor_*/*.wav under {root} (extract "
                                f"Audio_Song_Actors_01-24.zip there)")
    groups: dict[tuple[int, str, str], list[Path]] = defaultdict(list)
    skipped: dict[str, str] = {}
    for w in wavs:
        m = _NAME.fullmatch(w.stem)
        if not m:
            skipped[w.name] = "not a RAVDESS file name"
            continue
        modality, channel, emo, inten, stmt, _rep, actor = m.groups()
        if channel != "02":
            skipped[w.name] = "speech, not song"
            continue
        if stmt not in STATEMENTS or emo not in EMOTIONS:
            skipped[w.name] = f"unknown statement/emotion code {stmt}/{emo}"
            continue
        groups[(int(actor), emo, inten)].append(w)
    takes = []
    for (actor, emo, inten), segs in sorted(groups.items()):
        gender = gender_of(actor)
        label = f"actor{actor:02d}/{EMOTIONS[emo]}/{INTENSITIES.get(inten, inten)}"
        if gender not in genders:
            skipped[label] = f"gender {gender} not in {list(genders)}"
            continue
        segs.sort(key=lambda p: p.stem)  # statement-major, then repetition
        takes.append(Take(
            key=label,
            segments=segs,
            rel_dir=Path("RAW") / CORPUS / f"actor{actor:02d}" / f"{EMOTIONS[emo]}_{inten}",
            singer=f"{CORPUS}-actor{actor:02d}",
            gender=gender,
            language="en",
            song=f"statements [{EMOTIONS[emo]} / {INTENSITIES.get(inten, inten)}]",
            lyrics=[STATEMENTS[_NAME.fullmatch(s.stem).group(5)] for s in segs],
            extra={"ravdess": {"actor": actor, "emotion": EMOTIONS[emo],
                               "intensity": INTENSITIES.get(inten, inten),
                               "files": [s.name for s in segs]}},
        ))
    return takes, skipped


def import_ravdess(data_root, *, root, genders=("F", "M"), limit=None, dry_run=False):
    """-> (manifest, new_records, skipped, takes). Caller saves the manifest."""
    takes, skipped = find_takes(root, genders=genders)
    if limit is not None:
        takes = takes[:limit]
    manifest, new, more = import_takes(
        data_root, takes, corpus=CORPUS, license_note=LICENSE_NOTE, source_url=SOURCE_URL,
        lyrics_source="dataset:ravdess", dry_run=dry_run)
    skipped.update(more)
    return manifest, new, skipped, takes
