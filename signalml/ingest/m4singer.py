"""M4Singer adapter — Mandarin studio singing, 20 singers, phone + note labels (S2).

M4Singer (NeurIPS 2022 Datasets & Benchmarks): 20 professional singers covering SATB
(Soprano/Alto = F, Tenor/Bass = M; 10 + 10), ~29.7 h, 700 songs cut into phrases; WAVs
at 44.1, 48 or 96 kHz depending on the session. ``meta.json`` carries per phrase the
lyrics (``txt``), phones (Opencpop-style pinyin initials/finals), phone durations,
MIDI note per phone, note durations and slur flags; each phrase also ships a TextGrid
and a MIDI file. Licence string: CC BY-NC-SA 4.0.

Layout read: ``<root>/<Singer>#<Song>/NNNN.wav`` plus ``<root>/meta.json``
(Google Drive ``m4singer.zip``, linked from github.com/M4Singer/M4Singer).

One record per (singer, song): the phrases are joined end to end (language ``zh``). The
pinyin phones are NOT converted: our phone fields are IPA, and a pinyin -> IPA table
for Mandarin (tones live outside these labels) is a reviewed decision, not something to
guess at import. The corpus alignment and notes go to ``align/phones.native.json``
(phone set ``opencpop_pinyin``) for that later conversion and for variance / SOME
checks; nothing downstream reads it, so ``status.aligned`` stays false.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from .acapella import Take, import_takes
from .batch import short_name
from .gtsinger import gender_of

CORPUS = "m4singer"
LICENSE_NOTE = "M4Singer (Zhang et al. 2022) — CC BY-NC-SA 4.0"
SOURCE_URL = "https://github.com/M4Singer/M4Singer"
NATIVE_PHONE_SET = "opencpop_pinyin"
_SILENT = {"<SP>", "<AP>"}


def _native(items: list[dict], offsets: list[float]) -> dict:
    """meta.json items of one song (in phrase order) -> one time-aligned phone/note list."""
    phones = []
    for item, off in zip(items, offsets):
        t = off
        for ph, dur, note, slur in zip(item["phs"], item["ph_dur"], item["notes"],
                                       item.get("is_slur", [0] * len(item["phs"]))):
            phones.append({"ph": ph, "start": round(t, 4), "end": round(t + dur, 4),
                           "note": int(note) if note else None, "slur": bool(slur),
                           "silent": ph in _SILENT})
            t += dur
    return {"phone_set": NATIVE_PHONE_SET, "language": "zh", "aligner": "manual:m4singer",
            "phones": phones}


def find_takes(root: str | Path, *, genders=("F", "M")) -> tuple[list[Take], dict[str, str]]:
    root = Path(root)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.json under {root} (extract m4singer.zip; the "
                                f"folder holding meta.json and the Singer#Song folders)")
    by_song: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in json.loads(meta_path.read_text(encoding="utf-8")):
        singer, song, _seg = item["item_name"].split("#")
        by_song[(singer, song)].append(item)
    takes, skipped = [], {}
    for (singer, song), items in sorted(by_song.items()):
        label = f"{singer}#{song}"
        gender = gender_of(singer)
        if gender not in genders:
            skipped[label] = f"gender {gender} not in {list(genders)}"
            continue
        items.sort(key=lambda it: it["item_name"].rsplit("#", 1)[1])
        segs = [root / label / f"{it['item_name'].rsplit('#', 1)[1]}.wav" for it in items]
        if not all(s.exists() for s in segs):
            skipped[label] = "phrase WAV missing"
            continue
        offsets, t = [], 0.0
        for s in segs:
            info = sf.info(str(s))
            offsets.append(t)
            t += info.frames / info.samplerate
        takes.append(Take(
            key=label,
            segments=segs,
            rel_dir=Path("RAW") / CORPUS / singer.lower() / short_name(song),
            singer=f"{CORPUS}-{singer}",
            gender=gender,
            language="zh",
            song=song,
            lyrics=[it.get("txt", "") for it in items],
            native_alignment=_native(items, offsets),
            extra={"m4singer": {"singer": singer, "song": song, "phrases": len(segs),
                                "native_alignment": "align/phones.native.json "
                                                    f"({NATIVE_PHONE_SET}, not IPA)"}},
        ))
    return takes, skipped


def import_m4singer(data_root, *, root, genders=("F", "M"), limit=None, dry_run=False):
    """-> (manifest, new_records, skipped, takes). Caller saves the manifest."""
    takes, skipped = find_takes(root, genders=genders)
    if limit is not None:
        takes = takes[:limit]
    manifest, new, more = import_takes(
        data_root, takes, corpus=CORPUS, license_note=LICENSE_NOTE, source_url=SOURCE_URL,
        lyrics_source="dataset:m4singer", dry_run=dry_run)
    skipped.update(more)
    return manifest, new, skipped, takes
