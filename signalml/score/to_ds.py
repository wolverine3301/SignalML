"""score.json -> DiffSinger ``.ds`` — what a trained variance + acoustic pair sings from.

A ``.ds`` file is a JSON list of phrase segments, each with an ``offset`` into the song.
Segments come from :func:`score.segment.split_segments`, so the render unit here is the
same phrase unit the render cache (D12) keys on.

Per segment this writes the variance model's inputs and nothing it predicts:

- ``note_seq`` / ``note_dur`` / ``note_slur`` — the score, with ``rest`` for silence
  (a pad before and after the phrase, and any gap inside it)
- ``ph_seq`` / ``ph_num`` — phonemes, grouped **one group per non-slur note**, which
  is what the variance model means by a word (inference opens a group at every
  non-slur note: ``note2word = cumsum(~note_slur)`` in their ``ds_variance.py``)

``ph_dur`` and ``f0_seq`` are left out on purpose: they are what the variance model
predicts (``--predict dur pitch``), and the acoustic model then renders its output.

**Grouping must match training** (``segmentation.ph_num_mode`` in the dataset recipe):

- ``vowel_onset`` (default) — a note's onset consonants belong to the group *before*
  it, because the note lands on its vowel and the consonant is sung ahead of the beat.
  A syllable with no nucleus keeps its phones.
- ``syllable`` — each note keeps exactly its own phonemes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from .phoneset import get_phone_set
from .schema import NoteEvent, Score
from .segment import DEFAULT_MIN_REST_SEC, split_segments

SP = "SP"
REST = "rest"
_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# a gap inside a phrase shorter than this is phrasing, not a rest note (it merges into
# the previous note so the variance model is not handed slivers)
MIN_INNER_REST_SEC = 0.05

PhNumMode = Literal["vowel_onset", "syllable"]


def note_name(midi: int) -> str:
    """60 -> 'C4' (ASCII sharps, the spelling DiffSinger's note_seq parser reads)."""
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def _groups(units: list[tuple[str, list[str]]], mode: PhNumMode, phone_set: str
            ) -> list[list[str]]:
    """``units`` = (kind, phones) per non-slur note, kind 'rest' or 'note'. Returns one
    phone group per unit (never empty)."""
    is_nucleus = get_phone_set(phone_set).is_nucleus
    groups: list[list[str]] = []
    for kind, phones in units:
        if kind == REST:
            groups.append([SP])
            continue
        if mode == "vowel_onset" and groups:
            first = next((i for i, ph in enumerate(phones) if is_nucleus(ph)), None)
            if first:  # 0 = starts on its vowel; None = no nucleus, keep everything
                groups[-1].extend(phones[:first])
                phones = phones[first:]
        groups.append(list(phones))
    return groups


def segment_to_ds(notes: list[NoteEvent], *, phone_set: str, mode: PhNumMode = "vowel_onset",
                  pad_sec: float = 0.5) -> dict:
    """One phrase -> one ``.ds`` segment dict."""
    start = max(0.0, notes[0].start - pad_sec)
    seq: list[str] = []
    dur: list[float] = []
    slur: list[int] = []
    units: list[tuple[str, list[str]]] = []

    def add(name: str, length: float, is_slur: bool, kind: str, phones: list[str]):
        seq.append(name)
        dur.append(round(length, 6))
        slur.append(int(is_slur))
        if not is_slur:
            units.append((kind, phones))

    lead = notes[0].start - start
    if lead > 0:
        add(REST, lead, False, REST, [])
    for i, note in enumerate(notes):
        length = note.end - note.start
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        gap = (nxt.start - note.end) if nxt else 0.0
        if 0 < gap < MIN_INNER_REST_SEC:
            length += gap  # a sliver of silence is phrasing: hold the note through it
        add(note_name(note.midi), length, note.slur, "note", list(note.phonemes))
        if gap >= MIN_INNER_REST_SEC:
            add(REST, gap, False, REST, [])
    add(REST, pad_sec, False, REST, [])

    groups = _groups(units, mode, phone_set)
    return {
        "offset": round(start, 6),
        "text": " ".join(n.syllable for n in notes if not n.slur),
        "ph_seq": " ".join(ph for g in groups for ph in g),
        "ph_num": " ".join(str(len(g)) for g in groups),
        "note_seq": " ".join(seq),
        "note_dur": " ".join(f"{d:g}" for d in dur),
        "note_slur": " ".join(str(s) for s in slur),
    }


def score_to_ds(score: Score, *, mode: PhNumMode = "vowel_onset", pad_sec: float = 0.5,
                min_rest_sec: float = DEFAULT_MIN_REST_SEC) -> list[dict]:
    """A whole score -> the ``.ds`` list, one entry per phrase segment."""
    return [segment_to_ds(seg.notes, phone_set=score.phone_set, mode=mode, pad_sec=pad_sec)
            for seg in split_segments(score, min_rest_sec=min_rest_sec)]


def write_ds(segments: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
