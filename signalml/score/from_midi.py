"""MIDI + lyrics -> score JSON (docs/MIGRATION_PLAN.md P6.2).

Lowers a monophonic melody track plus syllable-tokenized lyrics into
``signalml-score/0.1``. Syllable-to-note binding is **by order**, so the lyrics must be
pre-syllabified by the writer:

- whitespace separates tokens; each token binds to the next note
- ``shin-ing`` (or ``shin- ing``) = one word, two syllables, two notes
- a bare ``-`` (or ``_``) token = melisma: the previous syllable holds through this
  note (``slur: true``, empty phonemes)

Documented limitations (v0.1): melody must be monophonic (overlaps are an error, not
merged); token count must equal note count exactly; a multi-syllable word written
unhyphenated is an error (we won't guess where to split the notes); MusicXML — which
carries syllabification natively — is the upgrade path (``from_musicxml``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from .schema import SCORE_FORMAT, NoteEvent, Score

CONTINUATION_TOKENS = {"-", "_", "+"}


@dataclass(frozen=True)
class LyricToken:
    text: str  # display form of the syllable ("shin")
    word: str | None  # full normalized word it belongs to; None for continuations
    syllable_index: int  # which of the word's syllables this is
    is_continuation: bool
    unit: int = -1  # index of the written word occurrence this token came from
    unit_size: int = 1  # how many syllable tokens that occurrence was written with


def parse_lyric_tokens(text: str) -> list[LyricToken]:
    """Lyrics text -> ordered syllable tokens (see module docstring for the syntax)."""
    # join trailing-hyphen splits: "shin- ing" -> "shin-ing"
    raw = text.split()
    joined: list[str] = []
    for tok in raw:
        if joined and joined[-1].endswith("-") and joined[-1] not in CONTINUATION_TOKENS:
            joined[-1] += tok
        else:
            joined.append(tok)

    tokens: list[LyricToken] = []
    for unit_idx, unit in enumerate(joined):
        if unit in CONTINUATION_TOKENS:
            if not tokens:
                raise ValueError("lyrics start with a melisma continuation token")
            tokens.append(LyricToken(text="", word=None, syllable_index=0,
                                     is_continuation=True, unit=unit_idx))
            continue
        parts = [p for p in unit.split("-") if p]
        word = "".join(parts)
        for i, part in enumerate(parts):
            tokens.append(LyricToken(text=part, word=word, syllable_index=i,
                                     is_continuation=False, unit=unit_idx,
                                     unit_size=len(parts)))
    return tokens


@dataclass(frozen=True)
class MidiNote:
    midi: int
    start_sec: float
    end_sec: float


def read_melody(
    midi_path: str | Path, track: int | None = None
) -> tuple[list[MidiNote], float, str | None]:
    """Parse a MIDI file -> (monophonic note list, bpm, key or None).

    Tempo changes anywhere in the file are honored when converting ticks to seconds;
    ``bpm`` reported in the score is the *first* tempo (good enough for a header field —
    timing itself is exact).
    """
    mid = mido.MidiFile(str(midi_path))

    # absolute-tick tempo map + key signature, gathered across all tracks
    tempo_events: list[tuple[int, int]] = []  # (abs_tick, us_per_beat)
    key: str | None = None
    for trk in mid.tracks:
        tick = 0
        for msg in trk:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((tick, msg.tempo))
            elif msg.type == "key_signature" and key is None:
                key = _midi_key_to_score(msg.key)
    tempo_events.sort()
    if not tempo_events or tempo_events[0][0] > 0:
        tempo_events.insert(0, (0, 500000))  # MIDI default 120 bpm

    def tick_to_sec(target: int) -> float:
        sec = 0.0
        for (t0, tempo), nxt in zip(tempo_events, tempo_events[1:] + [(None, None)]):
            t1 = nxt[0] if nxt[0] is not None and nxt[0] < target else target
            if t1 > t0:
                sec += mido.tick2second(t1 - t0, mid.ticks_per_beat, tempo)
            if nxt[0] is None or nxt[0] >= target:
                break
        return sec

    # pick the melody track
    note_tracks = [i for i, trk in enumerate(mid.tracks)
                   if any(m.type == "note_on" for m in trk)]
    if track is None:
        if not note_tracks:
            raise ValueError(f"{midi_path}: no track contains notes")
        track = note_tracks[0]
    elif track not in range(len(mid.tracks)):
        raise ValueError(f"{midi_path}: track {track} out of range ({len(mid.tracks)} tracks)")

    notes: list[MidiNote] = []
    active: tuple[int, int] | None = None  # (midi, start_tick)
    tick = 0
    for msg in mid.tracks[track]:
        tick += msg.time
        is_on = msg.type == "note_on" and msg.velocity > 0
        is_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
        if is_on:
            if active is not None:
                raise ValueError(
                    f"{midi_path}: polyphony at tick {tick} (note {msg.note} starts "
                    f"while {active[0]} is sounding) — melody track must be monophonic"
                )
            active = (msg.note, tick)
        elif is_off and active is not None and msg.note == active[0]:
            notes.append(MidiNote(
                midi=active[0],
                start_sec=round(tick_to_sec(active[1]), 6),
                end_sec=round(tick_to_sec(tick), 6),
            ))
            active = None
    if active is not None:
        raise ValueError(f"{midi_path}: note {active[0]} never released")

    bpm = round(mido.tempo2bpm(tempo_events[0][1]), 3)
    return notes, bpm, key


def _midi_key_to_score(key: str) -> str:
    """mido key_signature ('G', 'Am', 'Ebm') -> score format ('G:major', 'A:minor')."""
    if key.endswith("m"):
        return f"{key[:-1]}:minor"
    return f"{key}:major"


def score_from_midi(
    midi_path: str | Path,
    lyrics_text: str,
    *,
    g2p,
    language: str = "en",
    phone_set: str = "mfa_ipa/en_v1",
    track: int | None = None,
) -> Score:
    """Bind syllable tokens to melody notes in order and phonemize -> Score.

    ``g2p`` is any backend from :mod:`signalml.score.g2p` (``pronounce(word)`` ->
    ``WordPron``); backends with a batch ``pronounce_words`` get warmed up first.
    """
    tokens = parse_lyric_tokens(lyrics_text)
    notes, bpm, key = read_melody(midi_path, track=track)
    if len(tokens) != len(notes):
        raise ValueError(
            f"lyrics/melody mismatch: {len(tokens)} syllable tokens vs {len(notes)} notes "
            f"— every note needs exactly one token ('-' holds the previous syllable)"
        )

    words = sorted({t.word for t in tokens if t.word})
    if hasattr(g2p, "pronounce_words"):
        g2p.pronounce_words(words)
    prons = {w: g2p.pronounce(w) for w in words}

    events: list[NoteEvent] = []
    for token, note in zip(tokens, notes):
        if token.is_continuation:
            events.append(NoteEvent(
                start=note.start_sec, end=note.end_sec, midi=note.midi,
                syllable=events[-1].syllable, phonemes=[], stress=None, slur=True,
            ))
            continue
        assert token.word is not None
        pron = prons[token.word]
        if token.unit_size != len(pron.syllables):
            raise ValueError(
                f"word {token.word!r} is written as {token.unit_size} syllable token(s) "
                f"but its pronunciation has {len(pron.syllables)} syllables — hyphenate "
                f"the lyrics to match (e.g. 'shin-ing')"
            )
        syl = pron.syllables[token.syllable_index]
        events.append(NoteEvent(
            start=note.start_sec, end=note.end_sec, midi=note.midi,
            syllable=token.text, phonemes=list(syl.phones), stress=syl.stress, slur=False,
        ))

    return Score(format=SCORE_FORMAT, bpm=bpm, key=key, language=language,
                 phone_set=phone_set, notes=events)
