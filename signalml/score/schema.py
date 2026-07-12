"""Score JSON schema — ``signalml-score/0.1`` (docs/PIPELINE_AND_CONTRACTS.md §4, Q3).

A score is the singer's input: note events bound to syllables and IPA phonemes, with
stress as a separate field. Held vowels / melisma follow the DiffSinger convention:
one syllable spanning multiple notes = consecutive events where the continuation
events carry ``slur: true``, the same syllable text, and an **empty** phoneme list
(the synth stage extends the previous vowel through slurred notes).

TextGrid is not a score — it appears only as the S5 aligner intermediate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .phoneset import get_phone_set, known_phone_sets

SCORE_FORMAT = "signalml-score/0.1"

# Two adjacent notes may share a boundary; overlap beyond this is a real error.
_OVERLAP_TOLERANCE_SEC = 1e-6


class NoteEvent(BaseModel):
    start: float = Field(ge=0)  # seconds
    end: float
    midi: int = Field(ge=0, le=127)
    syllable: str
    phonemes: list[str] = Field(default_factory=list)
    stress: int | None = Field(default=None, ge=0, le=2)  # 0 none, 1 primary, 2 secondary
    slur: bool = False

    @model_validator(mode="after")
    def _check_note(self) -> NoteEvent:
        if self.end <= self.start:
            raise ValueError(f"note end {self.end} <= start {self.start}")
        if self.slur and self.phonemes:
            raise ValueError(
                f"slur continuation at {self.start}s must have empty phonemes "
                f"(got {self.phonemes}) — the previous vowel is held"
            )
        if not self.slur and not self.phonemes:
            raise ValueError(f"non-slur note at {self.start}s has no phonemes")
        return self


class Score(BaseModel):
    format: Literal["signalml-score/0.1"] = SCORE_FORMAT
    bpm: float = Field(gt=0)
    key: str | None = None  # e.g. "G:major"
    language: str = "en"
    phone_set: str
    notes: list[NoteEvent] = Field(min_length=1)

    @field_validator("key")
    @classmethod
    def _check_key(cls, v: str | None) -> str | None:
        if v is not None and ":" not in v:
            raise ValueError(f"key must look like 'G:major' / 'a:minor', got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_notes(self) -> Score:
        prev = self.notes[0]
        if prev.slur:
            raise ValueError("first note cannot be a slur continuation")
        for note in self.notes[1:]:
            if note.start < prev.start:
                raise ValueError(f"notes not sorted by start ({note.start} after {prev.start})")
            if note.start < prev.end - _OVERLAP_TOLERANCE_SEC:
                raise ValueError(
                    f"overlapping notes: [{prev.start}, {prev.end}] and "
                    f"[{note.start}, {note.end}] — scores are monophonic"
                )
            if note.slur and note.syllable != prev.syllable:
                raise ValueError(
                    f"slur at {note.start}s continues syllable {prev.syllable!r} "
                    f"but carries {note.syllable!r}"
                )
            prev = note
        return self

    def phone_set_problems(self) -> list[str]:
        """Phones outside the declared phone set (empty list = clean). Unknown phone-set
        *names* are reported as a single problem rather than raising, so future sets
        can circulate before they're registered here."""
        try:
            ps = get_phone_set(self.phone_set)
        except KeyError:
            return [
                f"unregistered phone set {self.phone_set!r} (known: {known_phone_sets()}) "
                f"— phones not checked"
            ]
        all_phones = [ph for note in self.notes for ph in note.phonemes]
        unknown = ps.unknown(all_phones)
        return [f"phone {ph!r} not in {self.phone_set}" for ph in unknown]


def load_score(path: str | Path) -> Score:
    return Score.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def save_score(score: Score, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(score.model_dump_json(indent=2), encoding="utf-8")
    return path


def validate_score_file(path: str | Path) -> list[str]:
    """All problems with a score file (schema violations or phone-set misses).
    Empty list = valid."""
    try:
        score = load_score(path)
    except Exception as exc:
        return [str(exc)]
    return score.phone_set_problems()
