"""Score JSON schema — ``signalml-score/0.2`` (docs/PIPELINE_AND_CONTRACTS.md §4, Q3).

A score is the singer's input: note events bound to syllables and IPA phonemes, with
stress as a separate field. Held vowels / melisma follow the DiffSinger convention:
one syllable spanning multiple notes = consecutive events where the continuation
events carry ``slur: true``, the same syllable text, and an **empty** phoneme list
(the synth stage extends the previous vowel through slurred notes).

TextGrid is not a score — it appears only as the S5 aligner intermediate.

**0.2 (D12) adds ``NoteEvent.id``** — a stable per-note handle so a region of a score
can be *named*: the addressing primitive targeted re-rendering is built on. Ids are
handles, not content — they deliberately take no part in the segment content hash
(``score/segment.py``), so renaming ids never invalidates a cached render. The one rule
that matters: **editors mint new ids, they never renumber existing ones**, or every
saved reference to a note (render history, edit lists) silently retargets.

0.1 files still load — :func:`load_score` upgrades them in memory by minting ids.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from .phoneset import get_phone_set, known_phone_sets

SCORE_FORMAT = "signalml-score/0.2"
LEGACY_SCORE_FORMATS = ("signalml-score/0.1",)
SUPPORTED_SCORE_FORMATS = (SCORE_FORMAT, *LEGACY_SCORE_FORMATS)

# Two adjacent notes may share a boundary; overlap beyond this is a real error.
_OVERLAP_TOLERANCE_SEC = 1e-6

# Ids are opaque handles; this only rules out shapes that break paths/JSON round-trips.
_NOTE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
# Minted ids look like n0001. Hand-written ids need not.
_MINTED_ID_RE = re.compile(r"^n(\d+)$")


class NoteEvent(BaseModel):
    id: str | None = None  # stable handle; minted on load/import, never renumbered
    start: float = Field(ge=0)  # seconds
    end: float
    midi: int = Field(ge=0, le=127)
    syllable: str
    phonemes: list[str] = Field(default_factory=list)
    stress: int | None = Field(default=None, ge=0, le=2)  # 0 none, 1 primary, 2 secondary
    slur: bool = False

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str | None) -> str | None:
        if v is not None and not _NOTE_ID_RE.match(v):
            raise ValueError(
                f"note id {v!r} must start alphanumeric and contain only "
                f"letters, digits, '_', '.', ':' or '-'"
            )
        return v

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
    format: str = SCORE_FORMAT
    bpm: float = Field(gt=0)
    key: str | None = None  # e.g. "G:major"
    language: str = "en"
    phone_set: str
    notes: list[NoteEvent] = Field(min_length=1)

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        if v not in SUPPORTED_SCORE_FORMATS:
            raise ValueError(
                f"unsupported score format {v!r} (supported: "
                f"{', '.join(SUPPORTED_SCORE_FORMATS)})"
            )
        return v

    @field_validator("key")
    @classmethod
    def _check_key(cls, v: str | None) -> str | None:
        if v is not None and ":" not in v:
            raise ValueError(f"key must look like 'G:major' / 'a:minor', got {v!r}")
        return v

    @model_validator(mode="after")
    def _check_notes(self) -> Score:
        seen: set[str] = set()
        for note in self.notes:
            if note.id is None:
                continue
            if note.id in seen:
                raise ValueError(
                    f"duplicate note id {note.id!r} — ids address a single note "
                    f"(a cached render would be attributed to the wrong one)"
                )
            seen.add(note.id)

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

    def ensure_ids(self) -> Score:
        """Mint ids for notes that lack one, in place; returns ``self``.

        Existing ids are left exactly as they are — this never renumbers. Minted ids
        start above the highest ``nNNNN`` already present, so ids of deleted notes are
        not handed out again and a stale reference fails loudly instead of pointing at
        a different note.
        """
        used = {n.id for n in self.notes if n.id is not None}
        nxt = 1 + max(
            (int(m.group(1)) for nid in used if (m := _MINTED_ID_RE.match(nid))),
            default=0,
        )
        for note in self.notes:
            if note.id is not None:
                continue
            while (candidate := f"n{nxt:04d}") in used:
                nxt += 1
            note.id = candidate
            used.add(candidate)
            nxt += 1
        return self

    def upgrade(self) -> Score:
        """Bring a legacy score up to :data:`SCORE_FORMAT`, in place; returns ``self``."""
        self.ensure_ids()
        self.format = SCORE_FORMAT
        return self

    def note_by_id(self, note_id: str) -> NoteEvent | None:
        return next((n for n in self.notes if n.id == note_id), None)

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
    """Load a score, upgrading legacy formats in memory.

    Every score returned here has ids on every note, so downstream code (segmentation,
    render provenance) can rely on them without defensive checks. The file on disk is
    untouched — ``signalml score upgrade`` is the explicit way to persist the upgrade.
    """
    score = Score.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    return score.upgrade()


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
