"""Score segmentation — the render unit, and the cache-invalidation primitive (D12).

Targeted editing ("this line is bad, redo *just* that") needs two things before any
model exists: a way to **name** a region of a score, and a way to tell whether that
region's audio would come out different. This module is both, and it is deliberately
model-free — it runs on a laptop with no GPU and no checkpoint.

A **segment** is a phrase: a maximal run of notes with no rest longer than
``min_rest_sec`` between them. That is the same unit the DiffSinger family renders
natively (its ``.ds`` files are a list of phrase segments with offsets), and rests are
where a splice is inaudible, so it is both the natural render unit and the natural
*edit* unit.

The **content hash** is what makes partial re-rendering safe. It covers exactly the
things that change how a phrase sounds and nothing else:

- note ids are excluded — they are handles, so renumbering never costs you a re-render
- note times are hashed **relative to the segment start** — moving a phrase later in the
  song reuses its render instead of invalidating it
- ``language`` and ``phone_set`` are included — they change pronunciation
- ``bpm``/``key`` are excluded — they are header metadata; note times are absolute

So: identical phrases anywhere in a song share one cache entry (a repeated chorus line
is rendered once), and editing one word invalidates one phrase.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..hashing import sha256_json
from .schema import NoteEvent, Score

# Below this, a gap is phrasing inside a line rather than a break between lines. Chosen
# as a starting point to be tuned by ear (D12 experiment 1), not from theory — it is a
# cache-key input, so changing it produces honest misses rather than stale hits.
DEFAULT_MIN_REST_SEC = 0.30

# Bump when the hash *payload* changes shape; invalidates every cached render.
SEGMENT_HASH_VERSION = 1

# Note times are rounded before hashing so float formatting can't shift a hash.
_TIME_PRECISION = 6


class Segment(BaseModel):
    """One phrase of a score: the unit that gets rendered, cached, and re-rendered."""

    index: int = Field(ge=0)  # position in the score; the stable handle for a UI
    start: float = Field(ge=0)
    end: float
    notes: list[NoteEvent] = Field(min_length=1)
    content_hash: str

    @property
    def note_ids(self) -> list[str]:
        """Ids of this segment's notes (``None`` for any note that lacks one)."""
        return [n.id for n in self.notes]  # type: ignore[misc]

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text(self) -> str:
        """Human-readable lyric of the phrase, melisma collapsed."""
        out: list[str] = []
        for note in self.notes:
            if not note.slur:
                out.append(note.syllable)
        return " ".join(out)


def segment_content_hash(
    notes: list[NoteEvent], *, language: str, phone_set: str
) -> str:
    """Hash of everything about ``notes`` that determines the rendered audio.

    See the module docstring for what is deliberately excluded. ``notes`` must be
    non-empty and sorted; times are hashed relative to ``notes[0].start``.
    """
    if not notes:
        raise ValueError("cannot hash an empty segment")
    origin = notes[0].start
    payload = {
        "v": SEGMENT_HASH_VERSION,
        "language": language,
        "phone_set": phone_set,
        "notes": [
            {
                "t0": round(n.start - origin, _TIME_PRECISION),
                "t1": round(n.end - origin, _TIME_PRECISION),
                "midi": n.midi,
                "syl": n.syllable,
                "ph": list(n.phonemes),
                "stress": n.stress,
                "slur": n.slur,
            }
            for n in notes
        ],
    }
    return sha256_json(payload)


def split_segments(
    score: Score, min_rest_sec: float = DEFAULT_MIN_REST_SEC
) -> list[Segment]:
    """Split ``score`` into phrase segments at rests of at least ``min_rest_sec``.

    A slur continuation can never open a segment: it carries no phonemes of its own
    (it holds the previous vowel), so a segment starting on one would be unrenderable.
    Held vowels therefore keep their phrase together no matter how long the note is.
    """
    if min_rest_sec < 0:
        raise ValueError(f"min_rest_sec must be >= 0, got {min_rest_sec}")

    runs: list[list[NoteEvent]] = [[score.notes[0]]]
    for note in score.notes[1:]:
        gap = note.start - runs[-1][-1].end
        if gap >= min_rest_sec and not note.slur:
            runs.append([note])
        else:
            runs[-1].append(note)

    return [
        Segment(
            index=i,
            start=run[0].start,
            end=run[-1].end,
            notes=run,
            content_hash=segment_content_hash(
                run, language=score.language, phone_set=score.phone_set
            ),
        )
        for i, run in enumerate(runs)
    ]


class RerenderPlan(BaseModel):
    """What a score edit costs: which phrases survive, which must be re-rendered.

    This is the whole point of D12 in one object — computable with no model, no GPU and
    no checkpoint, so an editor can show the cost of an edit before committing to it.
    """

    reuse: list[tuple[int, int]] = Field(default_factory=list)  # (new index, old index)
    render: list[int] = Field(default_factory=list)  # new indices needing a render
    dropped: list[int] = Field(default_factory=list)  # old indices no longer present

    @property
    def total(self) -> int:
        return len(self.reuse) + len(self.render)

    @property
    def reuse_fraction(self) -> float:
        return len(self.reuse) / self.total if self.total else 1.0

    def summary(self) -> str:
        return (
            f"{len(self.reuse)}/{self.total} segments reused "
            f"({self.reuse_fraction:.0%}), {len(self.render)} to render, "
            f"{len(self.dropped)} dropped"
        )


def plan_rerender(
    old: Score, new: Score, min_rest_sec: float = DEFAULT_MIN_REST_SEC
) -> RerenderPlan:
    """Diff two versions of a score by segment content -> what actually needs rendering.

    Matching is by content hash, not position, so inserting a line at the top of a song
    does not invalidate everything after it. A new segment whose content already existed
    anywhere in the old score is reusable — the cache is keyed on content, so where it
    used to sit is irrelevant.
    """
    old_segments = split_segments(old, min_rest_sec)
    new_segments = split_segments(new, min_rest_sec)

    first_old: dict[str, int] = {}
    for seg in old_segments:
        first_old.setdefault(seg.content_hash, seg.index)

    plan = RerenderPlan()
    matched: set[str] = set()
    for seg in new_segments:
        if seg.content_hash in first_old:
            plan.reuse.append((seg.index, first_old[seg.content_hash]))
            matched.add(seg.content_hash)
        else:
            plan.render.append(seg.index)
    plan.dropped = [s.index for s in old_segments if s.content_hash not in matched]
    return plan
