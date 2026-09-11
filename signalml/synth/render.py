"""Render addressing, provenance, and the segment cache (D12 pre-commitments 2 and 3).

**No model code lives here.** This module decides *what* to render, *whether it has been
rendered already*, and *where the result and its intermediates go* — so P8's renderer
only has to fill in the audio, and so all of this is testable today, with no checkpoint
and no GPU.

Two ideas:

1. **A render is a pure function of its inputs** (:class:`RenderInputs`), so its cache
   key is a hash of exactly those inputs. Rendering a song = rendering its segments and
   concatenating; an edit misses the cache for the edited segment and hits for the rest.
   Every input that changes the audio must be in the key — the audio profile is in there
   because a ``dev``-profile render and a ``prod``-profile render of the same phrase are
   different audio (Q11), and omitting it would serve 22.05 kHz audio into a 44.1 kHz
   song.

2. **The variance intermediate is an artifact, not a temporary** (:class:`VarianceTrack`).
   The F0 curve and phoneme durations exist in memory between the variance models and the
   acoustic model; persisting them is what makes "this note goes flat" fixable by editing
   the curve and re-running, instead of rerolling the seed and losing the take. An edited
   variance track enters the cache key via ``variance_sha256``, so a hand-fixed phrase
   caches like any other render.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..hashing import sha256_json

RENDER_RECORD_FORMAT = "signalml-render/0.1"
VARIANCE_FORMAT = "signalml-variance/0.1"

# Bump when the cache-key payload changes shape; invalidates every cached render.
RENDER_KEY_VERSION = 1

# Cached renders live in DATA_ROOT/renders/<key prefix>/.
RENDERS_DIRNAME = "renders"
_KEY_PREFIX_LEN = 16

RECORD_FILENAME = "record.json"
VARIANCE_FILENAME = "variance.json"


class VarianceTrack(BaseModel):
    """Per-phrase timing and pitch: the editable intermediate (pre-commitment 3).

    Deliberately plain JSON rather than a binary blob — same reasoning as score.json
    being hand-editable. A phrase at 44.1 kHz / hop 512 is ~86 values per second, so a
    five-second line is a few hundred floats: small enough to read, diff, and hand-fix.
    """

    format: str = VARIANCE_FORMAT
    origin: Literal["predicted", "edited", "imported"] = "predicted"
    sample_rate: int = Field(gt=0)
    frame_hop: int = Field(gt=0)
    phonemes: list[str]
    durations_sec: list[float]
    f0_hz: list[float]
    voiced: list[bool]
    notes: str = ""  # free text: what was edited and why

    @model_validator(mode="after")
    def _check_lengths(self) -> VarianceTrack:
        if len(self.phonemes) != len(self.durations_sec):
            raise ValueError(
                f"{len(self.phonemes)} phonemes but {len(self.durations_sec)} durations "
                f"— one duration per phoneme"
            )
        if len(self.f0_hz) != len(self.voiced):
            raise ValueError(
                f"{len(self.f0_hz)} f0 frames but {len(self.voiced)} voicing frames "
                f"— per-frame arrays must share a frame rate"
            )
        if any(d < 0 for d in self.durations_sec):
            raise ValueError("negative phoneme duration")
        return self

    @property
    def duration_sec(self) -> float:
        return sum(self.durations_sec)

    def content_hash(self) -> str:
        """Hash of the values that change the audio (not ``origin`` or ``notes``)."""
        return sha256_json(
            {
                "v": RENDER_KEY_VERSION,
                "sample_rate": self.sample_rate,
                "frame_hop": self.frame_hop,
                "phonemes": list(self.phonemes),
                "durations_sec": [round(d, 6) for d in self.durations_sec],
                "f0_hz": [round(f, 4) for f in self.f0_hz],
                "voiced": list(self.voiced),
            }
        )


class RenderInputs(BaseModel):
    """Everything that determines a segment's audio. The cache key is a hash of this.

    If you add a field that changes the audio, it must go in :meth:`cache_key` *and*
    ``RENDER_KEY_VERSION`` must be bumped — otherwise old cache entries answer for new
    inputs, which is the one failure mode of this design that is silent.
    """

    segment_hash: str  # from score.segment.segment_content_hash
    voice: str
    embedding_sha256: str
    checkpoint_sha256: str
    audio_profile: str  # "dev" / "prod" (Q11) — profiles never mix
    seed: int
    config_sha256: str
    # None = the variance models predict it (deterministic given the fields above).
    # Set = a hand-edited or imported track overrides the prediction.
    variance_sha256: str | None = None

    def cache_key(self) -> str:
        return sha256_json(
            {
                "v": RENDER_KEY_VERSION,
                "segment": self.segment_hash,
                "voice": self.voice,
                "embedding": self.embedding_sha256,
                "checkpoint": self.checkpoint_sha256,
                "profile": self.audio_profile,
                "seed": self.seed,
                "config": self.config_sha256,
                "variance": self.variance_sha256,
            }
        )

    def with_variance(self, track: VarianceTrack) -> RenderInputs:
        """Same render, but driven by an explicit variance track (the repair path)."""
        return self.model_copy(update={"variance_sha256": track.content_hash()})


class RenderRecord(BaseModel):
    """What produced one cached segment render. Written beside its artifacts."""

    format: str = RENDER_RECORD_FORMAT
    key: str
    inputs: RenderInputs
    created: str
    score_id: str | None = None
    segment_index: int | None = None
    note_ids: list[str] = Field(default_factory=list)
    # published name -> filename inside the render dir, e.g. {"audio": "audio.wav"}
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _check_key(self) -> RenderRecord:
        expected = self.inputs.cache_key()
        if self.key != expected:
            raise ValueError(
                f"render key {self.key[:16]}... does not match its inputs "
                f"(expected {expected[:16]}...) — the record was hand-edited, or the "
                f"key payload changed without a RENDER_KEY_VERSION bump"
            )
        return self


def new_record(
    inputs: RenderInputs,
    *,
    score_id: str | None = None,
    segment_index: int | None = None,
    note_ids: list[str] | None = None,
    artifacts: dict[str, str] | None = None,
    notes: str = "",
) -> RenderRecord:
    """Build a record for ``inputs``, stamping the key and creation time."""
    return RenderRecord(
        key=inputs.cache_key(),
        inputs=inputs,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        score_id=score_id,
        segment_index=segment_index,
        note_ids=list(note_ids or []),
        artifacts=dict(artifacts or {}),
        notes=notes,
    )


def renders_root(data_root: str | Path) -> Path:
    return Path(data_root) / RENDERS_DIRNAME


def render_dir(data_root: str | Path, key: str) -> Path:
    """Directory holding one cached render. Content-addressed: existence *is* the cache.

    Named by a prefix of the key for a readable path; the full key is stored in the
    record and checked on load, so a prefix collision is caught rather than served.
    """
    if len(key) < _KEY_PREFIX_LEN:
        raise ValueError(f"render key {key!r} is too short to address a directory")
    return renders_root(data_root) / key[:_KEY_PREFIX_LEN]


def save_record(data_root: str | Path, record: RenderRecord) -> Path:
    path = render_dir(data_root, record.key) / RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_record(path: str | Path) -> RenderRecord:
    return RenderRecord.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def save_variance(data_root: str | Path, key: str, track: VarianceTrack) -> Path:
    path = render_dir(data_root, key) / VARIANCE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(track.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_variance(path: str | Path) -> VarianceTrack:
    return VarianceTrack.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def find_cached(data_root: str | Path, inputs: RenderInputs) -> RenderRecord | None:
    """The cached render for ``inputs``, or None. A key-prefix collision returns None."""
    key = inputs.cache_key()
    path = render_dir(data_root, key) / RECORD_FILENAME
    if not path.exists():
        return None
    record = load_record(path)
    return record if record.key == key else None


def artifact_path(data_root: str | Path, record: RenderRecord, name: str) -> Path:
    """Absolute path of one of a record's artifacts (``KeyError`` if not present)."""
    return render_dir(data_root, record.key) / record.artifacts[name]
