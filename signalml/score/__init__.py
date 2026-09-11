"""Score format: schema, phone sets, G2P, and importers (contracts §4; Q2/Q3)."""

from .from_midi import score_from_midi
from .g2p import ChainG2P, EspeakG2P, G2PError, LexiconG2P, MfaG2P, WordPron, syllabify
from .phoneset import get_phone_set, known_phone_sets
from .schema import (
    LEGACY_SCORE_FORMATS,
    SCORE_FORMAT,
    SUPPORTED_SCORE_FORMATS,
    NoteEvent,
    Score,
    load_score,
    save_score,
    validate_score_file,
)
from .segment import (
    DEFAULT_MIN_REST_SEC,
    RerenderPlan,
    Segment,
    plan_rerender,
    segment_content_hash,
    split_segments,
)

__all__ = [
    "DEFAULT_MIN_REST_SEC",
    "LEGACY_SCORE_FORMATS",
    "SCORE_FORMAT",
    "SUPPORTED_SCORE_FORMATS",
    "ChainG2P",
    "EspeakG2P",
    "G2PError",
    "LexiconG2P",
    "MfaG2P",
    "NoteEvent",
    "RerenderPlan",
    "Score",
    "Segment",
    "WordPron",
    "get_phone_set",
    "known_phone_sets",
    "load_score",
    "plan_rerender",
    "save_score",
    "score_from_midi",
    "segment_content_hash",
    "split_segments",
    "syllabify",
    "validate_score_file",
]
