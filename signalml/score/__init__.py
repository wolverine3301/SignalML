"""Score format: schema, phone sets, G2P, and importers (contracts §4; Q2/Q3)."""

from .from_midi import score_from_midi
from .g2p import ChainG2P, EspeakG2P, G2PError, LexiconG2P, MfaG2P, WordPron, syllabify
from .phoneset import get_phone_set, known_phone_sets
from .schema import (
    SCORE_FORMAT,
    NoteEvent,
    Score,
    load_score,
    save_score,
    validate_score_file,
)

__all__ = [
    "SCORE_FORMAT",
    "ChainG2P",
    "EspeakG2P",
    "G2PError",
    "LexiconG2P",
    "MfaG2P",
    "NoteEvent",
    "Score",
    "WordPron",
    "get_phone_set",
    "known_phone_sets",
    "load_score",
    "save_score",
    "score_from_midi",
    "syllabify",
    "validate_score_file",
]
