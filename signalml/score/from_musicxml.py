"""MusicXML -> score JSON (deliberate stub — docs/MIGRATION_PLAN.md P6.4).

MusicXML carries lyric syllabification natively (<lyric><syllabic> single/begin/
middle/end plus <extend> for melisma), which removes every heuristic the MIDI importer
needs. Implement when a real MusicXML source appears. Notes for the implementer:

- use ``music21`` or plain ElementTree (the subset needed is small: part -> measure ->
  note {pitch, duration, tie, lyric})
- ``<syllabic>`` maps directly onto our hyphenation tokens; ``<extend>`` -> slur events
- tempo from <sound tempo=...> / metronome marks; key from <key><fifths>
"""

from __future__ import annotations

from pathlib import Path

from .schema import Score


def score_from_musicxml(path: str | Path, **_kwargs) -> Score:
    raise NotImplementedError(
        "MusicXML import is not implemented yet (Migration P6.4 stub) — convert to "
        "MIDI + syllabified lyrics and use score_from_midi, or implement this importer "
        "(see module docstring for the mapping notes)."
    )
