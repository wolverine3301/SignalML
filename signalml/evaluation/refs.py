"""Ground-truth loaders for the aligner eval.

Only formats that have actually been inspected get a bespoke loader — JamendoLyrics
here, whose layout was verified against the published repo (2026-09-18). Everything
else goes through the generic CSV/TextGrid readers below, which is enough to plug in a
new reference set without guessing at a schema.

**Licence discipline:** JamendoLyrics is evaluation-only. 13 of its 20 English songs are
No-Derivatives, so audio, stems, clips and any derived artefact are measured and
discarded — never retained, never trained on. See docs/notes/candidate_corpora.md.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .alignment import WordOnset


@dataclass(frozen=True)
class JamendoSong:
    """One row of ``JamendoLyrics.csv``."""

    name: str
    artist: str
    title: str
    language: str
    license_type: str
    polyphonic: bool
    lyric_overlap: bool
    non_lexical: bool

    @property
    def no_derivatives(self) -> bool:
        """True if the licence forbids derivative works (any ``-ND`` variant)."""
        return "nd" in _licence_parts(self.license_type)

    @property
    def clean(self) -> bool:
        """No overlapping vocal lines and no non-lexical vocalisations.

        These are real singing-aligner failure modes, but they are their own question
        and will dominate the headline error if mixed into it. Report them as a
        separate bucket rather than dropping them silently.
        """
        return not (self.polyphonic or self.lyric_overlap or self.non_lexical)


def _licence_parts(license_type: str) -> set[str]:
    return {p for p in license_type.replace("CC", " ").lower().replace("-", " ").split() if p}


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def jamendolyrics_songs(root: str | Path) -> list[JamendoSong]:
    """Parse ``JamendoLyrics.csv`` at the dataset root."""
    root = Path(root)
    path = root / "JamendoLyrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}: not a JamendoLyrics checkout")
    songs: list[JamendoSong] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            # Filepath is "mp3/<Artist>_-_<Title>.mp3"; the stem is the id used by every
            # other file in the dataset (lyrics/, annotations/words/, annotations/lines/).
            name = Path(row["Filepath"]).stem
            songs.append(JamendoSong(
                name=name,
                artist=row.get("Artist", "").strip(),
                title=row.get("Title", "").strip(),
                language=row.get("Language", "").strip(),
                license_type=row.get("LicenseType", "").strip(),
                polyphonic=_as_bool(row.get("Polyphonic", "")),
                lyric_overlap=_as_bool(row.get("LyricOverlap", "")),
                non_lexical=_as_bool(row.get("NonLexical", "")),
            ))
    return songs


def load_jamendolyrics_onsets(root: str | Path, song: str) -> list[WordOnset]:
    """Word onsets for one JamendoLyrics song.

    The annotation CSV (``word_start,word_end,line_end``) carries **no word text** — it
    is positional against ``lyrics/<song>.words.txt``, one word per line. Verified
    one-to-one on the published data; a mismatch means the checkout is inconsistent and
    is raised rather than zipped short, because silently truncating would shift every
    later onset and produce a plausible-looking wrong answer.
    """
    root = Path(root)
    words_path = root / "lyrics" / f"{song}.words.txt"
    times_path = root / "annotations" / "words" / f"{song}.csv"
    for p in (words_path, times_path):
        if not p.exists():
            raise FileNotFoundError(f"{p}: missing for song {song!r}")

    words = [w for w in words_path.read_text(encoding="utf-8").splitlines() if w.strip()]
    starts: list[float] = []
    with times_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            starts.append(float(row["word_start"]))

    if len(words) != len(starts):
        raise ValueError(
            f"{song}: {len(words)} words in {words_path.name} but {len(starts)} "
            f"rows in {times_path.name} — the annotation is positional, so these "
            f"must match exactly"
        )
    return [WordOnset(w, t) for w, t in zip(words, starts)]


def load_word_csv(path: str | Path, *, word_col: str = "word",
                  start_col: str = "start") -> list[WordOnset]:
    """Generic word-onset CSV: a text column and a start-time column, in time order."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {word_col, start_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)} "
                             f"(found {reader.fieldnames})")
        return [WordOnset(row[word_col], float(row[start_col])) for row in reader]


def load_boundary_csv(path: str | Path, *, start_col: str = "start",
                      end_col: str | None = "end") -> list[float]:
    """Generic boundary CSV -> sorted unique times. ``end_col`` may be absent."""
    path = Path(path)
    times: set[float] = set()
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        if start_col not in fields:
            raise ValueError(f"{path}: missing column {start_col!r} (found {reader.fieldnames})")
        use_end = end_col is not None and end_col in fields
        for row in reader:
            times.add(round(float(row[start_col]), 6))
            if use_end and row[end_col]:
                times.add(round(float(row[end_col]), 6))
    return sorted(times)


def load_textgrid_boundaries(path: str | Path, *, tier: str = "phones") -> list[float]:
    """Boundary times from a TextGrid tier — SOFA output, or a hand-annotated reference.

    Empty and silence-marked intervals still contribute boundaries: where speech stops
    is exactly as much a boundary as where it starts.
    """
    from praatio import textgrid as praatio_tg  # local import: keep CLI start light

    tg = praatio_tg.openTextgrid(str(path), includeEmptyIntervals=True)
    names = {n.lower(): n for n in tg.tierNames}
    if tier.lower() not in names:
        raise ValueError(f"{path}: no {tier!r} tier (tiers: {list(tg.tierNames)})")
    times: set[float] = set()
    for entry in tg.getTier(names[tier.lower()]).entries:
        times.add(round(float(entry.start), 6))
        times.add(round(float(entry.end), 6))
    return sorted(times)


def load_textgrid_onsets(path: str | Path, *, tier: str = "words",
                         silence: frozenset[str] = frozenset({"", "sil", "sp", "spn", "<eps>"}),
                         ) -> list[WordOnset]:
    """Word onsets from a TextGrid tier, skipping silence marks."""
    from praatio import textgrid as praatio_tg

    tg = praatio_tg.openTextgrid(str(path), includeEmptyIntervals=False)
    names = {n.lower(): n for n in tg.tierNames}
    if tier.lower() not in names:
        raise ValueError(f"{path}: no {tier!r} tier (tiers: {list(tg.tierNames)})")
    return [WordOnset(e.label, float(e.start))
            for e in tg.getTier(names[tier.lower()]).entries
            if e.label.strip().lower() not in silence]
