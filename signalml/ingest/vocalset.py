"""VocalSet corpus adapter — a cappella technique recordings (S2).

VocalSet is 10.1 h of 20 professional singers (9 female / 11 male) performing 17 vocal
techniques in four contexts (scales, arpeggios, long tones, excerpts), each sung on the
five vowels. Two properties make it worth onboarding despite its size:

* **CC BY 4.0** — the only *permissively* licensed corpus in the collection. Everything
  else here (MedleyDB, and GTSinger/MoisesDB if they land) is CC BY-NC-SA, so VocalSet
  plus own-recorded material is the only lineage a shippable checkpoint can claim.
* **Dry solo studio audio** with no production on the voice at all — the ideal shape for
  vocoder training and for the ECAPA timbre space.

All labels come from the filename, which VocalSet defines as the unique identifier
(``f2_arpeggios_f_slow_forte_e.wav``): singer id (``f2`` -> female #2, so **gender is in
the corpus by construction**, no flag needed), context, technique, and the trailing
vowel. Directory layout is only used as a fallback, so ``FULL/``-style and
``train``/``test`` splits both parse.

What it is NOT: there are no lyrics — the material is sung on isolated vowels, so these
records import with ``language=None`` deliberately. That keeps them out of an ``en``
acoustic dataset (which would otherwise train on vowel-only "words") while leaving them
fully available to the vocoder and to voice-bank work. The ``spoken`` technique in the
excerpts section imports as ``domain=spoken``.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from ..manifest import (
    FileInfo,
    Manifest,
    ManifestRecord,
    MetaInfo,
    SourceInfo,
    _normalize_singer,
    probe_audio,
    sha256_file,
)
from ..stages.common import song_dir, update_analysis

CORPUS = "vocalset"
LICENSE_NOTE = "VocalSet — CC BY 4.0 (attribution; commercial use permitted)"
ZENODO_URL = "https://zenodo.org/records/1193957"

# singer ids are f1..f9 / m1..m11 (the paper's abstract swaps the counts; the folder
# and filename prefixes are authoritative)
SINGER_RE = re.compile(r"^([fm])(\d+)$")
# singer *folders* spell it out; used when a filename lacks the id prefix
SINGER_DIR_RE = re.compile(r"^(female|male)\s*(\d+)$")
VOWELS = frozenset("aeiou")
CONTEXTS = ("long_tones", "arpeggios", "scales", "excerpts")
# real filenames carry abbreviations and typos ("arps", "arepggios"); the corpus was
# hand-named, so normalise rather than trust it
CONTEXT_ALIASES = {
    "arpeggio": "arpeggios", "arps": "arpeggios", "arp": "arpeggios",
    "arepggios": "arpeggios", "arpegios": "arpeggios",
    "scale": "scales", "longtones": "long_tones", "long tones": "long_tones",
    "excerpt": "excerpts",
}
TOKEN_FIXES = {"sow": "slow"}  # f_sow_forte -> f_slow_forte
# the excerpts section is the only material sung on WORDS: three short pieces.
# `row` is English; `caro` (Caro mio ben) is Italian; `dona` (Dona nobis pacem) Latin.
EXCERPTS = {"caro": "caro mio ben", "row": "row row row your boat",
            "dona": "dona nobis pacem"}
SPOKEN_TECHNIQUES = frozenset({"spoken", "speaking"})
DUPLICATE_RE = re.compile(r"^(.*?)\((\d+)\)$")  # "..._a(1).wav" duplicate marker


@dataclass(frozen=True)
class VocalSetFile:
    """One VocalSet recording with everything its filename encodes."""

    path: Path
    singer_id: str  # "f2"
    gender: str  # "F" / "M"
    context: str | None  # scales / arpeggios / long_tones / excerpts
    technique: str | None  # belt, vibrato, straight, lip_trill, spoken, ...
    vowel: str | None  # a e i o u
    excerpt: str | None = None  # caro / row / dona — the only material with words
    take: int | None = None  # trailing "(1)" or "_2" duplicate/take marker

    @property
    def singer(self) -> str:
        """Corpus-namespaced voice-bank identity — singer keys are global, and ``f2``
        alone would collide with anything else that ever uses that label."""
        return f"{CORPUS}-{self.singer_id}"

    @property
    def domain(self) -> str:
        parts = (self.technique or "").split("_")
        return "spoken" if SPOKEN_TECHNIQUES.intersection(parts) else "sung"

    @property
    def lyrics_hint(self) -> str | None:
        """The words being sung, for the excerpts — everything else is vowels."""
        return EXCERPTS.get(self.excerpt or "")

    @property
    def label(self) -> str:
        return self.path.stem


def _match_context(tokens: list[str]) -> tuple[str | None, list[str]]:
    """Pull a context out of the token list, returning (context, remaining tokens)."""
    for context in CONTEXTS:
        want = context.split("_")
        n = len(want)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == want:
                return context, tokens[:i] + tokens[i + n:]
    for i, token in enumerate(tokens):
        if token in CONTEXT_ALIASES:
            return CONTEXT_ALIASES[token], tokens[:i] + tokens[i + 1:]
    return None, tokens


def _singer_from_dirs(path: Path) -> str | None:
    """``.../female3/scales/...`` -> ``f3``. VocalSet spells singer folders out."""
    for parent in path.parents[:4]:
        m = SINGER_DIR_RE.match(parent.name.strip().lower())
        if m:
            return f"{m.group(1)[0]}{int(m.group(2))}"
    return None


def parse_filename(path: str | Path) -> VocalSetFile | None:
    """Parse one VocalSet WAV path. Returns None if it is not a VocalSet recording."""
    path = Path(path)
    # hand-named corpus: leading underscores, stray spaces, "(1)" duplicate markers
    tokens = [t.strip() for t in path.stem.lower().lstrip("_").split("_") if t.strip()]
    if not tokens:
        return None
    take = None
    dup = DUPLICATE_RE.match(tokens[-1])
    if dup:
        take = int(dup.group(2))
        tokens = tokens[:-1] + ([dup.group(1)] if dup.group(1) else [])
    m = SINGER_RE.match(tokens[0])
    if m:
        singer_id = f"{m.group(1)}{int(m.group(2))}"
        rest = tokens[1:]
    else:  # no id prefix on the file: fall back to the singer folder
        singer_id = _singer_from_dirs(path)
        if singer_id is None:
            return None
        rest = tokens

    vowel = None
    if rest and len(rest[-1]) == 1 and rest[-1] in VOWELS:
        vowel, rest = rest[-1], rest[:-1]
    if take is None and rest and rest[-1].isdigit():  # belt_2 = second take
        take, rest = int(rest[-1]), rest[:-1]

    excerpt = None
    for i, token in enumerate(rest):
        if token in EXCERPTS:
            excerpt, rest = token, rest[:i] + rest[i + 1:]
            break

    rest = [TOKEN_FIXES.get(t, t) for t in rest]
    context, rest = _match_context(rest)
    if context is None:  # fall back to the directory the file sits in
        parents = [p.name.lower().replace(" ", "_") for p in path.parents[:3]]
        for name in parents:
            if name in CONTEXTS:
                context = name
                break
            if name in CONTEXT_ALIASES:
                context = CONTEXT_ALIASES[name]
                break

    return VocalSetFile(
        path=path,
        singer_id=singer_id,
        gender="F" if singer_id[0] == "f" else "M",
        context=context,
        technique="_".join(rest) or None,
        vowel=vowel,
        excerpt=excerpt,
        take=take,
    )


def pick_root(root: str | Path) -> Path:
    """VocalSet 1.2 ships the same audio in several organisations (by singer, by
    technique, by vowel) alongside ``FULL/``. Prefer ``FULL`` when it is there:
    checksum dedupe would catch the copies anyway, but only after hashing them all.
    """
    root = Path(root)
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and child.name.upper() == "FULL":
            return child
    return root


def find_files(
    root: str | Path,
    *,
    genders: Sequence[str] = ("F", "M"),
    contexts: Sequence[str] | None = None,
    techniques: Sequence[str] | None = None,
) -> tuple[list[VocalSetFile], dict[str, str]]:
    """Walk ``root`` for VocalSet WAVs. Returns (files, skipped {path: reason})."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"VocalSet root not found: {root}\n"
            f"Download VocalSet.zip (2.1 GB, CC BY 4.0) from {ZENODO_URL} and extract "
            f"it there."
        )
    root = pick_root(root)
    want_gender = {g.upper() for g in genders}
    want_context = {c.lower() for c in contexts} if contexts else None
    want_technique = {t.lower() for t in techniques} if techniques else None

    files: list[VocalSetFile] = []
    skipped: dict[str, str] = {}
    for path in sorted(root.rglob("*.wav")):
        parsed = parse_filename(path)
        if parsed is None:
            skipped[path.name] = ("no VocalSet singer id in the filename or the "
                                  "enclosing folders")
            continue
        if parsed.gender not in want_gender:
            continue  # a deliberate filter, not a problem worth reporting per file
        if want_context and (parsed.context or "") not in want_context:
            continue
        if want_technique and (parsed.technique or "") not in want_technique:
            continue
        files.append(parsed)
    return files, skipped


def import_vocalset(
    data_root: str | Path,
    *,
    root: str | Path | None = None,
    genders: Sequence[str] = ("F",),
    contexts: Sequence[str] | None = None,
    techniques: Sequence[str] | None = None,
    language: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[Manifest, list[ManifestRecord], dict[str, str], list[VocalSetFile]]:
    """Import VocalSet recordings into the manifest. Caller saves.

    Defaults to female singers only (``genders=("F",)``) — the corpus is ~3.5k short
    files and importing the half that a female-only recipe will refuse anyway just
    burns disk. Records land ``separated=true`` (a cappella: nothing to separate),
    ``source_quality=studio``, ``processing=dry``. Idempotent by checksum. Returns
    ``(manifest, new_records, skipped, selected)``.
    """
    data_root = Path(data_root)
    root = Path(root) if root else data_root / CORPUS
    files, skipped = find_files(
        root, genders=genders, contexts=contexts, techniques=techniques)
    if limit is not None:
        files = files[:limit]

    manifest = Manifest.for_data_root(data_root)
    new_records: list[ManifestRecord] = []
    selected: list[VocalSetFile] = []
    for item in files:
        try:
            rel_path = item.path.relative_to(data_root).as_posix()
        except ValueError:
            skipped[item.label] = (
                f"audio lives outside DATA_ROOT ({item.path}); manifest paths are "
                f"DATA_ROOT-relative"
            )
            continue
        selected.append(item)
        if dry_run:
            continue

        digest = sha256_file(item.path)
        if manifest.by_sha256(digest):
            skipped[item.label] = "checksum already in manifest"
            selected.pop()
            continue

        duration, sr, channels = probe_audio(item.path)
        rec = ManifestRecord(
            id=manifest.next_id(),
            source=SourceInfo(kind="other", url=ZENODO_URL,
                              retrieved=date.today().isoformat()),
            file=FileInfo(
                path=rel_path,
                sha256=digest,
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
            ),
            meta=MetaInfo(
                singer=_normalize_singer(item.singer),
                gender=item.gender,
                song=item.label,
                # no lyrics: sung on isolated vowels, so language stays null and an
                # `en` acoustic recipe cannot pick these up by accident
                language=language,
                license_note=LICENSE_NOTE,
                has_lyrics=False,
                source_quality="studio",
                processing="dry",  # a cappella, no production on the voice
                domain=item.domain,
                corpus=CORPUS,
            ),
        )
        rec.status.separated = True  # a cappella solo: nothing to separate
        rec.quality.notes = (
            f"vocalset context={item.context or 'unknown'} "
            f"technique={item.technique or 'none'} vowel={item.vowel or 'none'}"
            + (f" excerpt={item.excerpt}" if item.excerpt else "")
        )

        sdir = song_dir(data_root, rec.id)
        stems_dir = sdir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item.path, stems_dir / "vocals.wav")
        update_analysis(sdir, "separate", {
            "model": "vocalset-acapella",
            "imported_from": rel_path,
            "vocalset": {
                "singer": item.singer_id,
                "context": item.context,
                "technique": item.technique,
                "vowel": item.vowel,
                "take": item.take,
                # excerpts are the only VocalSet material sung on words; the text is
                # known, so these are the files that could get lyrics sidecars later
                "excerpt": item.excerpt,
                "lyrics_hint": item.lyrics_hint,
                "license": LICENSE_NOTE,
            },
            "date": date.today().isoformat(),
        })

        manifest.add(rec)
        new_records.append(rec)

    return manifest, new_records, skipped, selected


def summarize(files: Sequence[VocalSetFile]) -> str:
    """Census of a selection: singers, techniques, contexts — the shape of the import."""
    singers = sorted({f.singer_id for f in files})
    techniques = sorted({f.technique or "none" for f in files})
    contexts = sorted({f.context or "unknown" for f in files})
    return (f"{len(files)} file(s), {len(singers)} singer(s) {singers}\n"
            f"  contexts: {', '.join(contexts)}\n"
            f"  techniques ({len(techniques)}): {', '.join(techniques)}")
