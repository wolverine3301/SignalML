"""GTSinger corpus adapter — studio singing with manual phoneme alignments (S2 + S5).

GTSinger (NeurIPS 2024): 80.59 h from 20 professional singers in 9 languages, recorded
in professional studios. Every song is sung as a *control* take and a *technique* take
(breathy, glissando, mixed voice / falsetto, pharyngeal, vibrato), cut into phrases,
and every phrase carries a hand-checked TextGrid (``word`` and ``phone`` tiers; English
phones in ARPAbet with stress digits) plus a MusicXML score.

**CC BY-NC-SA 4.0** — research / development only; `exclude_corpora: [gtsinger]` keeps
it out of anything that must ship.

Layout read: ``<root>/<Language>/<Singer>/<Technique>/<Song>/<Group>/NNNN.{wav,TextGrid}``
(``hf download GTSinger/GTSinger``). One record per (singer, technique, song, group):
the group's phrase WAVs are joined end to end into ``RAW/gtsinger/…`` and the stem,
and the TextGrids are converted straight to ``phones.json`` — the alignment is manual,
so neither Whisper nor MFA runs on these songs. It is written both as the canonical
alignment and as the ``phrase`` variant (one utterance per phrase, with the per-phrase
health numbers), so recipes on either reach it. ``Paired_Speech_Group`` (the lyrics
read aloud) is skipped: it is speech, not singing.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

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
from ..stages.align_chunks import Utterance, utterance_health
from ..stages.common import song_dir, update_analysis
from .batch import short_name

CORPUS = "gtsinger"
LICENSE_NOTE = "GTSinger — CC BY-NC-SA 4.0 (non-commercial: research / dev only)"
SOURCE_URL = "https://huggingface.co/datasets/GTSinger/GTSinger"
PHONE_SET = "mfa_ipa/en_v1"
ALIGNER = "manual:gtsinger"

# ARPAbet -> the IPA symbols MFA's english_mfa dictionary uses (so GTSinger phones share
# embeddings with the rest of the corpus). english_mfa writes the strut vowel as ɐ and
# has no ʌ; stress picks between the reduced and full vowel where English has both.
_ARPABET = {
    "AA": "ɑ", "AE": "æ", "AO": "ɔ", "AW": "aw", "AY": "aj", "EH": "ɛ", "EY": "ej",
    "IH": "ɪ", "IY": "i", "OW": "ow", "OY": "ɔj", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s",
    "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}
_STRESSED = {"AH": ("ə", "ɐ"), "ER": ("ɚ", "ɝ")}  # (unstressed, stressed)
_SILENT = {"", "<SP>", "<AP>", "SP", "AP", "sil", "sp", "<sil>"}
_SPEECH_GROUP = "Paired_Speech_Group"
_FEMALE_PARTS = ("soprano", "alto", "mezzo")
_MALE_PARTS = ("tenor", "bass", "baritone")


def arpabet_to_ipa(label: str) -> str:
    m = re.fullmatch(r"([A-Z]+)([012])?", label.strip().upper())
    if not m:
        raise ValueError(f"not an ARPAbet phone: {label!r}")
    base, stress = m.group(1), m.group(2)
    if base in _STRESSED:
        reduced, full = _STRESSED[base]
        return reduced if stress in (None, "0") else full
    if base not in _ARPABET:
        raise ValueError(f"not an ARPAbet phone: {label!r}")
    return _ARPABET[base]


def gender_of(singer: str) -> str | None:
    part = singer.lower()
    if any(p in part for p in _FEMALE_PARTS):
        return "F"
    if any(p in part for p in _MALE_PARTS):
        return "M"
    return None


def textgrid_for(wav: Path) -> Path | None:
    """The phrase's TextGrid: ``0000.TextGrid``, or ``0000_TextGrid`` (no extension) as
    about 40% of the English folders name it."""
    for tg in (wav.with_suffix(".TextGrid"), wav.with_name(f"{wav.stem}_TextGrid")):
        if tg.exists():
            return tg
    return None


def textgrid_phones(path: Path, offset: float = 0.0) -> tuple[list[dict], list[str]]:
    """(phones.json entries, words) of one phrase TextGrid, shifted by ``offset``."""
    from praatio import textgrid as praatio_tg

    tg = praatio_tg.openTextgrid(str(path), includeEmptyIntervals=False)
    tiers = {name.lower(): name for name in tg.tierNames}
    if "phone" not in tiers:
        raise ValueError(f"{path}: no 'phone' tier (tiers: {list(tg.tierNames)})")
    words = [(e.start, e.end, e.label.strip()) for e in tg.getTier(tiers["word"]).entries
             if e.label.strip() not in _SILENT] if "word" in tiers else []

    def word_at(t: float) -> str | None:
        return next((w.lower() for s, e, w in words if s <= t < e), None)

    phones = []
    for e in tg.getTier(tiers["phone"]).entries:
        label = e.label.strip()
        if label in _SILENT:
            continue
        phones.append({"ph": arpabet_to_ipa(label), "start": round(offset + e.start, 4),
                       "end": round(offset + e.end, 4), "word": word_at((e.start + e.end) / 2),
                       "stress": None})
    return phones, [w for _, _, w in words]


@dataclass
class Group:
    singer: str        # e.g. EN-Alto-1
    technique: str     # e.g. Breathy
    song: str
    group: str         # e.g. Control_Group
    segments: list[Path] = field(default_factory=list)  # the phrase WAVs, in order

    @property
    def label(self) -> str:
        return f"{self.singer}/{self.technique}/{self.song}/{self.group}"


def find_groups(root: str | Path, *, language: str = "English",
                genders: Sequence[str] = ("F",)) -> tuple[list[Group], dict[str, str]]:
    root = Path(root)
    lang_dir = root / language
    if not lang_dir.is_dir():
        raise FileNotFoundError(f"no {language}/ under {root} - download it with "
                                f"`hf download GTSinger/GTSinger --repo-type dataset "
                                f"--include '{language}/*'`")
    groups, skipped = [], {}
    for singer_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
        if gender_of(singer_dir.name) not in genders:
            skipped[singer_dir.name] = f"gender {gender_of(singer_dir.name)} not in {list(genders)}"
            continue
        for gdir in sorted(singer_dir.glob("*/*/*")):
            if not gdir.is_dir():
                continue
            technique, song, group = gdir.parent.parent.name, gdir.parent.name, gdir.name
            item = Group(singer_dir.name, technique, song, group)
            if group == _SPEECH_GROUP:
                skipped[item.label] = "spoken reading of the lyrics, not singing"
                continue
            wavs = sorted((w for w in gdir.glob("*.wav") if textgrid_for(w)),
                          key=lambda w: (len(w.stem), w.stem))
            if not wavs:
                skipped[item.label] = "no phrase WAV with a TextGrid"
                continue
            item.segments = wavs
            groups.append(item)
    return groups, skipped


def _join(segments: list[Path]) -> tuple[np.ndarray, int, list[float]]:
    """Phrases joined end to end -> (audio, sr, start offset of each phrase)."""
    parts, offsets, sr, t = [], [], None, 0.0
    for w in segments:
        y, r = sf.read(str(w), dtype="float32", always_2d=True)
        if sr is None:
            sr = r
        elif r != sr:
            raise ValueError(f"{w}: {r} Hz, the rest of the group is {sr} Hz")
        parts.append(y.mean(axis=1))
        offsets.append(t)
        t += len(parts[-1]) / r
    return np.concatenate(parts), int(sr), offsets


def import_gtsinger(
    data_root: str | Path,
    *,
    root: str | Path,
    genders: Sequence[str] = ("F",),
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[Manifest, list[ManifestRecord], dict[str, str], list[Group]]:
    """Import GTSinger's English singing takes, aligned. Caller saves the manifest.

    Returns ``(manifest, new_records, skipped, selected)``. Idempotent: a group whose
    joined WAV is already in the manifest (by checksum) is skipped."""
    data_root = Path(data_root)
    groups, skipped = find_groups(root, genders=genders)
    if limit is not None:
        groups = groups[:limit]
    manifest = Manifest.for_data_root(data_root)
    new_records: list[ManifestRecord] = []
    if dry_run:
        return manifest, new_records, skipped, groups
    today = date.today().isoformat()
    for g in groups:
        try:
            rel_dir = (Path("RAW") / CORPUS / short_name(g.singer.lower())
                       / short_name(g.technique) / short_name(f"{g.song}__{g.group}"))
            out_dir = data_root / rel_dir
            wav = out_dir / "take.wav"
            phones: list[dict] = []
            utts: list[Utterance] = []
            lines: list[str] = []
            audio, sr, offsets = _join(g.segments)
            for seg, off in zip(g.segments, offsets):
                ph, words = textgrid_phones(textgrid_for(seg), off)
                phones += ph
                if ph and words:
                    lines.append(" ".join(words))
                    utts.append(Utterance(round(off, 3), round(off + sf.info(str(seg)).duration, 3),
                                          lines[-1], (len(lines) - 1,)))
            if not phones:
                raise ValueError("no phones in any TextGrid")
            out_dir.mkdir(parents=True, exist_ok=True)
            if not wav.exists():
                sf.write(str(wav), audio, sr, subtype="PCM_24")
            digest = sha256_file(wav)
            if manifest.by_sha256(digest):
                skipped[g.label] = "checksum already in manifest"
                continue
            (out_dir / "lyrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            duration, rate, channels = probe_audio(wav)
            rec = ManifestRecord(
                id=manifest.next_id(),
                source=SourceInfo(kind="other", url=SOURCE_URL, retrieved=today),
                file=FileInfo(path=(rel_dir / "take.wav").as_posix(), sha256=digest,
                              duration_sec=duration, sample_rate=rate, channels=channels),
                meta=MetaInfo(
                    singer=_normalize_singer(f"{CORPUS}-{g.singer}"),
                    gender=gender_of(g.singer),
                    song=f"{g.song} [{g.technique} / {g.group}]",
                    language="en",
                    license_note=LICENSE_NOTE,
                    has_lyrics=True,
                    lyrics_path=(rel_dir / "lyrics.txt").as_posix(),
                    lyrics_source="dataset:gtsinger",
                    source_quality="studio",
                    processing="dry",
                    domain="sung",
                    corpus=CORPUS,
                ),
            )
            rec.status.separated = True   # studio a cappella: nothing to separate
            rec.status.aligned = True     # manual alignment, converted below
            rec.quality.align_score = 1.0
            sdir = song_dir(data_root, rec.id)
            (sdir / "stems").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wav, sdir / "stems" / "vocals.wav")
            payload = {"phone_set": PHONE_SET, "aligner": ALIGNER, "language": "en",
                       "phones": phones}
            (sdir / "align").mkdir(parents=True, exist_ok=True)
            for name in ("phones.json", "phones.phrase.json"):
                (sdir / "align" / name).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            (sdir / "align" / "utterances.phrase.json").write_text(
                json.dumps(utterance_health(utts, phones), ensure_ascii=False, indent=1),
                encoding="utf-8")
            update_analysis(sdir, "separate", {
                "model": "gtsinger-acapella", "imported_from": [str(s) for s in g.segments],
                "license": LICENSE_NOTE, "date": today})
            align_info = {"aligner": ALIGNER, "mode": "given", "phone_set": PHONE_SET,
                          "n_phones": len(phones), "n_utterances": len(utts),
                          "align_score": 1.0,
                          "align_score_method": "manual alignment shipped with the corpus",
                          "date": today}
            update_analysis(sdir, "align", align_info)
            update_analysis(sdir, "align.phrase", align_info)
            # the corpus' own transcript of this very take: nothing to resolve
            update_analysis(sdir, "lyrics_resolve", {
                "agreement": 1.0, "method": "dataset transcript of the take", "date": today})
            manifest.add(rec)
            new_records.append(rec)
        except Exception as exc:  # one bad group must not stop the corpus
            skipped[g.label] = f"failed: {exc}"
    return manifest, new_records, skipped, groups
