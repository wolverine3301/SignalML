"""NUS-48E adapter — sung + read lyrics by the same 12 singers, manual phone labels (S2+S5).

The NUS Sung and Spoken Lyrics Corpus (Duan et al., APSIPA ASC 2013, NUS Sound and
Music Computing Lab): 12 non-professional singers (6 F / 6 M per the paper; the folders
carry initials only, so gender is left for review) each sing 4 of 20 English songs and
read the same lyrics aloud — 169 min, 25,474 manually labelled sung phones. Both the
sung and the read takes carry time-aligned phone labels (``start end phone`` per line,
lowercase ARPAbet without stress digits, ``sil`` / ``sp`` for silence). 44.1 kHz (a few
48 kHz), mono or stereo, 16/24-bit. Licence string: CC BY 4.0 (Zenodo record);
README: research use, cite the paper.

Layout read: ``<root>/<SINGER>/{sing,read}/NN.{wav,txt}``. One record per (singer,
part, song): sung takes are ``domain=sung``, read takes ``domain=spoken`` (dataset
recipes filter on domain, so speech never leaks into a singing run), and each record
names its partner take in ``analysis.json`` — the parallel speech/singing pairs. The
labels become ``phones.json`` directly (manual alignment, no MFA), also as the
``phrase`` variant with one utterance per stretch between silences.

Stress is not in the labels. ``ah`` / ``er`` are mapped like any unstressed ARPAbet
vowel (``ə`` / ``ɚ``, see ``arpabet_to_ipa``); the count per take is recorded so a
later word-level re-labelling can find them. There is no word tier and no lyric text.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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
from .gtsinger import arpabet_to_ipa

CORPUS = "nus48e"
LICENSE_NOTE = "NUS-48E (Duan et al. 2013, NUS SMC Lab) — CC BY 4.0 (Zenodo 19595152)"
SOURCE_URL = "https://zenodo.org/records/19595152"
PHONE_SET = "mfa_ipa/en_v1"
ALIGNER = "manual:nus48e"
PARTS = {"sing": "sung", "read": "spoken"}
_SILENT = {"sil", "sp", ""}
_UTT_GAP_SEC = 0.25   # a silence at least this long ends a phrase


@dataclass
class Take:
    singer: str
    part: str      # "sing" | "read"
    song: str      # "01".."20"
    wav: Path
    labels: Path

    @property
    def label(self) -> str:
        return f"{self.singer}/{self.part}/{self.song}"


def read_labels(path: Path) -> list[tuple[float, float, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append((float(parts[0]), float(parts[1]), parts[2].strip().lower()))
    return rows


def convert(rows: list[tuple[float, float, str]]) -> tuple[list[dict], list[Utterance], int]:
    """Labels -> (phones.json entries, phrase utterances, number of ah/er with unknown
    stress). Raises ValueError on a label that is not ARPAbet."""
    phones, utts, unstressed = [], [], 0
    run: list[dict] = []

    def close() -> None:
        if run:
            utts.append(Utterance(run[0]["start"], run[-1]["end"],
                                  " ".join(p["ph"] for p in run), (len(utts),)))
            run.clear()

    for start, end, lab in rows:
        if lab in _SILENT:
            if end - start >= _UTT_GAP_SEC:
                close()
            continue
        unstressed += lab in ("ah", "er")
        p = {"ph": arpabet_to_ipa(lab), "start": round(start, 4), "end": round(end, 4),
             "word": None, "stress": None}
        phones.append(p)
        run.append(p)
    close()
    return phones, utts, unstressed


def find_takes(root: str | Path, *, parts=("sing", "read")) -> tuple[list[Take], dict[str, str]]:
    root = Path(root)
    singers = sorted(d for d in root.iterdir() if d.is_dir() and (d / "sing").is_dir()) \
        if root.is_dir() else []
    if not singers:
        raise FileNotFoundError(f"no <SINGER>/sing/ folders under {root} (extract "
                                f"nus-smc-corpus_48.zip; point at nus-smc-corpus_48/)")
    takes, skipped = [], {}
    for sdir in singers:
        for part in parts:
            for wav in sorted((sdir / part).glob("*.wav")):
                take = Take(sdir.name, part, wav.stem, wav, wav.with_suffix(".txt"))
                if not take.labels.exists():
                    skipped[take.label] = "no phone label file"
                    continue
                takes.append(take)
    return takes, skipped


def import_nus48e(data_root, *, root, parts=("sing", "read"), limit=None, dry_run=False):
    """-> (manifest, new_records, skipped, takes). Caller saves the manifest.

    Idempotent by checksum of the mono copy."""
    data_root = Path(data_root)
    takes, skipped = find_takes(root, parts=parts)
    if limit is not None:
        takes = takes[:limit]
    manifest = Manifest.for_data_root(data_root)
    new_records: list[ManifestRecord] = []
    if dry_run:
        return manifest, new_records, skipped, takes
    today = date.today().isoformat()
    for t in takes:
        try:
            phones, utts, unstressed = convert(read_labels(t.labels))
            if not phones:
                raise ValueError("no phones in the label file")
            rel_dir = Path("RAW") / CORPUS / t.singer.lower() / t.part / t.song
            out_dir = data_root / rel_dir
            wav = out_dir / "take.wav"
            out_dir.mkdir(parents=True, exist_ok=True)
            if not wav.exists():  # mono copy at the native rate; clean resamples
                y, sr = sf.read(str(t.wav), dtype="float32", always_2d=True)
                sf.write(str(wav), y.mean(axis=1), sr, subtype="PCM_24")
            digest = sha256_file(wav)
            if manifest.by_sha256(digest):
                skipped[t.label] = "checksum already in manifest"
                continue
            duration, rate, channels = probe_audio(wav)
            domain = PARTS[t.part]
            rec = ManifestRecord(
                id=manifest.next_id(),
                source=SourceInfo(kind="other", url=SOURCE_URL, retrieved=today),
                file=FileInfo(path=(rel_dir / "take.wav").as_posix(), sha256=digest,
                              duration_sec=duration, sample_rate=rate, channels=channels),
                meta=MetaInfo(
                    singer=_normalize_singer(f"{CORPUS}-{t.singer}"),
                    gender=None,  # not in the corpus files: set by review
                    song=f"nus48e song {t.song} [{t.part}]",
                    language="en",
                    license_note=LICENSE_NOTE,
                    has_lyrics=False,  # phone labels only, no lyric text
                    source_quality="studio",
                    processing="dry",
                    domain=domain,
                    corpus=CORPUS,
                ),
            )
            rec.status.separated = True
            rec.status.aligned = True
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
                "model": "nus48e-acapella", "imported_from": str(t.wav),
                "license": LICENSE_NOTE, "date": today})
            align_info = {"aligner": ALIGNER, "mode": "given", "phone_set": PHONE_SET,
                          "n_phones": len(phones), "n_utterances": len(utts),
                          "align_score": 1.0,
                          "align_score_method": "manual alignment shipped with the corpus",
                          "stress": f"not labelled: {unstressed} ah/er phones mapped to "
                                    f"the unstressed vowel (ə/ɚ)",
                          "date": today}
            update_analysis(sdir, "align", align_info)
            update_analysis(sdir, "align.phrase", align_info)
            update_analysis(sdir, "lyrics_resolve", {
                "agreement": 1.0, "method": "corpus phone transcription of the take",
                "date": today})
            other = "read" if t.part == "sing" else "sing"
            update_analysis(sdir, "import", {"nus48e": {
                "singer": t.singer, "song": t.song, "part": t.part,
                "pair": (Path(t.wav).parent.parent / other / t.wav.name).as_posix()}})
            manifest.add(rec)
            new_records.append(rec)
        except Exception as exc:  # one bad take must not stop the corpus
            skipped[t.label] = f"failed: {exc}"
    _link_pairs(data_root, manifest)
    return manifest, new_records, skipped, takes


def _link_pairs(data_root: Path, manifest: Manifest) -> None:
    """Write each take's partner record id (sung <-> read) into analysis.json."""
    ids: dict[tuple[str, str], dict[str, str]] = {}
    for rec in manifest.records:
        if rec.meta.corpus != CORPUS:
            continue
        path = song_dir(data_root, rec.id) / "analysis.json"
        try:
            info = json.loads(path.read_text(encoding="utf-8")).get("import", {}).get(CORPUS)
        except (OSError, ValueError):
            continue
        if info:
            ids.setdefault((info["singer"], info["song"]), {})[info["part"]] = rec.id
    for pair in ids.values():
        for part, rid in pair.items():
            other = pair.get("read" if part == "sing" else "sing")
            sdir = song_dir(data_root, rid)
            data = json.loads((sdir / "analysis.json").read_text(encoding="utf-8"))
            if other and data["import"][CORPUS].get("pair_id") != other:
                data["import"][CORPUS]["pair_id"] = other
                update_analysis(sdir, "import", data["import"])
