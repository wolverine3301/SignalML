"""Quick identity review: who sings each main vocal track.

The queue is every multitrack record whose role is a lead (or an unnamed vocal, which
is often the lead), one card per song. Each track gets a few short snippets from its
clearest sung passages and a pre-filled guess - gender from the median-F0 hint the
importer stored, singer from the artist name - so reviewing is mostly confirming.
Doubles fold under their lead: same singer unless the reviewer says otherwise. Tracks
that measure as screams/growls (:func:`..audio.voice.harshness`) get ``harsh`` pre-ticked,
and a confirmed track that measures harsh without the flag is marked for a second listen.

Answers append to ``DATA_ROOT/review/identity.jsonl`` (latest line per record wins, so
nothing is ever lost) and reach the manifest only through :func:`apply_labels`, which
sets ``gender``, ``singer``, ``language`` and ``excluded`` and records the verdict in
``analysis.json``. A bad take or a track that is not singing is excluded, not deleted:
dataset build skips it, the audio stays.
"""

from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio.voice import harshness
from ..ingest import multitrack as mt
from ..manifest import Manifest, _normalize_singer
from ..stages.common import song_dir, update_analysis

LABELS_FILE = Path("review") / "identity.jsonl"
SNIPPET_CACHE = Path("review") / "snippets.json"
HARSH_CACHE = Path("review") / "harsh.json"
MAIN_ROLES = frozenset({"lead", "lead_double", "vocal", "vocal_double"})
FLAGS = {
    "not_lead": "not the lead singer",
    "not_english": "not sung in English",
    "bad_take": "bad take (noise, clipping, mistakes)",
    "not_singing": "not singing (talk, whisper, effects, silence)",
    # screams, growls, fry/distorted delivery: no stable pitch for F0 extraction, the
    # aligner or the acoustic model. Excluded, not discarded - a style-aware model may
    # want them later, and re-labelling + `review apply` brings a track back.
    "harsh": "mostly harsh vocals (screams, growls, distorted fry) - sung grit is fine",
    # oohs/aahs are singing and stay in: the flag only tells the lyrics step not to expect
    # words (Whisper hears nothing, which is right, not a failed transcription)
    "wordless": "sung without words throughout (oohs, aahs) - still singing, kept",
    # heavy effects / pitch-shifting / a voice played as an instrument: sets
    # processing=heavy, which dataset recipes exclude by default (reversible)
    "effected": "heavily effected or pitch-shifted, used as an instrument",
    "duplicate": "the same recording as another track of the song (keep the other)",
}
EXCLUDING_FLAGS = ("bad_take", "not_singing", "harsh", "duplicate")
N_SNIPPETS, SNIPPET_SEC, MAX_SNIPPET_SEC = 3, 6.0, 15.0
# Pitch only rules a voice male: on a reviewed multitrack batch the men sang at a median
# F0 of 248 Hz (p95 474) - right over the women (277-425) - and a pitch "female" guess
# was wrong 114 times out of 114 misses. Above M_BELOW_HZ the voice classifier decides.
M_BELOW_HZ = 250.0
SNIPPET_PEAK = 10 ** (-3 / 20)  # snippets are peak-normalised to -3 dBFS for listening
MAX_GAIN = 10 ** (30 / 20)


def display_name(name: str) -> str:
    """``AnnaSomebodyAndTheBand`` -> ``Anna Somebody And The Band``."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name or "").strip()


def artist_key(name: str) -> str:
    """One key per artist however its folders spell it (``FytaKyte`` = ``Fytakyte``)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def gender_guess(f0_median_hz: float | None) -> str | None:
    if f0_median_hz is None:
        return None
    return "M" if f0_median_hz < M_BELOW_HZ else None


def lead_base(track: str) -> str:
    """The track a double doubles: ``LeadVox2DT1`` -> ``LeadVox2``."""
    return re.sub(r"dt\d*(adlib)?$", "", track, flags=re.I)


class LabelStore:
    """Append-only JSONL of review answers; the latest line per record wins."""

    def __init__(self, data_root: str | Path):
        self.path = Path(data_root) / LABELS_FILE

    def load(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
        return out

    def append(self, labels: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as f:
            for label in labels:
                f.write(json.dumps({**label, "ts": stamp}, ensure_ascii=False) + "\n")


def validate_label(label: dict, known_ids: set[str]) -> dict:
    """Normalise one posted answer; raises ValueError on anything malformed."""
    rid = label.get("id")
    if rid not in known_ids:
        raise ValueError(f"unknown record {rid!r}")
    gender = label.get("gender")
    if gender not in ("F", "M", None):
        raise ValueError(f"gender must be F, M or null, got {gender!r}")
    singer = label.get("singer")
    if singer is not None and (not isinstance(singer, str) or len(singer) > 80):
        raise ValueError("singer must be a string of at most 80 characters")
    flags = label.get("flags") or []
    if not isinstance(flags, list) or set(flags) - set(FLAGS):
        raise ValueError(f"flags must be a subset of {sorted(FLAGS)}")
    return {"id": rid, "gender": gender, "singer": _normalize_singer(singer),
            "flags": sorted(set(flags)), "same_as_lead": bool(label.get("same_as_lead")),
            # confirmed while the harsh measurement was on screen: a "not harsh" is a
            # decision, not an oversight, so the re-check warning retires
            "harsh_seen": bool(label.get("harsh_seen"))}


def _snippets(data_root: Path, rec_id: str, cache: dict) -> list[list[float]]:
    if rec_id not in cache:
        wav = song_dir(data_root, rec_id) / "stems" / "vocals.wav"
        y = mt._load(wav)
        rel, dbfs = mt._frame_db(y)
        mask = (rel > mt.SUNG_DB) & (dbfs > mt.AUDIBLE_DBFS)
        wins = mt._sung_windows(mask, n=N_SNIPPETS, window_sec=SNIPPET_SEC)
        cache[rec_id] = sorted([round(a, 2), round(b - a, 2)] for a, b in wins)
    return cache[rec_id]


def _harsh(data_root: Path, rec_id: str, cache: dict) -> dict | None:
    if rec_id not in cache:
        y, sr = sf.read(str(song_dir(data_root, rec_id) / "stems" / "vocals.wav"),
                        dtype="float32", always_2d=True)
        cache[rec_id] = harshness(y.mean(axis=1), sr)
    return cache[rec_id]


def _settled(label: dict | None) -> bool:
    return bool(label) and not label.get("recheck")


def reopen(data_root: str | Path, record_ids: list[str], note: str) -> int:
    """Put answered tracks back in the queue: their answers stay as the pre-fill."""
    labels = LabelStore(data_root).load()
    rows = [{**{k: v for k, v in labels[r].items() if k != "ts"}, "recheck": True,
             "note": note} for r in record_ids if r in labels]
    LabelStore(data_root).append(rows)
    return len(rows)


def _load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def _main_vocal_records(manifest: Manifest, data_root: Path):
    for rec in manifest.records:
        if rec.meta.corpus != mt.CORPUS:
            continue
        path = song_dir(data_root, rec.id) / "analysis.json"
        if not path.exists():
            continue
        info = json.loads(path.read_text(encoding="utf-8")).get("separate", {}).get("multitrack")
        if info and info.get("role") in MAIN_ROLES:
            yield rec, info


def _artist_names(records) -> dict[str, str]:
    """artist_key -> display name, from the spelling with the most capitals (it splits
    into the most readable words)."""
    spelling: dict[str, str] = {}
    for _, info in records:
        name = info.get("artist") or ""
        k = artist_key(name)
        if sum(c.isupper() for c in name) > sum(c.isupper() for c in spelling.get(k, "")):
            spelling[k] = name
    return {k: display_name(v) for k, v in spelling.items()}


def singer_base(info: dict, names: dict[str, str]) -> str:
    """The default singer id of a track: its artist, lowercased."""
    return (names.get(artist_key(info.get("artist") or "")) or info["folder"]).lower()


def consistent_singers(labels: dict[str, dict], base_of: dict[str, str]) -> dict[str, str | None]:
    """Singer ids that follow the genders the reviewer chose, per artist.

    A song has one singer unless it pairs a male and a female voice (what review of a
    first batch showed). So for ids that are the artist default (``base``, ``base-f``,
    ``base-m``): an artist whose confirmed tracks are all one gender gets one id; an
    artist with both genders gets ``base-f`` / ``base-m`` by each track's gender - in
    every song, whichever order they were reviewed in. Typed names are kept as typed.
    """
    genders: dict[str, set[str]] = {}
    for rid, lab in labels.items():
        if rid in base_of and lab.get("gender"):
            genders.setdefault(base_of[rid], set()).add(lab["gender"])
    out: dict[str, str | None] = {}
    for rid, lab in labels.items():
        singer, base = _normalize_singer(lab.get("singer")), base_of.get(rid)
        if base and singer in (base, f"{base}-f", f"{base}-m"):
            if len(genders.get(base, ())) <= 1:
                singer = base
            elif lab.get("gender"):
                singer = f"{base}-{lab['gender'].lower()}"
        out[rid] = singer
    return out


def build_cards(data_root: str | Path) -> dict:
    """The review queue: one card per song, main vocal tracks with guesses + answers."""
    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    labels = LabelStore(data_root).load()
    snips, harsh = _load_cache(data_root / SNIPPET_CACHE), _load_cache(data_root / HARSH_CACHE)
    n_snips, n_harsh = len(snips), len(harsh)

    records = list(_main_vocal_records(manifest, data_root))
    names = _artist_names(records)

    songs: dict[tuple[str, str], dict] = {}
    for rec, info in records:
        key = (info.get("batch") or "", info["folder"])
        card = songs.setdefault(key, {
            "key": f"{key[0]}/{key[1]}", "batch": key[0], "folder": key[1],
            "artist": names.get(artist_key(info.get("artist") or ""), ""),
            "song": display_name(info.get("song") or "")
            or f"untitled ({key[1].split('_')[0]})", "tracks": []})
        f0 = info.get("f0_median_hz")
        card["tracks"].append({
            "id": rec.id, "track": info["track"], "role": info["role"], "f0": f0,
            "guess_gender": gender_guess(f0), "sung_sec": info.get("sung_sec"),
            "snippets": _snippets(data_root, rec.id, snips),
            "harsh": _harsh(data_root, rec.id, harsh), "label": labels.get(rec.id)})

    if len(snips) != n_snips:
        _save_cache(data_root / SNIPPET_CACHE, snips)
    if len(harsh) != n_harsh:
        _save_cache(data_root / HARSH_CACHE, harsh)

    singers_by_artist: dict[str, set[str]] = {}
    for card in songs.values():
        tracks = sorted(card["tracks"], key=lambda t: (t["role"].endswith("_double"), t["track"]))
        by_name = {t["track"].lower(): t for t in tracks}
        for t in tracks:
            if t["role"].endswith("_double"):
                lead = by_name.get(lead_base(t["track"]).lower())
                t["lead_id"] = lead["id"] if lead and lead is not t else None
        # a band with both a male and a female lead needs two singer ids
        mains = {t["guess_gender"] for t in tracks if not t.get("lead_id")} - {None}
        base = card["artist"].lower() or card["folder"].lower()
        for t in tracks:
            g = t["guess_gender"]
            t["guess_singer"] = f"{base}-{g.lower()}" if len(mains) > 1 and g else base
            if t["label"] and t["label"].get("singer"):
                singers_by_artist.setdefault(card["artist"], set()).add(t["label"]["singer"])
        card["tracks"] = tracks
        # a "recheck" answer is kept as the pre-fill but the song is pending again
        card["confirmed"] = all(_settled(t["label"]) for t in tracks)
        # confirmed before the harsh flag existed, or missed: worth a second listen
        card["harsh_warning"] = any(
            _settled(t["label"]) and (t["harsh"] or {}).get("likely")
            and "harsh" not in t["label"]["flags"] and not t["label"].get("harsh_seen")
            for t in tracks)

    cards = sorted(songs.values(), key=lambda c: (c["batch"], c["folder"]))
    for card in cards:
        card["known_singers"] = sorted(singers_by_artist.get(card["artist"], ()))
    n_tracks = sum(len(c["tracks"]) for c in cards)
    return {
        "cards": cards,
        "flags": FLAGS,
        "progress": {"songs": len(cards), "confirmed": sum(c["confirmed"] for c in cards),
                     "tracks": n_tracks,
                     "labelled": sum(1 for c in cards for t in c["tracks"]
                                     if _settled(t["label"]))},
    }


def snippet_wav(data_root: str | Path, rec_id: str, start: float, dur: float) -> bytes:
    """A peak-normalised 16-bit WAV excerpt of a record's vocal stem."""
    if not re.fullmatch(r"sng_\d+", rec_id):
        raise ValueError(f"bad record id {rec_id!r}")
    if not (0 <= start and 0 < dur <= MAX_SNIPPET_SEC):
        raise ValueError("snippet must start at >= 0 s and last (0, 15] s")
    wav = song_dir(data_root, rec_id) / "stems" / "vocals.wav"
    info = sf.info(str(wav))
    y, sr = sf.read(str(wav), start=int(start * info.samplerate),
                    frames=int(dur * info.samplerate), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y * min(SNIPPET_PEAK / peak, MAX_GAIN)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


FULL_SR = 22050  # full-track listening copy: a 4-minute take is ~10 MB, scrubs instantly


def full_track_wav(data_root: str | Path, rec_id: str) -> bytes:
    """The whole vocal track as one listening copy (mono 22.05 kHz 16-bit, peak -3 dBFS),
    for voices three short snippets cannot settle."""
    from scipy.signal import resample_poly

    if not re.fullmatch(r"sng_\d+", rec_id):
        raise ValueError(f"bad record id {rec_id!r}")
    y, sr = sf.read(str(song_dir(data_root, rec_id) / "stems" / "vocals.wav"),
                    dtype="float32", always_2d=True)
    y = resample_poly(y.mean(axis=1), FULL_SR, sr).astype(np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y * min(SNIPPET_PEAK / peak, MAX_GAIN)
    buf = io.BytesIO()
    sf.write(buf, y, FULL_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _import_processing(rec) -> str | None:
    """What the importer tagged (``review`` only ever raises it to heavy): a record
    un-flagged as effected goes back to its import value, not to whatever was there."""
    if rec.meta.processing != "heavy":
        return rec.meta.processing
    return "produced" if re.search(r"(?<!un)tuned", rec.meta.song or "", re.I) else "dry"


def apply_labels(
    data_root: str | Path, *, dry_run: bool = False
) -> tuple[Manifest, list[tuple[str, dict]]]:
    """Write the latest answer per record into the manifest. Caller saves it.

    Returns ``(manifest, [(record_id, {field: (old, new)})])`` for the records whose
    fields change. Idempotent: re-applying the same answers changes nothing.
    """
    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    labels = LabelStore(data_root).load()
    records = list(_main_vocal_records(manifest, data_root))
    names = _artist_names(records)
    base_of = {rec.id: singer_base(info, names) for rec, info in records}
    singers = consistent_singers(labels, base_of)
    changed: list[tuple[str, dict]] = []
    for rid, label in sorted(labels.items()):
        rec = manifest.get(rid)
        if rec is None:
            continue
        flags = set(label.get("flags") or [])
        new = {
            "gender": label.get("gender"),
            "singer": singers[rid],
            "language": "und" if "not_english" in flags else "en",
            "excluded": ", ".join(f for f in EXCLUDING_FLAGS if f in flags) or None,
            "processing": "heavy" if "effected" in flags else _import_processing(rec),
        }
        diff = {k: (getattr(rec.meta, k), v) for k, v in new.items() if getattr(rec.meta, k) != v}
        if not diff:
            continue
        changed.append((rid, diff))
        if dry_run:
            continue
        for k, v in new.items():
            setattr(rec.meta, k, v)
        update_analysis(song_dir(data_root, rid), "review", {
            "identity": {k: label.get(k) for k in ("gender", "singer", "flags", "same_as_lead")},
            "reviewed": label.get("ts"),
        })
    return manifest, changed
