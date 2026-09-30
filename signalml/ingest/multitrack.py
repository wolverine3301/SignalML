"""Raw studio multitrack adapter — one record per usable vocal track (S2).

A multitrack library ships every instrument and voice of a song as its own raw WAV, one
folder per song (``<Artist>_<Song>_Full`` or ``<CODE>_<Artist>_Full``), tracks named by
role (``03_LeadVox.wav``, ``12_BackingVox2DT.wav``, ``01_Kick.wav``). The vocal tracks
are the untouched mic feeds, so nothing needs separating: they are the cleanest singing
the pipeline has seen. The catch is volume - many songs, hundreds of tracks, nobody
listening - so every keep/drop decision here is a rule with a stated reason, and the
batch report says what was dropped and why.

Decisions, in order:

* **Folder** - skipped when it is an excerpt of a sibling ``…_Full`` folder, or when no
  track is named like a voice (instrumental-only sessions).
* **Name** - tracks named like a voice become candidates; effect prints, samples,
  scratch/guide takes, whispers, choirs and stereo vocal tracks (stacked submixes or
  FX prints) are dropped; a pitch-corrected ``…Tuned`` copy loses to its ``…Untuned``
  sibling. Names that are neither voice- nor instrument-like are reported as
  *unclassified*, never guessed - an audio classifier decides those later.
* **Signal** - measured on short windows where the track is sung:
  - *empty*: under ``MIN_SUNG_SEC`` of singing that is also above ``AUDIBLE_DBFS``
    (silent and hiss-only tracks), or no dynamics at all (a constant noise or tone);
  - *live-room bleed*: the pauses sit less than ``BLEED_FLOOR_DB`` below the singing,
    or an instrument track of the same session is audible in the vocal mic: GCC-PHAT
    (phase-transform cross-correlation, median over the windows) above
    ``BLEED_PHAT``. PHAT responds to "same source at a fixed delay" and ignores "same
    notes" - plain correlation also lit up for studio overdubs where a guitar doubled
    the melody. Measured on a first batch: studio overdubs <= 0.026, live-room tracks
    0.038-0.40. A session with one vocal mic left open to the room (pauses above
    ``BLEED_FLOOR_DB``) was tracked live, so its edited vocal mics must stay under the
    stricter ``LIVE_SESSION_PHAT``. The report keeps every track's value, for re-tuning
    on later batches;
  - *same take on several mics*: waveform correlation above ``DUP_CORR`` with a track
    already kept (0.9+ for mic pairs; a double-tracked second performance is ~0.1).

Layout: song folders may sit directly under the batch folder or one level down in a
*container* folder (``pop/``, ``country rock/``) whose name is kept as a genre hint.
Session documents are read (:mod:`.sheets`): a ``meta.txt`` style becomes the genre,
a composition sheet's lyrics become the main vocals' ``lyrics.txt`` (official lyrics,
so these songs take the human-lyrics path), and the sections, key, tempo and cue times
go to ``song_structure.json``. A vocal that is the same recording as a track imported
from an earlier batch (same artist, audio correlation above ``DUP_CORR``) is dropped:
a ``_BackingStems`` export's guide vocal is often the ``_Full`` session's lead take.

Identity is NOT taken from folder names: a band has several singers and backing parts
are often someone else. Records land with ``singer`` and ``gender`` unset - dataset
build refuses such records - until a voice-embedding pass assigns both. A median-F0
hint is stored in ``analysis.json`` for that pass.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..manifest import (
    FileInfo,
    Manifest,
    ManifestRecord,
    MetaInfo,
    SourceInfo,
    sha256_file,
)
from ..stages.common import song_dir, update_analysis
from .batch import short_name
from .sheets import lyrics_text, read_session_docs

CORPUS = "multitrack"

ANALYSIS_SR = 8000  # plenty for level, correlation and a singing-range F0
MIN_SUNG_SEC = 5.0
SUNG_DB = -35.0  # a 50 ms frame counts as sung within 35 dB of the track's loud level
AUDIBLE_DBFS = -60.0  # ...and above this absolute level (silence/hiss never counts)
FLAT_DB = -10.0  # pauses this close to the loud level: no dynamics, not a performance
BLEED_FLOOR_DB = -41.0  # live-room mics measured -19..-38 dB; studio takes <= -44.6
BLEED_PHAT = 0.035
LIVE_SESSION_PHAT = 0.02  # once an open vocal mic proves the session was tracked live
PHAT_BIN_FLOOR = 1e-3  # bins 60 dB below a signal's peak carry no phase information
DUP_CORR = 0.5
N_WINDOWS, WINDOW_SEC = 3, 10.0

VOICE = re.compile(r"vox|vocal|voice|choir|singer|singing|harmon|adlib|\bbv\b|bgv|whisper|"
                   r"^lv(?=dt|\d|$)", re.I)
NOT_VOICE = re.compile(r"gtr|guitar|perc|vocoder", re.I)  # ElecGtrDT, PercChoir, Vocoder1
INSTRUMENT = re.compile(
    r"kick|snare|hi.?hat|\bhh\b|tom|overhead|\boh\b|room|drum|cymb|ride|crash|bass|gtr|"
    r"guitar|piano|keys|synth|pad|string|violin|viola|cello|fiddle|perc|clap|snap|tamb|"
    r"shaker|organ|hammond|rhodes|wurli|mellotron|sax|trumpet|trombone|horn|brass|flute|"
    r"clarinet|marimba|harp|mandolin|banjo|ukulele|ukelele|dobro|steel|accordion|clav|"
    r"conga|bongo|cajon|timbale|timp|woodblock|maraca|gong|bell|glock|vibe|orchestra|"
    r"sub|808|arp|pluck|noise|riser|loop|click|didgeridoo|radio|whistle|vocoder|midi|"
    r"\bdi\b|amp|bombo|cunono|paliteo|guasa|heartbe|hat|farfisa|kalimba|chime|fill|"
    r"break|rain|footstep|sfx|chamber|reverb|delay|\bamb",
    re.I,
)
DROP_BY_NAME: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"sfx|\bfx\b|effect|sample", re.I), "effects print / sample, not a performance"),
    (re.compile(r"scratch|guide", re.I), "scratch/guide take (replaced by the real one)"),
    (re.compile(r"whisper", re.I), "whispered, not sung"),
    (re.compile(r"choir", re.I), "choir: a group of voices, no single singer"),
)
EXCERPT_OF = "_full"
FOLDER_SUFFIX = re.compile(r"_(full|stems|backingstems)$", re.I)
MAIN_ROLES = frozenset({"lead", "lead_double", "vocal", "vocal_double"})  # sing the lyrics
# documents a session ships beside its audio: lyrics, lead sheets, notes, MIDI. Kept with
# the imported tracks so the lyrics step (and later work) can use them.
DOC_SUFFIXES = frozenset({".txt", ".rtf", ".pdf", ".doc", ".docx", ".odt", ".mid", ".midi",
                          ".xml", ".musicxml", ".mxl", ".csv", ".xls", ".xlsx"})


@dataclass
class TrackDecision:
    track: str  # file stem without the running-number prefix
    path: str  # source path
    kind: str  # vocal | instrument | unclassified
    role: str | None = None  # lead, backing, vocal (+ "_double")
    keep: bool = False
    reason: str = ""
    duplicate_of: str | None = None
    stats: dict = field(default_factory=dict)


@dataclass
class FolderPlan:
    folder: str
    artist: str
    song: str | None
    skip_reason: str = ""
    tracks: list[TrackDecision] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)  # source paths of session documents
    container: str | None = None  # the folder it sat in (a genre hint), if any
    style: str | None = None  # meta.txt "Style:" if the session has one
    sheets: list[str] = field(default_factory=list)  # "comp" / "cue" present

    @property
    def kept(self) -> list[TrackDecision]:
        return [t for t in self.tracks if t.keep]

    @classmethod
    def from_dict(cls, d: dict) -> FolderPlan:
        return cls(folder=d["folder"], artist=d["artist"], song=d.get("song"),
                   skip_reason=d.get("skip_reason", ""),
                   tracks=[TrackDecision(**t) for t in d.get("tracks", [])],
                   docs=list(d.get("docs", [])), container=d.get("container"),
                   style=d.get("style"), sheets=list(d.get("sheets", [])))


def parse_folder_name(name: str) -> tuple[str, str | None]:
    """``Artist_Song_Full`` / ``Artist_Song_Stems`` -> (Artist, Song);
    ``MR1001_Artist_Full`` -> (Artist, None)."""
    base = FOLDER_SUFFIX.sub("", name)
    coded = re.match(r"^[A-Z]{1,4}\d{3,}_(.+)$", base)
    if coded:
        return coded.group(1), None
    artist, _, song = base.partition("_")
    return artist, song or None


def track_name(path: Path) -> str:
    """``03_LeadVox`` -> ``LeadVox``; ``ImAlright_Stems_06_GydVox`` -> ``GydVox``."""
    return re.sub(r"^(?:.+?_stems_)?\d+[\s_-]*", "", path.stem, flags=re.I).strip()


def _is_junk(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")


def song_folders(root: Path) -> list[tuple[Path, str | None]]:
    """``(song_folder, container)`` for every song under ``root``: a folder named like a
    session (``_Full``/``_Stems``/``_BackingStems``) or holding WAVs itself is a song;
    any other folder is a container (a genre shelf) and is searched one level down."""
    out: list[tuple[Path, str | None]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not _is_junk(p)):
        if FOLDER_SUFFIX.search(d.name) or any(
                f.suffix.lower() == ".wav" and not _is_junk(f) for f in d.iterdir()):
            out.append((d, None))
            continue
        for sub in sorted(p for p in d.iterdir() if p.is_dir() and not _is_junk(p)):
            if FOLDER_SUFFIX.search(sub.name) or any(
                    f.suffix.lower() == ".wav" and not _is_junk(f) for f in sub.rglob("*.wav")):
                out.append((sub, d.name))
    return out


def session_docs(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() in DOC_SUFFIXES and not _is_junk(p)]


def song_tracks(folder: Path) -> list[Path]:
    """Every WAV of a song folder (nested subfolders included, macOS zip junk not)."""
    return [p for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() == ".wav" and not _is_junk(p)]


def classify_name(name: str) -> tuple[str, str | None]:
    """(kind, role) from a track name alone."""
    if VOICE.search(name) and not NOT_VOICE.search(name):
        low = name.lower()
        if re.search(r"lead|\bld\b|^lv", low):
            role = "lead"
        elif re.search(r"backing|\bbg\b|\bbv\b|bgv|harmon", low):
            role = "backing"
        else:
            role = "vocal"
        if re.search(r"dt\d*(adlib)?$|double", low):
            role += "_double"
        return "vocal", role
    if INSTRUMENT.search(name):
        return "instrument", None
    return "unclassified", None


def name_drop_reason(name: str, channels: int, siblings: set[str]) -> str:
    for pattern, reason in DROP_BY_NAME:
        if pattern.search(name):
            return reason
    if channels > 1:
        return "stereo vocal track (a stacked submix or FX print, not one voice)"
    low = name.lower()
    if re.search(r"(?<!un)tuned", low):
        twin = re.sub(r"(?<!un)tuned", "untuned", low)
        if twin in siblings:
            return "pitch-corrected copy (its untuned twin is kept)"
    return ""


def _load(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample_poly(y.mean(axis=1), ANALYSIS_SR, sr).astype(np.float32)


def _frame_db(y: np.ndarray, hop: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame level: (dB relative to the track's loud level, dBFS)."""
    n = len(y) // hop
    if n == 0:
        return np.full(1, -200.0), np.full(1, -200.0)
    rms = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-10
    dbfs = 20 * np.log10(rms)
    # the loud level comes from the audible frames: a backing part that sings for ten
    # seconds of a three-minute track is >95% digital silence, and a percentile over all
    # frames would make silence the reference
    audible = rms[dbfs > AUDIBLE_DBFS]
    ref = np.percentile(audible if len(audible) else rms, 95)
    return 20 * np.log10(rms / ref), dbfs


def _xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Peak normalised cross-correlation within +-max_lag samples."""
    n = min(len(a), len(b))
    if n < 2 * max_lag + 1:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    c = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)))
    c = np.concatenate([c[-max_lag:], c[: max_lag + 1]])
    return float(np.max(np.abs(c)) / denom)


def _phat(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """GCC-PHAT peak within +-max_lag: near 0 for independent sources however similar
    their notes; clearly positive when one signal is a filtered, delayed copy of the
    other (an instrument heard in the vocal mic)."""
    n = min(len(a), len(b))
    if n < 2 * max_lag + 1:
        return 0.0
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    fa, fb = np.fft.rfft(a, 2 * n), np.fft.rfft(b, 2 * n)
    # only bins where both signals have energy: PHAT gives every bin equal weight, so
    # near-empty bins (a DI bass has almost nothing above a few hundred Hz) would
    # otherwise contribute leakage phase as if it were signal
    live = (np.abs(fa) > PHAT_BIN_FLOOR * np.abs(fa).max()) \
        & (np.abs(fb) > PHAT_BIN_FLOOR * np.abs(fb).max())
    if live.sum() < 16:
        return 0.0
    cross = fa * np.conj(fb)
    c = np.fft.irfft(np.where(live, cross / (np.abs(cross) + 1e-30), 0))
    return float(np.max(np.abs(np.concatenate([c[-max_lag:], c[: max_lag + 1]]))))


def _sung_windows(sung_mask: np.ndarray, hop: int = 400, *, n: int = N_WINDOWS,
                  window_sec: float = WINDOW_SEC) -> list[tuple[float, float]]:
    """Up to ``n`` non-overlapping windows (seconds) with the most sung frames."""
    per = int(window_sec * ANALYSIS_SR / hop)
    sung = sung_mask.astype(float)
    if len(sung) < per:
        return [(0.0, len(sung) * hop / ANALYSIS_SR)]
    score = np.convolve(sung, np.ones(per), mode="valid")
    out: list[tuple[float, float]] = []
    for _ in range(n):
        i = int(np.argmax(score))
        if score[i] <= 0:
            break
        out.append((i * hop / ANALYSIS_SR, (i + per) * hop / ANALYSIS_SR))
        score[max(0, i - per): i + per] = -1
    return out


def _f0_median(y: np.ndarray, sung_mask: np.ndarray, hop: int = 400) -> float | None:
    import librosa

    f0 = librosa.yin(y, fmin=70, fmax=1000, sr=ANALYSIS_SR, frame_length=1024, hop_length=hop)
    n = min(len(f0), len(sung_mask))
    voiced = f0[:n][sung_mask[:n]]
    return round(float(np.median(voiced)), 1) if len(voiced) > 20 else None


def plan_folder(folder: Path, *, siblings: set[str] | None = None,
                earlier: list[dict] | None = None) -> FolderPlan:
    """Decide every track of one song folder. Reads audio (8 kHz analysis copies).
    ``earlier``: already-imported tracks of the same artist (see :func:`earlier_takes`)."""
    artist, song = parse_folder_name(folder.name)
    plan = FolderPlan(folder=folder.name, artist=artist, song=song,
                      docs=[str(p) for p in session_docs(folder)])
    base = FOLDER_SUFFIX.sub("", folder.name)
    if siblings and not folder.name.lower().endswith(EXCERPT_OF) \
            and f"{base}{EXCERPT_OF}".lower() in siblings:
        plan.skip_reason = (f"excerpt of {base}_Full" if base == folder.name
                            else f"another export of {base}_Full")
        return plan

    paths = song_tracks(folder)
    names = {track_name(p).lower() for p in paths}
    instruments: list[Path] = []
    for p in paths:
        name = track_name(p)
        kind, role = classify_name(name)
        d = TrackDecision(track=name, path=str(p), kind=kind, role=role)
        plan.tracks.append(d)
        if kind == "instrument":
            instruments.append(p)
            d.reason = "instrument"
        elif kind == "unclassified":
            d.reason = "name is neither voice- nor instrument-like (audio classifier decides)"
        else:
            d.reason = name_drop_reason(name, sf.info(str(p)).channels, names)
    vocal = [t for t in plan.tracks if t.kind == "vocal" and not t.reason]
    if not any(t.kind == "vocal" for t in plan.tracks):
        plan.skip_reason = "no vocal tracks (instrumental session)"
        return plan

    inst_audio = {p: _load(p) for p in instruments}  # one read per session track
    # pass 1: measure every candidate
    measured: dict[str, tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]] = {}
    for t in vocal:
        y = _load(Path(t.path))
        rel, dbfs = _frame_db(y)
        sung_mask = (rel > SUNG_DB) & (dbfs > AUDIBLE_DBFS)
        sung = float(sung_mask.sum()) * 400 / ANALYSIS_SR
        floor = float(np.percentile(rel, 10))
        t.stats = {"sec": round(len(y) / ANALYSIS_SR, 1), "sung_sec": round(sung, 1),
                   "floor_db": round(floor, 1), "peak_dbfs": round(float(dbfs.max()), 1)}
        if sung < MIN_SUNG_SEC:
            t.reason = f"empty: {sung:.1f} s of audible singing"
            continue
        if floor > FLAT_DB:
            t.reason = "empty: no dynamics (constant noise or tone, not a performance)"
            continue
        wins = [(int(a * ANALYSIS_SR), int(b * ANALYSIS_SR))
                for a, b in _sung_windows(sung_mask)]
        t.stats["instr_phat"] = round(max(
            (float(np.median([_phat(y[a:b], x[a:b], 160) for a, b in wins]))
             for x in inst_audio.values()), default=0.0), 3)
        measured[t.track] = (y, sung_mask, wins)

    # a vocal mic left open in the room (pauses full of band) proves the session was
    # tracked live; its edited vocal mics then get the stricter bar, since editing hides
    # the pauses but not the band under the singing
    live_session = any(t.stats["floor_db"] > BLEED_FLOOR_DB for t in vocal if t.track in measured)
    bar = LIVE_SESSION_PHAT if live_session else BLEED_PHAT

    # pass 2: decide - leads first, so a mic duplicate resolves to the lead's name
    kept: list[tuple[TrackDecision, np.ndarray]] = []
    for t in sorted(vocal, key=lambda t: (not t.role.startswith("lead"), t.track)):
        if t.track not in measured:
            continue
        y, sung_mask, wins = measured[t.track]
        floor, phat = t.stats["floor_db"], t.stats["instr_phat"]
        if floor > BLEED_FLOOR_DB or phat > bar:
            t.reason = (f"live-room bleed (pause floor {floor:.0f} dB, instrument PHAT "
                        f"{phat:.3f}{', live session' if live_session else ''})")
            continue
        for other, oy in kept:
            dup = max(_xcorr(y[a:b], oy[a:b], 80) for a, b in wins)
            if dup > DUP_CORR:
                t.reason = f"same take as {other.track} on another mic (corr {dup:.2f})"
                t.duplicate_of = other.track
                break
        if t.reason:
            continue
        for prev in earlier or []:
            dup = max(_xcorr(y[a:b], prev["audio"][a:b], 80) for a, b in wins)
            if dup > DUP_CORR:
                t.reason = (f"same take as {prev['id']} ({prev['folder']}:{prev['track']}), "
                            f"imported earlier (corr {dup:.2f})")
                t.duplicate_of = prev["id"]
                break
        if t.reason:
            continue
        t.stats["f0_median_hz"] = _f0_median(y, sung_mask)
        t.keep = True
        kept.append((t, y))
    return plan


def _song_key(song: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (song or "").lower())


def _artist_match(a: str, b: str) -> bool:
    """Same artist, allowing a featuring credit: ``Triviul`` vs ``TriviulFeatTheFiend``."""
    a, b = _song_key(a), _song_key(b)
    return a == b or (min(len(a), len(b)) >= 5 and (a.startswith(b) or b.startswith(a)))


def earlier_takes(data_root: Path, manifest: Manifest, artist: str, song: str | None,
                  exclude_batch: str) -> list[dict]:
    """Imported multitrack vocals of the same artist and song (or an untitled one) from
    other batches, with 8 kHz audio for the duplicate check."""
    out = []
    for rec in manifest.records:
        if rec.meta.corpus != CORPUS:
            continue
        path = song_dir(data_root, rec.id) / "analysis.json"
        info = json.loads(path.read_text(encoding="utf-8")).get("separate", {}) \
            .get("multitrack", {}) if path.exists() else {}
        if not info or info.get("batch") == exclude_batch \
                or not _artist_match(info.get("artist", ""), artist):
            continue
        if song and info.get("song") and _song_key(info["song"]) != _song_key(song):
            continue
        out.append({"id": rec.id, "folder": info["folder"], "track": info["track"],
                    "audio": _load(song_dir(data_root, rec.id) / "stems" / "vocals.wav")})
    return out


def plan_batch(root: str | Path, *, reuse: dict[str, dict] | None = None,
               earlier=None) -> list[FolderPlan]:
    """Plan every song folder. Folders found in ``reuse`` (a previous report's plans,
    keyed by folder name) keep their decisions and are not read again, so a batch that
    grows by a few folders costs only the new ones. ``earlier(artist, song)`` returns
    already-imported takes to check new vocals against (cross-batch duplicates)."""
    root = Path(root)
    reuse = reuse or {}
    folders = song_folders(root)
    names = {f.name.lower() for f, _ in folders}
    plans = []
    for f, container in folders:
        if f.name in reuse:
            plan = FolderPlan.from_dict(reuse[f.name])
            plan.docs = [str(p) for p in session_docs(f)]
        else:
            artist, song = parse_folder_name(f.name)
            plan = plan_folder(f, siblings=names,
                               earlier=earlier(artist, song) if earlier else None)
        plan.container = container
        docs = read_session_docs(plan.docs)
        plan.style = (docs.get("meta") or {}).get("style")
        plan.sheets = [k for k in ("comp", "cue") if k in docs]
        plans.append(plan)
    return plans


def batch_report(plans: list[FolderPlan]) -> dict:
    """Counts and sung hours per decision, plus every folder's full decision list."""
    kept_roles: dict[str, float] = {}
    dropped: dict[str, int] = {}
    for fp in plans:
        for t in fp.tracks:
            if t.keep:
                kept_roles[t.role] = kept_roles.get(t.role, 0.0) + t.stats["sung_sec"] / 3600
            elif t.kind == "vocal":
                key = re.sub(r"\s*\(.*|:.*| as .*", "", t.reason)
                dropped[key] = dropped.get(key, 0) + 1
    return {
        "folders": len(plans),
        "folders_skipped": {fp.folder: fp.skip_reason for fp in plans if fp.skip_reason},
        "kept_tracks": sum(len(fp.kept) for fp in plans),
        "kept_sung_hours_by_role": {k: round(v, 2) for k, v in sorted(kept_roles.items())},
        "dropped_vocal_tracks_by_reason": dict(sorted(dropped.items(), key=lambda kv: -kv[1])),
        "unclassified_tracks": sorted(f"{fp.folder}:{t.track}" for fp in plans
                                      for t in fp.tracks if t.kind == "unclassified"),
        "songs_with_lyrics_sheet": sum(1 for fp in plans if "comp" in fp.sheets and fp.kept),
        "songs_with_cue_sheet": sum(1 for fp in plans if "cue" in fp.sheets and fp.kept),
        "styles": sorted({fp.style or fp.container for fp in plans
                          if fp.kept and (fp.style or fp.container)}),
        "plans": [asdict(fp) for fp in plans],
    }


def _write_song_documents(data_root: Path, song_raw: Path, fp: FolderPlan) -> dict:
    """Parse a session's documents once; write lyrics.txt / song_structure.json beside
    its imported tracks. Returns what the records need (genre, lyrics, structure)."""
    docs = read_session_docs(fp.docs)
    out: dict = {"genre": (fp.style or fp.container or "").strip().lower() or None}
    if "comp" in docs or "cue" in docs or fp.style:
        song_raw.mkdir(parents=True, exist_ok=True)
        structure = {"style": fp.style, "container": fp.container,
                     "cue": docs.get("cue"), "comp": docs.get("comp")}
        path = song_raw / "song_structure.json"
        path.write_text(json.dumps(structure, indent=1, ensure_ascii=False), encoding="utf-8")
        cue = docs.get("cue") or {}
        out["structure"] = {"file": path.relative_to(data_root).as_posix(), "style": fp.style,
                            "key": cue.get("key"), "meter": cue.get("meter"),
                            "bpm": cue.get("bpm"), "sections": [
                                s["name"] for s in (docs.get("comp") or {}).get("sections", [])]}
    if docs.get("comp"):
        lyr = song_raw / "lyrics.txt"
        lyr.write_text(lyrics_text(docs["comp"]), encoding="utf-8")
        out["lyrics_path"] = lyr.relative_to(data_root).as_posix()
    return out


def _same_song_records(data_root: Path, manifest: Manifest, fp: FolderPlan,
                       batch: str) -> list[tuple[ManifestRecord, str]]:
    """Main-vocal records of this song imported from other batches - by duplicate link
    (a dropped take that *is* an earlier recording) or by artist and song name - so a
    later export's sheets reach the takes that sing them. ``(record, role)`` pairs."""
    linked = {_folder_of(data_root, t.duplicate_of) for t in fp.tracks
              if t.duplicate_of and t.duplicate_of.startswith("sng_")} - {None}
    out = []
    for rec in manifest.records:
        if rec.meta.corpus != CORPUS:
            continue
        path = song_dir(data_root, rec.id) / "analysis.json"
        analysis = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        info = analysis.get("separate", {}).get("multitrack", {})
        if not info or info.get("batch") == batch or info.get("role") not in MAIN_ROLES:
            continue
        same_name = (bool(fp.song and info.get("song"))
                     and _artist_match(info.get("artist", ""), fp.artist)
                     and _song_key(info["song"]) == _song_key(fp.song))
        if same_name or info["folder"] in linked:
            out.append((rec, info["role"]))
    return out


def _folder_of(data_root: Path, rec_id: str) -> str | None:
    path = song_dir(data_root, rec_id) / "analysis.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("separate", {}) \
        .get("multitrack", {}).get("folder")


def _attach_documents(data_root: Path, rec: ManifestRecord, t: TrackDecision,
                      docs: list[str], song_meta: dict) -> None:
    """Genre for every track; official lyrics for the ones that sing them (main roles -
    a backing part sings a subset, left to the ASR/resolve path); analysis sections."""
    sdir = song_dir(data_root, rec.id)
    if docs:
        update_analysis(sdir, "session_docs", {"files": docs})
    if song_meta.get("structure"):
        update_analysis(sdir, "song_structure", song_meta["structure"])
    if song_meta.get("genre") and not rec.meta.genre:
        rec.meta.genre = song_meta["genre"]
    if song_meta.get("lyrics_path") and t.role in MAIN_ROLES and not rec.meta.has_lyrics:
        rec.meta.has_lyrics = True
        rec.meta.lyrics_path = song_meta["lyrics_path"]
        rec.meta.lyrics_source = None  # the session's composition sheet: human lyrics


def import_multitrack(
    data_root: str | Path,
    *,
    root: str | Path,
    batch: str,
    dry_run: bool = False,
    replan: bool = False,
) -> tuple[Manifest, list[ManifestRecord], dict]:
    """Plan a batch and import its kept vocal tracks. Caller saves the manifest.

    Kept tracks are copied to ``RAW/<batch>/<folder>/`` (the source tree holds the whole
    session, instruments included) and to ``songs/<id>/stems/vocals.wav``; the session
    documents (lyrics, lead sheets, MIDI) go to ``RAW/<batch>/<folder>/docs/``. Re-running
    on a grown batch plans only the new folders (``replan`` re-checks every folder) and
    imports only what the manifest does not hold yet. The report is written to
    ``RAW/<batch>/import_report.json`` either way.
    """
    data_root = Path(data_root)
    raw_dir = data_root / "RAW" / batch
    report_path = raw_dir / "import_report.json"
    reuse = {}
    if report_path.exists() and not replan:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        reuse = {p["folder"]: p for p in previous["plans"]}
    manifest = Manifest.for_data_root(data_root)
    plans = plan_batch(root, reuse=reuse, earlier=lambda artist, song: earlier_takes(
        data_root, manifest, artist, song, exclude_batch=batch))
    report = batch_report(plans)
    report["planned_now"] = sum(1 for fp in plans if fp.folder not in reuse)
    new_records: list[ManifestRecord] = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    if dry_run:
        return manifest, new_records, report

    by_path = {r.file.path: r for r in manifest.records}
    for fp in plans:
        earlier_song = _same_song_records(data_root, manifest, fp, batch) \
            if (fp.docs and (fp.sheets or fp.style)) else []
        if not fp.kept and not earlier_song:
            continue
        docs = []
        for doc in map(Path, fp.docs):
            ddst = raw_dir / short_name(fp.folder) / "docs" / doc.name
            ddst.parent.mkdir(parents=True, exist_ok=True)
            if not ddst.exists():
                shutil.copyfile(doc, ddst)
            docs.append(ddst.relative_to(data_root).as_posix())
        song_meta = _write_song_documents(data_root, raw_dir / short_name(fp.folder), fp)
        for rec, role in earlier_song:  # a later export's sheets for takes imported before
            _attach_documents(data_root, rec, TrackDecision(track="", path="", kind="vocal",
                                                            role=role), docs, song_meta)
            manifest.upsert(rec)
        for t in fp.kept:
            src = Path(t.path)
            dst = raw_dir / short_name(fp.folder) / src.name
            rel = dst.relative_to(data_root).as_posix()
            if rel in by_path:  # imported by an earlier run: only refresh its documents
                _attach_documents(data_root, by_path[rel], t, docs, song_meta)
                continue
            digest = sha256_file(src)
            if manifest.by_sha256(digest):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
            info = sf.info(str(dst))
            title = f"{fp.artist} - {fp.song or fp.folder}"
            rec = ManifestRecord(
                id=manifest.next_id(),
                source=SourceInfo(kind="local", retrieved=date.today().isoformat()),
                file=FileInfo(path=rel, sha256=digest, duration_sec=float(info.duration),
                              sample_rate=int(info.samplerate), channels=int(info.channels)),
                meta=MetaInfo(
                    singer=None,  # assigned from the voice, not the folder name
                    gender=None,
                    song=f"{title} [{t.track}]",
                    language=None,  # set by the ASR pass
                    source_quality="studio",
                    processing="produced" if re.search(r"(?<!un)tuned", t.track, re.I)
                    else "dry",
                    corpus=CORPUS,
                ),
            )
            rec.status.separated = True  # a raw track IS the isolated voice
            rec.quality.notes = f"multitrack {fp.folder}:{t.track} role={t.role}"
            sdir = song_dir(data_root, rec.id)
            (sdir / "stems").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dst, sdir / "stems" / "vocals.wav")
            update_analysis(sdir, "separate", {
                "model": "multitrack-raw",
                "imported_from": rel,
                "multitrack": {"batch": batch, "folder": fp.folder, "artist": fp.artist,
                               "song": fp.song, "track": t.track, "role": t.role,
                               **t.stats},
                "date": date.today().isoformat(),
            })
            _attach_documents(data_root, rec, t, docs, song_meta)
            manifest.add(rec)
            by_path[rel] = rec
            new_records.append(rec)
    return manifest, new_records, report
