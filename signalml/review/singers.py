"""Who sings the backing vocals - voice-embedding grouping for multitrack tracks.

Identity review (:mod:`.identity`) names the singer of every main vocal by ear. The
backing parts - most of a multitrack library - stay unlabelled, and dataset build
refuses a record without a singer. Usually the lead sings their own backing, but not
always (a band's second singer, a session vocalist), and pitch cannot tell: a male
lead's harmonies sit as high as a woman's voice.

So voices are compared instead. ``scripts/ecapa_embed.py`` (ECAPA-TDNN trained on
VoxCeleb, in its own venv) turns each track's sung stretches into 3-second voiceprints,
stored per record as ``features/ecapa.json``. Matching works per artist - a band's
singers recur across its songs - against the singers the reviewer already named:

* a track whose voice is a named singer's (cosine to that singer's centroid at least
  ``same_singer``, ahead of every other named singer of the artist by ``margin``) takes
  that singer and gender with no listening;
* the rest are grouped by voice (average linkage; groups merge while their mean cosine
  is at least ``group_join``), and each group is ONE review question - one answer labels
  every track in it;
* a double (``BackingVox3DT1``) follows the track it doubles (``BackingVox3``) and is
  never asked about.

Tracks tagged ``processing: heavy`` (identity review's "effected": heavy FX, pitch-shifted,
a voice played as an instrument) take no part: their voiceprint is not the singer's.
The reviewer can tag a backing part the same way (verdict ``effected``).

Only groups holding a backing part that sings NEW lyrics - words the song's main vocals
never sing (an answer line, a second singer's verse) - are queued for review. Wordless
parts (oohs, aahs), parts re-singing the lead's lines and doubles are not wanted for
training (reviewer's call, 2026-10-03), so their singer does not need a human. Words
are counted from the lyrics step's transcript with vocables left out.

The model learned voices from speech; high and wordless singing is far from that, so
scores are only compared inside one artist's tracks (same studio, same mics), and the
thresholds are measured against the reviewer's own labels (:func:`calibrate`).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel

from ..config import CONFIGS_DIR
from ..manifest import Manifest, _normalize_singer
from ..stages.common import song_dir, update_analysis
from .identity import (
    MAIN_ROLES,
    SNIPPET_CACHE,
    LabelStore,
    _import_processing,
    _load_cache,
    _save_cache,
    _snippets,
    artist_key,
    display_name,
    lead_base,
)

EMBED_NAME = Path("features") / "ecapa.json"
MATCH_FILE = Path("review") / "singer_match.json"
LABELS_FILE = Path("review") / "singers.jsonl"
CORPUS = "multitrack"
SPAN_BRIDGE_SEC = 0.3  # gaps shorter than this (breaths, consonant dips) stay in one span
MIN_LEXICAL = 8  # fewer real words than this: a wordless part (oohs, aahs, "yeah")
MIN_DISTINCT = 4  # ...or fewer different ones: vocal percussion ("buh buh buh", "bop")
# new lyrics: at least this many different words the main vocals never sing, making up
# at least this share of the part's different words (a stray misheard word is not a line)
NEW_DISTINCT, NEW_SHARE = 5, 0.3
VERDICTS = ("singer", "new", "unclear", "effected")
UNNAMED = ("unclear", "effected")  # verdicts that leave the part without a singer
UNCLEAR = "singer unclear"  # meta.excluded for a part the reviewer could not place


class SingersConfig(BaseModel):
    """Embedding backend wiring + matching thresholds."""

    # argv; {jobs} is replaced by the batch file. Empty = refuse rather than guess.
    command: list[str] = []
    command_cwd: str | None = None
    timeout_sec: float = 2 * 3600.0
    min_windows: int = 2  # fewer 3 s windows than this: too little voice to compare
    same_singer: float = 0.75
    margin: float = 0.15
    group_join: float = 0.70


def load_singers_config(path: str | Path | None = None) -> SingersConfig:
    path = Path(path) if path else CONFIGS_DIR / "singers.yaml"
    if path.exists():
        return SingersConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return SingersConfig()


def _analysis(sdir: Path) -> dict:
    path = sdir / "analysis.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# -- embedding (external backend) ------------------------------------------------------

def sung_spans(intervals: list[list[float]], bridge: float = SPAN_BRIDGE_SEC) -> list[list[float]]:
    """The clean stage's non-silent intervals, with short gaps bridged."""
    spans: list[list[float]] = []
    for start, end in intervals:
        if spans and start - spans[-1][1] < bridge:
            spans[-1][1] = end
        else:
            spans.append([start, end])
    return spans


def embedding_jobs(data_root: str | Path, *, ids: list[str] | None = None,
                   force: bool = False) -> list[dict]:
    """``[{"id", "wav", "spans"}]`` for every cleaned, non-excluded multitrack record
    without an embedding yet."""
    data_root = Path(data_root)
    jobs = []
    for rec in Manifest.for_data_root(data_root).records:
        if rec.meta.corpus != CORPUS or rec.meta.excluded or not rec.status.cleaned:
            continue
        if ids and rec.id not in ids:
            continue
        sdir = song_dir(data_root, rec.id)
        wav = sdir / "clean" / "vocals.wav"
        if not wav.exists() or ((sdir / EMBED_NAME).exists() and not force):
            continue
        silence = _analysis(sdir).get("clean", {}).get("stems", {}).get("vocals", {}).get("silence")
        jobs.append({"id": rec.id, "wav": str(wav),
                     "spans": sung_spans(silence["intervals_sec"]) if silence else None})
    return jobs


def import_results(data_root: str | Path,
                   results: dict[str, Path]) -> tuple[list[str], dict[str, str]]:
    """Copy backend outputs (id -> json path) into place; returns (written, failed)."""
    data_root = Path(data_root)
    written, failed = [], {}
    for rid, out in results.items():
        if not out.exists():
            failed[rid] = "backend wrote no result"
            continue
        payload = json.loads(out.read_text(encoding="utf-8"))
        if "error" in payload:
            failed[rid] = payload["error"]
            continue
        sdir = song_dir(data_root, rid)
        dest = sdir / EMBED_NAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, dest)
        update_analysis(sdir, "singer_embed", {
            "model": payload.get("model"), "windows": len(payload.get("windows", [])),
            "date": _dt.date.today().isoformat()})
        written.append(rid)
    return written, failed


def _default_runner(argv: list[str], cwd: str | None, timeout: float) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        raise RuntimeError(f"embedding backend failed (exit {proc.returncode}):\n{tail}")


def embed(data_root: str | Path, *, cfg: SingersConfig | None = None,
          runner: Callable[[list[str], str | None, float], None] | None = None,
          ids: list[str] | None = None, force: bool = False) -> tuple[list[str], dict[str, str]]:
    """One backend invocation over every record that needs an embedding."""
    data_root = Path(data_root)
    cfg = cfg or load_singers_config()
    if not cfg.command:
        raise RuntimeError(
            "no embedding command configured - copy configs/singers.yaml to "
            "configs/singers.local.yaml and point `command` at the ecapa venv + "
            "scripts/ecapa_embed.py")
    jobs = embedding_jobs(data_root, ids=ids, force=force)
    if not jobs:
        return [], {}
    batch = data_root / "work" / f"ecapa_{_dt.datetime.now():%Y%m%d_%H%M%S_%f}"
    batch.mkdir(parents=True, exist_ok=True)
    outs = {j["id"]: batch / f"{j['id']}.json" for j in jobs}
    (batch / "jobs.json").write_text(json.dumps(
        [{"wav": j["wav"], "out": str(outs[j["id"]]), "spans": j["spans"]} for j in jobs],
        indent=1), encoding="utf-8")
    argv = [part.format(jobs=str(batch / "jobs.json")) for part in cfg.command]
    (runner or _default_runner)(argv, cfg.command_cwd, cfg.timeout_sec)
    return import_results(data_root, outs)


# -- what a part sings ------------------------------------------------------------------

VOCABLE = re.compile(
    r"^(o+h*|a+h*|u+h*|m+|hm+|mm+hm+|la+|na+|da+|do+|du+|ba+|ha+|ho+|hoo+|whoa+|woah+|"
    r"wo+|ay+|eh+|ye+ah+|yeah+|hey+|ya+|yo+|ooh+|aah+|ahh+)$")


def lexical_words(text: str) -> list[str]:
    """Words that carry lyrics - vocables (ooh, ah, la, yeah...) left out."""
    words = (w.strip("'") for w in re.findall(r"[a-z']+", text.lower().replace("-", " ")))
    return [w for w in words if w and not VOCABLE.match(w)]


def _lyrics_text(data_root: Path, rec) -> str | None:
    if not (rec.meta.has_lyrics and rec.meta.lyrics_path):
        return None
    path = data_root / rec.meta.lyrics_path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def content(text: str | None, main_vocab: set[str]) -> dict:
    """``{"words", "new_words", "wordless", "new_lyrics", "sample"}`` for one part.

    ``new_words`` are the different real words the song's main vocals never sing; no
    transcript at all (Whisper heard under ``min_words``) is wordless."""
    if text is None:
        return {"words": 0, "new_words": 0, "wordless": True, "new_lyrics": False,
                "sample": ""}
    lex = lexical_words(text)
    distinct = set(lex)
    new = distinct - main_vocab
    wordless = len(lex) < MIN_LEXICAL or len(distinct) < MIN_DISTINCT
    return {"words": len(lex), "new_words": len(new), "wordless": wordless,
            "new_lyrics": not wordless and len(new) >= NEW_DISTINCT
            and len(new) >= NEW_SHARE * len(distinct),
            "sample": " / ".join(line.strip() for line in text.splitlines() if line.strip())[:160]}


# -- matching ----------------------------------------------------------------------------

@dataclass
class Track:
    id: str
    artist: str  # artist_key: one per band however its folders spell it
    folder: str
    song: str
    track: str
    role: str
    singer: str | None
    gender: str | None
    sung_sec: float | None = None
    n_windows: int = 0
    centroid: np.ndarray | None = None
    parent: str | None = None  # the track this one doubles, when the session has it
    content: dict = field(default_factory=dict)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def load_tracks(data_root: str | Path, cfg: SingersConfig | None = None) -> list[Track]:
    """Every non-excluded multitrack record: voice centroid (None without an embedding or
    with fewer than ``min_windows`` windows), the track it doubles, what it sings."""
    data_root = Path(data_root)
    cfg = cfg or SingersConfig()
    manifest = Manifest.for_data_root(data_root)
    out: list[Track] = []
    texts: dict[str, str | None] = {}
    for rec in manifest.records:
        if rec.meta.corpus != CORPUS or rec.meta.excluded or rec.meta.processing == "heavy":
            continue
        sdir = song_dir(data_root, rec.id)
        info = _analysis(sdir).get("separate", {}).get("multitrack")
        if not info:
            continue
        t = Track(id=rec.id, artist=artist_key(info.get("artist") or info["folder"]),
                  folder=info["folder"],
                  song=display_name(info.get("song") or "") or info["folder"],
                  track=info["track"], role=info["role"], singer=rec.meta.singer,
                  gender=rec.meta.gender, sung_sec=info.get("sung_sec"))
        emb = sdir / EMBED_NAME
        if emb.exists():
            vecs = np.asarray(json.loads(emb.read_text(encoding="utf-8"))["embeddings"],
                              dtype=np.float32)
            t.n_windows = len(vecs)
            if len(vecs) >= cfg.min_windows:
                t.centroid = _unit(vecs.mean(axis=0))
        texts[rec.id] = _lyrics_text(data_root, rec)
        out.append(t)

    by_folder: dict[str, list[Track]] = defaultdict(list)
    for t in out:
        by_folder[t.folder].append(t)
    for ts in by_folder.values():
        names = {t.track.lower(): t for t in ts}
        vocab = {w for t in ts if t.role in MAIN_ROLES for w in lexical_words(texts[t.id] or "")}
        for t in ts:
            if t.role.endswith("_double"):
                p = names.get(lead_base(t.track).lower())
                t.parent = p.id if p is not None and p is not t else None
            t.content = content(texts[t.id], vocab)
    return out


def _singer_centroids(tracks: list[Track], exclude: str | None = None) -> dict[str, np.ndarray]:
    by: dict[str, list[np.ndarray]] = defaultdict(list)
    for t in tracks:
        if t.singer and t.centroid is not None and t.id != exclude:
            by[t.singer].append(t.centroid)
    return {s: _unit(np.mean(v, axis=0)) for s, v in by.items()}


def _singer_genders(tracks: list[Track]) -> dict[str, str | None]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for t in tracks:
        if t.singer and t.gender:
            votes[t.singer][t.gender] += 1
    return {s: c.most_common(1)[0][0] for s, c in votes.items()}


def _pct(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = np.asarray(values)
    return {"n": len(v), **{f"p{p}": round(float(np.percentile(v, p)), 3)
                            for p in (1, 5, 25, 50, 75, 95, 99)}}


def calibrate(tracks: list[Track]) -> dict:
    """Score distributions on the reviewer's labels, the way :func:`match` scores:

    * ``same``: a labelled track against its own singer's centroid built WITHOUT it
      (leave-one-out) - what a backing part sung by a named singer scores;
    * ``same_other_song``: the same, with the centroid built only from the singer's
      other songs - the hard case of a singer whose style changes song to song;
    * ``different``: a labelled track against every OTHER named singer of its artist
      (bands with two singers) - what a stranger in the band scores.
    """
    same, other_song, diff = [], [], []
    by_artist: dict[str, list[Track]] = defaultdict(list)
    for t in tracks:
        by_artist[t.artist].append(t)
    for ts in by_artist.values():
        for t in ts:
            if not t.singer or t.centroid is None:
                continue
            cents = _singer_centroids(ts, exclude=t.id)
            if t.singer in cents:
                same.append(float(t.centroid @ cents[t.singer]))
            elsewhere = _singer_centroids([u for u in ts if u.folder != t.folder])
            if t.singer in elsewhere:
                other_song.append(float(t.centroid @ elsewhere[t.singer]))
            diff.extend(float(t.centroid @ c) for s, c in cents.items() if s != t.singer)
    return {"same": _pct(same), "same_other_song": _pct(other_song), "different": _pct(diff)}


def _groups(tracks: list[Track], join: float) -> list[list[Track]]:
    """Average-linkage clusters: groups merge while their mean cosine is >= ``join``."""
    if len(tracks) == 1:
        return [tracks]
    from scipy.cluster.hierarchy import fcluster, linkage

    labels = fcluster(linkage(np.stack([t.centroid for t in tracks]), method="average",
                              metric="cosine"), t=1.0 - join, criterion="distance")
    out: dict[int, list[Track]] = defaultdict(list)
    for t, lab in zip(tracks, labels):
        out[int(lab)].append(t)
    return sorted(out.values(), key=lambda g: (-len(g), g[0].id))


def _member(t: Track, **extra) -> dict:
    return {"id": t.id, "folder": t.folder, "song": t.song, "track": t.track, "role": t.role,
            "sung_sec": t.sung_sec, **t.content, **extra}


@dataclass
class MatchResult:
    auto: list[dict] = field(default_factory=list)  # confident: singer assigned
    groups: list[dict] = field(default_factory=list)  # one review question each
    follows: dict[str, str] = field(default_factory=dict)  # double id -> the id it doubles
    unmatched: list[dict] = field(default_factory=list)  # no embedding / too little voice


def match(tracks: list[Track], cfg: SingersConfig) -> MatchResult:
    res = MatchResult()
    by_artist: dict[str, list[Track]] = defaultdict(list)
    for t in tracks:
        by_artist[t.artist].append(t)
    for artist in sorted(by_artist):
        ts = by_artist[artist]
        anchors = _singer_centroids(ts)
        genders = _singer_genders(ts)
        rest: list[Track] = []
        for t in ts:
            if t.singer:
                continue
            if t.parent:
                res.follows[t.id] = t.parent
                continue
            if t.centroid is None:
                res.unmatched.append(_member(t, artist=artist, windows=t.n_windows))
                continue
            scores = sorted(((float(t.centroid @ c), s) for s, c in anchors.items()),
                            reverse=True)
            best, singer = scores[0] if scores else (-1.0, None)
            second = scores[1][0] if len(scores) > 1 else None
            if singer and best >= cfg.same_singer and best - (second or -1.0) >= cfg.margin:
                res.auto.append(_member(t, artist=artist, singer=singer,
                                        gender=genders.get(singer), score=round(best, 3),
                                        second=None if second is None else round(second, 3)))
            else:
                rest.append(t)
        for k, members in enumerate(_groups(rest, cfg.group_join) if rest else []):
            c = _unit(np.mean([m.centroid for m in members], axis=0))
            near = sorted(((float(c @ a), s) for s, a in anchors.items()), reverse=True)
            inner = [float(a.centroid @ b.centroid) for i, a in enumerate(members)
                     for b in members[i + 1:]]
            res.groups.append({
                "key": f"{artist}/{k + 1}", "artist": artist,
                "members": [_member(m, score_to_group=round(float(m.centroid @ c), 3))
                            for m in members],
                "cohesion": round(float(np.mean(inner)), 3) if inner else None,
                "nearest": [{"singer": s, "gender": genders.get(s), "score": round(v, 3)}
                            for v, s in near],
                "needs_review": any(m.content.get("new_lyrics") for m in members),
            })
    return res


def run_match(data_root: str | Path, cfg: SingersConfig | None = None) -> dict:
    """Calibrate + match, and write ``review/singer_match.json``. Changes no record."""
    data_root = Path(data_root)
    cfg = cfg or load_singers_config()
    tracks = load_tracks(data_root, cfg)
    res = match(tracks, cfg)
    queued = [g for g in res.groups if g["needs_review"]]
    report = {
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": cfg.model_dump(exclude={"command", "command_cwd", "timeout_sec"}),
        "calibration": calibrate(tracks),
        "counts": {"tracks": len(tracks), "labelled": sum(bool(t.singer) for t in tracks),
                   "auto": len(res.auto), "follows": len(res.follows),
                   "groups": len(res.groups), "groups_to_review": len(queued),
                   "tracks_to_review": sum(len(g["members"]) for g in queued),
                   "unmatched": len(res.unmatched)},
        "auto": res.auto, "groups": res.groups, "follows": res.follows,
        "unmatched": res.unmatched,
    }
    path = data_root / MATCH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# -- review -------------------------------------------------------------------------------

def label_store(data_root: str | Path) -> LabelStore:
    """Answers append to ``review/singers.jsonl``; the latest line per record wins."""
    return LabelStore(data_root, LABELS_FILE)


def _base(artist: str, singers: list[str], names: dict[str, str]) -> str:
    named = [re.sub(r"-[fm]$", "", s) for s in singers]
    return Counter(named).most_common(1)[0][0] if named else names.get(artist, artist)


def build_cards(data_root: str | Path) -> dict:
    """The review queue from the last ``match``: one card per voice group that holds a
    part with real words. Each card carries snippets of its parts and of the artist's
    named singers (their track in the same song when there is one), for A/B listening."""
    data_root = Path(data_root)
    report = json.loads((data_root / MATCH_FILE).read_text(encoding="utf-8"))
    labels = label_store(data_root).load()
    snips = _load_cache(data_root / SNIPPET_CACHE)
    n_snips = len(snips)
    manifest = Manifest.for_data_root(data_root)

    refs_by_artist: dict[str, list] = defaultdict(list)  # named tracks to compare with
    names: dict[str, str] = {}
    for rec in manifest.records:
        if rec.meta.corpus != CORPUS or rec.meta.excluded or not rec.meta.singer \
                or rec.meta.processing == "heavy":
            continue
        info = _analysis(song_dir(data_root, rec.id)).get("separate", {}).get("multitrack")
        if info and info.get("role") in ("lead", "vocal"):
            a = artist_key(info.get("artist") or info["folder"])
            names.setdefault(a, display_name(info.get("artist") or ""))
            refs_by_artist[a].append((rec, info))

    cards = []
    for g in report["groups"]:
        if not g["needs_review"]:
            continue
        folders = {m["folder"] for m in g["members"]}
        refs = []
        for n in g["nearest"]:
            cands = [(r, i) for r, i in refs_by_artist[g["artist"]] if r.meta.singer == n["singer"]]
            cands.sort(key=lambda ri: (ri[1]["folder"] not in folders, ri[1]["role"] != "lead",
                                       -(ri[1].get("sung_sec") or 0)))
            if cands:
                r, i = cands[0]
                refs.append({**n, "id": r.id, "folder": i["folder"], "track": i["track"],
                             "snippets": _snippets(data_root, r.id, snips)})
        members = sorted(g["members"], key=lambda m: (m["wordless"], -(m["words"] or 0)))
        for m in members:
            m["snippets"] = _snippets(data_root, m["id"], snips)
            m["label"] = labels.get(m["id"])
        singers = [n["singer"] for n in g["nearest"]]
        cards.append({
            "key": g["key"], "artist": names.get(g["artist"]) or g["artist"],
            "songs": sorted({m["song"] for m in g["members"]}),
            "members": members, "refs": refs, "cohesion": g["cohesion"],
            "new_base": _base(g["artist"], singers, names).lower(),
            "done": all(m["label"] for m in members),
        })
    if len(snips) != n_snips:
        _save_cache(data_root / SNIPPET_CACHE, snips)
    cards.sort(key=lambda c: (c["artist"].lower(), c["key"]))
    return {"cards": cards, "matched": report["date"],
            "progress": {"groups": len(cards), "done": sum(c["done"] for c in cards),
                         "tracks": sum(len(c["members"]) for c in cards),
                         "answered": sum(1 for c in cards for m in c["members"] if m["label"])}}


def validate_label(label: dict, known_ids: set[str]) -> dict:
    """One posted answer, per track: ``verdict`` singer (an existing singer id), new (a
    name + gender), unclear (left out of training) or effected (pitch-shifted / heavy FX:
    ``processing: heavy``, which recipes leave out by default)."""
    rid = label.get("id")
    if rid not in known_ids:
        raise ValueError(f"unknown record {rid!r}")
    verdict = label.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    singer = label.get("singer")
    if verdict not in UNNAMED and (not isinstance(singer, str) or not singer.strip()
                                   or len(singer) > 80):
        raise ValueError("singer must be a non-empty string of at most 80 characters")
    gender = label.get("gender")
    if gender not in ("F", "M", None):
        raise ValueError(f"gender must be F, M or null, got {gender!r}")
    if verdict == "new" and gender is None:
        raise ValueError("a new singer needs a gender")
    return {"id": rid, "verdict": verdict, "group": label.get("group"),
            "singer": _normalize_singer(singer) if verdict not in UNNAMED else None,
            "gender": gender if verdict not in UNNAMED else None}


def known_ids(data_root: str | Path) -> set[str]:
    report = json.loads((Path(data_root) / MATCH_FILE).read_text(encoding="utf-8"))
    return {m["id"] for g in report["groups"] for m in g["members"]}


def apply(data_root: str | Path, *,
          dry_run: bool = False) -> tuple[Manifest, list[tuple[str, dict]]]:
    """Write the last match's confident assignments, the review answers and the doubles
    that follow them into the manifest. Caller saves.

    Only records without a singer, or whose singer came from this pass before, are
    touched - an identity-review label always wins. Idempotent.
    """
    data_root = Path(data_root)
    report = json.loads((data_root / MATCH_FILE).read_text(encoding="utf-8"))
    labels = label_store(data_root).load()
    manifest = Manifest.for_data_root(data_root)
    singer_gender: dict[str, str] = {}
    for rec in manifest.records:
        if rec.meta.corpus == CORPUS and rec.meta.singer and rec.meta.gender:
            singer_gender.setdefault(rec.meta.singer, rec.meta.gender)

    decided: dict[str, dict] = {}
    for a in report["auto"]:
        decided[a["id"]] = {"singer": a["singer"], "gender": a["gender"], "excluded": None,
                            "source": "auto", "score": a["score"]}
    for rid, lab in labels.items():
        if lab["verdict"] == "unclear":
            decided[rid] = {"singer": None, "gender": None, "excluded": UNCLEAR, "source": "review"}
        elif lab["verdict"] == "effected":
            decided[rid] = {"singer": None, "gender": None, "excluded": None,
                            "processing": "heavy", "source": "review"}
        else:
            gender = lab["gender"] if lab["verdict"] == "new" else (
                singer_gender.get(lab["singer"]) or lab["gender"])
            decided[rid] = {"singer": lab["singer"], "gender": gender, "excluded": None,
                            "source": "review"}
    for rid, parent in report["follows"].items():
        if parent in decided:
            decided[rid] = {**decided[parent], "source": "double", "of": parent}
        else:  # the parent was labelled by identity review
            p = manifest.get(parent)
            if p is not None and p.meta.singer:
                decided[rid] = {"singer": p.meta.singer, "gender": p.meta.gender,
                                "excluded": None, "source": "double", "of": parent}

    changed: list[tuple[str, dict]] = []
    for rid, d in sorted(decided.items()):
        rec = manifest.get(rid)
        if rec is None:
            continue
        sdir = song_dir(data_root, rid)
        ours = "singer_match" in _analysis(sdir)
        if rec.meta.singer and not ours:
            continue  # a human identity label
        if rec.meta.excluded and rec.meta.excluded != UNCLEAR:
            continue  # excluded for another reason - leave it alone
        new = {"singer": d["singer"], "gender": d["gender"], "excluded": d["excluded"],
               "processing": d.get("processing") or _import_processing(rec)}
        diff = {k: (getattr(rec.meta, k), v) for k, v in new.items() if getattr(rec.meta, k) != v}
        if not diff:
            continue
        changed.append((rid, diff))
        if dry_run:
            continue
        for k, v in new.items():
            setattr(rec.meta, k, v)
        update_analysis(sdir, "singer_match", {**d, "matched": report["date"]})
    return manifest, changed
