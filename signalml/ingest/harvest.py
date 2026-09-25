"""Channel harvest — plan what to take from a curated channel, then ingest what was
downloaded by hand.

Two halves around one manual step. ``plan`` lists a channel (metadata only — listing
works where downloading is refused) and keeps solo, unprocessed-sounding singing: no
talking videos, duets, TV/arena sets, remixes, medleys or studio releases; one version
per song; intimate recordings ranked first; a cap per singer so one voice cannot
dominate the timbre space. ``ingest`` takes whatever audio lands in an inbox folder,
matches it to a plan by video id (or title), and writes the corpus layout the rest of
the pipeline already reads: ``RAW/<batch>/<singer>/<title>-<id>/<title>-<id>.wav`` plus
a ``META.txt`` carrying singer, source URL and a licence note (-> license_note).

The downloading itself stays manual: YouTube refuses automated downloads from the
work PC, and working around that is not something this code does.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import Manifest, scan_directory

PLAN_SUFFIX = ".plan.json"
INBOX_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mp4",
              ".mkv", ".mov"}

MIN_SEC, MAX_SEC = 90, 420  # shorter = clips/shorts, longer = concerts and compilations

_TALK = re.compile(
    r"track by track|episode|trailer|interview|vlog|behind the scenes|documentary|"
    r"q ?& ?a|\breacts?\b|unboxing|announce|teaser|explains|story behind|commentary|"
    r"podcast|#shorts|tour diary|making (of|')|rehearsal|livestream|live stream|\bchat\b|"
    r"tiktok|\btag\b|day in the life|character breakdown|evolution of|ask:reply", re.I)
_DUET_ANY = re.compile(r"\b(feat\.?|ft\.?|featuring|duet)\b", re.I)
_DUET_HEAD = re.compile(r"\b(with|and|x|vs\.?)\b|&|,|\+", re.I)
_TV = re.compile(
    r"kimmel|today show|good morning|saturday night|\bsnl\b|ellen|the view|seth meyers|"
    r"fallon|colbert|corden|\bvmas?\b|\bmmvas?\b|grammy|awards|rockin|\bo2\b|austin city|"
    r"on tour|tour video|festival|iheart|bestival|arena|stadium", re.I)
_REMIX = re.compile(r"remix|instrumental|karaoke|sped up|slowed|nightcore|\b8d\b", re.I)
_MEDLEY = re.compile(r"medley|mashup|compilation|unreleased covers|full album|playlist", re.I)
_PRODUCED = re.compile(r"\(audio\)|official audio|official video|official music video|"
                       r"lyric|visuali[sz]er|music video|studio video", re.I)
_NON_LATIN = re.compile(r"[Ͱ-ϿЀ-ӿ֐-ۿऀ-෿"
                        r"　-鿿가-힯]")
# rank order: nothing to separate, then intimate sessions, then live-in-a-room, then covers
_TIERS = (
    re.compile(r"a ?cappella|acapella|vocals? only", re.I),
    re.compile(r"acoustic|stripped|piano|unplugged|session", re.I),
    re.compile(r"off the floor|warm up|living room|at home|bedroom|one take", re.I),
    re.compile(r"vevo|\blift\b|dscvr|sofar|live performance|\blive\b", re.I),
    re.compile(r"\bcover\b", re.I),
)
_YT_ID = re.compile(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")


@dataclass
class Entry:
    id: str
    title: str
    duration: int  # seconds

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


@dataclass
class Plan:
    singer: str
    channel: str
    picked: list[Entry]
    skipped: dict[str, str] = field(default_factory=dict)  # id -> reason
    gender: str = "F"
    genre: str = "acoustic"
    license: str | None = None
    created: str = ""


def _strip_singer(title: str, singer: str) -> str:
    out = title
    for name in {singer, singer.split()[0]} if len(singer.split()[0]) > 3 else {singer}:
        out = re.sub(re.escape(name), " ", out, flags=re.I)
    return out


def song_key(title: str, singer: str) -> str:
    """The song a title is a version *of* — the unit the one-version-per-song cap counts.
    'Artist - Harbor (Porch Version)' and 'Harbor (Live Acoustic)' are both 'harbor'."""
    t = _strip_singer(title, singer)
    t = re.sub(r"\(.*?\)|\[.*?\]|\|.*$|\".*?\"", " ", t)
    t = t.split(" - ")[-1] if " - " in t else t
    t = t.split(":")[-1]
    t = re.sub(r"(?i)\b(official|live|performance|acoustic|cover|version|by|audio|video)\b",
               " ", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())


def skip_reason(e: Entry, singer: str, *, have_ids: set[str],
                prefer: list[str] = ()) -> str | None:
    """Why an upload is not solo, natural-sounding singing — or None to keep it.
    ``prefer`` patterns (a channel's own name for its acoustic series) count
    as a keep marker, like "acoustic" does."""
    t = e.title
    rest = _strip_singer(t, singer)
    head = rest.split(" - ")[0] if " - " in rest else ""
    name = re.escape(singer)
    if e.id in have_ids:
        return "already in corpus"
    if _NON_LATIN.search(t):
        return "non-Latin title"
    if _TALK.search(t):
        return "talking / not a song"
    if not (MIN_SEC <= e.duration <= MAX_SEC):
        return "too short or too long"
    if (_DUET_ANY.search(t) or _DUET_HEAD.search(head)
            or re.search(rf"{name}\s+(and|&)\s+\w|\w\s+(and|&)\s+{name}", t, re.I)):
        return "duet / feature"
    if _MEDLEY.search(t):
        return "medley / compilation"
    if _REMIX.search(t):
        return "remix / instrumental"
    if _TV.search(t):
        return "TV / arena performance"
    if any(p.search(t) for p in _TIERS) or any(re.search(p, t, re.I) for p in prefer):
        return None
    if _PRODUCED.search(t):
        return "studio release"
    return "unclear (probably a music video)"


def _tier(title: str, prefer: list[str]) -> int:
    if any(re.search(p, title, re.I) for p in prefer):
        return -1
    return next((i for i, p in enumerate(_TIERS) if p.search(title)), len(_TIERS))


def plan_channel(entries: list[Entry], singer: str, *, channel: str = "",
                 have_ids: set[str] = frozenset(), cap: int = 25, versions: int = 1,
                 prefer: list[str] = (), license: str | None = None,
                 gender: str = "F") -> Plan:
    """Filter + rank a channel listing into at most ``cap`` songs (``versions`` per song)."""
    singer = singer.strip().lower()
    skipped: dict[str, str] = {}
    keep: list[Entry] = []
    for e in entries:
        reason = skip_reason(e, singer, have_ids=set(have_ids), prefer=list(prefer))
        if reason:
            skipped[e.id] = reason
        else:
            keep.append(e)
    keep.sort(key=lambda e: _tier(e.title, list(prefer)))  # stable: channel order within a tier
    per_song: Counter[str] = Counter()
    picked: list[Entry] = []
    for e in keep:
        k = song_key(e.title, singer) or e.id
        if len(picked) >= cap:
            skipped[e.id] = f"over the cap of {cap}"
        elif per_song[k] >= versions:
            skipped[e.id] = "another version of the same song"
        else:
            per_song[k] += 1
            picked.append(e)
    return Plan(singer=singer, channel=channel, picked=picked, skipped=skipped,
                gender=gender, license=license,
                created=_dt.date.today().isoformat())


def list_channel(url: str) -> list[Entry]:
    """A channel's uploads, metadata only (no audio is fetched)."""
    import yt_dlp  # lazy: network dep kept out of offline paths

    base = url.rstrip("/")
    if not re.search(r"/(videos|streams)$", base):
        base += "/videos"
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(base, download=False)
    return [Entry(id=e["id"], title=e.get("title") or "", duration=int(e.get("duration") or 0))
            for e in info.get("entries") or [] if e.get("id")]


def corpus_youtube_ids(manifest: Manifest) -> set[str]:
    """Every YouTube id the corpus already holds: in file/folder names and source URLs."""
    ids: set[str] = set()
    for rec in manifest.records:
        for text in (rec.file.path, rec.meta.song or "", rec.source.url or ""):
            m = re.search(r"(?:[-_=/])([A-Za-z0-9_-]{11})(?:\.[A-Za-z0-9]+)?$", text)
            if m:
                ids.add(m.group(1))
    return ids


def save_plan(plan: Plan, out_dir: Path) -> tuple[Path, Path]:
    """Write ``<date>_<singer>.plan.json`` (what ingest reads) and a ``.md`` checklist."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{plan.created}_{plan.singer.replace(' ', '_')}"
    jpath, mpath = out_dir / f"{stem}{PLAN_SUFFIX}", out_dir / f"{stem}.md"
    data = {k: v for k, v in plan.__dict__.items() if k not in ("picked", "skipped")}
    data["picked"] = [e.__dict__ for e in plan.picked]
    data["skipped"] = plan.skipped
    jpath.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    mins = sum(e.duration for e in plan.picked) / 60
    lines = [f"# Harvest plan: {plan.singer} ({plan.created})", "",
             f"Channel: {plan.channel}", "",
             f"**{len(plan.picked)} songs, {mins:.0f} min.** Download each into your inbox "
             "folder (any format, any name - the id or title is enough to match), then run "
             "`signalml harvest inbox`.", ""]
    lines += [f"- [ ] [{e.title}]({e.url}) `{e.duration // 60}:{e.duration % 60:02d}`"
              for e in plan.picked]
    reasons = Counter(plan.skipped.values())
    lines += ["", "Skipped: " + ", ".join(f"{r} {n}" for r, n in reasons.most_common())]
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jpath, mpath


def load_plans(paths: list[Path]) -> list[Plan]:
    plans = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        d["picked"] = [Entry(**e) for e in d["picked"]]
        plans.append(Plan(**d))
    return plans


# ---------------------------------------------------------------- ingest

Converter = Callable[[Path, Path], None]


def ffmpeg_to_wav(src: Path, dst: Path) -> None:
    """Decode anything ffmpeg reads to PCM WAV, keeping its rate and channels — the
    clean stage owns resampling, so ingest does not decide the profile."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                    "-vn", "-acodec", "pcm_s16le", str(dst)], check=True)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def safe_name(title: str) -> str:
    """A folder/file name Windows accepts, still readable."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .")
    return name[:120] or "untitled"


@dataclass
class IngestSummary:
    placed: list[tuple[str, Path]] = field(default_factory=list)  # (id, wav)
    unmatched: list[Path] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    missing: dict[str, list[Entry]] = field(default_factory=dict)  # singer -> entries
    new_records: list[str] = field(default_factory=list)


def _match(path: Path, by_id: dict[str, tuple[Plan, Entry]]) -> tuple[Plan, Entry] | None:
    for cand in _YT_ID.findall(path.stem):
        if cand in by_id:
            return by_id[cand]
    stem = _norm(path.stem)
    if len(stem) < 6:
        return None
    hits = [v for v in by_id.values() if _norm(v[1].title) and
            (_norm(v[1].title) == stem or _norm(v[1].title) in stem or stem in _norm(v[1].title))]
    return hits[0] if len(hits) == 1 else None  # ambiguous title -> leave it for a human


def ingest_inbox(data_root: Path, inbox: Path, plans: list[Plan], *, batch: str = "harvest",
                 converter: Converter = ffmpeg_to_wav, language: str = "en") -> IngestSummary:
    """Place inbox audio into the corpus layout and add it to the manifest. Idempotent:
    a song whose folder already holds a WAV is not re-converted, and processed inbox files
    move to ``inbox/_ingested`` so what remains in the inbox is what still needs a look."""
    data_root, inbox = Path(data_root), Path(inbox)
    summary = IngestSummary()
    by_id = {e.id: (p, e) for p in plans for e in p.picked}
    done_dir = inbox / "_ingested"
    touched: set[tuple[str, str]] = set()
    for src in sorted(inbox.iterdir()):
        if not (src.is_file() and src.suffix.lower() in INBOX_EXTS):
            continue
        hit = _match(src, by_id)
        if hit is None:
            summary.unmatched.append(src)
            continue
        plan, e = hit
        name = f"{safe_name(e.title)}-{e.id}"
        folder = data_root / "RAW" / batch / plan.singer / name
        wav = folder / f"{name}.wav"
        if wav.exists():
            summary.already_present.append(e.id)
        else:
            folder.mkdir(parents=True, exist_ok=True)
            converter(src, wav)
            meta = [f"SONG:{e.title}", f"SINGER:{plan.singer}", "ARTIST:",
                    f"GENRE:{plan.genre}", "TYPE:", "QUALITY:", f"SOURCE_URL:{e.url}"]
            if plan.license:
                meta.append(f"LICENSE:{plan.license}")
            (folder / "META.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
            summary.placed.append((e.id, wav))
        touched.add((plan.singer, plan.gender))
        done_dir.mkdir(exist_ok=True)
        shutil.move(str(src), str(done_dir / src.name))

    for singer, gender in sorted(touched):
        sub = (Path("RAW") / batch / singer).as_posix()
        manifest, new = scan_directory(data_root, subpath=sub, language=language,
                                       gender=gender)
        manifest.save()
        summary.new_records += [r.id for r in new]

    on_disk = {m.group(1) for m in (re.search(r"-([A-Za-z0-9_-]{11})$", d.name)
                                    for d in (data_root / "RAW" / batch).glob("*/*")
                                    if d.is_dir() and any(d.glob("*.wav"))) if m} \
        if (data_root / "RAW" / batch).exists() else set()
    for p in plans:
        miss = [e for e in p.picked if e.id not in on_disk]
        if miss:
            summary.missing[p.singer] = miss
    return summary
