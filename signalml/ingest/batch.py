"""Import a hand-curated batch: ``<src>/<singer>/<song>/{audio, lyrics .txt}``.

The shape a person assembles by hand - one folder per singer, one per song, the audio
and a lyrics file beside it. The source is copied, never moved or edited: audio is
converted to WAV into ``RAW/<batch>/<singer>/<song>/``, the lyrics are tidied into
``lyrics.txt`` (the original kept as ``lyrics.source.txt``), and a ``META.txt`` records
singer, genre, processing and an optional licence note. The manifest scan then picks
the songs up like any other corpus folder.

Lyrics tidying is conservative: section headers (``[Chorus]``) are not sung and go;
parentheses and ellipses go but their words stay, because backing lines and ad-libs in
parentheses *are* sung. Everything else is left for the aligner to judge.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import scan_directory

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac"}
Converter = Callable[[Path, Path, float | None], None]


def ffmpeg_to_wav(src: Path, dst: Path, end: float | None = None) -> None:
    """Decode to PCM WAV, optionally stopping at ``end`` seconds."""
    cut = ["-t", f"{end:.3f}"] if end is not None else []
    tmp = dst.with_name(dst.stem + ".tmp.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                    *cut, "-vn", "-acodec", "pcm_s16le", str(tmp)], check=True)
    tmp.replace(dst)


_CUT = re.compile(r"^\s*note\s*:\s*cut(?:\s+audio)?\s+after\s+(\d+):([0-5]\d)\s*$",
                  re.I | re.M)


def cut_note(text: str) -> tuple[str, float | None]:
    """``note: cut (audio) after M:SS`` lines -> (text without them, cut point in s).
    The person curating a batch marks where the singing ends and the audio should stop
    (heavy editing, talking after the song); the note is never part of the lyrics."""
    cut = None
    for m in _CUT.finditer(text):
        cut = int(m.group(1)) * 60 + int(m.group(2))
    return _CUT.sub("", text), cut


def tidy_lyrics(text: str) -> str:
    """Drop section headers, parentheses and ellipses; keep every sung word."""
    text = re.sub(r"^\s*\[[^\]]*\]\s*$", "", text, flags=re.M)  # a [Chorus] line
    text = re.sub(r"\[[^\]]*\]", " ", text)                      # inline header
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"\.{2,}|…", " ", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln) + "\n"


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:120] or "untitled"


@dataclass
class BatchSummary:
    placed: list[Path] = field(default_factory=list)
    already_present: list[Path] = field(default_factory=list)
    retrimmed: list[Path] = field(default_factory=list)  # existing songs cut to a new note
    skipped: dict[str, str] = field(default_factory=dict)  # song folder -> reason
    new_records: list[str] = field(default_factory=list)


def import_batch(
    src: str | Path,
    data_root: str | Path,
    *,
    batch: str,
    genre: str | None = None,
    processing: str | None = None,
    licenses: dict[str, str] | None = None,
    language: str = "en",
    gender: str = "F",
    converter: Converter = ffmpeg_to_wav,
) -> BatchSummary:
    """Copy ``src`` into ``RAW/<batch>/`` and add the songs to the manifest.

    ``licenses`` maps a singer (lowercase) to the licence note recorded as
    ``license_note``. Idempotent: a song whose WAV is already in place is not
    re-converted, and the manifest scan skips known checksums."""
    src, data_root = Path(src), Path(data_root)
    licenses = {k.strip().lower(): v for k, v in (licenses or {}).items()}
    summary = BatchSummary()
    base = data_root / "RAW" / batch
    for singer_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        singer = singer_dir.name.strip().lower()
        for song in sorted(p for p in singer_dir.iterdir() if p.is_dir()):
            audio = [p for p in song.iterdir() if p.suffix.lower() in AUDIO_EXTS]
            texts = [p for p in song.iterdir() if p.suffix.lower() == ".txt"
                     and p.name != "lyrics.source.txt"]
            if len(audio) != 1:
                summary.skipped[str(song)] = f"expected 1 audio file, found {len(audio)}"
                continue
            if len(texts) != 1:
                summary.skipped[str(song)] = f"expected 1 lyrics .txt, found {len(texts)}"
                continue
            raw = texts[0].read_text(encoding="utf-8", errors="replace")
            lyrics_text, cut = cut_note(raw)
            if not tidy_lyrics(lyrics_text).strip():
                summary.skipped[str(song)] = "lyrics .txt is empty - paste the lyrics first"
                continue
            name = safe_name(song.name)
            dst = base / singer / name
            wav = dst / f"{name}.wav"
            dst.mkdir(parents=True, exist_ok=True)
            if wav.exists() and not (cut is not None and _duration(wav) > cut + 0.5):
                summary.already_present.append(wav)
            else:
                if wav.exists():
                    summary.retrimmed.append(wav)  # a cut note arrived after import
                else:
                    summary.placed.append(wav)
                converter(audio[0], wav, cut)
            (dst / "lyrics.source.txt").write_text(raw, encoding="utf-8")
            (dst / "lyrics.txt").write_text(tidy_lyrics(lyrics_text), encoding="utf-8")
            meta = [f"SONG:{song.name}", f"SINGER:{singer}", "ARTIST:",
                    f"GENRE:{genre or ''}", "TYPE:", "QUALITY:"]
            url_file = song / "source.url"
            if url_file.exists():
                meta.append(f"SOURCE_URL:{url_file.read_text(encoding='utf-8').strip()}")
            if processing:
                meta.append(f"PROCESSING:{processing}")
            if singer in licenses:
                meta.append(f"LICENSE:{licenses[singer]}")
            (dst / "META.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    if summary.retrimmed:
        _refresh_records(data_root, summary.retrimmed)
    if base.exists():
        manifest, new = scan_directory(data_root, subpath=f"RAW/{batch}", language=language,
                                       gender=gender, source_quality="separated")
        manifest.save()
        summary.new_records = [r.id for r in new]
    return summary


def _duration(wav: Path) -> float:
    import soundfile as sf

    return sf.info(str(wav)).duration


def _refresh_records(data_root: Path, wavs: list[Path]) -> None:
    """A re-trimmed file keeps its record: new checksum and duration, and the stage
    flags reset so separate/clean/align redo it (their outputs were for the old audio)."""
    from ..manifest import Manifest, probe_audio, sha256_file

    manifest = Manifest.for_data_root(data_root)
    by_path = {r.file.path: r for r in manifest.records}
    for wav in wavs:
        rec = by_path.get(wav.relative_to(data_root).as_posix())
        if rec is None:
            continue
        rec.file.sha256 = sha256_file(wav)
        rec.file.duration_sec, rec.file.sample_rate, rec.file.channels = probe_audio(wav)
        for flag in ("separated", "cleaned", "aligned", "featurized", "transcribed"):
            setattr(rec.status, flag, False)
        rec.quality.align_score = None
        manifest.upsert(rec)
    manifest.save()
