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
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import scan_directory

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac"}
Converter = Callable[[Path, Path], None]


def ffmpeg_to_wav(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                    "-vn", "-acodec", "pcm_s16le", str(dst)], check=True)


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
            texts = [p for p in song.iterdir() if p.suffix.lower() == ".txt"]
            if len(audio) != 1:
                summary.skipped[str(song)] = f"expected 1 audio file, found {len(audio)}"
                continue
            if len(texts) != 1:
                summary.skipped[str(song)] = f"expected 1 lyrics .txt, found {len(texts)}"
                continue
            name = safe_name(song.name)
            dst = base / singer / name
            wav = dst / f"{name}.wav"
            dst.mkdir(parents=True, exist_ok=True)
            if wav.exists():
                summary.already_present.append(wav)
            else:
                if audio[0].suffix.lower() == ".wav":
                    shutil.copyfile(audio[0], wav)
                else:
                    converter(audio[0], wav)
                summary.placed.append(wav)
            raw = texts[0].read_text(encoding="utf-8", errors="replace")
            (dst / "lyrics.source.txt").write_text(raw, encoding="utf-8")
            (dst / "lyrics.txt").write_text(tidy_lyrics(raw), encoding="utf-8")
            meta = [f"SONG:{song.name}", f"SINGER:{singer}", "ARTIST:",
                    f"GENRE:{genre or ''}", "TYPE:", "QUALITY:"]
            if processing:
                meta.append(f"PROCESSING:{processing}")
            if singer in licenses:
                meta.append(f"LICENSE:{licenses[singer]}")
            (dst / "META.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")
    if base.exists():
        manifest, new = scan_directory(data_root, subpath=f"RAW/{batch}", language=language,
                                       gender=gender, source_quality="separated")
        manifest.save()
        summary.new_records = [r.id for r in new]
    return summary
