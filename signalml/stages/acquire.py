"""S1 acquire — download audio via yt-dlp into raw/ + manifest records.

Contract: docs/PIPELINE_AND_CONTRACTS.md §S1. Idempotent two ways: a URL already in
the manifest is skipped before download, and a downloaded file whose checksum is
already recorded is skipped after (the duplicate file is removed). The downloader is
injectable so tests run offline.

Audio is kept in its native container (no ffmpeg post-processing dependency); formats
libsndfile can't probe get their duration from yt-dlp's metadata instead.

Future enhancement (Q14): lyrics acquisition for newly downloaded songs
(fetch / Whisper-assisted draft + human verify).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import (
    FileInfo,
    Manifest,
    ManifestRecord,
    MetaInfo,
    SourceInfo,
    probe_audio,
    sha256_file,
)

# downloader(url, out_dir) -> (downloaded file path, info dict from the backend)
Downloader = Callable[[str, Path], tuple[Path, dict]]


@dataclass
class AcquireSummary:
    added: list[str] = field(default_factory=list)  # manifest ids
    skipped_known_url: list[str] = field(default_factory=list)
    skipped_known_checksum: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # url -> error


def ytdlp_downloader(url: str, out_dir: Path) -> tuple[Path, dict]:
    """Default backend: best audio, native container, no re-encode."""
    import yt_dlp  # lazy: keep import cost & network deps out of offline paths

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    return path, dict(info)


def read_url_list(path: str | Path) -> list[str]:
    """One URL per line; blank lines and '#' comments ignored."""
    urls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def acquire(
    urls: list[str],
    data_root: str | Path,
    *,
    downloader: Downloader = ytdlp_downloader,
    language: str | None = None,
) -> AcquireSummary:
    data_root = Path(data_root)
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.for_data_root(data_root)
    summary = AcquireSummary()

    for url in urls:
        if manifest.by_url(url):
            summary.skipped_known_url.append(url)
            continue

        try:
            path, info = downloader(url, raw_dir)
        except Exception as exc:  # a bad URL must not kill the batch
            summary.failed[url] = str(exc)
            continue

        digest = sha256_file(path)
        existing = manifest.by_sha256(digest)
        if existing:
            summary.skipped_known_checksum.append(url)
            path.unlink(missing_ok=True)  # duplicate payload; keep the first copy only
            continue

        duration, sr, channels = probe_audio(path)
        if duration is None and info.get("duration") is not None:
            duration = float(info["duration"])

        rec = ManifestRecord(
            id=manifest.next_id(),
            source=SourceInfo(
                kind="youtube",
                url=url,
                retrieved=_dt.date.today().isoformat(),
            ),
            file=FileInfo(
                path=path.relative_to(data_root).as_posix(),
                sha256=digest,
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
            ),
            meta=MetaInfo(song=info.get("title") or path.stem, language=language),
        )
        manifest.add(rec)
        summary.added.append(rec.id)

    manifest.save()
    return summary
