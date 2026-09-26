"""Fetch a singer's songs into the folder shape ``import-batch`` reads.

``<out>/<singer>/<title>-<id>/`` per URL, holding the downloaded audio, an empty
``lyrics.txt`` to paste into, and ``source.url``. Lyrics are not fetched: they are
typed or pasted by a person (and ``lyrics --check`` then flags where the pasted text
departs from the singing). ``import-batch`` skips a song whose lyrics.txt is still
empty, so a half-filled folder can be imported safely.

Downloads use plain yt-dlp - no cookies, no bot-check workarounds. Where a site
refuses automated downloads, fetch the audio by hand into the same folder.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# downloader(url, folder) -> (audio path, info dict)
Downloader = Callable[[str, Path], tuple[Path, dict]]


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:120] or "untitled"


def ytdlp_info(url: str) -> dict:
    import yt_dlp  # lazy: network dep kept out of offline paths

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as y:
        return y.extract_info(url, download=False)


def ytdlp_downloader(url: str, folder: Path) -> tuple[Path, dict]:
    import yt_dlp

    opts = {"format": "bestaudio/best", "outtmpl": str(folder / "%(title)s-%(id)s.%(ext)s"),
            "windowsfilenames": True, "noplaylist": True, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
        return Path(y.prepare_filename(info)), dict(info)


@dataclass
class FetchSummary:
    fetched: list[Path] = field(default_factory=list)
    already: list[Path] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # url -> error


def fetch(urls: list[str], singer: str, out: str | Path, *,
          info: Callable[[str], dict] = ytdlp_info,
          downloader: Downloader = ytdlp_downloader) -> FetchSummary:
    """One folder per URL under ``out/<singer>/``; idempotent per video id."""
    root = Path(out) / singer.strip()
    summary = FetchSummary()
    for url in [u.strip() for u in urls if u.strip() and not u.strip().startswith("#")]:
        try:
            meta = info(url)
            folder = root / safe_name(f"{meta.get('title') or 'untitled'}-{meta['id']}")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "source.url").write_text(url + "\n", encoding="utf-8")
            lyrics = folder / "lyrics.txt"
            if not lyrics.exists():
                lyrics.write_text("", encoding="utf-8")
            if any(p.suffix.lower() not in (".txt", ".url") for p in folder.iterdir()):
                summary.already.append(folder)
                continue
            downloader(url, folder)
            summary.fetched.append(folder)
        except Exception as exc:  # one bad URL must not stop the list
            summary.failed[url] = str(exc).splitlines()[0][:200]
    return summary
