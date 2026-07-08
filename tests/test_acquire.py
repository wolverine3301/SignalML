"""S1 acquire contract tests — offline, downloader injected (no yt-dlp, no network)."""

from __future__ import annotations

from pathlib import Path

from signalml.manifest import Manifest
from signalml.stages.acquire import acquire, read_url_list


def _fake_downloader_factory(make_wav, *, same_payload: bool = False):
    """Returns a downloader writing a distinct (or identical) sine per URL."""
    calls: list[str] = []

    def _download(url: str, out_dir: Path) -> tuple[Path, dict]:
        calls.append(url)
        vid = url.rsplit("=", 1)[-1]
        hz = 220.0 if same_payload else 220.0 + 55.0 * len(calls)
        path = make_wav(out_dir / f"{vid}.wav", seconds=0.5, hz=hz)
        return path, {"id": vid, "title": f"Title {vid}", "duration": 0.5}

    _download.calls = calls
    return _download


URLS = ["https://youtube.example/watch?v=aaa", "https://youtube.example/watch?v=bbb"]


def test_acquire_two_urls(tmp_path, make_wav):
    downloader = _fake_downloader_factory(make_wav)
    summary = acquire(URLS, tmp_path, downloader=downloader, language="en")

    assert len(summary.added) == 2 and not summary.failed
    assert (tmp_path / "raw" / "aaa.wav").exists()
    assert (tmp_path / "raw" / "bbb.wav").exists()

    manifest = Manifest.for_data_root(tmp_path)
    assert len(manifest) == 2
    rec = manifest.by_url(URLS[0])
    assert rec.source.kind == "youtube"
    assert rec.source.retrieved  # ISO date recorded
    assert rec.meta.song == "Title aaa"
    assert rec.meta.language == "en"
    assert rec.file.duration_sec is not None
    assert rec.status.separated is False


def test_reacquire_is_noop(tmp_path, make_wav):
    downloader = _fake_downloader_factory(make_wav)
    acquire(URLS, tmp_path, downloader=downloader)
    summary2 = acquire(URLS, tmp_path, downloader=downloader)

    assert summary2.added == []
    assert set(summary2.skipped_known_url) == set(URLS)
    assert len(downloader.calls) == 2  # second run never re-downloaded
    assert len(Manifest.for_data_root(tmp_path)) == 2


def test_checksum_dedupe_across_urls(tmp_path, make_wav):
    downloader = _fake_downloader_factory(make_wav, same_payload=True)
    summary = acquire(URLS, tmp_path, downloader=downloader)

    assert len(summary.added) == 1
    assert summary.skipped_known_checksum == [URLS[1]]
    assert not (tmp_path / "raw" / "bbb.wav").exists()  # duplicate payload removed
    assert len(Manifest.for_data_root(tmp_path)) == 1


def test_failed_url_does_not_kill_batch(tmp_path, make_wav):
    good = _fake_downloader_factory(make_wav)

    def flaky(url: str, out_dir: Path):
        if "aaa" in url:
            raise RuntimeError("boom")
        return good(url, out_dir)

    summary = acquire(URLS, tmp_path, downloader=flaky)
    assert list(summary.failed) == [URLS[0]]
    assert len(summary.added) == 1


def test_read_url_list(tmp_path):
    f = tmp_path / "urls.txt"
    f.write_text("# comment\nhttps://a\n\n  https://b  \n", encoding="utf-8")
    assert read_url_list(f) == ["https://a", "https://b"]
