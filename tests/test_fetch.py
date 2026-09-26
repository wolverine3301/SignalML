"""harvest fetch contract tests — offline: metadata and download are fakes."""

from __future__ import annotations

from signalml.ingest.batch import import_batch
from signalml.ingest.fetch import fetch
from signalml.manifest import Manifest

URLS = ["https://video.example/watch?v=AAAAAAAAAAA", "https://video.example/watch?v=BBBBBBBBBBB"]


def _fakes(make_wav, calls):
    def info(url):
        vid = url.rsplit("=", 1)[-1]
        return {"id": vid, "title": f"Harbor {vid[0]} (Acoustic)"}

    def download(url, folder):
        calls.append(url)
        vid = url.rsplit("=", 1)[-1]
        path = make_wav(folder / f"take-{vid}.wav", hz=220 + 110 * len(calls))
        return path, {"id": vid}
    return info, download


def test_fetch_makes_a_folder_per_url_and_is_idempotent(tmp_path, make_wav):
    calls: list[str] = []
    info, dl = _fakes(make_wav, calls)
    out = tmp_path / "batch"
    s = fetch([*URLS, "# a comment", ""], "Nova Reyes", out, info=info, downloader=dl)
    assert len(s.fetched) == 2 and not s.failed and len(calls) == 2
    folder = out / "Nova Reyes" / "Harbor A (Acoustic)-AAAAAAAAAAA"
    assert (folder / "lyrics.txt").read_text(encoding="utf-8") == ""
    assert (folder / "source.url").read_text(encoding="utf-8").strip() == URLS[0]
    again = fetch(URLS, "Nova Reyes", out, info=info, downloader=dl)
    assert again.fetched == [] and len(again.already) == 2 and len(calls) == 2


def test_import_skips_empty_lyrics_and_records_source(tmp_path, make_wav):
    calls: list[str] = []
    info, dl = _fakes(make_wav, calls)
    out, root = tmp_path / "batch", tmp_path / "dr"
    fetch(URLS, "Nova Reyes", out, info=info, downloader=dl)
    filled = out / "Nova Reyes" / "Harbor A (Acoustic)-AAAAAAAAAAA" / "lyrics.txt"
    filled.write_text("hold the light across the water\n", encoding="utf-8")

    s = import_batch(out, root, batch="b1",
                     converter=lambda a, b, e=None: b.write_bytes(a.read_bytes()))
    assert len(s.new_records) == 1
    assert any("empty" in why for why in s.skipped.values())
    (rec,) = Manifest.for_data_root(root).records
    assert rec.source.url == URLS[0] and rec.meta.singer == "nova reyes"
