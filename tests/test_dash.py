"""Dashboard contract tests — offline: build_status is a pure function of DATA_ROOT;
the server is exercised once on an ephemeral port."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from signalml.dash.server import DashServer, build_status
from signalml.manifest import scan_directory
from signalml.stages.common import song_dir, update_analysis


@pytest.fixture()
def data_root(tmp_path, make_wav):
    root = tmp_path / "dr"
    make_wav(root / "raw" / "song.wav", seconds=1.0)
    (root / "raw" / "song.txt").write_text("shine", encoding="utf-8")
    manifest, _ = scan_directory(root, language="en", gender="F", singer="alice",
                                 source_quality="separated")
    rec = manifest.records[0]
    rec.status.separated = True
    rec.status.cleaned = True
    rec.quality.align_score = 0.42  # below the attention threshold
    manifest.upsert(rec)
    manifest.save()
    update_analysis(song_dir(root, rec.id), "features",
                    {"bpm": 97.5, "key": "G:major", "profile": "dev"})
    update_analysis(song_dir(root, rec.id), "clean", {"profile": "dev"})
    return root


def test_build_status_shape(data_root):
    d = build_status(data_root)
    assert d["totals"] == {"songs": 1, "hours": pytest.approx(0.0, abs=0.01),
                           "singers": 1, "lyrics_pct": 100.0}
    stages = {s["name"]: s["done"] for s in d["stages"]}
    assert stages == {"scanned": 1, "separated": 1, "cleaned": 1,
                      "aligned": 0, "featurized": 0}
    assert d["singers"][0]["name"] == "alice"
    assert d["align_scores"] == [0.42]
    song = d["songs"][0]
    assert song["bpm"] == 97.5 and song["key"] == "G:major"
    assert song["stages"]["cleaned"] is True and song["stages"]["aligned"] is False
    assert d["profiles"] == ["dev"]


def test_build_status_empty_root(tmp_path):
    d = build_status(tmp_path)
    assert d["totals"]["songs"] == 0
    assert d["singers"] == [] and d["songs"] == []


def test_server_routes(data_root):
    server = DashServer(data_root, port=0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        html = urllib.request.urlopen(f"{base}/", timeout=5).read().decode("utf-8")
        assert "MISSION CONTROL" in html
        payload = json.loads(
            urllib.request.urlopen(f"{base}/api/status", timeout=5).read())
        assert payload["totals"]["songs"] == 1
        assert urllib.request.urlopen(f"{base}/", timeout=5).status == 200
        demo = urllib.request.urlopen(f"{base}/netviz", timeout=5).read().decode("utf-8")
        assert "NetViz.render" in demo
        js = urllib.request.urlopen(f"{base}/netviz.js", timeout=5).read().decode("utf-8")
        assert "const NetViz" in js
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/nope", timeout=5)
    finally:
        server.shutdown()
        server.server_close()
