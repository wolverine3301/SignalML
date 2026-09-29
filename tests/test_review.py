"""Identity review contract tests - offline; synthetic records, invented names."""

from __future__ import annotations

import io
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from signalml.manifest import FileInfo, Manifest, ManifestRecord, MetaInfo
from signalml.review import identity
from signalml.review.server import ReviewServer
from signalml.stages.common import song_dir, update_analysis

SR = 16000


def _voice(f0: float, seconds: float = 20.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = sum(0.2 / k * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 5))
    return (y * ((t % 4.0) < 3.0)).astype(np.float32)


@pytest.fixture()
def data_root(tmp_path) -> Path:
    """Two songs: a solo artist (lead + its double + a backing track, which is not a
    main vocal) and a band with a male and a female lead."""
    root = tmp_path / "dr"
    manifest = Manifest.for_data_root(root)
    tracks = [
        ("SoloArtist_FirstSong_Full", "SoloArtist", "FirstSong", "LeadVox", "lead", 300.0),
        ("SoloArtist_FirstSong_Full", "SoloArtist", "FirstSong", "LeadVoxDT", "lead_double", 302.0),
        ("SoloArtist_FirstSong_Full", "SoloArtist", "FirstSong", "BackingVox1", "backing", 400.0),
        ("SomeBand_Duet_Full", "SomeBand", "Duet", "LeadVox1", "lead", 150.0),
        ("SomeBand_Duet_Full", "SomeBand", "Duet", "LeadVox2", "lead", 320.0),
    ]
    for i, (folder, artist, song, track, role, f0) in enumerate(tracks):
        rid = f"sng_{9000 + i}"
        sdir = song_dir(root, rid)
        (sdir / "stems").mkdir(parents=True)
        sf.write(str(sdir / "stems" / "vocals.wav"), _voice(f0), SR, subtype="PCM_16")
        update_analysis(sdir, "separate", {"multitrack": {
            "batch": "b1", "folder": folder, "artist": artist, "song": song, "track": track,
            "role": role, "sung_sec": 15.0, "f0_median_hz": f0}})
        manifest.add(ManifestRecord(
            id=rid, file=FileInfo(path=f"RAW/b1/{folder}/{track}.wav", sha256=f"{i:064d}"),
            meta=MetaInfo(corpus="multitrack", source_quality="studio")))
    manifest.save()
    return root


def test_helpers():
    assert identity.display_name("AnnaSomebodyAndTheBand") == "Anna Somebody And The Band"
    assert identity.display_name("MR1001") == "MR1001"
    assert (identity.gender_guess(150), identity.gender_guess(230), identity.gender_guess(300)) \
        == ("M", None, "F")
    assert identity.lead_base("LeadVox2DT1") == "LeadVox2"
    assert identity.lead_base("LeadVoxDTAdLib") == "LeadVox"


def test_cards_group_main_vocals_with_guesses(data_root):
    payload = identity.build_cards(data_root)
    cards = {c["folder"]: c for c in payload["cards"]}
    assert payload["progress"] == {"songs": 2, "confirmed": 0, "tracks": 4, "labelled": 0}

    solo = {t["track"]: t for t in cards["SoloArtist_FirstSong_Full"]["tracks"]}
    assert set(solo) == {"LeadVox", "LeadVoxDT"}  # backing is not a main vocal
    assert solo["LeadVoxDT"]["lead_id"] == solo["LeadVox"]["id"]
    assert solo["LeadVox"]["guess_gender"] == "F"
    assert solo["LeadVox"]["guess_singer"] == "solo artist"
    assert len(solo["LeadVox"]["snippets"]) == 3
    assert all(0 < d <= identity.SNIPPET_SEC for _, d in solo["LeadVox"]["snippets"])

    duet = {t["track"]: t for t in cards["SomeBand_Duet_Full"]["tracks"]}
    assert duet["LeadVox1"]["guess_singer"] == "some band-m"
    assert duet["LeadVox2"]["guess_singer"] == "some band-f"
    assert (data_root / identity.SNIPPET_CACHE).exists()


def test_snippet_wav_is_normalised(data_root):
    wav = identity.snippet_wav(data_root, "sng_9000", 1.0, 2.0)
    y, sr = sf.read(io.BytesIO(wav))
    assert sr == SR and len(y) == 2 * SR
    assert np.max(np.abs(y)) == pytest.approx(10 ** (-3 / 20), abs=0.01)
    with pytest.raises(ValueError):
        identity.snippet_wav(data_root, "../etc", 0, 2)
    with pytest.raises(ValueError):
        identity.snippet_wav(data_root, "sng_9000", 0, 60)


def test_validate_label():
    ok = identity.validate_label({"id": "sng_1", "gender": "F", "singer": "  Some  Band-F ",
                                  "flags": ["bad_take"]}, {"sng_1"})
    assert ok["singer"] == "some band-f" and ok["flags"] == ["bad_take"]
    for bad in ({"id": "sng_2"}, {"id": "sng_1", "gender": "X"},
                {"id": "sng_1", "flags": ["delete_everything"]}):
        with pytest.raises(ValueError):
            identity.validate_label(bad, {"sng_1"})


def test_apply_writes_manifest_and_is_idempotent(data_root):
    store = identity.LabelStore(data_root)
    store.append([{"id": "sng_9003", "gender": "M", "singer": "some band-m", "flags": []}])
    store.append([{"id": "sng_9004", "gender": "F", "singer": "some band-f",
                   "flags": ["not_english"]}])
    store.append([{"id": "sng_9000", "gender": "F", "singer": "wrong", "flags": []}])
    store.append([{"id": "sng_9000", "gender": "F", "singer": "solo artist",
                   "flags": ["bad_take"]}])  # the later answer wins

    _, preview = identity.apply_labels(data_root, dry_run=True)
    assert len(preview) == 3
    assert Manifest.for_data_root(data_root).get("sng_9003").meta.gender is None

    manifest, changed = identity.apply_labels(data_root)
    manifest.save()
    m = Manifest.for_data_root(data_root)
    assert (m.get("sng_9003").meta.gender, m.get("sng_9003").meta.singer,
            m.get("sng_9003").meta.language) == ("M", "some band-m", "en")
    assert m.get("sng_9004").meta.language == "und"
    assert (m.get("sng_9000").meta.singer, m.get("sng_9000").meta.excluded) \
        == ("solo artist", "bad_take")
    review = json.loads((song_dir(data_root, "sng_9000") / "analysis.json")
                        .read_text(encoding="utf-8"))["review"]
    assert review["identity"]["flags"] == ["bad_take"]
    assert identity.apply_labels(data_root)[1] == []


def test_server_round_trip(data_root):
    server = ReviewServer(data_root, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
        assert "<title>Voice Review</title>" in page
        payload = json.loads(urllib.request.urlopen(f"{base}/api/identity").read())
        tid = payload["cards"][0]["tracks"][0]["id"]
        audio = urllib.request.urlopen(f"{base}/audio/{tid}?s=0&d=1").read()
        assert audio[:4] == b"RIFF"

        def post(body: dict) -> int:
            req = urllib.request.Request(f"{base}/api/identity/labels",
                                         data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as err:
                return err.code

        assert post({"labels": [{"id": tid, "gender": "F", "singer": "solo artist"}]}) == 200
        assert post({"labels": [{"id": "sng_1", "gender": "F"}]}) == 400  # not in the queue
        after = json.loads(urllib.request.urlopen(f"{base}/api/identity").read())
        assert after["progress"]["labelled"] == 1
    finally:
        server.shutdown()
        server.server_close()
