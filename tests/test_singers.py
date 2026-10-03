"""Backing-vocal singer match contract tests - offline; synthetic voices, invented names."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from signalml.manifest import FileInfo, Manifest, ManifestRecord, MetaInfo, StatusFlags
from signalml.review import singers
from signalml.review.server import ReviewServer
from signalml.stages.common import song_dir, update_analysis

SR = 16000
DIM = 192
RNG = np.random.default_rng(7)
# three voices that an embedding model keeps apart: the band's named lead, a second
# singer heard only on backing parts, and a third one heard only on oohs
VOICE = {k: v / np.linalg.norm(v) for k, v in zip("mfx", RNG.normal(size=(3, DIM)))}

LEAD_WORDS = "I walk the harbor line tonight\nand every lantern knows my name"
TRACKS = [
    # id, folder, track, role, singer, gender, voice, lyrics
    ("sng_9100", "KestrelLane_Harbor_Full", "LeadVox", "lead", "kestrel lane-m", "M", "m",
     LEAD_WORDS),
    ("sng_9101", "KestrelLane_Harbor_Full", "BackingVox1", "backing", None, None, "m",
     "walk the harbor line tonight\nevery lantern knows my name"),
    ("sng_9102", "KestrelLane_Harbor_Full", "BackingVox2", "backing", None, None, "f",
     "silver birds are calling over water\nfollow them home before the morning"),
    ("sng_9103", "KestrelLane_Harbor_Full", "BackingVox2DT1", "backing_double", None, None, "f",
     "silver birds are calling over water"),
    ("sng_9104", "KestrelLane_Harbor_Full", "BackingVox3", "backing", None, None, "x",
     "ooh ooh ah\nooh yeah ah ah\nla la la"),
    ("sng_9105", "KestrelLane_Nightjar_Full", "LeadVox", "lead", "kestrel lane-m", "M", "m",
     "nightjar singing in the dark field"),
    ("sng_9106", "KestrelLane_Nightjar_Full", "BackingVox1", "backing", None, None, "f",
     "dark field dreaming of a summer we never had"),
    ("sng_9107", "KestrelLane_Nightjar_Full", "BackingVoxShort", "backing", None, None, None, None),
]


def _windows(voice: str, n: int = 6) -> list[list[float]]:
    v = VOICE[voice] + RNG.normal(scale=0.03, size=(n, DIM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).round(5).tolist()


@pytest.fixture()
def data_root(tmp_path) -> Path:
    root = tmp_path / "dr"
    manifest = Manifest.for_data_root(root)
    t = np.arange(10 * SR) / SR
    audio = (0.2 * np.sin(2 * np.pi * 220 * t) * ((t % 3) < 2)).astype(np.float32)
    for i, (rid, folder, track, role, singer, gender, voice, lyrics) in enumerate(TRACKS):
        sdir = song_dir(root, rid)
        for sub in ("stems", "clean"):
            (sdir / sub).mkdir(parents=True)
            sf.write(str(sdir / sub / "vocals.wav"), audio, SR, subtype="PCM_16")
        update_analysis(sdir, "separate", {"multitrack": {
            "batch": "b1", "folder": folder, "artist": "KestrelLane",
            "song": folder.split("_")[1], "track": track, "role": role, "sung_sec": 7.0}})
        update_analysis(sdir, "clean", {"stems": {"vocals": {"silence": {
            "intervals_sec": [[0.0, 2.0], [3.0, 5.0], [5.1, 8.0]]}}}})
        if voice:
            (sdir / singers.EMBED_NAME).parent.mkdir(parents=True, exist_ok=True)
            (sdir / singers.EMBED_NAME).write_text(json.dumps(
                {"model": "fake", "embeddings": _windows(voice)}), encoding="utf-8")
        meta = MetaInfo(corpus="multitrack", source_quality="studio", singer=singer,
                        gender=gender, language="en")
        if lyrics:
            (sdir / "lyrics").mkdir()
            (sdir / "lyrics" / "lyrics.txt").write_text(lyrics + "\n", encoding="utf-8")
            meta.has_lyrics, meta.lyrics_path = True, f"songs/{rid}/lyrics/lyrics.txt"
        manifest.add(ManifestRecord(
            id=rid, file=FileInfo(path=f"RAW/b1/{folder}/{track}.wav", sha256=f"{i:064d}"),
            meta=meta, status=StatusFlags(separated=True, cleaned=True)))
    manifest.save()
    return root


def test_spans_and_words():
    assert singers.sung_spans([[0, 1], [1.2, 2], [3, 4]]) == [[0, 2], [3, 4]]
    assert singers.lexical_words("Ooh, ooh-ah YEAH la la, don't let go") == ["don't", "let", "go"]
    c = singers.content("walk the harbor line\nsilver birds calling home tonight now",
                        {"walk", "the", "harbor", "line"})
    assert c["words"] == 10 and c["new_words"] == 6 and not c["wordless"]
    assert singers.content("ooh ah ooh yeah la la", set())["wordless"]
    assert singers.content(None, set())["wordless"]


def test_match_assigns_groups_and_follows(data_root):
    tracks = {t.id: t for t in singers.load_tracks(data_root)}
    assert tracks["sng_9103"].parent == "sng_9102"
    assert tracks["sng_9102"].content["new_words"] >= 8  # a line the lead never sings

    res = singers.match(list(tracks.values()), singers.SingersConfig())
    assert [(a["id"], a["singer"], a["gender"]) for a in res.auto] == \
        [("sng_9101", "kestrel lane-m", "M")]
    assert res.follows == {"sng_9103": "sng_9102"}
    assert [u["id"] for u in res.unmatched] == ["sng_9107"]
    groups = {tuple(sorted(m["id"] for m in g["members"])): g for g in res.groups}
    # the second singer's parts group across both songs; the ooh-only voice is not queued
    assert set(groups) == {("sng_9102", "sng_9106"), ("sng_9104",)}
    assert groups[("sng_9102", "sng_9106")]["needs_review"]
    assert not groups[("sng_9104",)]["needs_review"]
    assert groups[("sng_9102", "sng_9106")]["nearest"][0]["singer"] == "kestrel lane-m"

    cal = singers.calibrate(list(tracks.values()))
    assert cal["same"]["n"] == 2 and cal["same"]["p50"] > 0.9
    assert cal["different"]["n"] == 0  # one named singer: nothing to tell apart


def test_embed_runs_backend_and_files_results(data_root):
    rid = "sng_9107"
    assert [j["id"] for j in singers.embedding_jobs(data_root)] == [rid]
    assert singers.embedding_jobs(data_root, ids=[rid])[0]["spans"] == [[0.0, 2.0], [3.0, 8.0]]
    seen = {}

    def runner(argv, cwd, timeout):
        jobs = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
        seen["argv"] = argv[:-1]
        for job in jobs:
            Path(job["out"]).write_text(json.dumps(
                {"model": "fake", "windows": [[0, 3]], "embeddings": _windows("f", 3)}),
                encoding="utf-8")

    cfg = singers.SingersConfig(command=["py", "ecapa_embed.py", "--jobs", "{jobs}"])
    written, failed = singers.embed(data_root, cfg=cfg, runner=runner)
    assert written == [rid] and not failed and seen["argv"] == ["py", "ecapa_embed.py", "--jobs"]
    assert (song_dir(data_root, rid) / singers.EMBED_NAME).exists()
    assert singers.embedding_jobs(data_root) == []  # idempotent
    with pytest.raises(RuntimeError, match="no embedding command"):
        singers.embed(data_root, cfg=singers.SingersConfig(), force=True)


def test_apply_auto_answers_and_doubles(data_root):
    singers.run_match(data_root, singers.SingersConfig())
    store = singers.label_store(data_root)
    known = singers.known_ids(data_root)
    store.append([singers.validate_label(
        {"id": rid, "verdict": "new", "singer": "Kestrel Lane-BV-F1", "gender": "F"}, known)
        for rid in ("sng_9102", "sng_9106")])
    store.append([singers.validate_label({"id": "sng_9104", "verdict": "unclear"}, known)])
    # an answer for a track a human already named never overrides that name
    store.append([{"id": "sng_9100", "verdict": "new", "singer": "someone else", "gender": "F"}])

    manifest, changed = singers.apply(data_root)
    manifest.save()
    got = {r.id: (r.meta.singer, r.meta.gender, r.meta.excluded)
           for r in Manifest.for_data_root(data_root).records}
    assert got["sng_9101"] == ("kestrel lane-m", "M", None)  # auto
    assert got["sng_9102"] == got["sng_9106"] == ("kestrel lane-bv-f1", "F", None)
    assert got["sng_9103"] == ("kestrel lane-bv-f1", "F", None)  # the double follows
    assert got["sng_9104"] == (None, None, singers.UNCLEAR)
    assert got["sng_9100"] == ("kestrel lane-m", "M", None)
    assert {rid for rid, _ in changed} == {"sng_9101", "sng_9102", "sng_9103", "sng_9104",
                                           "sng_9106"}
    analysis = json.loads((song_dir(data_root, "sng_9103") / "analysis.json").read_text())
    assert analysis["singer_match"]["source"] == "double"
    assert singers.apply(data_root)[1] == []  # idempotent

    # a corrected answer reaches a record this pass labelled before
    store.append([singers.validate_label(
        {"id": "sng_9102", "verdict": "singer", "singer": "kestrel lane-m"}, known)])
    manifest, changed = singers.apply(data_root)
    assert dict(changed)["sng_9102"]["singer"] == ("kestrel lane-bv-f1", "kestrel lane-m")
    assert manifest.get("sng_9102").meta.gender == "M"  # the named singer's gender
    assert manifest.get("sng_9103").meta.singer == "kestrel lane-m"


def test_validate_label():
    ok = singers.validate_label({"id": "sng_1", "verdict": "new", "singer": " Nova  Reyes ",
                                 "gender": "F", "group": "g/1"}, {"sng_1"})
    assert ok == {"id": "sng_1", "verdict": "new", "group": "g/1", "singer": "nova reyes",
                  "gender": "F"}
    assert singers.validate_label({"id": "sng_1", "verdict": "unclear", "singer": "x"},
                                  {"sng_1"})["singer"] is None
    for bad in ({"id": "sng_2", "verdict": "unclear"}, {"id": "sng_1", "verdict": "maybe"},
                {"id": "sng_1", "verdict": "singer"},
                {"id": "sng_1", "verdict": "new", "singer": "nova"}):  # new needs a gender
        with pytest.raises(ValueError):
            singers.validate_label(bad, {"sng_1"})


def test_server_singers_page(data_root):
    singers.run_match(data_root, singers.SingersConfig())
    server = ReviewServer(data_root, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Backing voices" in urllib.request.urlopen(f"{base}/singers").read()
        payload = json.loads(urllib.request.urlopen(f"{base}/api/singers").read())
        assert payload["progress"] == {"groups": 1, "done": 0, "tracks": 2, "answered": 0}
        card = payload["cards"][0]
        assert {m["id"] for m in card["members"]} == {"sng_9102", "sng_9106"}
        assert card["refs"][0]["singer"] == "kestrel lane-m"
        assert card["refs"][0]["id"] == "sng_9100"  # the lead of a song the group is in
        assert card["new_base"] == "kestrel lane" and card["songs"] == ["Harbor", "Nightjar"]
        assert len(card["members"][0]["snippets"]) >= 1

        body = json.dumps({"labels": [{"id": "sng_9102", "verdict": "singer",
                                       "singer": "kestrel lane-m"}]}).encode()
        req = urllib.request.Request(f"{base}/api/singers/labels", data=body,
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read()) == {"saved": 1}
        bad = urllib.request.Request(f"{base}/api/singers/labels", data=json.dumps(
            {"labels": [{"id": "sng_9100", "verdict": "unclear"}]}).encode())
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400  # not a queued part
    finally:
        server.shutdown()
        server.server_close()
    assert singers.label_store(data_root).load()["sng_9102"]["verdict"] == "singer"


def test_content_wordless_and_new_lyrics():
    vocab = set(singers.lexical_words(LEAD_WORDS))
    # vocal percussion: plenty of "words", hardly any different ones
    assert singers.content("buh buh buh bop bop " * 10, vocab)["wordless"]
    resung = singers.content("I walk the harbor line tonight and every lantern knows", vocab)
    assert not resung["wordless"] and not resung["new_lyrics"]  # the lead's own line
    one_new = singers.content("I walk the harbor line tonight and every lantern glows", vocab)
    assert not one_new["new_lyrics"]  # one misheard word is not a new line
    answer = singers.content("silver birds are calling over the water tonight", vocab)
    assert answer["new_lyrics"] and answer["new_words"] == 6
