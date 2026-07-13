from __future__ import annotations

import pytest

from signalml.manifest import (
    FileInfo,
    Manifest,
    ManifestRecord,
    resolve_data_root,
    scan_directory,
    sha256_file,
)


def _record(rid: str, sha: str = "abc") -> ManifestRecord:
    return ManifestRecord(id=rid, file=FileInfo(path=f"raw/{rid}.wav", sha256=sha))


def test_next_id_sequence(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    assert m.next_id() == "sng_0001"
    m.add(_record("sng_0001"))
    m.add(_record("sng_0007"))
    assert m.next_id() == "sng_0008"


def test_add_duplicate_id_raises(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    m.add(_record("sng_0001"))
    with pytest.raises(ValueError):
        m.add(_record("sng_0001"))
    m.upsert(_record("sng_0001", sha="changed"))  # upsert allowed
    assert m.get("sng_0001").file.sha256 == "changed"


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    m = Manifest(path)
    rec = _record("sng_0001")
    rec.meta.language = "gd"
    rec.status.separated = True
    m.add(rec)
    m.save()

    reloaded = Manifest(path)
    assert len(reloaded) == 1
    got = reloaded.get("sng_0001")
    assert got.meta.language == "gd"
    assert got.status.separated is True
    assert not path.with_suffix(".jsonl.tmp").exists()  # atomic temp cleaned up


def test_select_by_status(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    a, b = _record("sng_0001", "s1"), _record("sng_0002", "s2")
    a.status.separated = True
    m.add(a)
    m.add(b)
    assert [r.id for r in m.select(separated=True)] == ["sng_0001"]
    assert [r.id for r in m.select(separated=False, aligned=False)] == ["sng_0002"]
    with pytest.raises(KeyError):
        m.select(bogus=True)


def test_lookup_by_url_and_sha(tmp_path):
    m = Manifest(tmp_path / "manifest.jsonl")
    rec = _record("sng_0001", sha="deadbeef")
    rec.source.url = "https://example.com/v"
    m.add(rec)
    assert m.by_url("https://example.com/v").id == "sng_0001"
    assert m.by_sha256("deadbeef").id == "sng_0001"
    assert m.by_url("https://example.com/other") is None


def test_resolve_data_root_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("SIGNALML_DATA_ROOT", raising=False)
    assert str(resolve_data_root()) == "data"
    monkeypatch.setenv("SIGNALML_DATA_ROOT", str(tmp_path / "env_root"))
    assert resolve_data_root() == tmp_path / "env_root"
    assert resolve_data_root(tmp_path / "arg_root") == tmp_path / "arg_root"


class TestScan:
    def test_scan_creates_records_with_lyrics_detection(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        make_wav(root / "raw" / "songA.wav")
        (root / "raw" / "songA.txt").write_text("la la la", encoding="utf-8")
        make_wav(root / "raw" / "gaelic" / "songB.wav", hz=330)

        manifest, new = scan_directory(root, language="en")
        assert len(new) == 2
        by_song = {r.meta.song: r for r in new}

        a = by_song["songA"]
        assert a.meta.has_lyrics is True
        assert a.meta.lyrics_path == "raw/songA.txt"
        assert a.meta.language == "en"
        assert a.file.duration_sec == pytest.approx(2.0, abs=0.05)
        assert a.file.sha256 == sha256_file(root / "raw" / "songA.wav")

        b = by_song["songB"]
        assert b.meta.has_lyrics is False
        assert b.file.path == "raw/gaelic/songB.wav"  # recursive + posix-relative

    def test_scan_song_folder_convention(self, tmp_path, make_wav):
        """One song per folder with lyrics.txt + META.txt siblings (corpus layout)."""
        root = tmp_path / "dr"
        song_dir = root / "RAW" / "kim petras" / "Broken Glass-IZ0"
        make_wav(song_dir / "Kygo, Kim Petras - Broken Glass (Lyrics)-IZ0.wav")
        (song_dir / "lyrics.txt").write_text("we could fix it", encoding="utf-8")
        (song_dir / "META.txt").write_text(
            "SONG:\nSINGER:kim petras\nARTIST:kygo\nGENRE:edm\nTYPE:\nQUALITY:\n",
            encoding="utf-8")

        _, new = scan_directory(root, subpath="RAW", language="en", gender="F",
                                singer="fallback-tag", source_quality="separated")
        rec = new[0]
        assert rec.meta.has_lyrics is True
        assert rec.meta.lyrics_path.endswith("lyrics.txt")
        assert rec.meta.singer == "kim petras"  # META.txt beats the CLI tag
        assert rec.meta.song.startswith("Kygo")  # empty SONG: falls back to stem
        assert rec.meta.source_quality == "separated"

    def test_commit_survives_concurrent_stage_snapshots(self, tmp_path, make_wav):
        """Two stages holding independent manifest snapshots (parallel align +
        features, the 2026-07-12 race) must not clobber each other's songs."""
        root = tmp_path / "dr"
        make_wav(root / "raw" / "a.wav")
        make_wav(root / "raw" / "b.wav", hz=330)
        manifest, _ = scan_directory(root)
        manifest.save()

        m1 = Manifest.for_data_root(root)  # "features" process
        m2 = Manifest.for_data_root(root)  # "align" process, stale snapshot
        r1 = m1.records[0]
        r1.status.featurized = True
        m1.commit(r1)
        r2 = m2.records[1]  # m2 never saw r1's update
        r2.status.aligned = True
        r2.quality.align_score = 0.9
        m2.commit(r2)

        final = Manifest.for_data_root(root)
        assert final.records[0].status.featurized is True  # not wiped by m2's commit
        assert final.records[1].status.aligned is True
        assert final.records[1].quality.align_score == 0.9

    def test_rescan_is_idempotent(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        make_wav(root / "raw" / "song.wav")
        manifest, new = scan_directory(root)
        assert len(new) == 1
        manifest.save()

        manifest2, new2 = scan_directory(root)
        assert new2 == []
        assert len(manifest2) == 1

    def test_scan_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_directory(tmp_path / "nowhere")
