"""Multitrack adapter contract tests - offline; synthetic sessions, invented names."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from signalml.ingest.multitrack import (
    classify_name,
    import_multitrack,
    name_drop_reason,
    parse_folder_name,
    plan_batch,
)
from signalml.manifest import Manifest
from signalml.stages.common import song_dir

SR = 16000
SEC = 12.0


def _voice(f0: float, seed: int) -> np.ndarray:
    """Harmonic 'singing': 2 s phrases with 1 s of digital silence between them."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SEC * SR)) / SR
    f = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    y = sum(0.3 / k * np.sin(k * phase + rng.uniform(0, 6)) for k in range(1, 6))
    gate = (t % 3.0) < 2.0
    return (y * gate).astype(np.float32)


def _drums(seed: int, period: float = 0.47) -> np.ndarray:
    """Noise hits every `period` s - off the voice's 3 s phrase grid, as real drums are
    never sample-locked to a singer's onsets."""
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.3, int(SEC * SR))
    env = np.exp(-((np.arange(len(y)) % int(period * SR)) / (0.05 * SR)))
    return (y * env).astype(np.float32)


def _bass() -> np.ndarray:
    t = np.arange(int(SEC * SR)) / SR
    return (0.4 * np.sin(2 * np.pi * 55 * t + 0.3 * np.sin(2 * np.pi * 0.5 * t))).astype(np.float32)


def _write(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, SR, subtype="PCM_24")


@pytest.fixture()
def session_root(tmp_path) -> Path:
    root = tmp_path / "batch"
    drums, bass = _drums(1), _bass()
    lead = _voice(220, 2)

    s = root / "SomeBand_FirstSong_Full"
    _write(s / "01_Kick.wav", drums)
    _write(s / "02_Bass.wav", bass)
    _write(s / "03_LeadVox.wav", lead)
    _write(s / "04_LeadVoxDT.wav", _voice(221, 3))  # a second performance
    _write(s / "05_BackingVox1.wav", _voice(330, 4))
    _write(s / "06_VoxSFX.wav", _voice(440, 5))
    _write(s / "08_LeadDouble.wav", _voice(222, 6))
    stereo = np.stack([_voice(300, 7), _voice(301, 8)], axis=1)
    _write(s / "09_BackingVox2.wav", stereo)
    _write(s / "10_BackingVox3.wav", np.zeros(int(SEC * SR), np.float32))  # silent
    hiss = np.random.default_rng(21).normal(0, 3e-4, int(SEC * SR)).astype(np.float32)
    _write(s / "11_BackingVox4.wav", hiss)  # -70 dBFS noise, no singing
    (s / "__MACOSX").mkdir()
    (s / "__MACOSX" / "._03_LeadVox.wav").write_bytes(b"\x00\x05junk")

    m = root / "MR0001_SomeSinger_Full"
    _write(m / "01_Drums.wav", drums)
    a = _voice(180, 9)
    _write(m / "02_VOX LD-MicA.wav", a)
    b = np.roll(a, 3) * 0.7 + np.random.default_rng(10).normal(0, 1e-3, len(a))
    _write(m / "03_VOX LD-MicB.wav", b.astype(np.float32))
    _write(m / "04_LeadVox_Tuned.wav", _voice(250, 11))
    _write(m / "05_LeadVox_Untuned.wav", _voice(251, 12))

    live = root / "LiveBand_Session_Full"
    kick, snare = _drums(14), _drums(15, period=0.43)
    _write(live / "01_Kick.wav", kick)
    _write(live / "02_Snare.wav", snare)
    v = _voice(200, 13)
    gate = (np.abs(v) > 0).astype(np.float32)  # edited: silent pauses, band under the singing
    _write(live / "03_Vocals-Singer.wav", (v + gate * (kick + snare)).astype(np.float32))

    _write(root / "SomeBand_FirstSong" / "01_LeadVox.wav", lead[: SR * 8])  # excerpt
    _write(root / "Guitarist_Instrumental_Full" / "01_GtrMic1.wav", bass)
    return root


def test_folder_names():
    assert parse_folder_name("SomeBand_FirstSong_Full") == ("SomeBand", "FirstSong")
    assert parse_folder_name("MR0001_SomeSinger_Full") == ("SomeSinger", None)
    assert parse_folder_name("SomeBand_FirstSong") == ("SomeBand", "FirstSong")
    assert parse_folder_name("SomeBand_Other_Stems") == ("SomeBand", "Other")


@pytest.mark.parametrize("name,expected", [
    ("LeadVox", ("vocal", "lead")),
    ("LeadVox2DT3", ("vocal", "lead_double")),
    ("BackingVoxFemale1DT1", ("vocal", "backing_double")),
    ("VOX LD-U48-1081", ("vocal", "lead")),
    ("Vox07", ("vocal", "vocal")),
    ("ElecGtr1DT", ("instrument", None)),
    ("PercChoirMains", ("instrument", None)),
    ("Vocoder1", ("instrument", None)),
    ("Kick", ("instrument", None)),
    ("LeadDouble", ("unclassified", None)),
    ("BGV 1 251_01", ("vocal", "backing")),
    ("LVDT1", ("vocal", "lead_double")),
    ("LV", ("vocal", "lead")),
    ("Lvl", ("unclassified", None)),
])
def test_classify_name(name, expected):
    assert classify_name(name) == expected


def test_name_drops():
    assert "effects" in name_drop_reason("VoxSFX", 1, set())
    assert "whisper" in name_drop_reason("WhisperDT", 1, set())
    assert "choir" in name_drop_reason("Choir", 1, set())
    assert "stereo" in name_drop_reason("BackingVox3", 2, set())
    assert "pitch-corrected" in name_drop_reason("LeadVox_Tuned", 1, {"leadvox_untuned"})
    assert name_drop_reason("LeadVox_Tuned", 1, set()) == ""  # no untuned twin: keep it
    assert name_drop_reason("LeadVox_Untuned", 1, {"leadvox_tuned"}) == ""


def test_plan_batch_decisions(session_root):
    plans = {fp.folder: fp for fp in plan_batch(session_root)}
    assert plans["SomeBand_FirstSong"].skip_reason.startswith("excerpt of")
    assert plans["Guitarist_Instrumental_Full"].skip_reason.startswith("no vocal tracks")

    first = {t.track: t for t in plans["SomeBand_FirstSong_Full"].tracks}
    assert "._03_LeadVox" not in first and len(first) == 10  # macOS junk ignored
    assert first["LeadVox"].keep and first["LeadVoxDT"].keep and first["BackingVox1"].keep
    # an overdub shares the session with drums it never heard: no bleed. (Guitar
    # unison vs voice is validated on real recordings - see the module docstring -
    # steady synthetic tones stay phase-coherent where real performances never do.)
    assert first["LeadVox"].stats["instr_phat"] < 0.04
    assert not first["VoxSFX"].keep and not first["BackingVox2"].keep
    assert first["BackingVox3"].reason.startswith("empty")
    assert first["BackingVox4"].reason.startswith("empty")
    assert first["LeadDouble"].kind == "unclassified" and not first["LeadDouble"].keep
    assert first["LeadVox"].stats["f0_median_hz"] == pytest.approx(220, rel=0.05)

    coded = {t.track: t for t in plans["MR0001_SomeSinger_Full"].tracks}
    kept = sorted(t.track for t in coded.values() if t.keep)
    assert kept == ["LeadVox_Untuned", "VOX LD-MicA"]
    assert coded["VOX LD-MicB"].duplicate_of == "VOX LD-MicA"
    assert "pitch-corrected" in coded["LeadVox_Tuned"].reason

    live = plans["LiveBand_Session_Full"].tracks
    vocal = next(t for t in live if t.kind == "vocal")
    assert not vocal.keep and vocal.reason.startswith("live-room bleed")


def test_short_part_in_a_long_silent_track(tmp_path):
    """A backing part that sings a few seconds of a long track is >95% silence; it must
    be judged on its singing, not read as 'no dynamics'."""
    from signalml.ingest.multitrack import plan_folder

    long_sec, voice = 120, _voice(300, 16)  # 12 s: 2 s phrases with 1 s rests

    def part(seconds: int) -> np.ndarray:
        y = np.zeros(long_sec * SR, np.float32)
        y[30 * SR: 30 * SR + seconds * SR] = voice[: seconds * SR]
        return y

    s = tmp_path / "SomeBand_Sparse_Full"
    _write(s / "01_Kick.wav", np.tile(_drums(17), 10))
    _write(s / "02_BackingVox1.wav", part(9))  # three 2 s phrases, ~6 s sung
    _write(s / "03_BackingVox2.wav", part(3))  # one 2 s phrase
    plan = {t.track: t for t in plan_folder(s).tracks}
    assert plan["BackingVox1"].keep, plan["BackingVox1"].reason
    assert plan["BackingVox2"].reason.startswith("empty: 2.")


def test_import_writes_unassigned_studio_records(session_root, tmp_path):
    root = tmp_path / "dr"
    manifest, new, report = import_multitrack(root, root=session_root, batch="batch_x")
    manifest.save()
    assert len(new) == report["kept_tracks"] == 5
    rec = Manifest.for_data_root(root).get(new[0].id)
    assert rec.meta.corpus == "multitrack" and rec.meta.source_quality == "studio"
    assert rec.meta.singer is None and rec.meta.gender is None  # the voice pass decides
    assert rec.status.separated and (song_dir(root, rec.id) / "stems" / "vocals.wav").exists()
    assert (root / rec.file.path).exists() and rec.file.path.startswith("RAW/batch_x/")
    analysis = json.loads((song_dir(root, rec.id) / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["separate"]["multitrack"]["role"] in {"lead", "lead_double", "backing"}
    assert (root / "RAW" / "batch_x" / "import_report.json").exists()

    again, new2, _ = import_multitrack(root, root=session_root, batch="batch_x")
    assert new2 == [] and len(again) == 5  # idempotent by checksum


def test_grown_batch_plans_only_new_folders_and_keeps_docs(session_root, tmp_path):
    root = tmp_path / "dr"
    manifest, first, report = import_multitrack(root, root=session_root, batch="bx")
    manifest.save()
    assert report["planned_now"] == 5

    new = session_root / "NewBand_Ballad_Full"
    _write(new / "01_LeadVox.wav", _voice(260, 30))
    (new / "Lyrics and notes.rtf").write_text("{\\rtf1 la la}", encoding="utf-8")
    _write(session_root / "NewBand_Ballad_Stems" / "01_Vox.wav", _voice(260, 30))  # re-export
    manifest, second, report = import_multitrack(root, root=session_root, batch="bx")
    manifest.save()
    assert report["planned_now"] == 2  # the new song and its stems export, nothing else
    skipped = report["folders_skipped"]["NewBand_Ballad_Stems"]
    assert skipped == "another export of NewBand_Ballad_Full"
    assert len(second) == 1 and len(Manifest.for_data_root(root)) == len(first) + 1
    docs = json.loads((song_dir(root, second[0].id) / "analysis.json")
                      .read_text(encoding="utf-8"))["session_docs"]["files"]
    assert docs == ["RAW/bx/NewBand_Ballad_Full/docs/Lyrics and notes.rtf"]
    assert (root / docs[0]).read_text(encoding="utf-8") == "{\\rtf1 la la}"


def test_dry_run_copies_nothing(session_root, tmp_path):
    root = tmp_path / "dr"
    manifest, new, report = import_multitrack(root, root=session_root, batch="b", dry_run=True)
    assert new == [] and len(manifest) == 0 and report["kept_tracks"] == 5
    assert not (root / "songs").exists()


def test_containers_backing_stems_and_documents(tmp_path, monkeypatch):
    """Genre shelves are containers; a _BackingStems export's guide vocal is named from
    its file; the session's sheets become official lyrics, genre and structure."""
    from signalml.ingest import sheets

    monkeypatch.setattr(sheets, "pdf_text", lambda p: open(p, encoding="utf-8").read())
    batch = tmp_path / "b5"
    song = batch / "pop" / "SomeBand_Lantern_BackingStems"
    _write(song / "Lantern_Stems_01_Drums.wav", _drums(40))
    _write(song / "Lantern_Stems_06_GydVox.wav", _voice(240, 41))
    (song / "SomeBand_Lantern_CompSheet.pdf").write_text(
        "Some Band: Lantern\nVERSE 1\nWe row until the dawn\nCHORUS 1\nOh lantern\n",
        encoding="utf-8")
    (song / "SomeBand_Lantern_CueSheet.pdf").write_text(
        "Some Band: Lantern\n(A minor, 4/4, 100bpm)\nTime Bars Section Notes\n0:05 4 Verse 1a\n",
        encoding="utf-8")
    (song / "meta.txt").write_text("Style: Harbor Pop\n", encoding="utf-8")

    root = tmp_path / "dr"
    manifest, new, report = import_multitrack(root, root=batch, batch="b5")
    manifest.save()
    plan = report["plans"][0]
    assert (plan["folder"], plan["container"], plan["artist"], plan["song"]) == \
        ("SomeBand_Lantern_BackingStems", "pop", "SomeBand", "Lantern")
    assert [t["track"] for t in plan["tracks"] if t["keep"]] == ["GydVox"]
    rec = Manifest.for_data_root(root).get(new[0].id)
    assert rec.meta.genre == "harbor pop" and rec.meta.has_lyrics
    assert rec.meta.lyrics_source is None  # official lyrics take the human-lyrics path
    assert (root / rec.meta.lyrics_path).read_text(encoding="utf-8") == \
        "We row until the dawn\nOh lantern\n"
    structure = json.loads((song_dir(root, rec.id) / "analysis.json")
                           .read_text(encoding="utf-8"))["song_structure"]
    assert (structure["key"], structure["bpm"], structure["sections"]) == \
        ("A minor", [100.0], ["VERSE 1", "CHORUS 1"])
    assert report["songs_with_lyrics_sheet"] == 1 and report["styles"] == ["Harbor Pop"]


def test_same_take_from_an_earlier_batch_is_dropped(tmp_path, monkeypatch):
    """...and the later export's sheets reach the earlier take, which sings them."""
    from signalml.ingest import sheets

    monkeypatch.setattr(sheets, "pdf_text", lambda p: open(p, encoding="utf-8").read())
    lead, other = _voice(230, 50), _voice(260, 51)
    first = tmp_path / "b1"
    _write(first / "SomeBand_Lantern_Full" / "01_LeadVox.wav", lead)
    _write(first / "SomeBand_Lantern_Full" / "02_Kick.wav", _drums(52))
    second = tmp_path / "b2"
    # a later export of the same song by the same band (a featuring credit on the name):
    # its guide vocal is the very same take; a different song's guide vocal is not
    later = second / "SomeBandFeatGuest_Lantern_BackingStems"
    _write(later / "Lantern_Stems_06_GydVox.wav", lead * 0.8)
    (later / "SomeBand_Lantern_CompSheet.pdf").write_text(
        "Some Band: Lantern\nVERSE 1\nWe row until the dawn\n", encoding="utf-8")
    (later / "meta.txt").write_text("Style: Harbor Pop\n", encoding="utf-8")
    _write(second / "SomeBand_Harbor_BackingStems" / "Harbor_Stems_06_GydVox.wav", other)

    root = tmp_path / "dr"
    manifest, first_new, _ = import_multitrack(root, root=first, batch="b1")
    manifest.save()
    manifest, second_new, report = import_multitrack(root, root=second, batch="b2")
    plans = {p["folder"]: p for p in report["plans"]}
    dup = plans["SomeBandFeatGuest_Lantern_BackingStems"]["tracks"][0]
    assert not dup["keep"] and dup["duplicate_of"] == first_new[0].id
    assert plans["SomeBand_Harbor_BackingStems"]["tracks"][0]["keep"]
    assert len(second_new) == 1
    earlier = manifest.get(first_new[0].id)
    assert earlier.meta.has_lyrics and earlier.meta.genre == "harbor pop"
    lyrics = (root / earlier.meta.lyrics_path).read_text(encoding="utf-8")
    assert lyrics == "We row until the dawn\n"
