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


def test_dry_run_copies_nothing(session_root, tmp_path):
    root = tmp_path / "dr"
    manifest, new, report = import_multitrack(root, root=session_root, batch="b", dry_run=True)
    assert new == [] and len(manifest) == 0 and report["kept_tracks"] == 5
    assert not (root / "songs").exists()
