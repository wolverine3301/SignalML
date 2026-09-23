"""S5b contract tests — offline, CPU, no transcriber installed.

The backend is external by design (D1 is still open), so every test here injects one.
What is under test is our half: manifest-driven selection, the notes.json contract,
the coverage metric that makes a bake-off possible without ground truth, and reading
back the two output shapes openvpi's tools emit.
"""

from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

from signalml.manifest import Manifest, scan_directory
from signalml.stages.common import song_dir
from signalml.stages.transcribe import (
    CommandTranscriber,
    Note,
    TranscribeConfig,
    coverage,
    read_notes,
    run,
)

PHONES = [
    {"ph": "ʃ", "start": 0.5, "end": 0.8, "word": "shine"},
    {"ph": "aj", "start": 0.8, "end": 1.2, "word": "shine"},
    {"ph": "SP", "start": 1.2, "end": 2.0, "word": ""},
    {"ph": "n", "start": 2.0, "end": 2.4, "word": "night"},
]


class FakeBackend:
    """Stands in for SOME/GAME: returns fixed notes, records what it was asked."""

    name = "fake/v1"

    def __init__(self, notes=None):
        self.notes = notes if notes is not None else [
            Note(start=0.5, end=1.2, midi=67.3),
            Note(start=2.0, end=2.4, midi=69.0, confidence=0.8),
        ]
        self.calls = []

    def transcribe(self, wav, work_dir):
        self.calls.append(wav)
        return list(self.notes)


def _aligned_song(root, make_wav, *, singer="alice", aligned=True):
    # distinct audio per singer: scan_directory dedupes on sha256, so two identical
    # tones would land as a single record
    hz = 200.0 + sum(ord(c) for c in singer) % 300
    make_wav(root / "raw" / f"{singer}.wav", seconds=3.0, hz=hz)
    manifest, new = scan_directory(root, language="en", gender="F", singer=singer)
    rec = new[0]
    rec.status.separated = rec.status.cleaned = True
    rec.status.aligned = aligned
    manifest.upsert(rec)
    manifest.save()
    sdir = song_dir(root, rec.id)
    (sdir / "clean").mkdir(parents=True, exist_ok=True)
    make_wav(sdir / "clean" / "vocals.wav", seconds=3.0)
    if aligned:
        (sdir / "align").mkdir(parents=True, exist_ok=True)
        (sdir / "align" / "phones.json").write_text(
            json.dumps({"phone_set": "mfa_ipa/en_v1", "phones": PHONES}),
            encoding="utf-8")
    return rec.id


class TestCoverage:
    """The bake-off metric: how much of what was *sung* got a note at all."""

    def test_full_coverage(self):
        notes = [Note(start=0.0, end=3.0, midi=60.0)]
        assert coverage(notes, PHONES) == 1.0

    def test_silence_is_not_counted_against_a_transcriber(self):
        # SP spans 1.2-2.0; sung time is 0.7 + 0.4 = 1.1s
        notes = [Note(start=0.5, end=1.2, midi=60.0), Note(start=2.0, end=2.4, midi=62.0)]
        assert coverage(notes, PHONES) == 1.0

    def test_half_the_melody_missing_shows_up(self):
        notes = [Note(start=0.5, end=1.2, midi=60.0)]
        assert coverage(notes, PHONES) == pytest.approx(0.7 / 1.1, abs=1e-3)

    def test_no_sung_phones_is_none_not_zero(self):
        assert coverage([], [{"ph": "SP", "start": 0.0, "end": 1.0}]) is None


class TestReadNotes:
    def test_reads_midi(self, tmp_path):
        midi = mido.MidiFile()
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.Message("note_on", note=67, velocity=80, time=480))
        track.append(mido.Message("note_off", note=67, velocity=0, time=480))
        path = tmp_path / "notes.mid"
        midi.save(str(path))
        notes = read_notes(path)
        assert len(notes) == 1 and notes[0].midi == 67.0
        assert notes[0].duration > 0

    def test_reads_csv_with_onset_offset_pitch(self, tmp_path):
        path = tmp_path / "notes.csv"
        path.write_text("onset,offset,pitch,confidence\n0.5,1.2,67.35,0.9\n",
                        encoding="utf-8")
        notes = read_notes(path)
        assert notes[0].midi == pytest.approx(67.35)
        assert notes[0].confidence == 0.9

    def test_unknown_columns_say_what_was_expected(self, tmp_path):
        path = tmp_path / "notes.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="onset/offset/pitch"):
            read_notes(path)


class TestCommandBackend:
    def test_refuses_to_guess_a_command(self, tmp_path):
        backend = CommandTranscriber(TranscribeConfig())
        with pytest.raises(RuntimeError, match="configs/transcribe.yaml"):
            backend.transcribe(tmp_path / "x.wav", tmp_path)

    def test_substitutes_wav_and_out_and_reads_the_result(self, tmp_path):
        calls = []

        class Done:
            returncode = 0
            stdout = stderr = ""

        def fake_runner(argv, **kwargs):
            calls.append(argv)
            Path(argv[-1]).write_text("onset,offset,pitch\n0.1,0.9,60.0\n",
                                      encoding="utf-8")
            return Done()

        cfg = TranscribeConfig(command=["sometool", "--wav", "{wav}", "--out", "{out}"],
                               output_name="notes.csv")
        backend = CommandTranscriber(cfg, runner=fake_runner)
        notes = backend.transcribe(tmp_path / "vocals.wav", tmp_path)
        assert notes[0].midi == 60.0
        assert calls[0][2].endswith("vocals.wav") and calls[0][4].endswith("notes.csv")

    def test_backend_failure_carries_the_tool_output(self, tmp_path):
        class Failed:
            returncode = 1
            stdout = ""
            stderr = "CUDA out of memory"

        cfg = TranscribeConfig(command=["sometool", "{wav}", "{out}"])
        backend = CommandTranscriber(cfg, runner=lambda *a, **k: Failed())
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            backend.transcribe(tmp_path / "x.wav", tmp_path)


class TestRun:
    def test_end_to_end_writes_notes_and_status(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        sid = _aligned_song(root, make_wav)
        backend = FakeBackend()
        summary = run(root, cfg=TranscribeConfig(), backend=backend)

        assert summary.transcribed == [sid] and summary.notes_written == 2
        payload = json.loads(
            (song_dir(root, sid) / "transcribe" / "notes.json").read_text(encoding="utf-8"))
        assert payload["transcriber"] == "fake/v1"
        assert payload["notes"][0]["midi"] == 67.3  # float pitch preserved
        assert payload["note_coverage"] == 1.0

        analysis = json.loads(
            (song_dir(root, sid) / "analysis.json").read_text(encoding="utf-8"))
        assert analysis["transcribe"]["n_notes"] == 2
        assert Manifest.for_data_root(root).records[0].status.transcribed is True

    def test_unaligned_songs_are_skipped(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        sid = _aligned_song(root, make_wav, aligned=False)
        summary = run(root, cfg=TranscribeConfig(), backend=FakeBackend())
        assert summary.transcribed == [] and sid in summary.skipped

    def test_idempotent_until_forced(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        _aligned_song(root, make_wav)
        backend = FakeBackend()
        run(root, cfg=TranscribeConfig(), backend=backend)
        again = run(root, cfg=TranscribeConfig(), backend=backend)
        assert again.transcribed == [] and len(backend.calls) == 1
        forced = run(root, cfg=TranscribeConfig(), backend=backend, force=True)
        assert len(forced.transcribed) == 1 and len(backend.calls) == 2

    def test_slivers_below_min_note_sec_are_dropped(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        _aligned_song(root, make_wav)
        backend = FakeBackend(notes=[Note(start=0.5, end=0.51, midi=67.0),
                                     Note(start=1.0, end=1.5, midi=69.0)])
        summary = run(root, cfg=TranscribeConfig(min_note_sec=0.05), backend=backend)
        assert summary.notes_written == 1

    def test_a_failing_song_does_not_kill_the_batch(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        good = _aligned_song(root, make_wav, singer="alice")
        bad = _aligned_song(root, make_wav, singer="bob")
        (song_dir(root, bad) / "clean" / "vocals.wav").unlink()
        summary = run(root, cfg=TranscribeConfig(), backend=FakeBackend())
        assert summary.transcribed == [good]
        assert "no cleaned stem" in summary.failed[bad]
