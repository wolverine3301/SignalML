"""S5a lyrics contract tests — offline: the Whisper backend is a fake that reads the
batch file and writes canned transcripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signalml.manifest import Manifest, scan_directory
from signalml.stages.lyrics import LyricsConfig, loop_flag, lyrics_stats, run

CFG = LyricsConfig(command=["whisper", "--jobs", "{jobs}"])
LINE = "hold me close and never let me go tonight"


def _song(root: Path, make_wav, name: str, *, lyrics: str | None = None) -> str:
    hz = 200.0 + 37.0 * sum(map(ord, name))  # distinct audio per song: scan dedupes on sha
    audio = make_wav(root / "raw" / name / f"{name}.wav", hz=hz % 2000 + 100)
    if lyrics is not None:
        (audio.parent / "lyrics.txt").write_text(lyrics, encoding="utf-8")
    manifest, (rec,) = scan_directory(root, language="en", gender="F", singer="alice")
    rec.status.separated = rec.status.cleaned = True
    manifest.upsert(rec)
    manifest.save()
    make_wav(root / "songs" / rec.id / "clean" / "vocals.wav")
    return rec.id


def _fake(segments_by_wav: dict | None = None, calls: list | None = None):
    def runner(argv, cwd, timeout):
        if calls is not None:
            calls.append(argv)
        jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))
        for job in jobs:
            segs = (segments_by_wav or {}).get(Path(job["wav"]).parts[-3],
                                               [{"start": 0.0, "end": 2.0, "text": LINE,
                                                 "compression_ratio": 1.2}])
            payload = {"error": segs} if isinstance(segs, str) else \
                {"model": "large-v3", "segments": segs}
            Path(job["out"]).write_text(json.dumps(payload), encoding="utf-8")
    return runner


def test_writes_machine_lyrics_with_provenance(tmp_path, make_wav):
    root = tmp_path / "dr"
    rid = _song(root, make_wav, "a")
    calls: list = []
    s = run(root, cfg=CFG, runner=_fake(calls=calls))
    assert s.written == [rid] and not s.failed and len(calls) == 1  # one batch
    rec = Manifest.for_data_root(root).get(rid)
    assert rec.meta.has_lyrics and rec.meta.lyrics_source == "asr:large-v3"
    assert rec.meta.lyrics_path == f"songs/{rid}/lyrics/lyrics.txt"
    assert (root / rec.meta.lyrics_path).read_text(encoding="utf-8").strip() == LINE
    analysis = json.loads((root / "songs" / rid / "analysis.json").read_text())
    assert analysis["lyrics"]["n_words"] == 9 and analysis["lyrics"]["flag"] is None


def test_hand_written_lyrics_never_replaced(tmp_path, make_wav):
    root = tmp_path / "dr"
    rid = _song(root, make_wav, "a", lyrics="my own corrected lyrics here")
    s = run(root, cfg=CFG, runner=_fake(), force=True)
    assert s.written == [] and rid in s.skipped
    rec = Manifest.for_data_root(root).get(rid)
    assert rec.meta.lyrics_source is None and rec.meta.lyrics_path.endswith("lyrics.txt")


def test_force_redoes_machine_lyrics_only(tmp_path, make_wav):
    root = tmp_path / "dr"
    rid = _song(root, make_wav, "a")
    run(root, cfg=CFG, runner=_fake())
    assert run(root, cfg=CFG, runner=_fake()).written == []  # idempotent
    assert run(root, cfg=CFG, runner=_fake(), force=True).written == [rid]


def test_too_few_words_and_backend_errors_fail_per_song(tmp_path, make_wav):
    root = tmp_path / "dr"
    quiet = _song(root, make_wav, "quiet")
    broken = _song(root, make_wav, "broken")
    ok = _song(root, make_wav, "ok")
    runner = _fake({quiet: [{"start": 0, "end": 1, "text": "oh"}], broken: "RuntimeError: x"})
    s = run(root, cfg=CFG, runner=runner)
    assert s.written == [ok]
    assert "only 1 words" in s.failed[quiet] and "RuntimeError" in s.failed[broken]
    assert not Manifest.for_data_root(root).get(quiet).meta.has_lyrics


def test_loop_is_written_but_flagged(tmp_path, make_wav):
    root = tmp_path / "dr"
    rid = _song(root, make_wav, "a")
    loop = [{"start": i, "end": i + 1, "text": "oh oh oh baby"} for i in range(6)]
    s = run(root, cfg=CFG, runner=_fake({rid: loop}))
    assert s.written == [rid] and "6x in a row" in s.flagged[rid]


def test_refuses_without_a_command(tmp_path, make_wav):
    root = tmp_path / "dr"
    _song(root, make_wav, "a")
    with pytest.raises(RuntimeError, match="no lyrics command"):
        run(root, cfg=LyricsConfig(), runner=_fake())


def test_stats_and_flag():
    segs = [{"text": "A line!", "compression_ratio": 1.1}, {"text": "a line"},
            {"text": "another"}]
    st = lyrics_stats(segs)
    assert st["max_consecutive_repeat"] == 2 and st["n_words"] == 5
    assert loop_flag(st) is None
