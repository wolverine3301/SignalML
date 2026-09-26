"""lyrics --check contract tests — offline; the Whisper backend is a fake."""

from __future__ import annotations

import json
from pathlib import Path

from signalml.manifest import Manifest, scan_directory
from signalml.stages.lyrics import LyricsConfig
from signalml.stages.lyrics_check import compare, run_check

CFG = LyricsConfig(command=["whisper", "--jobs", "{jobs}"])
TEXT = ("hold the light across the water\nwe walk the harbor road tonight\n"
        "and every lantern knows my name\n")


def _seg(start, words):
    return {"start": start, "end": start + len(words) * 0.5, "text": " ".join(words),
            "words": [{"w": w, "start": start + i * 0.5, "end": start + i * 0.5 + 0.4}
                      for i, w in enumerate(words)]}


LINES = [ln.split() for ln in TEXT.strip().split("\n")]


def test_compare_flags_a_repeated_line():
    segs = [_seg(0.0, LINES[0]), _seg(10.0, LINES[1]), _seg(20.0, LINES[1]),
            _seg(30.0, LINES[2])]
    (f,) = compare(TEXT, segs).flags
    # either copy of a repeated line is "the extra one" - both are right answers
    assert f.kind == "extra" and f.time in (10.0, 20.0) and "harbor road" in f.sung


def test_compare_flags_a_skipped_line():
    r = compare(TEXT, [_seg(0.0, LINES[0]), _seg(10.0, LINES[2])])
    (f,) = r.flags
    assert f.kind == "missing" and "harbor road" in f.text and r.agreement < 0.8


def test_compare_flags_a_changed_line():
    segs = [_seg(0.0, LINES[0]), _seg(10.0, LINES[1]),
            _seg(20.0, "but no one ever called me home again".split())]
    (f,) = compare(TEXT, segs).flags
    assert f.kind == "differs" and f.time >= 20.0 and "lantern" in f.text


def test_compare_ignores_single_word_noise():
    words = TEXT.split()
    words[3] = "cross"           # one misheard word: Whisper noise, not a departure
    r = compare(TEXT, [_seg(0.0, words)])
    assert r.flags == [] and r.agreement > 0.9


def _song(root: Path, make_wav, name: str, lyrics: str | None, hz: float) -> str:
    audio = make_wav(root / "RAW" / "b1" / name / f"{name}.wav", hz=hz)
    if lyrics is not None:
        (audio.parent / "lyrics.txt").write_text(lyrics, encoding="utf-8")
    manifest, (rec,) = scan_directory(root, subpath="RAW", language="en", gender="F")
    rec.status.separated = rec.status.cleaned = True
    manifest.upsert(rec)
    manifest.save()
    make_wav(root / "songs" / rec.id / "clean" / "vocals.wav", hz=hz)
    return rec.id


def _fake(segments_by_id):
    def runner(argv, cwd, timeout):
        jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text(encoding="utf-8"))
        for job in jobs:
            rid = Path(job["wav"]).parts[-3]
            Path(job["out"]).write_text(json.dumps(
                {"model": "large-v3", "segments": segments_by_id[rid]}), encoding="utf-8")
    return runner


def test_run_check_writes_review_and_never_touches_lyrics(tmp_path, make_wav):
    root = tmp_path / "dr"
    ok = _song(root, make_wav, "clean-song", TEXT, 220)
    bad = _song(root, make_wav, "repeat-song", TEXT, 330)
    _song(root, make_wav, "no-lyrics", None, 440)   # nothing to check against
    repeat = "hold the light across the".split()
    runner = _fake({ok: [_seg(0.0, TEXT.split())],
                    bad: [_seg(0.0, TEXT.split()), _seg(40.0, repeat)]})
    s = run_check(root, cfg=CFG, runner=runner, prefix="RAW/b1")
    assert set(s.checked) == {ok, bad}
    assert s.checked[ok].flags == [] and s.checked[bad].flags
    report = s.report.read_text(encoding="utf-8")
    assert "1 to look at" in report and "0:40 **SUNG, NOT IN TEXT:**" in report
    lyr = root / Manifest.for_data_root(root).get(ok).meta.lyrics_path
    assert lyr.read_text(encoding="utf-8") == TEXT          # untouched
    assert (root / "songs" / bad / "lyrics" / "asr.json").exists()
    # already checked -> skipped unless forced
    assert run_check(root, cfg=CFG, runner=runner).checked == {}
    assert set(run_check(root, cfg=CFG, runner=runner, force=True).checked) == {ok, bad}
