"""lyrics --resolve contract tests - offline, synthetic Whisper segments."""

from __future__ import annotations

import json

from signalml.manifest import Manifest, scan_directory
from signalml.stages.align import align
from signalml.stages.common import song_dir, update_analysis
from signalml.stages.lyrics_resolve import load_resolution, plain_lyrics, resolve, run_resolve


def _segs(*lines: str) -> list[dict]:
    """Whisper-shaped segments, one per sung line, words 0.5 s apart."""
    out, t = [], 0.0
    for line in lines:
        words = line.split()
        out.append({"start": t, "end": t + 0.5 * len(words), "text": line,
                    "words": [{"w": w, "start": t + 0.5 * i, "end": t + 0.5 * i + 0.4}
                              for i, w in enumerate(words)]})
        t += 0.5 * len(words) + 1.0
    return out


VERSE = ["hold the light across the water", "we walk the harbor road tonight",
         "and every lantern knows my name"]


def test_sung_echo_is_kept_without_brackets():
    text = "hold the light across the water (across the water)\n" + "\n".join(VERSE[1:])
    res = resolve(text, _segs(VERSE[0] + " across the water", *VERSE[1:]))
    assert res.text.splitlines()[0] == "hold the light across the water across the water"
    assert [g["decision"] for g in res.groups] == ["keep"]
    assert res.unsure_lines == []


def test_unsung_echo_is_dropped_and_the_main_copy_kept():
    # "I know (I know)" sung once: the echo goes, not the main text
    text = "I know (I know)\n" + "\n".join(VERSE)
    res = resolve(text, _segs("I know", *VERSE))
    assert res.text.splitlines()[0] == "I know"
    assert [g["decision"] for g in res.groups] == ["drop"]


def test_echo_replaced_by_other_singing_is_unsure():
    text = VERSE[0] + " (ooh the water)\n" + "\n".join(VERSE[1:])
    res = resolve(text, _segs(VERSE[0] + " yeah baby come on", *VERSE[1:]))
    assert [g["decision"] for g in res.groups] == ["unsure"]
    assert res.unsure_lines == [0]
    assert "ooh the water" in res.text.splitlines()[0]  # kept, just not vouched for


def test_cut_verse_is_dropped():
    extra = ["these words were never sung by the cover", "and neither was this line at all"]
    text = "\n".join([*VERSE, *extra, VERSE[0]])
    res = resolve(text, _segs(*VERSE, VERSE[0]))
    assert res.text.splitlines() == [*VERSE, VERSE[0]]
    assert [ln["status"] for ln in res.lines].count("dropped_unheard") == 2


def test_short_whisper_miss_is_not_a_cut_verse():
    # one short unheard line is Whisper noise, not a departure: keep it
    text = "\n".join([VERSE[0], "oh oh", *VERSE[1:]])
    res = resolve(text, _segs(*VERSE))
    assert "oh oh" in res.text.splitlines()


def test_sections_and_instructions_never_reach_the_aligner():
    text = "[Chorus]\nVerse 2:\n" + VERSE[0] + " (x2)\n" + "\n".join(VERSE[1:])
    res = resolve(text, _segs(*VERSE))
    assert res.text.splitlines() == VERSE
    assert not any(c in res.text for c in "()[]")


def test_sung_line_starting_with_a_section_word_is_lyrics():
    line = "hook line and sinker you caught me"
    res = resolve(line + "\n" + VERSE[0], _segs(line, VERSE[0]))
    assert res.text.splitlines()[0] == line


def test_low_agreement_song_decides_nothing():
    text = VERSE[0] + " (across the water)\n" + "\n".join(VERSE[1:])
    res = resolve(text, _segs("completely different words here", "nothing matches at all"))
    assert res.agreement < 0.4
    assert [g["decision"] for g in res.groups] == ["unsure"]
    assert res.unsure_lines == list(range(len(res.text.splitlines())))


def test_plain_lyrics_strips_brackets_keeps_words():
    assert plain_lyrics("[Intro]\nyou never saw me (you never saw me)\n(oh)\n") == \
        "you never saw me you never saw me\noh"


# --- stage -------------------------------------------------------------------------

def _root(tmp_path, make_wav, lyrics: str):
    root = tmp_path / "dr"
    make_wav(root / "raw" / "song.wav", seconds=1.0)
    (root / "raw" / "song.txt").write_text(lyrics, encoding="utf-8")
    manifest, _ = scan_directory(root, language="en")
    rec = manifest.records[0]
    rec.status.separated = rec.status.cleaned = True
    manifest.upsert(rec)
    manifest.save()
    sdir = song_dir(root, rec.id)
    make_wav(sdir / "clean" / "vocals.wav", seconds=1.0)
    update_analysis(sdir, "clean", {"stems": {"vocals": {"silence": {"voiced_sec": 0.5}}}})
    return root, sdir


def test_stage_writes_resolution_and_leaves_human_file_alone(tmp_path, make_wav):
    lyrics = "I know (I know)\n" + "\n".join(VERSE) + "\n"
    root, sdir = _root(tmp_path, make_wav, lyrics)
    assert run_resolve(root).skipped == ["sng_0001"]  # no asr.json yet
    (sdir / "lyrics").mkdir(parents=True)
    (sdir / "lyrics" / "asr.json").write_text(
        json.dumps({"model": "large-v3", "segments": _segs("I know", *VERSE)}),
        encoding="utf-8")

    s = run_resolve(root)
    assert list(s.resolved) == ["sng_0001"] and not s.failed
    text, unsure = load_resolution(sdir)
    assert text.splitlines()[0] == "I know" and unsure == []
    rec = Manifest.for_data_root(root).get("sng_0001")
    assert (root / rec.meta.lyrics_path).read_text(encoding="utf-8") == lyrics
    assert run_resolve(root).skipped == ["sng_0001"]  # idempotent
    assert list(run_resolve(root, force=True).resolved) == ["sng_0001"]


def test_align_reads_the_resolved_text(tmp_path, make_wav):
    root, sdir = _root(tmp_path, make_wav, "shine (shine)\n")
    (sdir / "lyrics").mkdir(parents=True)
    (sdir / "lyrics" / "asr.json").write_text(
        json.dumps({"segments": _segs("shine")}), encoding="utf-8")
    run_resolve(root)
    seen = {}

    def runner(cmd):
        corpus = cmd[cmd.index("align") + 2]
        from pathlib import Path
        seen["lab"] = (Path(corpus) / "sng_0001" / "sng_0001.lab").read_text(encoding="utf-8")

    align(root, runner=runner)
    assert seen["lab"].strip() == "shine"


def test_normalize_for_alignment():
    from signalml.stages.lyrics_resolve import normalize_for_alignment as n

    assert n("Thank God for heartbreak showers \u2019cause now I'm growin' wildflowers") == \
        "Thank God for heartbreak showers 'cause now I'm growin' wildflowers"
    assert n("to-to-touch me, oh-oh \u2013 Shenandoah River \u2014") == \
        "to to touch me, oh oh Shenandoah River"
    assert n("you & me, 2 hearts, 21 guns, 1999") == "you and me, two hearts, twenty one guns, 1999"
    assert n("caf\u00e9 * # ~ \u201cquoted\u201d") == "caf\u00e9 quoted"


def test_import_batch_songs_resolve_from_the_pasted_source(tmp_path, make_wav):
    # import-batch leaves lyrics.txt bracket-free and keeps the pasted text beside it
    root, sdir = _root(tmp_path, make_wav, "I know I know\n" + "\n".join(VERSE) + "\n")
    rec = Manifest.for_data_root(root).get("sng_0001")
    lyr = root / rec.meta.lyrics_path
    lyr.rename(lyr.with_name("lyrics.txt"))
    src = lyr.with_name("lyrics.source.txt")
    src.write_text("I know (I know)\n" + "\n".join(VERSE) + "\nNote: cut after 2:03\n",
                   encoding="utf-8")
    m = Manifest.for_data_root(root)
    rec = m.get("sng_0001")
    rec.meta.lyrics_path = lyr.with_name("lyrics.txt").relative_to(root).as_posix()
    m.upsert(rec)
    m.save()
    (sdir / "lyrics").mkdir(parents=True)
    (sdir / "lyrics" / "asr.json").write_text(
        json.dumps({"segments": _segs("I know", *VERSE)}), encoding="utf-8")
    s = run_resolve(root)
    res = s.resolved["sng_0001"]
    assert [g["decision"] for g in res.groups] == ["drop"]      # the parens were seen
    assert "note" not in res.text.lower()                       # the cut note never is
    meta = json.loads((sdir / "lyrics" / "resolve.json").read_text(encoding="utf-8"))
    assert meta["source"].endswith("lyrics.source.txt")


def _gapped(*parts):
    """Segments with explicit start times: (start, line)."""
    return [{"start": t, "end": t + 0.5 * len(ln.split()), "text": ln,
             "words": [{"w": w, "start": t + 0.5 * i, "end": t + 0.5 * i + 0.4}
                       for i, w in enumerate(ln.split())]} for t, ln in parts]


MISSED = "these words were sung but whisper never heard them"


def test_unheard_line_with_room_to_be_sung_is_kept_unsure():
    # Whisper skipped a sung line: its neighbours are 8 s apart
    text = "\n".join([VERSE[0], MISSED, VERSE[1]])
    segs = _gapped((0.0, VERSE[0]), (12.0, VERSE[1]))
    res = resolve(text, segs)
    assert MISSED in res.text.splitlines()
    assert res.unsure_lines == [1]


def test_unheard_line_over_silent_vocal_is_dropped():
    text = "\n".join([VERSE[0], MISSED, VERSE[1]])
    segs = _gapped((0.0, VERSE[0]), (12.0, VERSE[1]))
    res = resolve(text, segs, voiced=lambda t0, t1: 0.05)   # an instrumental break
    assert MISSED not in res.text.splitlines() and res.unsure_lines == []
    sung = resolve(text, segs, voiced=lambda t0, t1: 0.9)  # singing there
    assert MISSED in sung.text.splitlines() and sung.unsure_lines == [1]


def test_voiced_fraction_reads_the_vocal():
    import numpy as np

    from signalml.stages.lyrics_resolve import voiced_fraction

    sr = 1000
    y = np.concatenate([0.5 * np.ones(2 * sr), 1e-4 * np.ones(2 * sr)]).astype(np.float32)
    f = voiced_fraction(y, sr)
    assert f(0.0, 1.9) > 0.9 and f(2.1, 4.0) < 0.1
