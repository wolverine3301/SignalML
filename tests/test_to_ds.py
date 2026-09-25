"""score.json -> .ds contract tests (offline, no model)."""

from __future__ import annotations

import json

from signalml.cli import main
from signalml.score.schema import NoteEvent, Score, save_score
from signalml.score.to_ds import note_name, score_to_ds


def _n(start, end, midi, syl, phones, slur=False):
    return NoteEvent(start=start, end=end, midi=midi, syllable=syl,
                     phonemes=[] if slur else phones, slur=slur)


def _score():
    # phrase 1: "be-  -ing" held + "stay"; a 2 s rest; phrase 2: "hmm"
    return Score(bpm=100, phone_set="mfa_ipa/en_v1", notes=[
        _n(1.0, 1.5, 60, "be", ["b", "iː"]),
        _n(1.5, 2.0, 62, "ing", ["ɪ", "ŋ"]),
        _n(2.0, 2.5, 64, "ing", [], slur=True),
        _n(2.5, 3.0, 65, "stay", ["s", "t", "ej"]),
        _n(5.0, 6.0, 67, "hmm", ["h", "m"]),   # no nucleus
    ])


def _check(seg):
    notes, durs, slurs = (seg[k].split() for k in ("note_seq", "note_dur", "note_slur"))
    assert len(notes) == len(durs) == len(slurs)
    groups = [int(n) for n in seg["ph_num"].split()]
    assert sum(groups) == len(seg["ph_seq"].split())
    # the variance model's invariant: one phone group per non-slur note
    assert len(groups) == slurs.count("0")
    return notes, [float(d) for d in durs], slurs, groups


def test_note_name():
    assert [note_name(m) for m in (60, 61, 69, 48)] == ["C4", "C#4", "A4", "C3"]


def test_vowel_onset_segments():
    ds = score_to_ds(_score(), pad_sec=0.5)
    assert len(ds) == 2
    first, second = ds
    notes, durs, slurs, groups = _check(first)
    assert first["offset"] == 0.5
    assert notes == ["rest", "C4", "D4", "E4", "F4", "rest"]
    assert slurs == ["0", "0", "0", "1", "0", "0"]
    assert abs(sum(durs) - (3.0 + 0.5 - 0.5)) < 1e-6
    # onset consonants ride with the group before: [SP b] [iː] [ɪ ŋ s t] [ej] [SP]
    assert first["ph_seq"] == "SP b iː ɪ ŋ s t ej SP"
    assert groups == [2, 1, 4, 1, 1]
    _, _, _, g2 = _check(second)
    assert second["ph_seq"] == "SP h m SP" and g2 == [1, 2, 1]  # no nucleus: kept whole
    assert second["text"] == "hmm"


def test_syllable_mode_keeps_each_notes_phones():
    first = score_to_ds(_score(), mode="syllable")[0]
    _, _, _, groups = _check(first)
    assert first["ph_seq"] == "SP b iː ɪ ŋ s t ej SP" and groups == [1, 2, 2, 3, 1]


def test_inner_rest_and_sliver():
    s = Score(bpm=100, phone_set="mfa_ipa/en_v1", notes=[
        _n(1.0, 1.5, 60, "a", ["ej"]),
        _n(1.52, 2.0, 60, "a", ["ej"]),   # 20 ms sliver: merged into the first note
        _n(2.2, 2.6, 62, "b", ["b", "iː"]),  # 200 ms gap: a rest note
    ])
    notes, durs, _, _ = _check(score_to_ds(s, pad_sec=0.5)[0])
    assert notes == ["rest", "C4", "C4", "rest", "D4", "rest"]
    assert abs(durs[1] - 0.52) < 1e-6 and abs(durs[3] - 0.2) < 1e-6


def test_cli_writes_ds(tmp_path):
    src = save_score(_score(), tmp_path / "song.json")
    assert main(["score", "to-ds", str(src)]) == 0
    ds = json.loads((tmp_path / "song.ds").read_text(encoding="utf-8"))
    assert len(ds) == 2 and ds[0]["ph_num"] == "2 1 4 1 1"
