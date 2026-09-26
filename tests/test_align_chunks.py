"""Chunked-alignment planner contract tests - offline, synthetic words and levels."""

from __future__ import annotations

import numpy as np
from praatio import textgrid as praatio_tg

from signalml.stages.align_chunks import (
    HOP_SEC,
    frame_level_db,
    line_spans,
    plan_utterances,
    utterance_health,
    utterances_textgrid,
)

TEXT = ("hold the light across the water\nwe walk the harbor road tonight\n"
        "and every lantern knows my name\n")
LINES = [ln.split() for ln in TEXT.strip().split("\n")]
DUR = 40.0


def _seg(start, words):
    return {"start": start, "end": start + len(words) * 0.5, "text": " ".join(words),
            "words": [{"w": w, "start": start + i * 0.5, "end": start + i * 0.5 + 0.4}
                      for i, w in enumerate(words)]}


def _level(sung: list[tuple[float, float]], floor_db=-30.0, sung_db=-10.0):
    """-10 dB while singing, a quieter floor elsewhere (a reverb tail, not silence)."""
    t = np.arange(int(DUR / HOP_SEC)) * HOP_SEC
    lvl = np.full(t.shape, floor_db)
    for s, e in sung:
        lvl[(t >= s) & (t < e)] = sung_db
    return lvl


SEGS = [_seg(2.0, LINES[0]), _seg(10.0, LINES[1]), _seg(18.0, LINES[2])]
SUNG = [(2.0, 5.0), (10.0, 13.0), (18.0, 21.0)]


def test_cuts_at_the_dip_between_each_anchored_line():
    utts = plan_utterances(TEXT, SEGS, _level(SUNG), DUR)
    assert [u.lines for u in utts] == [(0,), (1,), (2,)]
    assert [u.text for u in utts] == [" ".join(w) for w in LINES]
    # the cut lands in the quiet stretch between lines, never inside the singing
    for u, (s, e) in zip(utts, SUNG):
        assert u.start <= s and u.end >= e
    assert utts[0].start == 1.0          # one-second margin before the first word
    assert utts[-1].end == 21.9          # ... and after the last (last word ends 20.9)


def test_no_cut_through_legato_singing():
    utts = plan_utterances(TEXT, SEGS, _level([(2.0, 21.0)]), DUR)
    assert len(utts) == 1 and utts[0].lines == (0, 1, 2)


def test_an_unanchored_line_rides_inside_its_neighbours_utterance():
    segs = [_seg(2.0, LINES[0]), _seg(18.0, LINES[2])]  # Whisper missed line 1
    utts = plan_utterances(TEXT, segs, _level(SUNG), DUR)
    assert [u.lines for u in utts] == [(0, 1, 2)]


def test_lone_word_matches_do_not_anchor_a_line():
    segs = [_seg(2.0, LINES[0]), _seg(14.0, ["tonight"]), _seg(18.0, LINES[2])]
    spans = line_spans([" ".join(w) for w in LINES], segs)
    assert spans[1] is None and spans[0] and spans[2]


def test_unanchored_leading_lines_keep_the_song_start():
    segs = [_seg(10.0, LINES[1]), _seg(18.0, LINES[2])]
    utts = plan_utterances(TEXT, segs, _level(SUNG), DUR)
    assert utts[0].start == 0.0 and utts[0].lines == (0, 1)


def test_textgrid_round_trips_through_praatio(tmp_path):
    utts = plan_utterances(TEXT, SEGS, _level(SUNG), DUR)
    p = tmp_path / "song.TextGrid"
    p.write_text(utterances_textgrid(utts, DUR, 'sng "x"'), encoding="utf-8")
    tg = praatio_tg.openTextgrid(str(p), includeEmptyIntervals=False)
    (name,) = tg.tierNames
    assert name == 'sng "x"'
    labels = [e.label for e in tg.getTier(name).entries]
    assert labels == [u.text for u in utts]


def test_unlyricked_singing_between_lines_is_left_out():
    # an ad-lib between lines 0 and 1 (in no line) is sung with rests either side
    segs = [_seg(2.0, LINES[0]), _seg(6.5, ["yeah", "baby", "come", "on"]),
            _seg(10.0, LINES[1]), _seg(18.0, LINES[2])]
    lvl = _level([(2.0, 5.0), (6.5, 8.5), (10.0, 13.0), (18.0, 21.0)])
    utts = plan_utterances(TEXT, segs, lvl, DUR)
    assert [u.lines for u in utts] == [(0,), (1,), (2,)]
    # neither neighbour's utterance covers the ad-lib
    assert utts[0].end <= 6.5 and utts[1].start >= 8.5


def test_no_excision_without_real_dips():
    # the same ad-lib sung legato into both lines: nothing is cut out
    segs = [_seg(2.0, LINES[0]), _seg(6.5, ["yeah", "baby", "come", "on"]),
            _seg(10.0, LINES[1]), _seg(18.0, LINES[2])]
    utts = plan_utterances(TEXT, segs, _level([(2.0, 13.0), (18.0, 21.0)]), DUR)
    assert utts[0].lines == (0, 1)


def test_excision_can_be_turned_off():
    segs = [_seg(2.0, LINES[0]), _seg(6.5, ["yeah", "baby", "come", "on"]),
            _seg(10.0, LINES[1]), _seg(18.0, LINES[2])]
    lvl = _level([(2.0, 5.0), (6.5, 8.5), (10.0, 13.0), (18.0, 21.0)])
    utts = plan_utterances(TEXT, segs, lvl, DUR, excise_gaps=False)
    assert all(a.end == b.start for a, b in zip(utts, utts[1:]))


def test_utterance_health_counts_floor_stacking():
    utts = plan_utterances(TEXT, SEGS, _level(SUNG), DUR)
    phones = ([{"ph": "a", "start": 2.0 + 0.03 * i, "end": 2.03 + 0.03 * i}
               for i in range(10)]                                 # 10 floor phones
              + [{"ph": "b", "start": 10.0, "end": 16.0}]          # one 6 s phone
              + [{"ph": "spn", "start": 18.0, "end": 18.5, "noise": True}])
    h = utterance_health(utts, phones, unsure_lines={2})
    assert h[0]["floor_frac"] == 1.0 and h[0]["max_floor_run"] == 10
    assert h[1]["max_phone_sec"] == 6.0 and h[1]["floor_frac"] == 0.0
    assert h[2]["n_noise"] == 1 and h[2]["unsure_lyrics"] is True
    assert not h[0]["unsure_lyrics"]


def test_frame_level_db_tracks_loudness():
    sr = 1000
    y = np.concatenate([np.zeros(sr), 0.5 * np.ones(sr)]).astype(np.float32)
    lvl = frame_level_db(y, sr)
    assert lvl[10] < -100 and abs(lvl[150] - 20 * np.log10(0.5)) < 0.1
