"""Chunked-alignment planner contract tests - offline, synthetic words and levels."""

from __future__ import annotations

import numpy as np
from praatio import textgrid as praatio_tg

from signalml.stages.align_chunks import (
    HOP_SEC,
    frame_level_db,
    line_spans,
    plan_utterances,
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


def test_frame_level_db_tracks_loudness():
    sr = 1000
    y = np.concatenate([np.zeros(sr), 0.5 * np.ones(sr)]).astype(np.float32)
    lvl = frame_level_db(y, sr)
    assert lvl[10] < -100 and abs(lvl[150] - 20 * np.log10(0.5)) < 0.1
