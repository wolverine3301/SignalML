"""Session-document parser tests - invented sheets shaped like the real ones."""

from __future__ import annotations

from signalml.ingest.sheets import (
    fix_quotes,
    lyrics_text,
    parse_comp_sheet,
    parse_cue_sheet,
    parse_meta_txt,
)

COMP = """Some Band: �Lantern Song�
VERSE 1
The harbor�s lights are low tonight
We row until the dawn
CHORUS 1
Oh lantern, lantern
PRECHORUS 1
Hold on
MIDDLE SECTION
Slow and steady
OUTRO (CHORUS 2)
Oh lantern, lantern
"""

CUE = """Some Band: �Lantern Song�
(E minor, 6/8, approx 87bpm)
Time Bars Section Notes
0:00 - Upbeat
0:01 2 Intro 1a
0.:17 8 Verse 1b
1:53: 8 Chorus 2b
2.16 4 Chorus 3a
"""


def test_quotes_from_pdf_text():
    assert fix_quotes("I�m fine") == "I'm fine"
    assert fix_quotes("�Title�") == "Title"


def test_comp_sheet_sections_and_plain_lyrics():
    comp = parse_comp_sheet(COMP)
    assert comp["title"] == "Some Band: Lantern Song"
    assert [s["name"] for s in comp["sections"]] == [
        "VERSE 1", "CHORUS 1", "PRECHORUS 1", "MIDDLE SECTION", "OUTRO (CHORUS 2)"]
    assert comp["sections"][0]["lines"][0] == "The harbor's lights are low tonight"
    text = lyrics_text(comp)
    assert "VERSE" not in text and "CHORUS" not in text  # the aligner never sings a header
    assert text.splitlines()[0] == "The harbor's lights are low tonight"
    assert len(text.splitlines()) == 6


def test_cue_sheet_header_and_messy_times():
    cue = parse_cue_sheet(CUE)
    assert (cue["key"], cue["meter"], cue["bpm"]) == ("E minor", "6/8", [87.0])
    rows = [(s["start_sec"], s["bars"], s["name"]) for s in cue["sections"]]
    assert rows == [(0, None, "Upbeat"), (1, 2, "Intro 1a"), (17, 8, "Verse 1b"),
                    (113, 8, "Chorus 2b"), (136, 4, "Chorus 3a")]


def test_cue_sheet_unknown_key_and_tempo_range():
    cue = parse_cue_sheet("T\n(key, 4/4,186-216bpm)\nTime Bars Section Notes\n")
    assert (cue["key"], cue["meter"], cue["bpm"]) == (None, "4/4", [186.0, 216.0])


def test_meta_txt():
    assert parse_meta_txt("Style: Smooth Acoustic Rock\n\nnoise") == \
        {"style": "Smooth Acoustic Rock"}
