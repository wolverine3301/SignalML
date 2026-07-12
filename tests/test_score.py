"""P6 contract tests: score schema, phone set, G2P, MIDI importer — offline, CPU-only.

Acceptance criterion from the migration plan: a hand-made MIDI + lyrics for one verse
produces a valid, human-readable score.json that round-trips the validator.
"""

from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest
from pydantic import ValidationError

from signalml.cli import main as cli_main
from signalml.score import (
    ChainG2P,
    G2PError,
    LexiconG2P,
    MfaG2P,
    NoteEvent,
    Score,
    load_score,
    save_score,
    score_from_midi,
    syllabify,
    validate_score_file,
)
from signalml.score.from_midi import parse_lyric_tokens, read_melody
from signalml.score.phoneset import (
    MFA_IPA_EN_V1,
    diff_against_mfa_dictionary,
    get_phone_set,
)

LEXICON = {
    "shine": [{"phones": ["ʃ", "aj", "n"], "stress": 1}],
    "shining": [{"phones": ["ʃ", "aj"], "stress": 1}, {"phones": ["n", "ɪ", "ŋ"], "stress": 0}],
    "star": [{"phones": ["s", "t", "ɑː", "ɹ"], "stress": 1}],
    "on": [{"phones": ["ɑ", "n"], "stress": 0}],
}


@pytest.fixture()
def g2p():
    return LexiconG2P(lexicon=LEXICON)


def _note(start, end, midi, syllable, phonemes, stress=1, slur=False):
    return NoteEvent(start=start, end=end, midi=midi, syllable=syllable,
                     phonemes=phonemes, stress=stress, slur=slur)


def _valid_score() -> Score:
    return Score(
        bpm=120, key="G:major", language="en", phone_set="mfa_ipa/en_v1",
        notes=[
            _note(0.0, 0.5, 67, "shine", ["ʃ", "aj", "n"]),
            NoteEvent(start=0.5, end=1.0, midi=69, syllable="shine",
                      phonemes=[], stress=None, slur=True),
        ],
    )


class TestSchema:
    def test_valid_score_round_trips(self, tmp_path):
        score = _valid_score()
        path = save_score(score, tmp_path / "score.json")
        assert load_score(path) == score
        assert validate_score_file(path) == []

    def test_rejects_overlapping_notes(self):
        with pytest.raises(ValidationError, match="overlapping"):
            Score(bpm=100, phone_set="mfa_ipa/en_v1", notes=[
                _note(0.0, 0.6, 67, "shine", ["ʃ", "aj", "n"]),
                _note(0.5, 1.0, 69, "star", ["s", "t", "ɑː", "ɹ"]),
            ])

    def test_rejects_slur_with_phonemes(self):
        with pytest.raises(ValidationError, match="slur"):
            _note(0.0, 0.5, 67, "shine", ["aj"], slur=True)

    def test_rejects_note_without_phonemes(self):
        with pytest.raises(ValidationError, match="no phonemes"):
            _note(0.0, 0.5, 67, "shine", [])

    def test_rejects_leading_slur(self):
        with pytest.raises(ValidationError, match="first note"):
            Score(bpm=100, phone_set="mfa_ipa/en_v1", notes=[
                NoteEvent(start=0.0, end=0.5, midi=67, syllable="x",
                          phonemes=[], stress=None, slur=True),
            ])

    def test_rejects_slur_syllable_change(self):
        with pytest.raises(ValidationError, match="continues syllable"):
            Score(bpm=100, phone_set="mfa_ipa/en_v1", notes=[
                _note(0.0, 0.5, 67, "shine", ["ʃ", "aj", "n"]),
                NoteEvent(start=0.5, end=1.0, midi=69, syllable="star",
                          phonemes=[], stress=None, slur=True),
            ])

    def test_rejects_bad_key_format(self):
        with pytest.raises(ValidationError, match="key"):
            Score(bpm=100, key="G major", phone_set="mfa_ipa/en_v1",
                  notes=[_note(0.0, 0.5, 67, "shine", ["ʃ", "aj", "n"])])

    def test_flags_phones_outside_set(self, tmp_path):
        score = _valid_score()
        score.notes[0].phonemes[0] = "ZZ"  # bypass note validation via direct mutation
        path = tmp_path / "bad.json"
        path.write_text(score.model_dump_json(), encoding="utf-8")
        problems = validate_score_file(path)
        assert problems and "ZZ" in problems[0]

    def test_unregistered_phone_set_is_reported_not_fatal(self, tmp_path):
        score = _valid_score().model_copy(update={"phone_set": "mfa_ipa/xx_v9"})
        path = tmp_path / "s.json"
        path.write_text(score.model_dump_json(), encoding="utf-8")
        problems = validate_score_file(path)
        assert problems and "unregistered" in problems[0]


class TestPhoneSet:
    def test_inventory_sanity(self):
        ps = get_phone_set("mfa_ipa/en_v1")
        assert {"ʃ", "aj", "ɹ", "ŋ", "ɫ̩"} <= ps.phones
        assert ps.is_nucleus("aj") and ps.is_nucleus("m̩") and not ps.is_nucleus("ʃ")

    def test_unknown_phones_ignores_silence_and_noise(self):
        assert MFA_IPA_EN_V1.unknown(["ʃ", "", "sil", "spn", "ZZ", "ZZ"]) == ["ZZ"]

    def test_unknown_phone_set_name(self):
        with pytest.raises(KeyError, match="mfa_ipa/en_v1"):
            get_phone_set("nope/v0")

    def test_dictionary_diff(self, tmp_path):
        d = tmp_path / "test.dict"
        d.write_text("shine\t0.99\tʃ aj n\nweird\t1.0\tQQ aj\n", encoding="utf-8")
        diff = diff_against_mfa_dictionary(MFA_IPA_EN_V1, d)
        assert diff.missing_from_set == ["QQ"]
        assert not diff.clean


class TestSyllabify:
    def test_single_syllable(self):
        assert syllabify(["ʃ", "aj", "n"], MFA_IPA_EN_V1) == [["ʃ", "aj", "n"]]

    def test_max_onset_split(self):
        assert syllabify(["ʃ", "aj", "n", "ɪ", "ŋ"], MFA_IPA_EN_V1) == [
            ["ʃ", "aj"], ["n", "ɪ", "ŋ"]]

    def test_no_nucleus_fallback(self):
        assert syllabify(["h", "m"], MFA_IPA_EN_V1) == [["h", "m"]]


class TestG2P:
    def test_lexicon_lookup_normalizes(self, g2p):
        pron = g2p.pronounce("Shine!")
        assert pron.syllables[0].phones == ["ʃ", "aj", "n"]
        assert pron.syllables[0].stress == 1

    def test_lexicon_miss_raises(self, g2p):
        with pytest.raises(G2PError):
            g2p.pronounce("moonbeam")

    def test_chain_prefers_first_backend(self, g2p):
        override = LexiconG2P(lexicon={"shine": [{"phones": ["s", "aj", "n"], "stress": 1}]})
        chain = ChainG2P(override, g2p)
        assert chain.pronounce("shine").syllables[0].phones == ["s", "aj", "n"]
        assert chain.pronounce("star").syllables[0].phones == ["s", "t", "ɑː", "ɹ"]

    def test_mfa_backend_with_fake_runner(self):
        def fake_runner(cmd):
            words = Path(cmd[cmd.index("g2p") + 1]).read_text(encoding="utf-8").split()
            out = cmd[cmd.index("g2p") + 3]
            lines = {"shining": "shining\tʃ aj n ɪ ŋ"}
            Path(out).write_text(
                "\n".join(lines[w] for w in words if w in lines) + "\n", encoding="utf-8")

        backend = MfaG2P(runner=fake_runner)
        pron = backend.pronounce("shining")
        assert [s.phones for s in pron.syllables] == [["ʃ", "aj"], ["n", "ɪ", "ŋ"]]
        assert pron.syllables[0].stress is None  # MFA IPA carries no stress

    def test_mfa_backend_rejects_bad_phones(self):
        def fake_runner(cmd):
            out = cmd[cmd.index("g2p") + 3]
            Path(out).write_text("x\tQQ aj\n", encoding="utf-8")

        with pytest.raises(ValueError, match="outside"):
            MfaG2P(runner=fake_runner).pronounce("x")


class TestLyricTokens:
    def test_hyphenation_and_melisma(self):
        tokens = parse_lyric_tokens("shin-ing - star")
        assert [(t.text, t.is_continuation) for t in tokens] == [
            ("shin", False), ("ing", False), ("", True), ("star", False)]
        assert tokens[0].word == tokens[1].word == "shining"
        assert (tokens[0].syllable_index, tokens[1].syllable_index) == (0, 1)
        assert tokens[0].unit_size == 2

    def test_trailing_hyphen_joins_next_token(self):
        tokens = parse_lyric_tokens("shin- ing star")
        assert [t.text for t in tokens] == ["shin", "ing", "star"]
        assert tokens[0].word == "shining"

    def test_leading_continuation_is_error(self):
        with pytest.raises(ValueError, match="start with"):
            parse_lyric_tokens("- shine")


def _write_midi(path, notes, *, tempo_bpm=120.0, key="G", ticks_per_beat=480):
    """notes: list of (midi, start_beats, dur_beats)."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    if key:
        track.append(mido.MetaMessage("key_signature", key=key, time=0))
    events = []
    for midi_note, start, dur in notes:
        events.append((round(start * ticks_per_beat), "on", midi_note))
        events.append((round((start + dur) * ticks_per_beat), "off", midi_note))
    events.sort(key=lambda e: (e[0], e[1] == "on"))  # offs before ons at equal ticks
    tick = 0
    for abs_tick, kind, midi_note in events:
        delta, tick = abs_tick - tick, abs_tick
        msg = "note_on" if kind == "on" else "note_off"
        track.append(mido.Message(msg, note=midi_note, velocity=90, time=delta))
    mid.save(str(path))
    return path


class TestFromMidi:
    def test_golden_verse(self, tmp_path, g2p):
        """MIDI acceptance test: 'shin-ing - star on' -> golden score JSON."""
        midi = _write_midi(tmp_path / "verse.mid", [
            (67, 0, 1), (69, 1, 1), (71, 2, 2), (74, 4, 1), (67, 5, 1)])
        score = score_from_midi(midi, "shin-ing - star on", g2p=g2p)

        golden = json.loads(
            (Path(__file__).parent / "fixtures" / "score_golden.json")
            .read_text(encoding="utf-8"))
        assert json.loads(score.model_dump_json()) == golden
        # and it round-trips the validator
        path = save_score(score, tmp_path / "score.json")
        assert validate_score_file(path) == []

    def test_tempo_change_respected(self, tmp_path, g2p):
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
        track.append(mido.Message("note_on", note=67, velocity=90, time=0))
        track.append(mido.Message("note_off", note=67, velocity=0, time=480))  # 0.5 s
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(60), time=0))
        track.append(mido.Message("note_on", note=69, velocity=90, time=0))
        track.append(mido.Message("note_off", note=69, velocity=0, time=480))  # 1.0 s
        mid.save(str(tmp_path / "t.mid"))

        notes, bpm, _ = read_melody(tmp_path / "t.mid")
        assert bpm == 120
        assert notes[0].end_sec == pytest.approx(0.5)
        assert notes[1].end_sec == pytest.approx(1.5)

    def test_polyphony_is_error(self, tmp_path):
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=67, velocity=90, time=0))
        track.append(mido.Message("note_on", note=71, velocity=90, time=240))
        track.append(mido.Message("note_off", note=67, velocity=0, time=240))
        track.append(mido.Message("note_off", note=71, velocity=0, time=240))
        mid.save(str(tmp_path / "poly.mid"))
        with pytest.raises(ValueError, match="monophonic"):
            read_melody(tmp_path / "poly.mid")

    def test_token_note_count_mismatch(self, tmp_path, g2p):
        midi = _write_midi(tmp_path / "m.mid", [(67, 0, 1)])
        with pytest.raises(ValueError, match="mismatch"):
            score_from_midi(midi, "shine star", g2p=g2p)

    def test_unhyphenated_multisyllable_word_is_error(self, tmp_path, g2p):
        midi = _write_midi(tmp_path / "m.mid", [(67, 0, 1)])
        with pytest.raises(ValueError, match="hyphenate"):
            score_from_midi(midi, "shining", g2p=g2p)


class TestCli:
    def test_validate_ok_and_bad(self, tmp_path, capsys):
        path = save_score(_valid_score(), tmp_path / "score.json")
        assert cli_main(["score", "validate", str(path)]) == 0
        bad = tmp_path / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        assert cli_main(["score", "validate", str(bad)]) == 1

    def test_from_midi_with_lexicon(self, tmp_path):
        midi = _write_midi(tmp_path / "verse.mid", [(67, 0, 1), (69, 1, 1)])
        lex = tmp_path / "lex.json"
        lex.write_text(json.dumps(LEXICON), encoding="utf-8")
        lyrics = tmp_path / "lyrics.txt"
        lyrics.write_text("shine star", encoding="utf-8")
        out = tmp_path / "out.json"
        rc = cli_main(["score", "from-midi", str(midi), "--lyrics", str(lyrics),
                       "--lexicon", str(lex), "--g2p", "none", "--out", str(out)])
        assert rc == 0
        assert validate_score_file(out) == []

    def test_phoneset_listing(self, capsys):
        assert cli_main(["score", "phoneset"]) == 0
        assert "mfa_ipa/en_v1" in capsys.readouterr().out
