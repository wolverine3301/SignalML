"""GTSinger adapter contract tests — offline; a synthetic corpus tree, invented names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from praatio import textgrid as praatio_tg

from signalml.ingest.gtsinger import arpabet_to_ipa, find_groups, gender_of, import_gtsinger
from signalml.manifest import Manifest
from signalml.stages.common import song_dir

# "where the sea" over 2.0 s: leading <SP>, trailing <AP>
WORDS = [(0.0, 0.5, "<SP>"), (0.5, 0.9, "where"), (0.9, 1.1, "the"), (1.1, 1.8, "sea"),
         (1.8, 2.0, "<AP>")]
PHONES = [(0.0, 0.5, "<SP>"), (0.5, 0.6, "W"), (0.6, 0.8, "EH1"), (0.8, 0.9, "R"),
          (0.9, 1.0, "DH"), (1.0, 1.1, "AH0"), (1.1, 1.2, "S"), (1.2, 1.8, "IY1"),
          (1.8, 2.0, "<AP>")]


def _phrase(path: Path, make_wav, hz: float = 220.0) -> None:
    make_wav(path.with_suffix(".wav"), seconds=2.0, hz=hz)
    tg = praatio_tg.Textgrid()
    tg.addTier(praatio_tg.IntervalTier("word", WORDS, 0, 2.0))
    tg.addTier(praatio_tg.IntervalTier("phone", PHONES, 0, 2.0))
    tg.save(str(path.with_suffix(".TextGrid")), format="long_textgrid",
            includeBlankSpaces=True)


@pytest.fixture()
def corpus(tmp_path, make_wav) -> Path:
    root = tmp_path / "GTSinger"
    song = root / "English" / "EN-Alto-1" / "Breathy" / "harbor lights"
    for k, group in enumerate(("Control_Group", "Breathy_Group", "Paired_Speech_Group")):
        for i in range(2):  # distinct audio per take, as in the real corpus
            _phrase(song / group / f"{i:04d}", make_wav, hz=220.0 + 40 * k + 10 * i)
    _phrase(root / "English" / "EN-Tenor-1" / "Vibrato" / "kestrel" / "Control_Group"
            / "0000", make_wav)
    return root


def test_arpabet_mapping_uses_the_english_mfa_symbols():
    assert [arpabet_to_ipa(p) for p in ("EH1", "AH0", "AH1", "ER0", "ER1", "IY1", "HH")] == \
        ["ɛ", "ə", "ɐ", "ɚ", "ɝ", "i", "h"]
    with pytest.raises(ValueError):
        arpabet_to_ipa("QQ1")


def test_find_groups_skips_speech_and_other_genders(corpus):
    assert gender_of("EN-Alto-2") == "F" and gender_of("EN-Tenor-1") == "M"
    groups, skipped = find_groups(corpus)
    assert sorted(g.group for g in groups) == ["Breathy_Group", "Control_Group"]
    assert all(len(g.segments) == 2 for g in groups)
    assert "EN-Tenor-1" in skipped and any("Paired_Speech" in k for k in skipped)


def _phrase_with(path: Path, make_wav, words, phones, hz: float = 220.0) -> None:
    make_wav(path.with_suffix(".wav"), seconds=2.0, hz=hz)
    tg = praatio_tg.Textgrid()
    tg.addTier(praatio_tg.IntervalTier("word", words, 0, 2.0))
    tg.addTier(praatio_tg.IntervalTier("phone", phones, 0, 2.0))
    tg.save(str(path.with_suffix(".TextGrid")), format="long_textgrid",
            includeBlankSpaces=True)


def test_ipa_language_writes_phones_json(tmp_path, make_wav):
    src = tmp_path / "GTSinger"
    _phrase_with(src / "French" / "FR-Tenor-9" / "Vibrato" / "phare" / "Control_Group" / "0000",
                 make_wav, [(0.2, 1.8, "phare")],
                 [(0.2, 0.6, "f"), (0.6, 1.4, "a"), (1.4, 1.8, "ʁ")])
    root = tmp_path / "dr"
    manifest, new, skipped, _ = import_gtsinger(root, root=src, genders=("M",),
                                                language="French")
    rec = new[0]
    assert rec.meta.language == "fr" and rec.status.aligned
    ph = json.loads((song_dir(root, rec.id) / "align" / "phones.json").read_text("utf-8"))
    assert ph["phone_set"] == "mfa_ipa/fr_v1" and [p["ph"] for p in ph["phones"]] == \
        ["f", "a", "ʁ"]


def test_pinyin_and_unmappable_phones_stay_native(tmp_path, make_wav):
    src = tmp_path / "GTSinger"
    _phrase_with(src / "Chinese" / "ZH-Tenor-9" / "Vibrato" / "灯塔" / "Control_Group" / "0000",
                 make_wav, [(0.2, 1.0, "灯"), (1.0, 1.8, "塔")],
                 [(0.2, 0.4, "d"), (0.4, 1.0, "eng"), (1.0, 1.2, "t"), (1.2, 1.8, "a")])
    _phrase_with(src / "English" / "EN-Tenor-9" / "Vibrato" / "kestrel" / "Control_Group"
                 / "0000", make_wav, [(0.2, 1.8, "sea")],
                 [(0.2, 0.8, "S"), (0.8, 1.8, "QQ1")], hz=300.0)
    root = tmp_path / "dr"
    for language, code, native_set in (("Chinese", "zh", "opencpop_pinyin"),
                                       ("English", "en", "gtsinger_en")):
        _, new, skipped, _ = import_gtsinger(root, root=src, genders=("M",),
                                             language=language)
        rec = new[0]
        assert rec.meta.language == code and not rec.status.aligned
        sdir = song_dir(root, rec.id)
        assert not (sdir / "align" / "phones.json").exists()
        native = json.loads((sdir / "align" / "phones.native.json").read_text("utf-8"))
        assert native["phone_set"] == native_set
        assert any(r.startswith("native:") for r in skipped.values())
        if language == "English":
            assert native["exceptions"] == ["'QQ1'"]
            assert [p["ph"] for p in native["phones"]] == ["S", "QQ1"]  # not guessed
        else:
            assert (root / rec.meta.lyrics_path).read_text("utf-8").strip() == "灯塔"


def test_male_singers_import_with_genders_m(corpus, tmp_path):
    root = tmp_path / "dr"
    manifest, new, _, groups = import_gtsinger(root, root=corpus, genders=("M",))
    assert [g.singer for g in groups] == ["EN-Tenor-1"]
    rec = new[0]
    assert (rec.meta.singer, rec.meta.gender) == ("gtsinger-en-tenor-1", "M")


def test_import_joins_phrases_and_converts_the_manual_alignment(corpus, tmp_path):
    root = tmp_path / "dr"
    manifest, new, skipped, _ = import_gtsinger(root, root=corpus)
    manifest.save()
    assert len(new) == 2 and not [r for r in skipped.values() if r.startswith("failed")]
    rec = Manifest.for_data_root(root).get(new[0].id)
    assert rec.meta.corpus == "gtsinger" and rec.meta.gender == "F"
    assert rec.meta.language == "en" and rec.status.separated and rec.status.aligned
    assert rec.file.duration_sec == pytest.approx(4.0, abs=0.01)  # 2 phrases joined
    sdir = song_dir(root, rec.id)
    ph = json.loads((sdir / "align" / "phones.phrase.json").read_text(encoding="utf-8"))
    assert ph["phone_set"] == "mfa_ipa/en_v1"
    tokens = [p["ph"] for p in ph["phones"]]
    assert tokens == ["w", "ɛ", "ɹ", "ð", "ə", "s", "i"] * 2    # silences dropped
    second = ph["phones"][7]
    assert second["start"] == pytest.approx(2.5) and second["word"] == "where"  # offset
    utts = json.loads((sdir / "align" / "utterances.phrase.json").read_text(encoding="utf-8"))
    assert [u["text"] for u in utts] == ["where the sea"] * 2
    analysis = json.loads((sdir / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["align.phrase"]["align_score"] == 1.0
    assert analysis["lyrics_resolve"]["agreement"] == 1.0
    assert (sdir / "stems" / "vocals.wav").exists()
    assert (root / rec.meta.lyrics_path).read_text(encoding="utf-8").splitlines() == \
        ["where the sea"] * 2


def test_import_is_idempotent(corpus, tmp_path):
    root = tmp_path / "dr"
    manifest, new, _, _ = import_gtsinger(root, root=corpus)
    manifest.save()
    manifest, again, skipped, _ = import_gtsinger(root, root=corpus)
    assert again == [] and sum("checksum" in r for r in skipped.values()) == 2


def test_textgrids_named_without_extension_are_found(tmp_path, make_wav):
    root = tmp_path / "GTSinger"
    gdir = root / "English" / "EN-Alto-2" / "Vibrato" / "kestrel" / "Vibrato_Group"
    _phrase(gdir / "0000", make_wav)
    (gdir / "0000.TextGrid").rename(gdir / "0000_TextGrid")   # the corpus' other spelling
    groups, _ = find_groups(root)
    assert len(groups) == 1 and groups[0].segments == [gdir / "0000.wav"]
    manifest, new, skipped, _ = import_gtsinger(tmp_path / "dr", root=root)
    assert len(new) == 1 and not skipped
