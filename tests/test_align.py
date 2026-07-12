"""S5 align contract tests — offline, CPU-only (MFA itself is faked via the injectable
runner; the converter is tested against praatio-written TextGrids).

Acceptance criteria covered: TextGrid fixture -> phones.json (golden payload), converter
rejects phones outside the declared phone set, alignment confidence recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from praatio import textgrid as praatio_tg

from signalml.manifest import Manifest, scan_directory
from signalml.stages.align import AlignConfig, align, load_align_config, textgrid_to_phones
from signalml.stages.common import song_dir, update_analysis

WORDS = [(0.1, 0.6, "shine")]
PHONES = [(0.1, 0.25, "ʃ"), (0.25, 0.45, "aj"), (0.45, 0.6, "n")]


def write_textgrid(path: Path, words=WORDS, phones=PHONES, max_t: float = 1.0) -> Path:
    tg = praatio_tg.Textgrid()
    tg.addTier(praatio_tg.IntervalTier("words", words, 0, max_t))
    tg.addTier(praatio_tg.IntervalTier("phones", phones, 0, max_t))
    path.parent.mkdir(parents=True, exist_ok=True)
    tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)
    return path


class TestConverter:
    def test_golden_payload(self, tmp_path):
        tg = write_textgrid(tmp_path / "a.TextGrid",
                            phones=PHONES + [(0.6, 0.8, "sil")])
        payload = textgrid_to_phones(tg, phone_set_name="mfa_ipa/en_v1", language="en")
        assert payload == {
            "phone_set": "mfa_ipa/en_v1",
            "aligner": "mfa",
            "language": "en",
            "phones": [
                {"ph": "ʃ", "start": 0.1, "end": 0.25, "word": "shine", "stress": None},
                {"ph": "aj", "start": 0.25, "end": 0.45, "word": "shine", "stress": None},
                {"ph": "n", "start": 0.45, "end": 0.6, "word": "shine", "stress": None},
            ],
        }

    def test_noise_kept_and_flagged(self, tmp_path):
        tg = write_textgrid(tmp_path / "a.TextGrid",
                            phones=[(0.1, 0.25, "spn")] + PHONES[1:])
        payload = textgrid_to_phones(tg, phone_set_name="mfa_ipa/en_v1", language="en")
        assert payload["phones"][0]["ph"] == "spn"
        assert payload["phones"][0]["noise"] is True

    def test_strict_rejects_unknown_phone(self, tmp_path):
        tg = write_textgrid(tmp_path / "a.TextGrid",
                            phones=[(0.1, 0.25, "QQ")] + PHONES[1:])
        with pytest.raises(ValueError, match="QQ"):
            textgrid_to_phones(tg, phone_set_name="mfa_ipa/en_v1", language="en")
        payload = textgrid_to_phones(tg, phone_set_name="mfa_ipa/en_v1", language="en",
                                     strict=False)
        assert payload["unknown_phones"] == ["QQ"]
        assert len(payload["phones"]) == 3  # unknown phone still present, just flagged

    def test_missing_phones_tier(self, tmp_path):
        tg = praatio_tg.Textgrid()
        tg.addTier(praatio_tg.IntervalTier("words", WORDS, 0, 1.0))
        path = tmp_path / "b.TextGrid"
        tg.save(str(path), format="long_textgrid", includeBlankSpaces=True)
        with pytest.raises(ValueError, match="phones"):
            textgrid_to_phones(path, phone_set_name="mfa_ipa/en_v1", language="en")


@pytest.fixture()
def data_root(tmp_path, make_wav):
    """One scanned song: cleaned, english, with a lyrics sidecar + silence map."""
    root = tmp_path / "dr"
    make_wav(root / "raw" / "song.wav", seconds=1.0)
    (root / "raw" / "song.txt").write_text("shine\n", encoding="utf-8")
    manifest, _ = scan_directory(root, language="en")
    rec = manifest.records[0]
    rec.status.separated = True
    rec.status.cleaned = True
    manifest.upsert(rec)
    manifest.save()

    sdir = song_dir(root, rec.id)
    make_wav(sdir / "clean" / "vocals.wav", seconds=1.0)
    update_analysis(sdir, "clean",
                    {"stems": {"vocals": {"silence": {"voiced_sec": 0.5}}}})
    return root


def fake_mfa_runner(cmd):
    """Pretend to be `mfa align`: read the corpus dir from the command, emit one
    TextGrid per staged song into the output dir (mirroring MFA's speaker layout)."""
    i = cmd.index("align")
    corpus, out_dir = Path(cmd[i + 2]), Path(cmd[i + 5])
    for spk in corpus.iterdir():
        if spk.is_dir():
            write_textgrid(out_dir / spk.name / f"{spk.name}.TextGrid")


class TestAlignStage:
    def test_end_to_end(self, data_root):
        summary = align(data_root, runner=fake_mfa_runner)
        assert summary.aligned == ["sng_0001"] and not summary.failed

        adir = song_dir(data_root, "sng_0001") / "align"
        assert (adir / "vocals.TextGrid").exists()  # aligner-native audit copy
        payload = json.loads((adir / "phones.json").read_text(encoding="utf-8"))
        assert payload["phone_set"] == "mfa_ipa/en_v1"
        assert [p["ph"] for p in payload["phones"]] == ["ʃ", "aj", "n"]

        rec = Manifest.for_data_root(data_root).get("sng_0001")
        assert rec.status.aligned is True
        assert rec.quality.align_score == 1.0  # speech 0.5 s / voiced 0.5 s

        analysis = json.loads(
            (song_dir(data_root, "sng_0001") / "analysis.json").read_text(encoding="utf-8"))
        assert analysis["align"]["n_phones"] == 3
        assert analysis["align"]["align_score"] == 1.0

    def test_idempotent_and_force(self, data_root):
        align(data_root, runner=fake_mfa_runner)
        again = align(data_root, runner=fake_mfa_runner)
        assert again.aligned == [] and again.skipped == ["sng_0001"]
        forced = align(data_root, force=True, runner=fake_mfa_runner)
        assert forced.aligned == ["sng_0001"]

    def test_wrong_language_is_skipped(self, data_root):
        manifest = Manifest.for_data_root(data_root)
        rec = manifest.records[0]
        rec.meta.language = "ga"  # Gaelic is wave-2: no aligner yet
        manifest.upsert(rec)
        manifest.save()
        summary = align(data_root, runner=fake_mfa_runner)
        assert summary.skipped == ["sng_0001"] and not summary.failed

    def test_missing_lyrics_fails_loudly(self, data_root):
        manifest = Manifest.for_data_root(data_root)
        rec = manifest.records[0]
        rec.meta.has_lyrics = False
        rec.meta.lyrics_path = None
        manifest.upsert(rec)
        manifest.save()
        summary = align(data_root, runner=fake_mfa_runner)
        assert "lyrics" in summary.failed["sng_0001"]

    def test_mfa_producing_nothing_fails_song(self, data_root):
        def silent_runner(cmd):
            pass  # MFA "ran" but emitted no TextGrid

        summary = align(data_root, runner=silent_runner)
        assert "sng_0001" in summary.failed
        assert Manifest.for_data_root(data_root).get("sng_0001").status.aligned is False


def test_load_align_config_defaults_and_file(tmp_path):
    assert load_align_config(tmp_path / "missing.yaml") == AlignConfig()
    f = tmp_path / "align.yaml"
    f.write_text("beam: 40\nstrict_phones: false\n", encoding="utf-8")
    cfg = load_align_config(f)
    assert cfg.beam == 40
    assert cfg.strict_phones is False
    assert cfg.acoustic_model == "english_mfa"  # default preserved
