"""MedleyDB adapter contract tests — offline, CPU-only.

Fixtures are synthesised: a two-track metadata folder plus tone WAVs laid out the way
the corpus is (``<Track>/<Track>_STEMS/<Track>_STEM_NN.wav``). The load-bearing
property is that gender comes from the per-stem instrument label, never from a flag.
"""

from __future__ import annotations

import json

import pytest
import yaml

from signalml.ingest.medleydb import (
    LICENSE_NOTE,
    discover_audio_roots,
    find_vocal_stems,
    import_medleydb,
    load_metadata,
    load_overrides,
)
from signalml.stages.common import song_dir

TRACK_F = "TestBand_SongOne"
TRACK_M = "OtherBand_SongTwo"


def _metadata(track, artist, stems, *, bleed="no", genre="Rock", title="Song"):
    return {
        "artist": artist,
        "title": title,
        "genre": genre,
        "has_bleed": bleed,
        "instrumental": "no",
        "excerpt": "no",
        "mix_filename": f"{track}_MIX.wav",
        "raw_dir": f"{track}_RAW",
        "stem_dir": f"{track}_STEMS",
        "stems": stems,
    }


def _stem(track, num, instrument, component="", raws=1):
    return {
        "component": component,
        "filename": f"{track}_STEM_{num}.wav",
        "instrument": instrument,
        "raw": {
            f"R{i:02d}": {
                "filename": f"{track}_RAW_{num}_{i:02d}.wav",
                "instrument": instrument,
            }
            for i in range(1, raws + 1)
        },
    }


@pytest.fixture()
def corpus(tmp_path, make_wav):
    """DATA_ROOT with medleydb/Metadata + audio for two tracks (one F, one M)."""
    root = tmp_path / "dr"
    meta_dir = root / "medleydb" / "Metadata"
    meta_dir.mkdir(parents=True)
    audio_root = root / "MedleyDB" / "V2"

    specs = {
        TRACK_F: _metadata(TRACK_F, "Test Band", {
            "S01": _stem(TRACK_F, "01", "drum set"),
            "S02": _stem(TRACK_F, "02", "female singer", component="melody"),
            "S05": _stem(TRACK_F, "05", "female singer", raws=2),  # harmony double
            "S06": _stem(TRACK_F, "06", "vocalists"),              # group: never gendered
        }),
        TRACK_M: _metadata(TRACK_M, "Other Band", {
            "S03": _stem(TRACK_M, "03", "male singer", component="melody"),
        }, bleed="yes"),
    }
    for track, meta in specs.items():
        (meta_dir / f"{track}_METADATA.yaml").write_text(
            yaml.safe_dump(meta), encoding="utf-8")
        for key, stem in meta["stems"].items():
            hz = 150.0 + int(key[1:]) * 20
            make_wav(audio_root / track / meta["stem_dir"] / stem["filename"], hz=hz)
            for raw in stem["raw"].values():
                make_wav(audio_root / track / meta["raw_dir"] / raw["filename"], hz=hz + 3)
    return root, audio_root


class TestSelection:
    def test_gender_comes_from_the_instrument_label(self, corpus):
        root, audio_root = corpus
        stems, _ = find_vocal_stems(
            load_metadata(root / "medleydb" / "Metadata"), [audio_root],
            instruments=["female singer", "male singer"])
        by_label = {s.label: s for s in stems}
        assert set(by_label) == {f"{TRACK_F}:S02", f"{TRACK_F}:S05", f"{TRACK_M}:S03"}
        assert by_label[f"{TRACK_F}:S02"].gender == "F"
        assert by_label[f"{TRACK_M}:S03"].gender == "M"
        assert by_label[f"{TRACK_M}:S03"].has_bleed is True

    def test_default_is_female_only(self, corpus):
        root, audio_root = corpus
        stems, _ = find_vocal_stems(
            load_metadata(root / "medleydb" / "Metadata"), [audio_root])
        assert {s.gender for s in stems} == {"F"}

    def test_melody_only_drops_harmony_stems(self, corpus):
        root, audio_root = corpus
        stems, skipped = find_vocal_stems(
            load_metadata(root / "medleydb" / "Metadata"), [audio_root],
            melody_only=True)
        assert [s.label for s in stems] == [f"{TRACK_F}:S02"]
        assert "melody" in skipped[f"{TRACK_F}:S05"]

    def test_exclude_bleed(self, corpus):
        root, audio_root = corpus
        stems, skipped = find_vocal_stems(
            load_metadata(root / "medleydb" / "Metadata"), [audio_root],
            instruments=["male singer"], include_bleed=False)
        assert stems == []
        assert "bleed" in skipped[TRACK_M]

    def test_raw_level_yields_one_record_per_take(self, corpus):
        root, audio_root = corpus
        stems, _ = find_vocal_stems(
            load_metadata(root / "medleydb" / "Metadata"), [audio_root], level="raw")
        assert [s.label for s in stems] == [
            f"{TRACK_F}:S02/R01", f"{TRACK_F}:S05/R01", f"{TRACK_F}:S05/R02"]

    def test_group_vocals_are_refused_not_guessed(self, corpus):
        root, audio_root = corpus
        with pytest.raises(ValueError, match="group label"):
            find_vocal_stems(load_metadata(root / "medleydb" / "Metadata"),
                             [audio_root], instruments=["vocalists"])

    def test_unknown_instrument_is_an_error(self, corpus):
        root, audio_root = corpus
        with pytest.raises(ValueError, match="unknown"):
            find_vocal_stems(load_metadata(root / "medleydb" / "Metadata"),
                             [audio_root], instruments=["banjo"])

    def test_missing_audio_reported_per_track(self, corpus, tmp_path):
        root, _ = corpus
        stems, skipped = find_vocal_stems(
            load_metadata(root / "medleydb" / "Metadata"), [tmp_path / "nowhere"])
        assert stems == []
        assert skipped[TRACK_F] == "audio not on this machine"

    def test_discover_audio_roots(self, corpus):
        root, audio_root = corpus
        assert discover_audio_roots(root, [TRACK_F, TRACK_M]) == [audio_root]


class TestImport:
    def test_import_tags_and_provenance(self, corpus):
        root, audio_root = corpus
        manifest, new, skipped, planned = import_medleydb(
            root, audio_roots=[audio_root], melody_only=True)
        manifest.save()
        assert len(new) == 1
        rec = new[0]
        assert (rec.meta.gender, rec.meta.singer) == ("F", "test band")
        assert rec.meta.source_quality == "studio"
        assert rec.meta.processing == "produced"  # stem level = engineer's submix
        assert rec.meta.license_note == LICENSE_NOTE
        assert rec.meta.genre == "rock"
        assert rec.meta.has_lyrics is False  # MedleyDB ships no transcripts
        assert rec.status.separated is True  # a stem is already an isolated source

        sdir = song_dir(root, rec.id)
        assert (sdir / "stems" / "vocals.wav").exists()
        analysis = json.loads((sdir / "analysis.json").read_text(encoding="utf-8"))
        assert analysis["separate"]["medleydb"]["stem"] == "S02"
        assert analysis["separate"]["medleydb"]["instrument"] == "female singer"

    def test_raw_level_is_tagged_dry(self, corpus):
        root, audio_root = corpus
        _, new, _, _ = import_medleydb(root, audio_roots=[audio_root], level="raw",
                                       melody_only=True)
        assert [r.meta.processing for r in new] == ["dry"]

    def test_idempotent_by_checksum(self, corpus):
        root, audio_root = corpus
        manifest, first, _, _ = import_medleydb(root, audio_roots=[audio_root])
        manifest.save()
        manifest, second, skipped, _ = import_medleydb(root, audio_roots=[audio_root])
        manifest.save()
        assert len(first) == 2 and second == []
        assert all("checksum" in r for r in skipped.values())
        assert len(manifest) == 2

    def test_dry_run_writes_nothing(self, corpus):
        root, audio_root = corpus
        manifest, new, _, planned = import_medleydb(
            root, audio_roots=[audio_root], dry_run=True)
        assert [(p.stem.label, p.gender, p.singer) for p in planned] == [
            (f"{TRACK_F}:S02", "F", "test band"), (f"{TRACK_F}:S05", "F", "test band")]
        assert new == []
        assert len(manifest) == 0
        assert not (root / "songs").exists()

    def test_overrides_correct_singer_and_language(self, corpus, tmp_path):
        root, audio_root = corpus
        over = tmp_path / "over.yaml"
        over.write_text(yaml.safe_dump({"tracks": {
            TRACK_F: {"singer": "Curated Soprano", "language": "it"},
            f"{TRACK_F}:S05": {"exclude": True},
        }}), encoding="utf-8")
        _, new, skipped, _ = import_medleydb(
            root, audio_roots=[audio_root], overrides_path=over)
        assert [(r.meta.singer, r.meta.language) for r in new] == [
            ("curated soprano", "it")]
        assert skipped[f"{TRACK_F}:S05"] == "excluded by overrides file"

    def test_unknown_override_field_is_an_error(self, tmp_path):
        path = tmp_path / "over.yaml"
        path.write_text(yaml.safe_dump({"tracks": {TRACK_F: {"vocalist": "x"}}}),
                        encoding="utf-8")
        with pytest.raises(ValueError, match="unknown field"):
            load_overrides(path)

    def test_lyrics_sidecar_next_to_the_stem_is_picked_up(self, corpus):
        root, audio_root = corpus
        stem_wav = (audio_root / TRACK_F / f"{TRACK_F}_STEMS"
                    / f"{TRACK_F}_STEM_02.wav")
        stem_wav.with_suffix(".txt").write_text("shine on\n", encoding="utf-8")
        _, new, _, _ = import_medleydb(root, audio_roots=[audio_root], melody_only=True)
        assert new[0].meta.has_lyrics is True
        assert new[0].meta.lyrics_path.endswith("_STEM_02.txt")


def test_shipped_overrides_file_is_valid():
    """configs/medleydb_overrides.yaml must stay loadable (field names drift)."""
    from pathlib import Path

    from signalml.config import CONFIGS_DIR

    path = Path(CONFIGS_DIR) / "medleydb_overrides.yaml"
    entries = load_overrides(path)
    assert "Verdi_IlTrovatore" in entries
