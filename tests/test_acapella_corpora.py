"""RAVDESS / vocadito adapter contract tests — offline; synthetic trees, invented lyrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from signalml.ingest.ravdess import import_ravdess
from signalml.ingest.vocadito import import_vocadito
from signalml.manifest import Manifest
from signalml.stages.common import song_dir


@pytest.fixture()
def ravdess(tmp_path, make_wav) -> Path:
    root = tmp_path / "ravdess"
    # actor 01 (M): neutral (one intensity), 2 statements x 2 repetitions
    for stmt in ("01", "02"):
        for rep in ("01", "02"):
            make_wav(root / "Actor_01" / f"03-02-01-01-{stmt}-{rep}-01.wav", seconds=0.5)
    # actor 02 (F): one angry/strong clip; plus a speech clip that must be ignored
    make_wav(root / "Actor_02" / "03-02-05-02-01-01-02.wav", seconds=0.5)
    make_wav(root / "Actor_02" / "03-01-05-02-01-01-02.wav", seconds=0.5)
    return root


def test_ravdess_groups_clips_into_takes(ravdess, tmp_path):
    root = tmp_path / "dr"
    manifest, new, skipped, takes = import_ravdess(root, root=ravdess)
    manifest.save()
    assert [t.key for t in takes] == ["actor01/neutral/normal", "actor02/angry/strong"]
    assert "03-01-05-02-01-01-02.wav" in skipped  # speech
    rec = Manifest.for_data_root(root).get(new[0].id)
    assert (rec.meta.corpus, rec.meta.singer, rec.meta.gender, rec.meta.language) == \
        ("ravdess", "ravdess-actor01", "M", "en")
    assert rec.meta.source_quality == "studio" and rec.status.separated
    assert sf.info(str(root / rec.file.path)).duration == pytest.approx(2.0, abs=0.01)
    lines = (root / rec.meta.lyrics_path).read_text(encoding="utf-8").splitlines()
    assert lines == ["kids are talking by the door"] * 2 + ["dogs are sitting by the door"] * 2
    analysis = json.loads((song_dir(root, rec.id) / "analysis.json").read_text("utf-8"))
    assert analysis["separate"]["segment_offsets_sec"] == pytest.approx([0, 0.5, 1.0, 1.5])
    assert analysis["import"]["ravdess"]["emotion"] == "neutral"
    assert Manifest.for_data_root(root).get(new[1].id).meta.gender == "F"


def test_ravdess_gender_filter_and_idempotency(ravdess, tmp_path):
    root = tmp_path / "dr"
    _, _, _, takes = import_ravdess(root, root=ravdess, genders=("F",), dry_run=True)
    assert [t.gender for t in takes] == ["F"]
    assert not (root / "manifest.jsonl").exists()  # dry run writes nothing
    manifest, _, _, _ = import_ravdess(root, root=ravdess)
    manifest.save()
    _, again, skipped, _ = import_ravdess(root, root=ravdess)
    assert again == [] and sum("checksum" in r for r in skipped.values()) == 2


@pytest.fixture()
def vocadito(tmp_path, make_wav) -> Path:
    root = tmp_path / "vocadito"
    (root / "Annotations" / "Lyrics").mkdir(parents=True)
    (root / "Annotations" / "Notes").mkdir(parents=True)
    (root / "vocadito_metadata.csv").write_text(
        "track_id,singer_id,average_pitch,language\n1,S1,55,English\n2,S2,60,French\n"
        "3,S3,50,English\n", encoding="utf-8")
    for tid in (1, 2, 3):
        make_wav(root / "Audio" / f"vocadito_{tid}.wav", seconds=0.5)
    for tid in (1, 2):  # track 3 has no lyrics file
        (root / "Annotations" / "Lyrics" / f"vocadito_{tid}_lyrics.txt").write_text(
            "the kestrel turns\n\nover the harbor\n", encoding="utf-8")
    (root / "Annotations" / "Notes" / "vocadito_1_notesA1.csv").write_text(
        "0.1,220.0,0.3\n", encoding="utf-8")
    return root


def test_vocadito_english_only_with_sidecars(vocadito, tmp_path):
    root = tmp_path / "dr"
    manifest, new, skipped, takes = import_vocadito(root, root=vocadito)
    manifest.save()
    assert [t.key for t in takes] == ["vocadito_1"]
    assert "language" in skipped["vocadito_2"] and "missing" in skipped["vocadito_3"]
    rec = new[0]
    assert (rec.meta.corpus, rec.meta.singer, rec.meta.language, rec.meta.gender) == \
        ("vocadito", "vocadito-s1", "en", None)
    assert rec.meta.source_quality is None  # consumer devices, not studio
    take_dir = (root / rec.file.path).parent
    assert (take_dir / "lyrics.txt").read_text(encoding="utf-8").splitlines() == \
        ["the kestrel turns", "over the harbor"]
    assert (take_dir / "vocadito_1_notesA1.csv").exists()
    _, _, _, both = import_vocadito(root, root=vocadito, languages=("en", "fr"), dry_run=True)
    assert [t.key for t in both] == ["vocadito_1", "vocadito_2"]
