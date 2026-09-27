"""CSD adapter contract tests — offline; a synthetic corpus tree, invented lyrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from signalml.ingest.csd import import_csd
from signalml.manifest import Manifest
from signalml.stages.common import song_dir


@pytest.fixture()
def corpus(tmp_path, make_wav) -> Path:
    root = tmp_path / "CSD" / "english"
    for i, (stem, hz) in enumerate([("en001a", 220.0), ("en001b", 260.0), ("en002a", 300.0)]):
        make_wav(root / "wav" / f"{stem}.wav", seconds=1.0, hz=hz)
        if stem != "en002a":  # one recording without lyrics
            (root / "lyric").mkdir(parents=True, exist_ok=True)
            (root / "lyric" / f"{stem}.txt").write_text("lanterns glow over the harbor\n",
                                                       encoding="utf-8")
    return tmp_path / "CSD"


def test_import_places_songs_ready_for_clean(corpus, tmp_path):
    root = tmp_path / "dr"
    manifest, new, skipped = import_csd(root, root=corpus)
    manifest.save()
    assert [r.meta.song for r in new] == ["en001a", "en001b"]
    assert "en002a" in skipped
    rec = Manifest.for_data_root(root).get(new[0].id)
    assert (rec.meta.corpus, rec.meta.singer, rec.meta.gender, rec.meta.language) == \
        ("csd", "csd-en", "F", "en")
    assert rec.status.separated and not rec.status.cleaned
    assert (song_dir(root, rec.id) / "stems" / "vocals.wav").exists()
    assert (root / rec.meta.lyrics_path).read_text(encoding="utf-8").strip() == \
        "lanterns glow over the harbor"


def test_import_is_idempotent(corpus, tmp_path):
    root = tmp_path / "dr"
    manifest, _, _ = import_csd(root, root=corpus)
    manifest.save()
    _, again, skipped = import_csd(root, root=corpus)
    assert again == [] and sum("checksum" in r for r in skipped.values()) == 2
