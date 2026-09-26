"""import-batch contract tests — offline; the converter is a copy (no ffmpeg)."""

from __future__ import annotations

from signalml.ingest.batch import import_batch, tidy_lyrics
from signalml.manifest import Manifest


def _copy(src, dst, end=None):
    """Stands in for ffmpeg: copies, and honours a cut by truncating the samples."""
    import soundfile as sf

    y, sr = sf.read(str(src))
    sf.write(str(dst), y[: int(end * sr)] if end is not None else y, sr)


def _src(tmp_path, make_wav):
    src = tmp_path / "incoming"
    a = src / "Nova Reyes" / "Harbor (Porch Version)"
    make_wav(a / "Harbor.mp3.wav", hz=220).rename(a / "Harbor.mp3")
    (a / "lyrics.txt").write_text("[Chorus]\nhold the (hold the) light...\n", encoding="utf-8")
    b = src / "Wren Hale" / "Low Tide"
    make_wav(b / "Low Tide.wav", hz=330)
    (b / "New Text Document.txt").write_text("the water rising\n", encoding="utf-8")
    c = src / "Wren Hale" / "no lyrics yet"
    make_wav(c / "x.wav", hz=440)
    return src


def test_tidy_lyrics_keeps_sung_words():
    raw = "[Verse 1]\nI know (I know) you...\n\n[Chorus] la la\n"
    assert tidy_lyrics(raw) == "I know I know you\nla la\n"


def test_import_places_tags_and_scans(tmp_path, make_wav):
    root = tmp_path / "dr"
    s = import_batch(_src(tmp_path, make_wav), root, batch="b1", genre="acoustic",
                     processing="dry", licenses={"Nova Reyes": "licence note"},
                     converter=_copy)
    assert len(s.placed) == 2 and len(s.new_records) == 2
    assert "no lyrics yet" in next(iter(s.skipped)) and "lyrics" in next(iter(s.skipped.values()))

    song = root / "RAW" / "b1" / "nova reyes" / "Harbor (Porch Version)"
    assert (song / "Harbor (Porch Version).wav").exists()
    assert (song / "lyrics.txt").read_text(encoding="utf-8") == "hold the hold the light\n"
    assert "[Chorus]" in (song / "lyrics.source.txt").read_text(encoding="utf-8")

    recs = {r.meta.singer: r for r in Manifest.for_data_root(root).records}
    nova, wren = recs["nova reyes"], recs["wren hale"]
    assert nova.meta.license_note == "licence note" and wren.meta.license_note is None
    assert nova.meta.genre == "acoustic" and nova.meta.processing == "dry"
    assert nova.meta.has_lyrics and wren.meta.has_lyrics  # any .txt name is accepted
    assert nova.meta.gender == "F" and nova.meta.language == "en"


def test_reimport_is_idempotent(tmp_path, make_wav):
    root, src = tmp_path / "dr", _src(tmp_path, make_wav)
    import_batch(src, root, batch="b1", converter=_copy)
    again = import_batch(src, root, batch="b1", converter=_copy)
    assert again.placed == [] and len(again.already_present) == 2
    assert again.new_records == [] and len(Manifest.for_data_root(root).records) == 2


def test_cut_note_trims_audio_and_leaves_the_lyrics(tmp_path, make_wav):
    """`note: cut after M:SS` in a lyrics file marks where the singing ends; it is an
    instruction, never lyrics. Arriving after import, it re-trims the same record."""
    src, root = tmp_path / "incoming", tmp_path / "dr"
    song = src / "Nova Reyes" / "Harbor"
    make_wav(song / "Harbor.wav", seconds=3.0, hz=220)
    (song / "lyrics.txt").write_text("hold the light\n", encoding="utf-8")
    import_batch(src, root, batch="b1", converter=_copy)
    (rec,) = Manifest.for_data_root(root).records
    assert abs(rec.file.duration_sec - 3.0) < 0.01
    rec.status.separated = True
    m = Manifest.for_data_root(root)
    m.upsert(rec)
    m.save()

    (song / "lyrics.txt").write_text("hold the light\nnote: cut audio after 0:01\n",
                                     encoding="utf-8")
    s = import_batch(src, root, batch="b1", converter=_copy)
    assert len(s.retrimmed) == 1 and s.new_records == []
    (rec,) = Manifest.for_data_root(root).records          # same record, new audio
    assert abs(rec.file.duration_sec - 1.0) < 0.01 and rec.status.separated is False
    lyrics = root / "RAW" / "b1" / "nova reyes" / "Harbor" / "lyrics.txt"
    assert lyrics.read_text(encoding="utf-8") == "hold the light\n"
