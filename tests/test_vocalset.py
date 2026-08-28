"""VocalSet adapter contract tests — offline, CPU-only.

The corpus encodes every label in the filename, so most of the contract is parsing:
`f2_arpeggios_f_slow_forte_e.wav` -> singer f2, female, arpeggios, technique
`f_slow_forte`, vowel `e`.
"""

from __future__ import annotations

import json

import pytest

from signalml.ingest.vocalset import (
    CORPUS,
    LICENSE_NOTE,
    find_files,
    import_vocalset,
    parse_filename,
    pick_root,
    summarize,
)
from signalml.stages.common import song_dir

FILES = [
    "FULL/female1/arpeggios/straight/f1_arpeggios_straight_a.wav",
    "FULL/female2/arpeggios/vibrato/f2_arpeggios_f_slow_forte_e.wav",
    "FULL/female2/long_tones/belt/f2_long_tones_belt_o.wav",
    "FULL/female3/excerpts/spoken/f3_excerpts_spoken.wav",
    "FULL/male1/scales/lip_trill/m1_scales_lip_trill_u.wav",
]


@pytest.fixture()
def corpus(tmp_path, make_wav):
    root = tmp_path / "dr"
    for i, rel in enumerate(FILES):  # distinct tones: identical files would dedupe
        make_wav(root / CORPUS / rel, seconds=1.0, hz=180.0 + 25 * i)
    (root / CORPUS / "FULL" / "notes.txt").write_text("readme", encoding="utf-8")
    make_wav(root / CORPUS / "FULL" / "stray_recording.wav")  # not a VocalSet name
    return root


class TestParseFilename:
    @pytest.mark.parametrize("name,expected", [
        ("f1_arpeggios_straight_a.wav", ("f1", "F", "arpeggios", "straight", "a")),
        ("f2_arpeggios_f_slow_forte_e.wav",
         ("f2", "F", "arpeggios", "f_slow_forte", "e")),
        ("f2_long_tones_belt_o.wav", ("f2", "F", "long_tones", "belt", "o")),
        ("m11_scales_lip_trill_u.wav", ("m11", "M", "scales", "lip_trill", "u")),
        ("f3_excerpts_spoken.wav", ("f3", "F", "excerpts", "spoken", None)),
        # real corpus messiness: typo'd/abbreviated contexts, stray spaces
        ("f4_arepggios_c_fast_forte_a.wav", ("f4", "F", "arpeggios", "c_fast_forte", "a")),
        ("m2_arps_c_fast_piano_e.wav", ("m2", "M", "arpeggios", "c_fast_piano", "e")),
        ("f5_scales_f_sow_forte_o.wav", ("f5", "F", "scales", "f_slow_forte", "o")),
        ("f6_ long_trillo_a.wav", ("f6", "F", None, "long_trillo", "a")),
    ])
    def test_tokens(self, name, expected):
        p = parse_filename(name)
        assert (p.singer_id, p.gender, p.context, p.technique, p.vowel) == expected

    def test_excerpts_split_out_the_sung_words(self):
        p = parse_filename("FULL/female9/excerpts/vibrato/f9_caro_vibrato.wav")
        assert (p.excerpt, p.technique, p.context) == ("caro", "vibrato", "excerpts")
        assert p.lyrics_hint == "caro mio ben"
        # `row_spoken` must still register as speech, not as a technique called
        # "row_spoken" (this is what makes domain=spoken work)
        p = parse_filename("FULL/female9/excerpts/spoken/f9_row_spoken.wav")
        assert (p.excerpt, p.technique, p.domain) == ("row", "spoken", "spoken")

    def test_duplicate_and_take_markers(self):
        p = parse_filename("f2_scales_vibrato_a(1).wav")
        assert (p.technique, p.vowel, p.take) == ("vibrato", "a", 1)
        p = parse_filename("f2_arpeggios_belt_2.wav")
        assert (p.technique, p.take) == ("belt", 2)

    def test_singer_falls_back_to_the_folder(self):
        p = parse_filename("VocalSet/FULL/female3/scales/belt/_scales_belt_a.wav")
        assert (p.singer_id, p.gender, p.context) == ("f3", "F", "scales")

    def test_context_falls_back_to_the_directory(self):
        p = parse_filename("VocalSet/female4/long_tones/vibrato/f4_vibrato_a.wav")
        assert p.context == "long_tones" and p.technique == "vibrato"

    def test_singer_id_is_namespaced_by_corpus(self):
        assert parse_filename("f2_scales_belt_a.wav").singer == "vocalset-f2"

    def test_spoken_technique_is_the_spoken_domain(self):
        assert parse_filename("f3_excerpts_spoken.wav").domain == "spoken"
        assert parse_filename("f3_scales_belt_a.wav").domain == "sung"

    def test_non_vocalset_names_are_rejected(self):
        assert parse_filename("stray_recording.wav") is None
        assert parse_filename("mix.wav") is None


class TestFindFiles:
    def test_default_finds_both_genders_and_flags_strays(self, corpus):
        files, skipped = find_files(corpus / CORPUS)
        assert len(files) == 5
        assert "stray_recording.wav" in skipped

    def test_gender_filter(self, corpus):
        files, _ = find_files(corpus / CORPUS, genders=["F"])
        assert {f.gender for f in files} == {"F"} and len(files) == 4

    def test_context_and_technique_filters(self, corpus):
        files, _ = find_files(corpus / CORPUS, contexts=["arpeggios"])
        assert {f.context for f in files} == {"arpeggios"}
        files, _ = find_files(corpus / CORPUS, techniques=["belt"], genders=["F", "M"])
        assert [f.technique for f in files] == ["belt"]

    def test_missing_root_names_the_download(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="zenodo"):
            find_files(tmp_path / "nope")

    def test_prefers_the_FULL_organization(self, tmp_path, make_wav):
        """VocalSet 1.2 ships the same audio organised by singer, technique and vowel;
        walking all of them would hash thousands of duplicates."""
        root = tmp_path / "vs"
        make_wav(root / "FULL" / "female1" / "scales" / "f1_scales_belt_a.wav")
        make_wav(root / "by_technique" / "belt" / "f1_scales_belt_a.wav", hz=250.0)
        assert pick_root(root).name == "FULL"
        files, _ = find_files(root)
        assert len(files) == 1

    def test_summarize(self, corpus):
        files, _ = find_files(corpus / CORPUS)
        text = summarize(files)
        assert "5 file(s)" in text and "'f1'" in text


class TestImport:
    def test_import_tags_and_provenance(self, corpus):
        manifest, new, _, _ = import_vocalset(corpus, genders=["F"])
        manifest.save()
        assert len(new) == 4
        rec = next(r for r in new if r.meta.song == "f2_long_tones_belt_o")
        assert rec.meta.corpus == CORPUS
        assert (rec.meta.gender, rec.meta.singer) == ("F", "vocalset-f2")
        assert rec.meta.source_quality == "studio"
        assert rec.meta.processing == "dry"
        assert rec.meta.license_note == LICENSE_NOTE
        assert rec.meta.language is None  # vowels, not words
        assert rec.meta.has_lyrics is False
        assert rec.status.separated is True

        analysis = json.loads(
            (song_dir(corpus, rec.id) / "analysis.json").read_text(encoding="utf-8"))
        assert analysis["separate"]["vocalset"]["technique"] == "belt"
        assert analysis["separate"]["vocalset"]["vowel"] == "o"

    def test_spoken_technique_lands_in_the_spoken_domain(self, corpus):
        _, new, _, _ = import_vocalset(corpus, genders=["F"])
        spoken = next(r for r in new if "spoken" in r.meta.song)
        assert spoken.meta.domain == "spoken"

    def test_female_only_by_default(self, corpus):
        _, new, _, _ = import_vocalset(corpus)
        assert {r.meta.gender for r in new} == {"F"}

    def test_idempotent_by_checksum(self, corpus):
        manifest, first, _, _ = import_vocalset(corpus, genders=["F", "M"])
        manifest.save()
        manifest, second, skipped, _ = import_vocalset(corpus, genders=["F", "M"])
        manifest.save()
        assert len(first) == 5 and second == []
        # the stray non-VocalSet wav is also in `skipped`, for its own reason
        assert sum("checksum" in r for r in skipped.values()) == 5
        assert len(manifest) == 5

    def test_limit_and_dry_run(self, corpus):
        manifest, new, _, selected = import_vocalset(
            corpus, genders=["F", "M"], limit=2, dry_run=True)
        assert len(selected) == 2 and new == [] and len(manifest) == 0
        assert not (corpus / "songs").exists()
