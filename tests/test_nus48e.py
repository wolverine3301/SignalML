"""NUS-48E adapter contract tests — offline; a synthetic corpus tree, invented singers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from signalml.ingest.nus48e import convert, import_nus48e
from signalml.manifest import Manifest
from signalml.stages.common import song_dir

# "sea love" with a phrase-ending silence between the words
LABELS = """0.000000 0.200000 sil
0.200000 0.300000 s
0.300000 0.700000 iy
0.700000 1.000000 sil
1.000000 1.100000 l
1.100000 1.500000 ah
1.500000 1.600000 v
1.600000 2.000000 sp
"""


@pytest.fixture()
def corpus(tmp_path, make_wav) -> Path:
    root = tmp_path / "nus-smc-corpus_48"
    for singer, hz in (("KSTR", 220.0), ("HRBR", 330.0)):
        for part, shift in (("sing", 0.0), ("read", 15.0)):
            wav = make_wav(root / singer / part / "07.wav", seconds=2.0, hz=hz + shift)
            wav.with_suffix(".txt").write_text(LABELS, encoding="utf-8")
    make_wav(root / "HRBR" / "sing" / "08.wav", seconds=1.0, hz=500.0)  # no labels
    return root


def test_convert_maps_arpabet_and_cuts_phrases_at_silences():
    rows = [tuple(float(x) if i < 2 else x for i, x in enumerate(line.split()))
            for line in LABELS.splitlines()]
    phones, utts, unstressed = convert(rows)
    assert [p["ph"] for p in phones] == ["s", "i", "l", "ə", "v"]
    assert [(u.start, u.end) for u in utts] == [(0.2, 0.7), (1.0, 1.6)]
    assert unstressed == 1 and all(p["stress"] is None for p in phones)


def test_import_pairs_sung_and_read_takes(corpus, tmp_path):
    root = tmp_path / "dr"
    manifest, new, skipped, takes = import_nus48e(root, root=corpus)
    manifest.save()
    assert len(takes) == 4 and "HRBR/sing/08" in skipped
    recs = Manifest.for_data_root(root).records
    assert sorted(r.meta.domain for r in recs) == ["spoken", "spoken", "sung", "sung"]
    sung = next(r for r in recs if r.meta.singer == "nus48e-kstr" and r.meta.domain == "sung")
    assert sung.status.aligned and sung.meta.gender is None and sung.meta.language == "en"
    sdir = song_dir(root, sung.id)
    ph = json.loads((sdir / "align" / "phones.phrase.json").read_text("utf-8"))
    assert ph["phone_set"] == "mfa_ipa/en_v1" and len(ph["phones"]) == 5
    utts = json.loads((sdir / "align" / "utterances.phrase.json").read_text("utf-8"))
    assert len(utts) == 2
    analysis = json.loads((sdir / "analysis.json").read_text("utf-8"))
    partner = Manifest.for_data_root(root).get(analysis["import"]["nus48e"]["pair_id"])
    assert (partner.meta.singer, partner.meta.domain) == ("nus48e-kstr", "spoken")
    assert analysis["lyrics_resolve"]["agreement"] == 1.0
    assert sf.info(str(root / sung.file.path)).channels == 1


def test_parts_filter_and_idempotency(corpus, tmp_path):
    root = tmp_path / "dr"
    _, _, _, sung_only = import_nus48e(root, root=corpus, parts=("sing",), dry_run=True)
    assert {t.part for t in sung_only} == {"sing"}
    manifest, _, _, _ = import_nus48e(root, root=corpus)
    manifest.save()
    _, again, skipped, _ = import_nus48e(root, root=corpus)
    assert again == [] and sum("checksum" in r for r in skipped.values()) == 4
