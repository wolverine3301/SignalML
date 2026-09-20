"""Aligner-eval harness tests — offline, CPU-only, no corpus.

Covers the P5.4 metrics (docs/notes/aligner_eval.md): word-onset agreement, phone
boundary error, the separation-drift ablation, and the ``align_score`` calibration.
The interesting cases are the ones where a naive implementation silently produces a
plausible wrong number — dropped words, the PCO denominator, and pooling across songs.
"""

from __future__ import annotations

import json

import pytest

from signalml.evaluation.alignment import (
    boundary_metrics,
    calibrate_align_score,
    drift,
    normalize_word,
    onset_metrics,
    pair_onsets,
    phone_boundaries,
    pooled_onset_metrics,
    spearman,
    word_onsets,
)
from signalml.evaluation.refs import (
    jamendolyrics_songs,
    load_boundary_csv,
    load_jamendolyrics_onsets,
    load_word_csv,
)
from signalml.evaluation.runner import load_song_map, run_jamendolyrics


def phones(*entries) -> dict:
    """(ph, start, end, word) tuples -> a phones.json payload."""
    return {
        "phone_set": "mfa_ipa/en_v1",
        "aligner": "mfa",
        "language": "en",
        "phones": [{"ph": p, "start": s, "end": e, "word": w, "stress": None}
                   for p, s, e, w in entries],
    }


def onsets(*pairs):
    from signalml.evaluation.alignment import WordOnset
    return [WordOnset(w, t) for w, t in pairs]


class TestWordOnsets:
    def test_groups_consecutive_phones_into_one_occurrence(self):
        payload = phones(("ʃ", 0.1, 0.25, "shine"), ("aj", 0.25, 0.45, "shine"),
                         ("n", 0.45, 0.6, "shine"), ("ɒ", 0.7, 0.9, "on"))
        assert word_onsets(payload) == onsets(("shine", 0.1), ("on", 0.7))

    def test_repeated_word_yields_one_onset_per_occurrence(self):
        payload = phones(("ɒ", 0.0, 0.2, "on"), ("n", 0.2, 0.3, "on"),
                         ("æ", 0.3, 0.5, "and"),
                         ("ɒ", 0.5, 0.7, "on"), ("n", 0.7, 0.8, "on"))
        assert [o.word for o in word_onsets(payload)] == ["on", "and", "on"]
        assert [o.start for o in word_onsets(payload)] == [0.0, 0.3, 0.5]

    def test_null_word_breaks_a_run(self):
        # An OOV hole genuinely separates two occurrences of the same word.
        payload = phones(("ɒ", 0.0, 0.2, "on"), ("spn", 0.2, 0.4, None),
                         ("ɒ", 0.4, 0.6, "on"))
        assert [o.start for o in word_onsets(payload)] == [0.0, 0.4]

    def test_empty_payload(self):
        assert word_onsets({"phones": []}) == []
        assert word_onsets({}) == []


class TestPhoneBoundaries:
    def test_starts_and_ends_deduped_and_sorted(self):
        payload = phones(("a", 0.1, 0.2, "x"), ("b", 0.2, 0.35, "x"))
        assert phone_boundaries(payload) == [0.1, 0.2, 0.35]

    def test_gap_keeps_both_edges(self):
        payload = phones(("a", 0.0, 0.1, "x"), ("b", 0.5, 0.6, "y"))
        assert phone_boundaries(payload) == [0.0, 0.1, 0.5, 0.6]

    def test_starts_only(self):
        payload = phones(("a", 0.0, 0.1, "x"), ("b", 0.5, 0.6, "y"))
        assert phone_boundaries(payload, include_ends=False) == [0.0, 0.5]


class TestPairing:
    def test_normalize_keeps_apostrophes(self):
        assert normalize_word("We're,") == "we're"
        assert normalize_word("  Shine!  ") == "shine"

    def test_dropped_word_does_not_shift_later_pairs(self):
        """The case that makes index-based pairing produce a plausible wrong answer."""
        ref = onsets(("we", 0.0), ("are", 1.0), ("going", 2.0), ("on", 3.0))
        hyp = onsets(("we", 0.05), ("going", 2.05), ("on", 3.05))  # 'are' went OOV
        pairs = pair_onsets(ref, hyp)
        assert [(r.word, h.word) for r, h in pairs] == [
            ("we", "we"), ("going", "going"), ("on", "on")]
        # every matched pair is ~50 ms out, not 1 s out
        assert all(abs(r.start - h.start) < 0.1 for r, h in pairs)

    def test_common_words_are_not_treated_as_junk(self):
        """difflib's autojunk would discard exactly the words lyrics are made of."""
        ref = onsets(*[("the", float(i)) for i in range(300)])
        hyp = onsets(*[("the", i + 0.01) for i in range(300)])
        assert len(pair_onsets(ref, hyp)) == 300

    def test_case_and_punctuation_insensitive(self):
        ref = onsets(("Shine,", 0.0), ("On!", 1.0))
        hyp = onsets(("shine", 0.02), ("on", 1.02))
        assert len(pair_onsets(ref, hyp)) == 2


class TestOnsetMetrics:
    def test_known_values(self):
        ref = onsets(("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 3.0))
        hyp = onsets(("a", 0.05), ("b", 1.05), ("c", 2.5), ("d", 3.05))
        m = onset_metrics(ref, hyp)
        assert m.n_matched == 4
        assert m.coverage == 1.0
        assert m.pco["0.1"] == pytest.approx(0.75)  # three within 100 ms
        assert m.pco["0.3"] == pytest.approx(0.75)  # the 0.5 s miss fails both cuts
        assert m.median_ae == pytest.approx(0.05)
        assert m.aae == pytest.approx((0.05 * 3 + 0.5) / 4)

    def test_pco_denominator_is_reference_count_not_matches(self):
        """An aligner must not be rewarded for dropping the words it finds hard."""
        ref = onsets(("a", 0.0), ("b", 1.0), ("c", 2.0), ("d", 3.0))
        hyp = onsets(("a", 0.0), ("b", 1.0))  # dropped half the song, perfectly placed
        m = onset_metrics(ref, hyp)
        assert m.pco["0.1"] == pytest.approx(0.5)
        assert m.coverage == pytest.approx(0.5)

    def test_mean_median_gap_flags_a_single_derailment(self):
        ref = onsets(*[(f"w{i}", float(i)) for i in range(20)])
        hyp = onsets(*[(f"w{i}", i + (30.0 if i == 7 else 0.01)) for i in range(20)])
        m = onset_metrics(ref, hyp)
        assert m.median_ae == pytest.approx(0.01)
        assert m.aae > 1.0  # mean wrecked, median intact — the diagnostic

    def test_no_overlap(self):
        m = onset_metrics(onsets(("a", 0.0)), onsets(("z", 0.0)))
        assert m.n_matched == 0 and m.aae is None and m.coverage == 0.0
        assert m.summary().startswith("no word matches")

    def test_empty_reference(self):
        m = onset_metrics([], onsets(("a", 0.0)))
        assert m.n_matched == 0 and m.coverage == 0.0


class TestPooling:
    def test_long_song_dominates(self):
        short = (onsets(("a", 0.0)), onsets(("a", 0.0)))                  # 1 word, perfect
        long_ref = onsets(*[(f"w{i}", float(i)) for i in range(99)])
        long_hyp = onsets(*[(f"w{i}", i + 1.0) for i in range(99)])       # 99 words, 1 s out
        m = pooled_onset_metrics([short, (long_ref, long_hyp)])
        assert m.n_ref == 100
        assert m.pco["0.3"] == pytest.approx(0.01)  # only the single short-song word

    def test_matching_does_not_cross_song_boundaries(self):
        a = (onsets(("x", 0.0)), onsets(("y", 0.0)))   # no match in song A
        b = (onsets(("y", 5.0)), onsets(("x", 5.0)))   # no match in song B
        m = pooled_onset_metrics([a, b])
        assert m.n_matched == 0  # would be 2 if songs were concatenated first

    def test_empty(self):
        m = pooled_onset_metrics([])
        assert m.n_ref == 0 and m.aae is None


class TestBoundaryMetrics:
    def test_nearest_neighbour(self):
        m = boundary_metrics([0.0, 1.0, 2.0], [0.01, 1.04, 2.10])
        assert m.mean_ms == pytest.approx((10 + 40 + 100) / 3)
        assert m.median_ms == pytest.approx(40.0)
        assert m.within["20"] == pytest.approx(1 / 3)
        assert m.within["50"] == pytest.approx(2 / 3)

    def test_picks_closest_not_positional(self):
        # A spurious extra hypothesis boundary must not shift the comparison.
        m = boundary_metrics([1.0], [0.0, 0.5, 1.02, 5.0])
        assert m.mean_ms == pytest.approx(20.0)

    def test_edges(self):
        assert boundary_metrics([1.0], [3.0]).mean_ms == pytest.approx(2000.0)
        assert boundary_metrics([], [1.0]).mean_ms is None
        assert boundary_metrics([1.0], []).mean_ms is None


class TestSpearman:
    def test_perfect_monotonic(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_nonlinear_but_monotonic_is_still_one(self):
        assert spearman([1, 2, 3, 4], [1, 4, 9, 1000]) == pytest.approx(1.0)

    def test_ties_averaged(self):
        assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)

    def test_undefined_cases(self):
        assert spearman([1, 2], [1, 2]) is None       # too few points
        assert spearman([1, 1, 1], [1, 2, 3]) is None  # zero variance
        with pytest.raises(ValueError):
            spearman([1, 2], [1, 2, 3])


class TestCalibration:
    def test_working_score_gives_negative_rho(self):
        """High align_score should mean low error, so rho is negative when it works."""
        cal = calibrate_align_score([(0.95, 0.02), (0.9, 0.05), (0.6, 0.4), (0.3, 1.2)])
        assert cal.rho == pytest.approx(-1.0)
        assert "predictive" in cal.summary()
        assert "NOT predictive" not in cal.summary()

    def test_useless_score_is_called_out(self):
        cal = calibrate_align_score([(0.9, 0.5), (0.8, 0.1), (0.7, 0.6), (0.6, 0.2)])
        assert cal.rho is not None and cal.rho > -0.5
        assert "NOT predictive" in cal.summary()

    def test_buckets_ordered_by_score(self):
        cal = calibrate_align_score([(0.1, 1.0), (0.4, 0.5), (0.7, 0.2), (0.95, 0.01)],
                                    n_buckets=4)
        mins = [b["align_score_min"] for b in cal.buckets]
        assert mins == sorted(mins)
        assert sum(b["n"] for b in cal.buckets) == 4

    def test_too_few_points(self):
        cal = calibrate_align_score([(0.9, 0.1)])
        assert cal.rho is None and "too few" in cal.summary()


class TestDrift:
    def test_identical_alignments_have_zero_drift(self):
        payload = phones(("a", 0.0, 0.5, "one"), ("b", 0.5, 1.0, "two"))
        m = drift(payload, payload)
        assert m.aae == pytest.approx(0.0) and m.coverage == 1.0

    def test_separation_shift_is_measured(self):
        stem = phones(("a", 0.0, 0.5, "one"), ("b", 1.0, 1.5, "two"))
        sep = phones(("a", 0.08, 0.55, "one"), ("b", 1.09, 1.55, "two"))
        m = drift(stem, sep)
        assert m.aae == pytest.approx(0.085)
        assert m.pco["0.1"] == pytest.approx(1.0)


# --- reference loaders ------------------------------------------------------

JL_HEADER = ("URL,Filepath,Artist,Title,Genre,LicenseType,Language,"
             "LyricOverlap,Polyphonic,NonLexical")


def write_jamendolyrics(root, songs):
    """Build a minimal JamendoLyrics checkout. ``songs`` maps name -> dict."""
    (root / "lyrics").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "words").mkdir(parents=True, exist_ok=True)
    rows = [JL_HEADER]
    for name, spec in songs.items():
        rows.append(
            f"http://x,mp3/{name}.mp3,{spec.get('artist', 'A')},{spec.get('title', 'T')},"
            f"rock,{spec.get('license', 'BY')},{spec.get('language', 'English')},"
            f"{str(spec.get('overlap', False)).lower()},"
            f"{str(spec.get('polyphonic', False)).lower()},"
            f"{str(spec.get('nonlexical', False)).lower()}"
        )
        words = spec.get("words", [])
        (root / "lyrics" / f"{name}.words.txt").write_text(
            "\n".join(w for w, _ in words) + "\n", encoding="utf-8")
        lines = ["word_start,word_end,line_end"]
        lines += [f"{t},{t + 0.2},nan" for _, t in words]
        (root / "annotations" / "words" / f"{name}.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
    (root / "JamendoLyrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


class TestJamendoLyricsRefs:
    def test_metadata_parsing_and_flags(self, tmp_path):
        root = write_jamendolyrics(tmp_path / "jl", {
            "A_-_One": {"license": "BY-NC-ND", "words": [("hello", 1.0)]},
            "B_-_Two": {"license": "BY-SA", "polyphonic": True, "words": [("x", 0.0)]},
            "C_-_Three": {"license": "CC BY", "language": "German", "words": [("y", 0.0)]},
        })
        songs = {s.name: s for s in jamendolyrics_songs(root)}
        assert songs["A_-_One"].no_derivatives is True
        assert songs["A_-_One"].clean is True
        assert songs["B_-_Two"].no_derivatives is False
        assert songs["B_-_Two"].clean is False      # polyphonic
        assert songs["C_-_Three"].language == "German"

    def test_nd_detection_does_not_false_positive(self, tmp_path):
        """'BY-NC-SA' contains no ND; substring matching would say it does."""
        root = write_jamendolyrics(tmp_path / "jl",
                                   {"S": {"license": "BY-NC-SA", "words": [("a", 0.0)]}})
        assert jamendolyrics_songs(root)[0].no_derivatives is False

    def test_onsets_are_positional_against_the_words_file(self, tmp_path):
        root = write_jamendolyrics(tmp_path / "jl", {
            "S": {"words": [("through", 32.4), ("days", 32.76), ("of", 32.97)]}})
        got = load_jamendolyrics_onsets(root, "S")
        assert [o.word for o in got] == ["through", "days", "of"]
        assert got[0].start == pytest.approx(32.4)

    def test_length_mismatch_raises_rather_than_zipping_short(self, tmp_path):
        root = write_jamendolyrics(tmp_path / "jl", {"S": {"words": [("a", 0.0), ("b", 1.0)]}})
        (root / "lyrics" / "S.words.txt").write_text("a\nb\nc\n", encoding="utf-8")
        with pytest.raises(ValueError, match="positional"):
            load_jamendolyrics_onsets(root, "S")

    def test_missing_files(self, tmp_path):
        root = write_jamendolyrics(tmp_path / "jl", {"S": {"words": [("a", 0.0)]}})
        with pytest.raises(FileNotFoundError):
            load_jamendolyrics_onsets(root, "Nope")
        with pytest.raises(FileNotFoundError):
            jamendolyrics_songs(tmp_path / "empty")


class TestGenericLoaders:
    def test_word_csv(self, tmp_path):
        p = tmp_path / "w.csv"
        p.write_text("word,start\nhello,1.5\nworld,2.0\n", encoding="utf-8")
        assert load_word_csv(p) == onsets(("hello", 1.5), ("world", 2.0))

    def test_word_csv_missing_column(self, tmp_path):
        p = tmp_path / "w.csv"
        p.write_text("token,start\na,1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column"):
            load_word_csv(p)

    def test_boundary_csv_dedupes(self, tmp_path):
        p = tmp_path / "b.csv"
        p.write_text("start,end\n0.0,0.5\n0.5,1.0\n", encoding="utf-8")
        assert load_boundary_csv(p) == [0.0, 0.5, 1.0]

    def test_boundary_csv_without_end_column(self, tmp_path):
        p = tmp_path / "b.csv"
        p.write_text("start\n0.0\n0.5\n", encoding="utf-8")
        assert load_boundary_csv(p, end_col=None) == [0.0, 0.5]


# --- runner -----------------------------------------------------------------

def write_alignment(data_root, song_id, entries):
    path = data_root / "songs" / song_id / "align" / "phones.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(phones(*entries)), encoding="utf-8")
    return path


class TestRunner:
    def test_end_to_end_buckets_and_report(self, tmp_path):
        ref_root = write_jamendolyrics(tmp_path / "jl", {
            "Clean_-_Song": {"words": [("one", 1.0), ("two", 2.0)]},
            "Flagged_-_Song": {"nonlexical": True, "words": [("three", 3.0)]},
            "German_-_Song": {"language": "German", "words": [("vier", 4.0)]},
        })
        data_root = tmp_path / "data"
        write_alignment(data_root, "Clean_-_Song",
                        [("a", 1.02, 1.3, "one"), ("b", 2.05, 2.4, "two")])
        write_alignment(data_root, "Flagged_-_Song", [("c", 3.5, 3.9, "three")])

        report = run_jamendolyrics(ref_root, data_root, language="English")
        assert report.n_songs == 2          # the German song is filtered out
        assert report.n_scored == 2
        assert set(report.pooled) == {"clean", "flagged"}
        assert report.pooled["clean"]["pco"]["0.1"] == pytest.approx(1.0)
        assert report.pooled["flagged"]["pco"]["0.1"] == pytest.approx(0.0)
        assert all("·" in line or line.strip() for line in report.summary_lines())

    def test_missing_alignment_is_reported_not_fatal(self, tmp_path):
        ref_root = write_jamendolyrics(tmp_path / "jl", {
            "Has_-_Align": {"words": [("one", 1.0)]},
            "No_-_Align": {"words": [("two", 2.0)]},
        })
        data_root = tmp_path / "data"
        write_alignment(data_root, "Has_-_Align", [("a", 1.0, 1.2, "one")])

        report = run_jamendolyrics(ref_root, data_root)
        assert report.n_songs == 2 and report.n_scored == 1
        failed = [s for s in report.songs if s.error]
        assert len(failed) == 1 and "no alignment" in failed[0].error

    def test_song_map_redirects_ids(self, tmp_path):
        ref_root = write_jamendolyrics(tmp_path / "jl", {"Ref_-_Name": {"words": [("one", 1.0)]}})
        data_root = tmp_path / "data"
        write_alignment(data_root, "sng_0042", [("a", 1.01, 1.2, "one")])

        map_path = tmp_path / "map.csv"
        map_path.write_text("ref,song_id\nRef_-_Name,sng_0042\n", encoding="utf-8")
        report = run_jamendolyrics(ref_root, data_root, song_map=load_song_map(map_path))
        assert report.n_scored == 1

    def test_report_roundtrips_to_json(self, tmp_path):
        ref_root = write_jamendolyrics(tmp_path / "jl", {"S_-_One": {"words": [("one", 1.0)]}})
        data_root = tmp_path / "data"
        write_alignment(data_root, "S_-_One", [("a", 1.0, 1.2, "one")])
        report = run_jamendolyrics(ref_root, data_root, aligner="sofa")
        out = report.write(tmp_path / "out" / "report.json")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["aligner"] == "sofa"
        assert loaded["ref_set"] == "jamendolyrics"
        assert loaded["songs"][0]["song"] == "S_-_One"


class TestCli:
    def test_eval_drift_smoke(self, tmp_path, capsys):
        from signalml.cli import main

        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps(phones(("x", 0.0, 0.5, "one"))), encoding="utf-8")
        b.write_text(json.dumps(phones(("x", 0.05, 0.55, "one"))), encoding="utf-8")
        out = tmp_path / "drift.json"
        assert main(["eval", "drift", str(a), str(b), "--out", str(out)]) == 0
        assert "drift" in capsys.readouterr().out
        assert json.loads(out.read_text(encoding="utf-8"))["n_matched"] == 1

    def test_eval_align_smoke(self, tmp_path, capsys):
        from signalml.cli import main

        ref_root = write_jamendolyrics(tmp_path / "jl", {"S_-_One": {"words": [("one", 1.0)]}})
        data_root = tmp_path / "data"
        write_alignment(data_root, "S_-_One", [("a", 1.02, 1.2, "one")])
        rc = main(["eval", "align", "--ref-root", str(ref_root),
                   "--data-root", str(data_root), "--no-calibrate"])
        assert rc == 0
        assert "jamendolyrics" in capsys.readouterr().out
