"""D12 contract tests: note ids, segmentation, content hashing, render addressing.

These are the three pre-commitments that keep targeted re-rendering possible whatever
method P8 ends up using — *name a region*, *know whether its audio would change*, *know
what produced it*. All model-free: CPU-only, offline, no checkpoint.

The load-bearing property under test is **cache correctness**: a hash that changes when
it shouldn't costs a needless re-render (annoying), and one that fails to change when it
should serves stale audio (silent, and the reason this file is thorough).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from signalml.cli import main as cli_main
from signalml.hashing import canonical_json_bytes, sha256_json
from signalml.score import (
    SCORE_FORMAT,
    NoteEvent,
    Score,
    load_score,
    plan_rerender,
    save_score,
    segment_content_hash,
    split_segments,
)
from signalml.score.segment import DEFAULT_MIN_REST_SEC
from signalml.synth import (
    RenderInputs,
    RenderRecord,
    VarianceTrack,
    find_cached,
    load_record,
    new_record,
    render_dir,
    save_record,
    save_variance,
)
from signalml.synth.render import RECORD_FILENAME

FIXTURES = Path(__file__).parent / "fixtures"


def note(
    start: float,
    end: float,
    *,
    midi: int = 60,
    syllable: str = "la",
    phonemes: tuple[str, ...] = ("l", "ɑ"),
    stress: int | None = 1,
    slur: bool = False,
    id: str | None = None,
) -> NoteEvent:
    return NoteEvent(
        id=id,
        start=start,
        end=end,
        midi=midi,
        syllable=syllable,
        phonemes=[] if slur else list(phonemes),
        stress=None if slur else stress,
        slur=slur,
    )


def score_of(*notes: NoteEvent, language: str = "en",
             phone_set: str = "mfa_ipa/en_v1") -> Score:
    return Score(bpm=120.0, language=language, phone_set=phone_set, notes=list(notes))


def two_phrase_score() -> Score:
    """Two phrases separated by a 0.5 s rest."""
    return score_of(
        note(0.0, 0.5, midi=67, syllable="shine"),
        note(0.5, 1.0, midi=69, syllable="on"),
        note(1.5, 2.0, midi=71, syllable="bright"),
        note(2.0, 2.5, midi=72, syllable="star"),
    ).ensure_ids()


# --------------------------------------------------------------------------- note ids


class TestNoteIds:
    def test_ensure_ids_mints_sequentially(self):
        score = score_of(note(0, 1), note(1, 2)).ensure_ids()
        assert [n.id for n in score.notes] == ["n0001", "n0002"]

    def test_ensure_ids_is_idempotent(self):
        score = score_of(note(0, 1), note(1, 2)).ensure_ids()
        before = [n.id for n in score.notes]
        assert [n.id for n in score.ensure_ids().notes] == before

    def test_existing_ids_are_never_renumbered(self):
        """The rule the whole scheme rests on: an edit must not retarget saved refs."""
        score = score_of(
            note(0, 1, id="n0001"),
            note(1, 2),  # inserted by hand, no id
            note(2, 3, id="n0002"),
        ).ensure_ids()
        assert score.notes[0].id == "n0001"
        assert score.notes[2].id == "n0002"
        assert score.notes[1].id not in {"n0001", "n0002"}

    def test_minted_ids_do_not_reuse_deleted_ones(self):
        """Minting above the high-water mark, so a stale ref fails loudly."""
        score = score_of(note(0, 1, id="n0007"), note(1, 2)).ensure_ids()
        assert score.notes[1].id == "n0008"

    def test_hand_written_ids_survive(self):
        score = score_of(note(0, 1, id="chorus-hook"), note(1, 2)).ensure_ids()
        assert score.notes[0].id == "chorus-hook"

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValidationError, match="duplicate note id"):
            score_of(note(0, 1, id="dup"), note(1, 2, id="dup"))

    @pytest.mark.parametrize("bad", ["", "-leading", "has space", "slash/es"])
    def test_malformed_ids_rejected(self, bad):
        with pytest.raises(ValidationError):
            note(0, 1, id=bad)

    def test_note_by_id(self):
        score = two_phrase_score()
        assert score.note_by_id("n0003").syllable == "bright"
        assert score.note_by_id("nope") is None


class TestLegacyUpgrade:
    def test_v01_file_loads_and_upgrades_in_memory(self):
        score = load_score(FIXTURES / "score_v01_legacy.json")
        assert score.format == SCORE_FORMAT
        assert all(n.id is not None for n in score.notes)

    def test_loading_does_not_touch_the_file(self):
        path = FIXTURES / "score_v01_legacy.json"
        before = path.read_text(encoding="utf-8")
        load_score(path)
        assert path.read_text(encoding="utf-8") == before

    def test_unknown_format_rejected(self):
        with pytest.raises(ValidationError, match="unsupported score format"):
            Score(format="signalml-score/9.9", bpm=120, phone_set="mfa_ipa/en_v1",
                  notes=[note(0, 1)])

    def test_cli_upgrade_rewrites_in_place(self, tmp_path, capsys):
        path = tmp_path / "score.json"
        path.write_text(
            (FIXTURES / "score_v01_legacy.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        assert cli_main(["score", "upgrade", str(path)]) == 0
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["format"] == SCORE_FORMAT
        assert all(n["id"] for n in written["notes"])
        assert "upgraded" in capsys.readouterr().out


# ------------------------------------------------------------------------ segmentation


class TestSegmentation:
    def test_splits_at_a_rest(self):
        segments = split_segments(two_phrase_score())
        assert [s.index for s in segments] == [0, 1]
        assert segments[0].note_ids == ["n0001", "n0002"]
        assert segments[1].start == 1.5 and segments[1].end == 2.5

    def test_short_gaps_stay_in_one_phrase(self):
        score = score_of(note(0, 0.5), note(0.6, 1.0)).ensure_ids()  # 0.1 s gap
        assert len(split_segments(score)) == 1

    def test_slur_never_opens_a_segment(self):
        """A slur carries no phonemes — a segment starting on one is unrenderable."""
        score = score_of(
            note(0.0, 0.5, syllable="held"),
            note(2.0, 3.0, syllable="held", slur=True),  # 1.5 s gap, but a slur
        ).ensure_ids()
        segments = split_segments(score)
        assert len(segments) == 1
        assert segments[0].end == 3.0

    def test_min_rest_is_tunable(self):
        score = two_phrase_score()
        assert len(split_segments(score, min_rest_sec=1.0)) == 1
        assert len(split_segments(score, min_rest_sec=0.4)) == 2

    def test_negative_min_rest_rejected(self):
        with pytest.raises(ValueError, match="min_rest_sec"):
            split_segments(two_phrase_score(), min_rest_sec=-1)

    def test_every_note_lands_in_exactly_one_segment(self):
        score = two_phrase_score()
        ids = [nid for seg in split_segments(score) for nid in seg.note_ids]
        assert ids == [n.id for n in score.notes]

    def test_text_collapses_melisma(self):
        score = score_of(
            note(0.0, 0.5, syllable="shine"),
            note(0.5, 1.5, syllable="shine", slur=True),
        ).ensure_ids()
        assert split_segments(score)[0].text == "shine"

    def test_default_threshold_is_the_documented_one(self):
        assert DEFAULT_MIN_REST_SEC == 0.30


# ----------------------------------------------------------------------- content hash


class TestContentHash:
    def _hash(self, score: Score, index: int = 0) -> str:
        return split_segments(score)[index].content_hash

    def test_stable_for_identical_content(self):
        assert self._hash(two_phrase_score()) == self._hash(two_phrase_score())

    def test_invariant_to_time_shift(self):
        """Moving a phrase later must reuse its render, not invalidate it."""
        early = score_of(note(0.0, 0.5), note(0.5, 1.0)).ensure_ids()
        late = score_of(note(10.0, 10.5), note(10.5, 11.0)).ensure_ids()
        assert self._hash(early) == self._hash(late)

    def test_invariant_to_note_ids(self):
        """Ids are handles, not content — renumbering must cost nothing."""
        a = score_of(note(0, 0.5, id="n0001"), note(0.5, 1, id="n0002"))
        b = score_of(note(0, 0.5, id="zzz"), note(0.5, 1, id="chorus-hook"))
        assert self._hash(a) == self._hash(b)

    def test_invariant_to_header_metadata(self):
        a = two_phrase_score()
        b = two_phrase_score()
        b.bpm, b.key = 200.0, "C:minor"
        assert self._hash(a) == self._hash(b)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda n: setattr(n, "midi", 61), id="pitch"),
            pytest.param(lambda n: setattr(n, "syllable", "other"), id="syllable"),
            pytest.param(lambda n: setattr(n, "phonemes", ["s"]), id="phonemes"),
            pytest.param(lambda n: setattr(n, "stress", 0), id="stress"),
            pytest.param(lambda n: setattr(n, "end", 0.9), id="duration"),
        ],
    )
    def test_changes_when_the_audio_would_change(self, mutate):
        base = two_phrase_score()
        edited = base.model_copy(deep=True)
        mutate(edited.notes[0])
        assert self._hash(edited) != self._hash(base)

    def test_changes_with_language_or_phone_set(self):
        base = score_of(note(0, 1)).ensure_ids()
        other_lang = score_of(note(0, 1), language="gd").ensure_ids()
        other_set = score_of(note(0, 1), phone_set="mfa_ipa/ga_v1").ensure_ids()
        assert self._hash(other_lang) != self._hash(base)
        assert self._hash(other_set) != self._hash(base)

    def test_relative_timing_within_a_phrase_matters(self):
        """Shifting the phrase is free; changing timing *inside* it is not."""
        a = score_of(note(0.0, 0.5), note(0.5, 1.0)).ensure_ids()
        b = score_of(note(0.0, 0.5), note(0.7, 1.2)).ensure_ids()
        assert self._hash(a) != self._hash(b)

    def test_empty_segment_rejected(self):
        with pytest.raises(ValueError, match="empty segment"):
            segment_content_hash([], language="en", phone_set="mfa_ipa/en_v1")


class TestCanonicalHashing:
    def test_key_order_does_not_matter(self):
        assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})

    def test_ipa_written_through_as_utf8(self):
        """Escaping would still be deterministic, but IPA in artifacts must stay legible."""
        assert "ɑ".encode() in canonical_json_bytes({"ph": "ɑ"})

    def test_nan_refused(self):
        with pytest.raises(ValueError):
            canonical_json_bytes({"x": float("nan")})

    def test_stable_across_interpreter_runs(self):
        """PYTHONHASHSEED must not reach the digest (sort_keys is what guarantees it)."""
        code = (
            "from signalml.hashing import sha256_json;"
            "print(sha256_json({'b':2,'a':[1,'ɑ'],'c':{'z':0,'y':1}}))"
        )
        digests = set()
        for seed in ("0", "1", "12345"):
            out = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, encoding="utf-8",
                env={**dict(__import__("os").environ), "PYTHONHASHSEED": seed},
                cwd=str(Path(__file__).resolve().parents[1]),
                check=True,
            )
            digests.add(out.stdout.strip())
        assert len(digests) == 1


# ------------------------------------------------------------------------- edit plans


class TestRerenderPlan:
    def test_editing_one_phrase_rerenders_only_that_phrase(self):
        old = two_phrase_score()
        new = old.model_copy(deep=True)
        new.notes[3].syllable = "moon"  # second phrase
        plan = plan_rerender(old, new)
        assert plan.render == [1]
        assert plan.reuse == [(0, 0)]
        assert plan.dropped == [1]

    def test_unchanged_score_renders_nothing(self):
        plan = plan_rerender(two_phrase_score(), two_phrase_score())
        assert plan.render == []
        assert plan.reuse_fraction == 1.0

    def test_repeated_phrases_share_one_render(self):
        """A chorus line sung twice is rendered once — the cache is keyed on content."""
        chorus = [note(0.0, 0.5, midi=67, syllable="shine"),
                  note(0.5, 1.0, midi=69, syllable="on")]
        repeat = [note(4.0, 4.5, midi=67, syllable="shine"),
                  note(4.5, 5.0, midi=69, syllable="on")]
        score = score_of(*chorus, *repeat).ensure_ids()
        segments = split_segments(score)
        assert len(segments) == 2
        assert segments[0].content_hash == segments[1].content_hash

    def test_insertion_at_the_top_does_not_invalidate_the_rest(self):
        """Matching is by content, not position — this is why a diff beats an index."""
        old = two_phrase_score()
        new = old.model_copy(deep=True)
        new.notes.insert(0, note(-0.0, 0.2, syllable="oh", id="n0099"))
        for n in new.notes[1:]:
            n.start += 1.0
            n.end += 1.0
        plan = plan_rerender(old, new)
        assert len(plan.render) == 1  # only the new phrase
        assert len(plan.reuse) == 2

    def test_summary_reads_cleanly(self):
        plan = plan_rerender(two_phrase_score(), two_phrase_score())
        assert "2/2 segments reused (100%)" in plan.summary()


# --------------------------------------------------------------- render addressing


def inputs(**overrides) -> RenderInputs:
    base = dict(
        segment_hash="a" * 64,
        voice="aurora",
        embedding_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        audio_profile="prod",
        seed=1234,
        config_sha256="d" * 64,
    )
    base.update(overrides)
    return RenderInputs(**base)


class TestRenderKey:
    def test_deterministic(self):
        assert inputs().cache_key() == inputs().cache_key()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("segment_hash", "f" * 64),
            ("voice", "other"),
            ("embedding_sha256", "e" * 64),
            ("checkpoint_sha256", "e" * 64),
            ("audio_profile", "dev"),
            ("seed", 9999),
            ("config_sha256", "e" * 64),
        ],
    )
    def test_every_input_reaches_the_key(self, field, value):
        """An input that misses the key is a silent stale-audio bug."""
        assert inputs(**{field: value}).cache_key() != inputs().cache_key()

    def test_audio_profile_separates_dev_from_prod(self):
        """Q11: profiles never mix — a dev render must not answer a prod request."""
        assert (inputs(audio_profile="dev").cache_key()
                != inputs(audio_profile="prod").cache_key())

    def test_edited_variance_changes_the_key(self):
        """The repair path caches like any other render instead of colliding with it."""
        track = VarianceTrack(
            origin="edited", sample_rate=44100, frame_hop=512,
            phonemes=["ʃ", "aɪ", "n"], durations_sec=[0.1, 0.3, 0.1],
            f0_hz=[440.0, 441.0], voiced=[True, True],
        )
        edited = inputs().with_variance(track)
        assert edited.variance_sha256 == track.content_hash()
        assert edited.cache_key() != inputs().cache_key()

    def test_variance_annotations_do_not_change_its_hash(self):
        common = dict(sample_rate=44100, frame_hop=512, phonemes=["a"],
                      durations_sec=[0.5], f0_hz=[440.0], voiced=[True])
        a = VarianceTrack(origin="predicted", **common)
        b = VarianceTrack(origin="edited", notes="fixed the flat note", **common)
        assert a.content_hash() == b.content_hash()


class TestVarianceTrack:
    def test_duration_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="one duration per phoneme"):
            VarianceTrack(sample_rate=44100, frame_hop=512, phonemes=["a", "b"],
                          durations_sec=[0.1], f0_hz=[], voiced=[])

    def test_frame_array_mismatch_rejected(self):
        with pytest.raises(ValidationError, match="share a frame rate"):
            VarianceTrack(sample_rate=44100, frame_hop=512, phonemes=["a"],
                          durations_sec=[0.1], f0_hz=[440.0, 441.0], voiced=[True])

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError, match="negative phoneme duration"):
            VarianceTrack(sample_rate=44100, frame_hop=512, phonemes=["a"],
                          durations_sec=[-0.1], f0_hz=[], voiced=[])

    def test_round_trip_keeps_ipa_legible(self, tmp_path):
        track = VarianceTrack(sample_rate=44100, frame_hop=512,
                              phonemes=["ʃ", "aɪ"], durations_sec=[0.1, 0.2],
                              f0_hz=[440.0], voiced=[True])
        path = save_variance(tmp_path, "0" * 64, track)
        assert "ʃ" in path.read_text(encoding="utf-8")
        assert track.duration_sec == pytest.approx(0.3)


class TestRenderCache:
    def test_save_then_find(self, tmp_path):
        record = new_record(inputs(), score_id="sng_0042", segment_index=1,
                            note_ids=["n0003", "n0004"], artifacts={"audio": "audio.wav"})
        save_record(tmp_path, record)
        found = find_cached(tmp_path, inputs())
        assert found is not None
        assert found.score_id == "sng_0042"
        assert found.artifacts["audio"] == "audio.wav"

    def test_miss_on_changed_inputs(self, tmp_path):
        save_record(tmp_path, new_record(inputs()))
        assert find_cached(tmp_path, inputs(seed=4321)) is None

    def test_miss_on_empty_cache(self, tmp_path):
        assert find_cached(tmp_path, inputs()) is None

    def test_record_round_trips(self, tmp_path):
        record = new_record(inputs(), notes="dev preview")
        path = save_record(tmp_path, record)
        assert load_record(path) == record

    def test_key_prefix_collision_is_detected_not_served(self, tmp_path):
        """Dirs are named by a key prefix; the full key in the record is the check."""
        other = inputs(seed=4321)
        record = new_record(inputs())
        target = render_dir(tmp_path, other.cache_key())
        target.mkdir(parents=True)
        (target / RECORD_FILENAME).write_text(record.model_dump_json(), encoding="utf-8")
        assert find_cached(tmp_path, other) is None

    def test_hand_edited_record_rejected(self):
        with pytest.raises(ValidationError, match="does not match its inputs"):
            RenderRecord(key="0" * 64, inputs=inputs(), created="2026-09-11T00:00:00+00:00")

    def test_short_key_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="too short"):
            render_dir(tmp_path, "abc")


# ------------------------------------------------------------------------------- CLI


class TestScoreSegmentsCLI:
    def _write(self, tmp_path: Path, score: Score, name: str = "score.json") -> Path:
        path = tmp_path / name
        save_score(score, path)
        return path

    def test_lists_segments(self, tmp_path, capsys):
        path = self._write(tmp_path, two_phrase_score())
        assert cli_main(["score", "segments", str(path)]) == 0
        out = capsys.readouterr().out
        assert "2 segments" in out
        assert "shine on" in out

    def test_json_output_is_machine_readable(self, tmp_path, capsys):
        path = self._write(tmp_path, two_phrase_score())
        assert cli_main(["score", "segments", str(path), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert [s["index"] for s in payload] == [0, 1]
        assert len(payload[0]["content_hash"]) == 64

    def test_min_rest_flag(self, tmp_path, capsys):
        path = self._write(tmp_path, two_phrase_score())
        assert cli_main(["score", "segments", str(path), "--min-rest", "1.0"]) == 0
        assert "1 segments" in capsys.readouterr().out

    def test_compare_reports_the_rerender_cost(self, tmp_path, capsys):
        old = two_phrase_score()
        new = old.model_copy(deep=True)
        new.notes[3].syllable = "moon"
        old_path = self._write(tmp_path, old, "old.json")
        new_path = self._write(tmp_path, new, "new.json")
        assert cli_main(
            ["score", "segments", str(old_path), "--compare", str(new_path)]
        ) == 0
        out = capsys.readouterr().out
        assert "1/2 segments reused" in out
        assert "RENDER  segment 1" in out
