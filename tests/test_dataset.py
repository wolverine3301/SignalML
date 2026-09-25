"""S6b contract tests — offline. The dataset is a pure function of manifest+recipe:
clips cut at aligned silence, trainer raw format + dictionary + config + card out."""

from __future__ import annotations

import csv
import json

import pytest
import soundfile as sf
import yaml

from signalml.config import CONFIGS_DIR, active_profile
from signalml.manifest import Manifest, scan_directory
from signalml.score.phoneset import get_phone_set
from signalml.stages.common import song_dir, update_analysis
from signalml.stages.dataset import (
    SP,
    DatasetRecipe,
    SegmentationCfg,
    build,
    load_dataset_recipe,
    segment_phones,
    vowel_onset_groups,
    write_variance_config,
)

DEV_SR = active_profile("dev").sample_rate

PHONES = [
    {"ph": "ʃ", "start": 0.5, "end": 0.8, "word": "shine", "stress": None},
    {"ph": "aj", "start": 0.8, "end": 1.2, "word": "shine", "stress": None},
    {"ph": "n", "start": 1.25, "end": 1.6, "word": "shine", "stress": None},
    {"ph": "s", "start": 2.6, "end": 3.0, "word": "stay", "stress": None},
    {"ph": "t", "start": 3.0, "end": 3.4, "word": "stay", "stress": None},
]

SEG = SegmentationCfg(split_gap_sec=0.4, min_clip_sec=0.5, max_clip_sec=15.0,
                      pad_sec=0.15, inner_sp_min_sec=0.05)


def recipe(**overrides) -> DatasetRecipe:
    base = {
        "name": "t1", "trainer": "acoustic", "profile": "dev",
        "filters": {"min_align_score": 0.8, "exclude_processing": ["heavy"]},
        "segmentation": SEG.model_dump(),
        "test_clips_per_speaker": 1,
    }
    base.update(overrides)
    return DatasetRecipe.model_validate(base)


def _ready_song(root, make_wav, *, singer="alice", score=0.9, phones=PHONES,
                profile="dev", processing=None):
    hz = 200.0 + sum(ord(c) for c in singer) % 300  # unique checksum per singer
    make_wav(root / "raw" / f"{singer}_song.wav", seconds=1.0, hz=hz)
    manifest, new = scan_directory(root, language="en", gender="F", singer=singer)
    rec = new[0]
    rec.status.separated = rec.status.cleaned = rec.status.aligned = True
    rec.quality.align_score = score
    rec.meta.processing = processing
    rec.meta.license_note = "personal research use"
    manifest.upsert(rec)
    manifest.save()

    sdir = song_dir(root, rec.id)
    make_wav(sdir / "clean" / "vocals.wav", seconds=4.0)
    update_analysis(sdir, "clean", {"profile": profile})
    align_dir = sdir / "align"
    align_dir.mkdir(parents=True, exist_ok=True)
    (align_dir / "phones.json").write_text(json.dumps({
        "phone_set": "mfa_ipa/en_v1", "aligner": "mfa", "language": "en",
        "phones": phones,
    }, ensure_ascii=False), encoding="utf-8")
    return rec.id


class TestSegmentPhones:
    def test_splits_at_gap_and_wraps_sp(self):
        clips, dropped = segment_phones(PHONES, SEG, audio_len_sec=4.0)
        assert dropped == 0
        assert len(clips) == 2
        first = clips[0]
        assert first.tokens == ("SP", "ʃ", "aj", "SP", "n", "SP")
        assert first.start == pytest.approx(0.35)
        assert first.end == pytest.approx(1.75)
        assert sum(first.durations) == pytest.approx(first.end - first.start)
        assert clips[1].tokens == ("SP", "s", "t", "SP")

    def test_max_clip_forces_split(self):
        long_phones = [{"ph": "aj", "start": i * 1.0, "end": i * 1.0 + 0.9}
                       for i in range(6)]  # 0.1s gaps only, ~5.9s total
        cfg = SEG.model_copy(update={"max_clip_sec": 2.0, "inner_sp_min_sec": 0.05})
        clips, _ = segment_phones(long_phones, cfg, audio_len_sec=10.0)
        assert len(clips) >= 2
        assert all(c.end - c.start <= 2.0 + 2 * cfg.pad_sec + 0.2 for c in clips)

    def test_noise_clip_dropped(self):
        phones = [dict(PHONES[0]), {**PHONES[1], "noise": True}]
        clips, dropped = segment_phones(phones, SEG, audio_len_sec=4.0)
        assert clips == [] and dropped == 1

    def test_too_short_dropped(self):
        phones = [{"ph": "aj", "start": 1.0, "end": 1.2}]
        clips, dropped = segment_phones(phones, SEG, audio_len_sec=4.0)
        assert clips == [] and dropped == 1



class TestVarianceConfig:
    """A variance dataset IS the acoustic dataset plus columns, so this writes a
    config rather than copying audio — and refuses when the columns are not there."""

    def _built(self, tmp_path, make_wav, *, notes=False):
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        summary = build(root, recipe=recipe())
        if notes:
            for folder in summary.out_dir.glob("*-en"):
                path = folder / "transcriptions.csv"
                rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=[*rows[0], "note_seq", "note_dur"])
                    w.writeheader()
                    for row in rows:
                        total = sum(float(d) for d in row["ph_dur"].split())
                        w.writerow({**row, "note_seq": "rest C4+12",
                                    "note_dur": f"0.1 {total - 0.1:.6f}"})
        return summary.out_dir

    def test_writes_a_config_that_extends_the_acoustic_one(self, tmp_path, make_wav):
        out = self._built(tmp_path, make_wav, notes=True)
        got = write_variance_config(out)
        config = yaml.safe_load(got.path.read_text(encoding="utf-8"))

        assert config["base_config"] == ["configs/variance.yaml"]
        assert config["predict_dur"] is True and config["predict_pitch"] is True
        # same raw data, different binarized output: different binarizer, different
        # tensors, and one directory for both would silently mix them
        acoustic = yaml.safe_load((out / "config_acoustic.yaml").read_text(encoding="utf-8"))
        assert config["datasets"] == acoustic["datasets"]
        assert config["binary_data_dir"] != acoustic["binary_data_dir"]
        # the audio contract is inherited, never re-derived (D5)
        for key in ("audio_sample_rate", "hop_size", "fft_size", "win_size"):
            assert config[key] == acoustic[key]
        # features needing the NN harmonic-noise separator stay off
        assert config["predict_breathiness"] is False
        assert config["predict_voicing"] is False

    def test_refuses_pitch_without_note_columns(self, tmp_path, make_wav):
        out = self._built(tmp_path, make_wav, notes=False)
        with pytest.raises(RuntimeError, match="note_seq"):
            write_variance_config(out)

    def test_duration_only_works_without_notes(self, tmp_path, make_wav):
        """Their own table: duration prediction needs ph_num, not note_seq."""
        out = self._built(tmp_path, make_wav, notes=False)
        got = write_variance_config(out, predict_pitch=False)
        config = yaml.safe_load(got.path.read_text(encoding="utf-8"))
        assert config["predict_pitch"] is False and config["predict_dur"] is True

    def test_refuses_to_clobber_without_force(self, tmp_path, make_wav):
        out = self._built(tmp_path, make_wav, notes=True)
        write_variance_config(out)
        with pytest.raises(FileExistsError):
            write_variance_config(out)
        assert write_variance_config(out, force=True).path.exists()

    def test_unbuilt_dataset_points_at_dataset_build(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dataset build"):
            write_variance_config(tmp_path / "nope")

    def test_trainer_opts_size_the_run(self, tmp_path, make_wav):
        out = self._built(tmp_path, make_wav, notes=True)
        from signalml.stages.dataset import TrainerOpts
        got = write_variance_config(out, trainer_opts=TrainerOpts(
            max_batch_frames=40000, max_batch_size=24, permanent_ckpt_start=2000,
            permanent_ckpt_interval=2000))
        config = yaml.safe_load(got.path.read_text(encoding="utf-8"))
        assert config["max_batch_frames"] == 40000
        assert config["permanent_ckpt_start"] == 2000


class TestBuild:
    def test_end_to_end(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        summary = build(root, recipe=recipe())
        assert summary.clips == 2 and not summary.skipped

        folder = summary.out_dir / "alice-en"
        with open(folder / "transcriptions.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2
        assert rows[0]["ph_seq"].startswith("SP ʃ aj")
        # durations must match the written wav exactly
        wav_path = folder / "wavs" / f"{rows[0]['name']}.wav"
        info = sf.info(str(wav_path))
        assert info.samplerate == DEV_SR
        durations = [float(d) for d in rows[0]["ph_dur"].split()]
        assert sum(durations) == pytest.approx(info.frames / info.samplerate, abs=1e-6)

        dictionary = (summary.out_dir / "dictionary_en.txt").read_text(encoding="utf-8")
        assert "ʃ\tʃ" in dictionary and "SP" not in dictionary

        config = yaml.safe_load(
            (summary.out_dir / "config_acoustic.yaml").read_text(encoding="utf-8"))
        assert config["use_spk_id"] is True and config["num_spk"] == 1
        assert config["datasets"][0]["speaker"] == "alice"
        assert config["datasets"][0]["spk_id"] == 0
        assert config["audio_sample_rate"] == DEV_SR
        assert config["val_with_vocoder"] is False  # dev profile: no matching vocoder

        card = (summary.out_dir / "dataset_card.md").read_text(encoding="utf-8")
        assert "alice" in card and "personal research use" in card

    def test_global_breath_token_is_merged_into_silence(self, tmp_path, make_wav):
        """AP and SP always exist in their phoneme set, and their binarizer refuses
        a dataset that never uses one. We do not detect breaths yet."""
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        summary = build(root, recipe=recipe())
        config = yaml.safe_load(
            (summary.out_dir / "config_acoustic.yaml").read_text(encoding="utf-8"))
        assert config["merged_phoneme_groups"] == [["AP", "SP"]]
        # hn-sep defaults to WORLD: 'vr' loads an NN checkpoint during binarization
        # even though no breathiness/voicing/tension embed consumes it
        assert config["hnsep"] == "world" and "hnsep_ckpt" not in config

    def test_breath_tokens_end_the_merge(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        phones = [dict(p) for p in PHONES]
        phones[1] = {**phones[1], "ph": "AP"}
        _ready_song(root, make_wav, phones=phones)
        summary = build(root, recipe=recipe())
        config = yaml.safe_load(
            (summary.out_dir / "config_acoustic.yaml").read_text(encoding="utf-8"))
        assert config["merged_phoneme_groups"] == []

    def test_trainer_opts_reach_the_generated_config(self, tmp_path, make_wav):
        """Batch sizing is a property of the box, so it belongs in the recipe."""
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        summary = build(root, recipe=recipe(trainer_opts={
            "max_batch_frames": 80000, "max_batch_size": 64,
            "binarization_workers": 8, "num_ckpt_keep": 3, "max_updates": 20000,
            "hnsep": "vr", "hnsep_ckpt": "checkpoints/vr/model.pt",
            "permanent_ckpt_start": 1000, "permanent_ckpt_interval": 1000,
            "val_with_vocoder": True, "extra": {"lr": 0.0004},
        }))
        config = yaml.safe_load(
            (summary.out_dir / "config_acoustic.yaml").read_text(encoding="utf-8"))
        assert config["max_batch_frames"] == 80000
        assert config["max_batch_size"] == 64
        assert config["binarization_args"]["num_workers"] == 8
        assert config["num_ckpt_keep"] == 3 and config["max_updates"] == 20000
        # rolling window + a permanent ladder: the rolling one alone deleted the best
        # checkpoint of the 2026-09-20 run
        assert config["permanent_ckpt_start"] == 1000
        assert config["permanent_ckpt_interval"] == 1000
        # explicit override beats the profile-derived default (dev would be False)
        assert config["val_with_vocoder"] is True
        assert config["lr"] == 0.0004
        assert config["hnsep"] == "vr"
        assert config["hnsep_ckpt"] == "checkpoints/vr/model.pt"

    def test_unset_trainer_opts_are_left_to_the_vendored_config(self, tmp_path,
                                                                make_wav):
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        summary = build(root, recipe=recipe())
        config = yaml.safe_load(
            (summary.out_dir / "config_acoustic.yaml").read_text(encoding="utf-8"))
        assert "max_updates" not in config and "num_ckpt_keep" not in config
        assert "permanent_ckpt_start" not in config

    def test_extra_cannot_overwrite_the_audio_contract(self):
        """D5: mel params come from the profile or the run is silently corrupt."""
        with pytest.raises(ValueError, match="audio_sample_rate"):
            recipe(trainer_opts={"extra": {"audio_sample_rate": 22050}})

    def test_shipped_rig_recipes_are_valid(self):
        for name in ("dataset.overfit.yaml", "dataset.full_v2.yaml"):
            r = load_dataset_recipe(CONFIGS_DIR / name)
            assert r.profile == "prod"  # the rig trains at prod, dev is throwaway
        overfit_opts = load_dataset_recipe(CONFIGS_DIR / "dataset.overfit.yaml").trainer_opts
        full_opts = load_dataset_recipe(CONFIGS_DIR / "dataset.full_v2.yaml").trainer_opts
        # measured on the 24 GB rig: above ~20000 frames at prod the batch spills out of
        # VRAM into system RAM and a step takes seconds instead of milliseconds
        for opts in (overfit_opts, full_opts):
            assert opts.max_batch_frames <= 20000
        # both keep a permanent checkpoint ladder, whatever the rolling window does
        for opts in (overfit_opts, full_opts):
            assert opts.permanent_ckpt_start and opts.permanent_ckpt_interval
        # tracked recipes never name the private corpus: singer lists live in the
        # gitignored .local.yaml overlays, so the public versions carry none
        for name in ("dataset.overfit.yaml", "dataset.full_v2.yaml"):
            public = load_dataset_recipe(CONFIGS_DIR / name, local=False)
            assert public.filters.singers == [] and public.filters.exclude_singers == []
        full = load_dataset_recipe(CONFIGS_DIR / "dataset.full_v2.yaml", local=False)
        assert full.filters.min_singer_minutes == 0.0

    def test_local_overlay_deep_merges(self, tmp_path):
        base = tmp_path / "dataset.x.yaml"
        base.write_text(
            "name: x\n"
            "filters:\n  gender: F\n  singers: []\n"
            "trainer_opts:\n  max_batch_frames: 100\n", encoding="utf-8")
        assert load_dataset_recipe(base).filters.singers == []
        (tmp_path / "dataset.x.local.yaml").write_text(
            "filters:\n  singers: [ann, bo]\n", encoding="utf-8")
        merged = load_dataset_recipe(base)
        assert merged.filters.singers == ["ann", "bo"] and merged.filters.gender == "F"
        assert merged.trainer_opts.max_batch_frames == 100
        assert load_dataset_recipe(base, local=False).filters.singers == []

    def test_ph_num_word_division(self, tmp_path, make_wav):
        """ph_num is phones-per-word: required for variance duration prediction and
        by SOME's dataset mode, and it must sum to the ph_seq length or the trainer
        rejects the row."""
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        summary = build(root, recipe=recipe())
        with open(summary.out_dir / "alice-en" / "transcriptions.csv",
                  encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "no clips"
        for row in rows:
            phones = row["ph_seq"].split()
            counts = [int(n) for n in row["ph_num"].split()]
            assert sum(counts) == len(phones)
            assert all(n >= 1 for n in counts)
            # every group is either one SP, or a run of real phones: a silence always
            # closes a word, even mid-word (the fixture's 'shine' has a 50 ms gap in
            # it, so it groups as 2 + SP + 1, and that is the honest reading)
            i = 0
            for count in counts:
                group = phones[i:i + count]
                assert group, "empty word group"
                if SP in group:
                    assert group == [SP], f"SP must be its own word, got {group}"
                i += count
        # a word whose phones are contiguous stays one group
        assert any(int(n) > 1 for row in rows for n in row["ph_num"].split())

    def test_vowel_onset_groups(self):
        """One group per note onset: the variance model's 'word' is the phones within
        a note, and a note lands on its vowel. A two-syllable dictionary word becomes
        two groups; SP opens its own and takes the next onset consonant with it."""
        nucleus = {"ɪ", "ə", "aj", "ej"}.__contains__
        toks = ["SP", "b", "ɪ", "j", "ə", "SP", "s", "t", "ej", "SP"]
        assert vowel_onset_groups(toks, nucleus) == (2, 2, 1, 3, 1, 1)
        assert vowel_onset_groups(["s", "t", "ej"], nucleus) == (2, 1)  # leading onset
        assert sum(vowel_onset_groups(toks, nucleus)) == len(toks)

    def test_ph_num_mode_vowel_onset_in_a_build(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        seg = {**recipe().segmentation.model_dump(), "ph_num_mode": "vowel_onset"}
        summary = build(root, recipe=recipe(segmentation=seg))
        with open(summary.out_dir / "alice-en" / "transcriptions.csv",
                  encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows
        for row in rows:
            phones, counts = row["ph_seq"].split(), [int(n) for n in row["ph_num"].split()]
            assert sum(counts) == len(phones)
            i = 0
            for count in counts:  # every group opens on SP or a nucleus
                head = phones[i]
                assert head == SP or get_phone_set("mfa_ipa/en_v1").is_nucleus(head) \
                    or i == 0, f"group opens on consonant {head!r}"
                i += count

    def test_variance_build_explains_the_transcriber_pass(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        with pytest.raises(NotImplementedError, match="batch_infer"):
            build(root, recipe=recipe(trainer="variance"))

    def test_filters_and_reasons(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        _ready_song(root, make_wav, singer="alice")
        _ready_song(root, make_wav, singer="bob", processing="heavy")
        _ready_song(root, make_wav, singer="cara", score=0.5)
        _ready_song(root, make_wav, singer="dana", profile="prod")
        summary = build(root, recipe=recipe())
        assert len(summary.songs_used) == 1
        reasons = " | ".join(summary.skipped.values())
        assert "'heavy' excluded" in reasons
        assert "align_score 0.5" in reasons
        assert "clean profile 'prod'" in reasons and "re-run" in reasons

    def test_min_singer_minutes_drops_thin_speakers(self, tmp_path, make_wav):
        """The floor is measured on clip seconds and drops a speaker's songs together."""
        root = tmp_path / "dr"
        _ready_song(root, make_wav, singer="alice")
        _ready_song(root, make_wav, singer="bob")
        # PHONES yields two clips of ~0.75s and ~1.1s -> well under a minute each
        summary = build(root, recipe=recipe(filters={
            "min_align_score": 0.8, "min_singer_minutes": 1.0}))
        assert not summary.songs_used
        assert len(summary.skipped) == 2
        reasons = " | ".join(summary.skipped.values())
        assert "min_singer_minutes 1.0" in reasons and "min of clips" in reasons
        # a zero floor is the default and keeps everyone
        summary = build(root, recipe=recipe(), force=True)
        assert len(summary.songs_used) == 2

    def test_refuses_existing_without_force(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        _ready_song(root, make_wav)
        build(root, recipe=recipe())
        with pytest.raises(FileExistsError, match="force"):
            build(root, recipe=recipe())
        summary = build(root, recipe=recipe(), force=True)
        assert summary.clips == 2

    def test_variance_blocked_on_d1(self, tmp_path):
        with pytest.raises(NotImplementedError, match="D1"):
            build(tmp_path, recipe=recipe(trainer="variance"))

    def test_male_singer_excluded_by_gender_filter(self, tmp_path, make_wav):
        """The female-only guarantee: a correctly tagged male record never reaches the
        trainer, and the card says so (mixed-gender corpora, e.g. MedleyDB)."""
        root = tmp_path / "dr"
        female = _ready_song(root, make_wav, singer="alice")
        male = _ready_song(root, make_wav, singer="bob")
        manifest = Manifest.for_data_root(root)
        rec = manifest.get(male)
        rec.meta.gender = "M"
        manifest.upsert(rec)
        manifest.save()

        summary = build(root, recipe=recipe())
        assert summary.songs_used == [female]
        assert summary.skipped[male] == "gender 'M' != 'F'"
        card = (summary.out_dir / "dataset_card.md").read_text(encoding="utf-8")
        assert "gender: F" in card and "bob" not in card

    def test_corpus_scoping(self, tmp_path, make_wav):
        """Recipes select a corpus, a combination, or everything-but — the knob for
        'train on VocalSet + my own material but not the NC corpora'."""
        root = tmp_path / "dr"
        own = _ready_song(root, make_wav, singer="alice")
        borrowed = _ready_song(root, make_wav, singer="bob")
        manifest = Manifest.for_data_root(root)
        for rid, corpus in ((own, "own"), (borrowed, "medleydb")):
            rec = manifest.get(rid)
            rec.meta.corpus = corpus
            manifest.upsert(rec)
        manifest.save()

        summary = build(root, recipe=recipe(
            name="t_only", filters={"corpora": ["own"]}))
        assert summary.songs_used == [own]
        assert "not in recipe corpora" in summary.skipped[borrowed]

        summary = build(root, recipe=recipe(
            name="t_excl", filters={"exclude_corpora": ["medleydb"]}))
        assert summary.songs_used == [own]
        assert summary.skipped[borrowed] == "corpus 'medleydb' excluded by recipe"

        summary = build(root, recipe=recipe(name="t_both"))  # no corpus filter
        assert sorted(summary.songs_used) == sorted([own, borrowed])
        card = (summary.out_dir / "dataset_card.md").read_text(encoding="utf-8")
        assert "- own" in card and "- medleydb" in card

    def test_exclude_singers(self, tmp_path, make_wav):
        """A tag that pools several real vocalists (a producer credited as the singer)
        is dropped by name, all of its songs, and the skip says why."""
        root = tmp_path / "dr"
        keep = _ready_song(root, make_wav, singer="alice")
        pooled = _ready_song(root, make_wav, singer="some producer")
        summary = build(root, recipe=recipe(
            name="t_excl_spk", filters={"exclude_singers": ["some producer"]}))
        assert summary.songs_used == [keep]
        assert summary.skipped[pooled] == "singer 'some producer' excluded by recipe"

    def test_machine_lyrics_switch(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        human = _ready_song(root, make_wav, singer="alice")
        asr = _ready_song(root, make_wav, singer="bea")
        manifest = Manifest.for_data_root(root)
        rec = manifest.get(asr)
        rec.meta.lyrics_source = "asr:large-v3"
        manifest.upsert(rec)
        manifest.save()
        both = build(root, recipe=recipe(name="t_all"))
        assert sorted(both.songs_used) == sorted([human, asr])
        only = build(root, recipe=recipe(name="t_h", filters={"machine_lyrics": False}))
        assert only.songs_used == [human]
        assert only.skipped[asr] == "machine lyrics (asr) excluded by recipe"

    def test_null_gender_refused(self, tmp_path, make_wav):
        root = tmp_path / "dr"
        make_wav(root / "raw" / "x.wav")
        manifest, new = scan_directory(root, language="en", singer="alice")
        rec = new[0]
        rec.status.cleaned = rec.status.aligned = True
        rec.quality.align_score = 0.9
        manifest.upsert(rec)
        manifest.save()
        summary = build(root, recipe=recipe())
        assert "gender is null" in summary.skipped[rec.id]


def test_load_recipe_default_file():
    r = load_dataset_recipe()  # configs/dataset.yaml must stay valid
    assert r.trainer == "acoustic"
    assert r.profile == "prod"
    assert "heavy" in r.filters.exclude_processing


def test_manifest_module_still_imports():
    # dataset.py replaced the P0 stub; make sure nothing else referenced it
    from signalml.stages import dataset  # noqa: F401

    assert Manifest is not None
