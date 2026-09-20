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
from signalml.stages.common import song_dir, update_analysis
from signalml.stages.dataset import (
    DatasetRecipe,
    SegmentationCfg,
    build,
    load_dataset_recipe,
    segment_phones,
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
            "val_with_vocoder": True, "extra": {"lr": 0.0004},
        }))
        config = yaml.safe_load(
            (summary.out_dir / "config_acoustic.yaml").read_text(encoding="utf-8"))
        assert config["max_batch_frames"] == 80000
        assert config["max_batch_size"] == 64
        assert config["binarization_args"]["num_workers"] == 8
        assert config["num_ckpt_keep"] == 3 and config["max_updates"] == 20000
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

    def test_extra_cannot_overwrite_the_audio_contract(self):
        """D5: mel params come from the profile or the run is silently corrupt."""
        with pytest.raises(ValueError, match="audio_sample_rate"):
            recipe(trainer_opts={"extra": {"audio_sample_rate": 22050}})

    def test_shipped_rig_recipes_are_valid(self):
        for name in ("dataset.overfit.yaml", "dataset.full_v2.yaml"):
            r = load_dataset_recipe(CONFIGS_DIR / name)
            assert r.profile == "prod"  # the rig trains at prod, dev is throwaway
            assert r.trainer_opts.max_batch_frames > 30000  # sized for the 5090
        overfit = load_dataset_recipe(CONFIGS_DIR / "dataset.overfit.yaml")
        assert len(overfit.filters.singers) == 3
        assert load_dataset_recipe(
            CONFIGS_DIR / "dataset.full_v2.yaml").filters.min_singer_minutes == 5.0

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
