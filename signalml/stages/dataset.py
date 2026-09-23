"""S6b dataset — manifest query -> vendored-trainer raw dataset + dataset card.

Contract: docs/PIPELINE_AND_CONTRACTS.md §S6b; integration target:
docs/notes/vendor_diffsinger.md (D2, adopt-their-world). A dataset is a *pure
function of the manifest + recipe*: filters select records, clips are cut at
aligned silence gaps (phones.json is the authority, so clips always start/end in
silence), and the output is exactly what the vendored binarizer consumes:

    datasets/<name>/<singer>-<lang>/wavs/*.wav + transcriptions.csv
    datasets/<name>/{dictionary.txt, config_acoustic.yaml, dataset_card.md}

Refusals are per-song and loud: null gender, below-threshold alignment, profile
mismatch (clips must come from the recipe's audio profile end-to-end, Q11).
``trainer: variance`` is blocked until D1 lands note labels (note_seq/note_dur).
Phoneme tokens are IPA straight from phones.json plus SP (silence) — the D3
smoke test passed on 2026-09-20: the vendored binarizer accepts all 87 IPA names,
so the ASCII-transliteration fallback stays unused. AP (breath) is a global token
we never emit, so the generated config merges it into SP (see
``_write_trainer_config``).
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import soundfile as sf
import yaml
from pydantic import BaseModel, Field, field_validator

from ..config import CONFIGS_DIR, active_profile
from ..manifest import Manifest, ManifestRecord
from .common import song_dir

SP = "SP"  # silence token (DiffSinger global phoneme)
AP = "AP"  # aspiration token (reserved; we do not detect breaths yet)


class DatasetFilters(BaseModel):
    domain: str = "sung"
    language: str = "en"
    gender: str | None = "F"
    # corpus scoping: [] = every corpus. `corpora` whitelists (one corpus, or a
    # combination); `exclude_corpora` blacklists (e.g. drop non-commercial sources
    # from a run that has to ship). Untagged records read as "(untagged)".
    corpora: list[str] = []
    exclude_corpora: list[str] = []
    min_align_score: float = 0.8
    exclude_processing: list[str] = ["heavy"]
    singers: list[str] = []  # empty = all singers
    # Drop a speaker whole when its *clipped* audio falls under this (0 = keep all).
    # Measured after segmentation, never on manifest duration: raw vocals are ~44%
    # silence and sub-minimum fragments, so a singer who reads as 5 minutes in the
    # manifest can be under 3 in the dataset.
    min_singer_minutes: float = 0.0
    ids: list[str] = []  # optional explicit whitelist (intersected with filters)


class SegmentationCfg(BaseModel):
    split_gap_sec: float = 0.4  # a phone gap this long ends a clip
    min_clip_sec: float = 2.0
    max_clip_sec: float = 15.0
    pad_sec: float = 0.15  # silence kept around each clip
    inner_sp_min_sec: float = 0.05  # smaller gaps merge into the previous phone
    drop_noise_clips: bool = True  # clips containing spn (alignment holes)


# The mel/audio contract is owned end-to-end by the active audio profile (D5,
# docs/notes/vendor_diffsinger.md). These keys are written from the profile and may
# never be hand-set in a recipe: a dataset binarized at one sample rate and trained
# under another fails silently, which is the exact bug class CLAUDE.md names.
PROFILE_OWNED_KEYS = frozenset({
    "audio_sample_rate", "audio_num_mel_bins", "hop_size", "fft_size", "win_size",
    "fmin", "fmax", "mel_base", "dictionaries", "datasets", "binary_data_dir",
    "num_spk", "use_spk_id", "num_lang", "use_lang_id",
})


class TrainerOpts(BaseModel):
    """Machine-shaped knobs for the generated trainer config.

    Everything here is about *the box the run happens on*, not about the data — the
    defaults are sized for an 8 GB card (the 2080S work PC) so a laptop build is
    runnable, and the rig recipes raise them. ``extra`` is the escape hatch for any
    other vendored-trainer key, merged last, with the audio contract fenced off.
    """

    max_batch_frames: int = 30000   # their acoustic template: 50000
    max_batch_size: int = 16        # their acoustic template: 64
    binarization_workers: int = 4
    pe: str = "parselmouth"         # no checkpoint needed; rmvpe is a quality upgrade
    # Harmonic-noise separation. Their base.yaml ships 'vr', which loads an NN
    # checkpoint eagerly during binarization even when nothing consumes its output
    # — and we train no breathiness/voicing/tension embeds, so nothing does. 'world'
    # is their documented default, needs no checkpoint, and stays lazy. Switch to
    # 'vr' (with hnsep_ckpt) the day those embeds go on.
    hnsep: str = "world"
    hnsep_ckpt: str | None = None
    vocoder: str = "NsfHifiGAN"
    # community NC checkpoint = dev preview ONLY (Q4); own vocoder replaces it
    vocoder_ckpt: str =         "checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt"
    # None = auto: vocoder validation only at prod (no dev-profile vocoder exists)
    val_with_vocoder: bool | None = None
    max_updates: int | None = None      # None = their config's value
    val_check_interval: int | None = None
    # num_ckpt_keep is a ROLLING window: the last N checkpoints, nothing else. A run
    # whose validation loss bottoms early (overfit_v1 hit its best at step 3000 of
    # 20000 on 2026-09-20) deletes the good checkpoint long before it finishes. The
    # permanent_* pair is their answer - every Nth checkpoint from `start` is kept
    # regardless - and their defaults (80000/20000) only bite in runs far longer than
    # ours. Cost is disk: an acoustic checkpoint is ~850 MB.
    num_ckpt_keep: int | None = None
    permanent_ckpt_start: int | None = None
    permanent_ckpt_interval: int | None = None
    extra: dict[str, object] = {}

    @field_validator("extra")
    @classmethod
    def _no_profile_keys(cls, v: dict[str, object]) -> dict[str, object]:
        clash = sorted(set(v) & PROFILE_OWNED_KEYS)
        if clash:
            raise ValueError(
                f"trainer_opts.extra may not set {clash}: those come from the audio "
                f"profile and the manifest (D5 mel contract). Change the profile or "
                f"the recipe filters instead."
            )
        return v


class DatasetRecipe(BaseModel):
    name: str
    trainer: Literal["acoustic", "variance"] = "acoustic"
    profile: str = "prod"  # clips must come from clean/ at this profile (Q11)
    filters: DatasetFilters = Field(default_factory=DatasetFilters)
    segmentation: SegmentationCfg = Field(default_factory=SegmentationCfg)
    test_clips_per_speaker: int = 2
    trainer_opts: TrainerOpts = Field(default_factory=TrainerOpts)


def load_dataset_recipe(path: str | Path | None = None) -> DatasetRecipe:
    path = Path(path) if path else CONFIGS_DIR / "dataset.yaml"
    return DatasetRecipe.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    )


@dataclass
class BuildSummary:
    out_dir: Path
    clips: int = 0
    seconds: float = 0.0
    songs_used: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # id -> reason
    dropped_clips: int = 0  # noise/too-short/unsplittable


@dataclass(frozen=True)
class Clip:
    start: float  # in-song seconds, pad included
    end: float
    tokens: tuple[str, ...]
    durations: tuple[float, ...]
    has_noise: bool


def segment_phones(
    phones: list[dict], cfg: SegmentationCfg, *, audio_len_sec: float
) -> tuple[list[Clip], int]:
    """Cut aligned phones into training clips at silence gaps.

    Greedy: a gap >= split_gap_sec always splits; exceeding max_clip_sec splits at
    the best inner gap seen so far (>= inner_sp_min_sec), or force-splits at the
    current phone boundary as a last resort. Returns (clips, dropped_count) where
    dropped covers too-short and (optionally) noise-containing clips.
    """
    clips: list[Clip] = []
    dropped = 0
    group: list[dict] = []
    best_gap_idx: int | None = None  # index in group AFTER which the best gap sits
    best_gap = 0.0

    def flush(g: list[dict]) -> None:
        nonlocal dropped
        if not g:
            return
        span = g[-1]["end"] - g[0]["start"]
        has_noise = any(p.get("noise") for p in g)
        if span < cfg.min_clip_sec or (cfg.drop_noise_clips and has_noise):
            dropped += 1
            return
        start = max(0.0, g[0]["start"] - cfg.pad_sec)
        end = min(audio_len_sec, g[-1]["end"] + cfg.pad_sec)
        tokens: list[str] = []
        durations: list[float] = []
        cursor = start
        for p in g:
            gap = p["start"] - cursor
            if gap >= cfg.inner_sp_min_sec:
                tokens.append(SP)
                durations.append(gap)
                cursor = p["start"]
            ph_end = max(p["end"], cursor)  # tiny gaps merge into this phone
            tokens.append(p["ph"])
            durations.append(ph_end - cursor)
            cursor = ph_end
        if end - cursor >= 1e-4:
            tokens.append(SP)
            durations.append(end - cursor)
        clips.append(Clip(start=start, end=end, tokens=tuple(tokens),
                          durations=tuple(durations), has_noise=has_noise))

    for p in phones:
        if not group:
            group = [p]
            best_gap_idx, best_gap = None, 0.0
            continue
        gap = p["start"] - group[-1]["end"]
        if gap >= cfg.split_gap_sec:
            flush(group)
            group = [p]
            best_gap_idx, best_gap = None, 0.0
            continue
        if gap >= cfg.inner_sp_min_sec and gap > best_gap:
            best_gap_idx, best_gap = len(group), gap
        span = p["end"] - group[0]["start"]
        if span > cfg.max_clip_sec:
            if best_gap_idx is not None:
                flush(group[:best_gap_idx])
                group = group[best_gap_idx:] + [p]
            else:  # no usable gap: force split before this phone
                flush(group)
                group = [p]
            best_gap_idx, best_gap = None, 0.0
            continue
        group.append(p)
    flush(group)
    return clips, dropped


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def _profile_of(sdir: Path) -> str | None:
    analysis = sdir / "analysis.json"
    if not analysis.exists():
        return None
    try:
        return json.loads(analysis.read_text(encoding="utf-8"))["clean"]["profile"]
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def _select(manifest: Manifest, recipe: DatasetRecipe, summary: BuildSummary,
            data_root: Path) -> list[ManifestRecord]:
    f = recipe.filters
    selected: list[ManifestRecord] = []
    for rec in manifest.records:
        reason = None
        if f.ids and rec.id not in f.ids:
            continue  # not even reported: an explicit whitelist is self-documenting
        if rec.meta.domain != f.domain:
            reason = f"domain {rec.meta.domain!r} != {f.domain!r}"
        elif rec.meta.language != f.language:
            reason = f"language {rec.meta.language!r} != {f.language!r}"
        elif f.gender and rec.meta.gender != f.gender:
            reason = "gender is null (tag it)" if rec.meta.gender is None else \
                f"gender {rec.meta.gender!r} != {f.gender!r}"
        elif f.corpora and rec.meta.corpus not in f.corpora:
            reason = (f"corpus {rec.meta.corpus or '(untagged)'!r} not in "
                      f"recipe corpora {f.corpora}")
        elif rec.meta.corpus in f.exclude_corpora:
            reason = f"corpus {rec.meta.corpus!r} excluded by recipe"
        elif f.singers and rec.meta.singer not in f.singers:
            reason = "singer not in recipe whitelist"
        elif rec.meta.singer is None:
            reason = "singer is null — the timbre space needs singer labels"
        elif rec.meta.processing in f.exclude_processing:
            reason = f"processing {rec.meta.processing!r} excluded by recipe"
        elif not (rec.status.cleaned and rec.status.aligned):
            reason = "not cleaned+aligned yet"
        elif rec.quality.align_score is None or \
                rec.quality.align_score < f.min_align_score:
            reason = f"align_score {rec.quality.align_score} < {f.min_align_score}"
        else:
            profile = _profile_of(song_dir(data_root, rec.id))
            if profile != recipe.profile:
                reason = (f"clean profile {profile!r} != recipe {recipe.profile!r} "
                          f"(re-run: signalml clean --profile {recipe.profile} --force, "
                          f"phones.json stays valid — timings are in seconds)")
        if reason:
            summary.skipped[rec.id] = reason
        else:
            selected.append(rec)
    return selected


def _cut_all(
    selected: list[ManifestRecord], recipe: DatasetRecipe, summary: BuildSummary,
    data_root: Path,
) -> tuple[list[ManifestRecord], dict[str, tuple[list[Clip], int]]]:
    """Segment every selected song up front, before a single wav is written.

    Clip time is the only honest measure of how much a speaker actually brings, and
    ``min_singer_minutes`` needs it before the write loop starts — otherwise a
    below-floor speaker's clips would be written and then have to be deleted. Reading
    the header (``sf.info``) instead of the samples keeps this pass cheap.
    """
    cut: dict[str, tuple[list[Clip], int]] = {}
    kept: list[ManifestRecord] = []
    for rec in selected:
        sdir = song_dir(data_root, rec.id)
        payload = json.loads(
            (sdir / "align" / "phones.json").read_text(encoding="utf-8"))
        info = sf.info(str(sdir / "clean" / "vocals.wav"))
        clips, dropped = segment_phones(
            payload["phones"], recipe.segmentation,
            audio_len_sec=info.frames / info.samplerate)
        if not clips:
            summary.skipped[rec.id] = "no usable clips after segmentation"
            continue
        cut[rec.id] = (clips, dropped)
        kept.append(rec)
    return kept, cut


def _drop_thin_singers(
    selected: list[ManifestRecord], cut: dict[str, tuple[list[Clip], int]],
    recipe: DatasetRecipe, summary: BuildSummary,
) -> list[ManifestRecord]:
    """Refuse speakers under ``min_singer_minutes``, all of their songs at once.

    A speaker with a couple of minutes still trains the shared acoustic backbone, but
    its embedding is noise-dominated — and for the voice bank (ARCHITECTURE §4) an
    undertrained timbre that renders as the corpus average is worse than an absent
    one, because the ECAPA novelty guard has nothing real to measure against.
    """
    floor = recipe.filters.min_singer_minutes
    if floor <= 0:
        return selected
    minutes: dict[str, float] = {}
    for rec in selected:
        clips, _ = cut[rec.id]
        minutes[rec.meta.singer] = minutes.get(rec.meta.singer, 0.0) + sum(
            c.end - c.start for c in clips) / 60
    kept = []
    for rec in selected:
        got = minutes[rec.meta.singer]
        if got < floor:
            summary.skipped[rec.id] = (
                f"singer {rec.meta.singer!r} has {got:.1f} min of clips across the "
                f"build < min_singer_minutes {floor}")
        else:
            kept.append(rec)
    return kept


def build(
    data_root: str | Path,
    *,
    recipe: DatasetRecipe | None = None,
    force: bool = False,
) -> BuildSummary:
    data_root = Path(data_root)
    recipe = recipe or load_dataset_recipe()
    if recipe.trainer == "variance":
        raise NotImplementedError(
            "variance datasets need note_seq/note_dur — blocked on DECISION_POINTS D1 "
            "(SOME/MakeDiffSinger note-annotation bake-off)"
        )

    out_dir = data_root / "datasets" / recipe.name
    summary = BuildSummary(out_dir=out_dir)
    if out_dir.exists():
        if not force:
            raise FileExistsError(
                f"{out_dir} exists — datasets are pure functions of manifest+recipe; "
                f"rebuild with --force (or pick a new recipe name)"
            )
        shutil.rmtree(out_dir)

    manifest = Manifest.for_data_root(data_root)
    selected = _select(manifest, recipe, summary, data_root)
    if not selected:
        return summary

    selected, cut = _cut_all(selected, recipe, summary, data_root)
    selected = _drop_thin_singers(selected, cut, recipe, summary)
    if not selected:
        return summary

    singers = sorted({rec.meta.singer for rec in selected})
    spk_ids = {s: i for i, s in enumerate(singers)}
    phones_used: set[str] = set()
    per_folder_rows: dict[str, list[tuple[str, str, str]]] = {}
    per_speaker_stats: dict[str, dict] = {
        s: {"clips": 0, "seconds": 0.0, "songs": 0,
            # carried into the card so a gender-filtered build is auditable after
            # the fact (mixed-gender corpora like MedleyDB tag gender per stem)
            "gender": next(r.meta.gender for r in selected if r.meta.singer == s)}
        for s in singers}
    licenses: set[str] = set()
    corpora: set[str] = set()
    align_scores: list[float] = []

    for rec in selected:
        sdir = song_dir(data_root, rec.id)
        audio, sr = sf.read(str(sdir / "clean" / "vocals.wav"), dtype="float32")
        clips, dropped = cut[rec.id]
        summary.dropped_clips += dropped

        folder = f"{_sanitize(rec.meta.singer)}-{rec.meta.language}"
        wavs_dir = out_dir / folder / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        for i, clip in enumerate(clips):
            name = f"{rec.id}_{i:03d}"
            lo, hi = int(round(clip.start * sr)), int(round(clip.end * sr))
            sf.write(str(wavs_dir / f"{name}.wav"), audio[lo:hi], sr,
                     subtype="PCM_16")
            # keep durations consistent with the written sample count
            durations = list(clip.durations)
            durations[-1] += (hi - lo) / sr - sum(durations)
            per_folder_rows.setdefault(folder, []).append((
                name,
                " ".join(clip.tokens),
                " ".join(f"{d:.6f}" for d in durations),
            ))
            phones_used.update(clip.tokens)
            per_speaker_stats[rec.meta.singer]["clips"] += 1
            per_speaker_stats[rec.meta.singer]["seconds"] += clip.end - clip.start
        per_speaker_stats[rec.meta.singer]["songs"] += 1
        summary.songs_used.append(rec.id)
        summary.clips += len(clips)
        summary.seconds += sum(c.end - c.start for c in clips)
        licenses.add(rec.meta.license_note or "(none recorded)")
        corpora.add(rec.meta.corpus or "(untagged)")
        align_scores.append(rec.quality.align_score)

    for folder, rows in per_folder_rows.items():
        with open(out_dir / folder / "transcriptions.csv", "w", newline="",
                  encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "ph_seq", "ph_dur"])
            writer.writerows(rows)

    _write_dictionary(out_dir, phones_used, recipe.filters.language)
    _write_trainer_config(out_dir, recipe, selected, spk_ids, per_folder_rows,
                          phones_used)
    _write_card(out_dir, recipe, summary, per_speaker_stats, spk_ids, licenses,
                corpora, align_scores)
    return summary


def _write_dictionary(out_dir: Path, phones_used: set[str], language: str) -> None:
    """Identity dictionary: phoneme-level input, each phone maps to itself (plus the
    global SP/AP tokens, which the trainer treats specially and must not be listed)."""
    phones = sorted(p for p in phones_used if p not in (SP, AP))
    lines = [f"{p}\t{p}" for p in phones]
    (out_dir / f"dictionary_{language}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _write_trainer_config(
    out_dir: Path,
    recipe: DatasetRecipe,
    selected: list[ManifestRecord],
    spk_ids: dict[str, int],
    per_folder_rows: dict[str, list[tuple[str, str, str]]],
    phones_used: set[str],
) -> None:
    profile = active_profile(recipe.profile)
    opts = recipe.trainer_opts
    lang = recipe.filters.language
    datasets_cfg = []
    for folder in sorted(per_folder_rows):
        singer = next(rec.meta.singer for rec in selected
                      if f"{_sanitize(rec.meta.singer)}-{rec.meta.language}" == folder)
        names = [row[0] for row in per_folder_rows[folder]]
        datasets_cfg.append({
            "raw_data_dir": (out_dir / folder).resolve().as_posix(),
            "speaker": _sanitize(singer),
            "spk_id": spk_ids[singer],
            "language": lang,
            "test_prefixes": names[: recipe.test_clips_per_speaker],
        })

    config = {
        "base_config": ["configs/acoustic.yaml"],
        "dictionaries": {
            lang: (out_dir / f"dictionary_{lang}.txt").resolve().as_posix()},
        "extra_phonemes": [],
        # AP (breath) and SP (silence) are *global* phonemes in their phoneme set:
        # both always exist, and their binarizer refuses a dataset that never uses
        # one ("phonemes are not covered in transcriptions"). We do not detect
        # breaths yet, so AP is merged into SP — breaths train as silence, which is
        # what they already are in our alignments. Undo the merge (and rebuild) the
        # day S4/S5 emits AP.
        "merged_phoneme_groups": ([] if AP in phones_used else [[AP, SP]]),
        "datasets": datasets_cfg,
        "binary_data_dir": (out_dir / "binary").resolve().as_posix(),
        "binarization_args": {"num_workers": opts.binarization_workers},
        "pe": opts.pe,
        "hnsep": opts.hnsep,
        "use_lang_id": False,
        "num_lang": 1,
        "use_spk_id": True,
        "num_spk": len(spk_ids),
        # audio contract comes from OUR profile (matches their defaults at prod, Q11)
        "audio_sample_rate": profile.sample_rate,
        "audio_num_mel_bins": profile.n_mels,
        "hop_size": profile.hop_length,
        "fft_size": profile.n_fft,
        "win_size": profile.win_length,
        "fmax": profile.fmax or profile.sample_rate // 2,
        "mel_base": "e",
        # community NC checkpoint = dev preview ONLY (Q4); own vocoder replaces it
        "vocoder": opts.vocoder,
        "vocoder_ckpt": opts.vocoder_ckpt,
        # dev profile has no matching vocoder checkpoint — skip vocoder validation
        "val_with_vocoder": (recipe.profile == "prod"
                             if opts.val_with_vocoder is None
                             else opts.val_with_vocoder),
        # batch sizing is a property of the box, not the data: defaults fit an 8 GB
        # card, the rig recipes raise them (their template defaults: 50000 / 64)
        "max_batch_frames": opts.max_batch_frames,
        "max_batch_size": opts.max_batch_size,
    }
    for key, value in (("hnsep_ckpt", opts.hnsep_ckpt),
                       ("max_updates", opts.max_updates),
                       ("val_check_interval", opts.val_check_interval),
                       ("num_ckpt_keep", opts.num_ckpt_keep),
                       ("permanent_ckpt_start", opts.permanent_ckpt_start),
                       ("permanent_ckpt_interval", opts.permanent_ckpt_interval)):
        if value is not None:
            config[key] = value
    config.update(opts.extra)  # validated against PROFILE_OWNED_KEYS at load time
    (out_dir / f"config_{recipe.trainer}.yaml").write_text(
        "# GENERATED by `signalml dataset build` — edit the recipe, not this file.\n"
        "# License note: the referenced community vocoder ckpt is CC BY-NC (dev\n"
        "# preview only, Q4); nothing rendered with it ships.\n"
        + yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8")


def _write_card(
    out_dir: Path,
    recipe: DatasetRecipe,
    summary: BuildSummary,
    per_speaker_stats: dict[str, dict],
    spk_ids: dict[str, int],
    licenses: set[str],
    corpora: set[str],
    align_scores: list[float],
) -> None:
    lines = [
        f"# Dataset card — {recipe.name}",
        f"\nBuilt {_dt.date.today().isoformat()} by `signalml dataset build` "
        f"(pure function of manifest + this recipe; rebuildable byte-for-byte).",
        f"\n- trainer: {recipe.trainer}   profile: **{recipe.profile}**",
        f"- songs: {len(summary.songs_used)}   clips: {summary.clips}   "
        f"hours: {summary.seconds / 3600:.2f}   dropped clips: {summary.dropped_clips}",
        f"- align_score range: {min(align_scores):.3f}–{max(align_scores):.3f}"
        if align_scores else "- align_score range: n/a",
        "\n## Recipe\n\n```yaml",
        yaml.safe_dump(recipe.model_dump(), allow_unicode=True, sort_keys=False).rstrip(),
        "```",
        "\n## Speakers (spk_id table — voice-bank identities)\n",
        "| spk_id | singer | gender | songs | clips | minutes |",
        "|---|---|---|---|---|---|",
    ]
    for singer, sid in sorted(spk_ids.items(), key=lambda kv: kv[1]):
        s = per_speaker_stats[singer]
        lines.append(f"| {sid} | {singer} | {s['gender'] or '?'} | {s['songs']} | "
                     f"{s['clips']} | {s['seconds'] / 60:.1f} |")
    lines += [
        "\n## Corpora\n",
        *[f"- {name}" for name in sorted(corpora)],
        "\n## License roll-up\n",
        *[f"- {note}" for note in sorted(licenses)],
        "\n## Skipped records\n",
        *([f"- {rid}: {reason}" for rid, reason in sorted(summary.skipped.items())]
          or ["- none"]),
        "",
    ]
    (out_dir / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")
