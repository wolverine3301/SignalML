"""S5 align — MFA (IPA phone set) -> align/phones.json; SOFA fallback.

Contract: docs/PIPELINE_AND_CONTRACTS.md §S5. Manifest-driven over cleaned songs with
lyrics sidecars; one batched MFA invocation per run (MFA startup is expensive), then a
per-song convert step: the aligner-native TextGrid is kept for audit at
``align/vocals.TextGrid`` and converted to the pipeline-native ``align/phones.json``
via praatio (replacing the buggy legacy hand parser — docs/CODE_SURVEY.md).

Downstream stages read *only* phones.json — swapping aligners (SOFA, Gaelic wave-2)
means one new converter/config, nothing downstream changes.

``quality.align_score`` (v1 heuristic, recorded as such): aligned-speech seconds over
the S4 silence-map's voiced seconds. Near 1.0 = the aligner accounted for everything
the energy detector calls speech; low values flag songs to exclude from training.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from json import dumps, loads
from pathlib import Path

import yaml
from pydantic import BaseModel

from ..config import CONFIGS_DIR
from ..manifest import Manifest, ManifestRecord
from ..score.phoneset import NOISE_MARKS, SILENCE_MARKS, get_phone_set
from .align_chunks import (
    Utterance,
    frame_level_db,
    plan_utterances,
    utterance_health,
    utterances_textgrid,
)
from .common import song_dir, update_analysis
from .lyrics import ASR_NAME
from .lyrics_resolve import load_resolution, plain_lyrics

Runner = Callable[[list[str]], object]


class AlignConfig(BaseModel):
    language: str = "en"
    phone_set: str = "mfa_ipa/en_v1"
    acoustic_model: str = "english_mfa"
    dictionary: str = "english_mfa"
    mfa_command: list[str] = ["conda", "run", "-n", "aligner", "--no-capture-output", "mfa"]
    beam: int = 100
    retry_beam: int = 400
    num_jobs: int = 4
    strict_phones: bool = True


def load_align_config(path: str | Path | None = None) -> AlignConfig:
    path = Path(path) if path else CONFIGS_DIR / "align.yaml"
    if path.exists():
        return AlignConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return AlignConfig()


@dataclass
class AlignSummary:
    aligned: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # not ready / already aligned / other language
    failed: dict[str, str] = field(default_factory=dict)


def textgrid_to_phones(
    tg_path: str | Path,
    *,
    phone_set_name: str,
    language: str,
    aligner: str = "mfa",
    strict: bool = True,
) -> dict:
    """MFA output TextGrid ('words' + 'phones' tiers) -> phones.json payload.

    Silence marks are dropped (silence is implicit in the time gaps); noise marks
    ("spn" = OOV/unknown speech) are kept and flagged — they mark alignment holes.
    Unknown phones fail loudly under ``strict`` so a phone-set/dictionary mismatch
    can't poison training data.
    """
    from praatio import textgrid as praatio_tg  # local import: keep CLI start light

    phone_set = get_phone_set(phone_set_name)
    tg = praatio_tg.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    # MFA names tiers "words"/"phones", or "<speaker> - words" for some inputs
    tier_names = {name.lower().rsplit(" - ", 1)[-1]: name for name in tg.tierNames}
    if "phones" not in tier_names:
        raise ValueError(f"{tg_path}: no 'phones' tier (tiers: {list(tg.tierNames)})")

    words: list[tuple[float, float, str]] = []
    if "words" in tier_names:
        words = [(e.start, e.end, e.label) for e in tg.getTier(tier_names["words"]).entries
                 if e.label not in SILENCE_MARKS]

    def word_at(t: float) -> str | None:
        for start, end, label in words:
            if start <= t < end:
                return label
        return None

    entries: list[dict] = []
    labels: list[str] = []
    for iv in tg.getTier(tier_names["phones"]).entries:
        label = iv.label.strip()
        if label in SILENCE_MARKS:
            continue
        labels.append(label)
        mid = (iv.start + iv.end) / 2
        entries.append({
            "ph": label,
            "start": round(iv.start, 4),
            "end": round(iv.end, 4),
            "word": word_at(mid),
            "stress": None,  # MFA IPA carries no stress; stress enters via the score (Q2)
            **({"noise": True} if label in NOISE_MARKS else {}),
        })

    unknown = phone_set.unknown(labels)
    if unknown and strict:
        raise ValueError(
            f"{tg_path}: phones outside {phone_set_name}: {unknown} — the aligner "
            f"dictionary and the declared phone set disagree "
            f"(check `signalml score phoneset --dict`)"
        )

    payload = {
        "phone_set": phone_set_name,
        "aligner": aligner,
        "language": language,
        "phones": entries,
    }
    if unknown:
        payload["unknown_phones"] = unknown
    return payload


def _default_runner(cmd: list[str]) -> object:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"could not launch {cmd[0]!r} — is conda (and the 'aligner' env) installed? "
            f"See README 'Alignment (MFA) install'."
        ) from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "")[-2000:]
        raise RuntimeError(f"mfa failed (exit {exc.returncode}):\n{tail}") from exc


def _voiced_sec(sdir: Path) -> float | None:
    analysis = sdir / "analysis.json"
    if not analysis.exists():
        return None
    data = loads(analysis.read_text(encoding="utf-8"))
    try:
        return float(data["clean"]["stems"]["vocals"]["silence"]["voiced_sec"])
    except (KeyError, TypeError, ValueError):
        return None


def _find_textgrid(out_dir: Path, song_id: str) -> Path | None:
    """MFA mirrors the corpus layout; accept both flat and speaker-subdir outputs."""
    for candidate in (out_dir / song_id / f"{song_id}.TextGrid",
                      out_dir / f"{song_id}.TextGrid"):
        if candidate.exists():
            return candidate
    return None


def _variant_suffix(variant: str | None) -> str:
    if variant is None:
        return ""
    if not re.fullmatch(r"[a-z0-9_]+", variant):
        raise ValueError(f"alignment variant {variant!r}: use [a-z0-9_] only")
    return f".{variant}"


def _stage_phrases(sdir: Path, rec: ManifestRecord, text: str, wav: Path,
                   spk: Path) -> list[Utterance]:
    """Plan phrase utterances and write MFA's TextGrid transcript beside the wav."""
    import soundfile as sf

    asr = sdir / "lyrics" / ASR_NAME
    if not asr.exists():
        raise FileNotFoundError(
            f"phrase alignment needs Whisper word times at {asr} "
            f"(signalml lyrics --check writes it)")
    payload = loads(asr.read_text(encoding="utf-8"))
    if "error" in payload:
        raise RuntimeError(f"asr.json holds an error: {payload['error']}")
    y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    duration = len(y) / sr
    utts = plan_utterances(text, payload.get("segments", []), frame_level_db(y, sr),
                           duration)
    if not utts:
        raise RuntimeError("no lyric lines to align")
    (spk / f"{rec.id}.TextGrid").write_text(
        utterances_textgrid(utts, duration, rec.id), encoding="utf-8")
    return utts


def align(
    data_root: str | Path,
    *,
    cfg: AlignConfig | None = None,
    force: bool = False,
    limit: int | None = None,
    runner: Runner | None = None,
    phrases: bool = False,
    variant: str | None = None,
    ids: list[str] | None = None,
) -> AlignSummary:
    """Align cleaned songs with MFA.

    ``phrases``: cut each song into phrase utterances anchored on Whisper word times
    (``align_chunks``) instead of one whole-song utterance; also writes
    ``align/utterances.json`` with per-phrase health for the dataset's phrase gate.
    Either mode reads ``lyrics/resolved.txt`` when ``lyrics --resolve`` made one;
    phrase mode otherwise strips brackets (MFA turns ``(...)`` into spn).

    ``variant``: write ``phones.<variant>.json`` (and ``vocals.<variant>.TextGrid``,
    ``utterances.<variant>.json``) beside the canonical files instead of replacing
    them, for side-by-side experiments; the manifest's status and align_score are left
    alone, and already-aligned songs are eligible.
    """
    data_root = Path(data_root)
    cfg = cfg or load_align_config()
    runner = runner or _default_runner
    manifest = Manifest.for_data_root(data_root)
    summary = AlignSummary()
    suffix = _variant_suffix(variant)

    work: list[ManifestRecord] = []
    for rec in manifest.records:
        if ids and rec.id not in ids:
            continue
        if variant is None:
            ready = rec.status.cleaned and (force or not rec.status.aligned)
        else:
            done = (song_dir(data_root, rec.id) / "align" / f"phones{suffix}.json").exists()
            ready = rec.status.cleaned and (force or not done)
        if not ready or rec.meta.language != cfg.language:
            summary.skipped.append(rec.id)
        elif not rec.meta.has_lyrics or not rec.meta.lyrics_path:
            summary.failed[rec.id] = "no lyrics .txt sidecar (Q14 says every song has one)"
        else:
            work.append(rec)
    if limit is not None:
        work = work[:limit]
    if not work:
        return summary

    with tempfile.TemporaryDirectory(prefix="signalml_mfa_", dir=str(data_root)) as tmp:
        corpus = Path(tmp) / "corpus"
        out_dir = Path(tmp) / "aligned"

        staged: list[ManifestRecord] = []
        plans: dict[str, tuple[list[Utterance], list[int], bool]] = {}
        for rec in work:
            try:
                sdir = song_dir(data_root, rec.id)
                wav = sdir / "clean" / "vocals.wav"
                if not wav.exists():
                    raise FileNotFoundError(f"clean vocal missing: {wav}")
                resolution = load_resolution(sdir)
                if resolution is not None:
                    lyrics, unsure = resolution
                else:
                    lyrics = (data_root / rec.meta.lyrics_path).read_text(encoding="utf-8")
                    lyrics, unsure = (plain_lyrics(lyrics) if phrases else lyrics), []
                spk = corpus / rec.id  # one speaker dir per song
                spk.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(wav, spk / f"{rec.id}.wav")
                if phrases:
                    utts = _stage_phrases(sdir, rec, lyrics, wav, spk)
                    plans[rec.id] = (utts, unsure, resolution is not None)
                else:
                    (spk / f"{rec.id}.lab").write_text(lyrics.strip() + "\n",
                                                       encoding="utf-8")
                staged.append(rec)
            except Exception as exc:
                summary.failed[rec.id] = str(exc)
        if not staged:
            return summary

        runner([
            *cfg.mfa_command, "align", "--clean",
            str(corpus), cfg.dictionary, cfg.acoustic_model, str(out_dir),
            "--beam", str(cfg.beam), "--retry_beam", str(cfg.retry_beam),
            "--num_jobs", str(cfg.num_jobs),
        ])

        for rec in staged:
            try:
                tg_src = _find_textgrid(out_dir, rec.id)
                if tg_src is None:
                    raise RuntimeError("MFA produced no TextGrid (song failed to align)")
                sdir = song_dir(data_root, rec.id)
                align_dir = sdir / "align"
                align_dir.mkdir(parents=True, exist_ok=True)
                tg_dst = align_dir / f"vocals{suffix}.TextGrid"
                shutil.copyfile(tg_src, tg_dst)

                payload = textgrid_to_phones(
                    tg_dst, phone_set_name=cfg.phone_set, language=cfg.language,
                    aligner="mfa", strict=cfg.strict_phones,
                )

                speech_sec = sum(
                    p["end"] - p["start"] for p in payload["phones"] if not p.get("noise")
                )
                voiced = _voiced_sec(sdir)
                score = round(min(1.0, speech_sec / voiced), 4) if voiced else None

                (align_dir / f"phones{suffix}.json").write_text(
                    dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                extra: dict = {"mode": "phrases" if phrases else "whole_song"}
                if phrases:
                    utts, unsure, resolved = plans[rec.id]
                    health = utterance_health(utts, payload["phones"], set(unsure))
                    (align_dir / f"utterances{suffix}.json").write_text(
                        dumps(health, ensure_ascii=False, indent=1), encoding="utf-8")
                    extra.update(n_utterances=len(utts), lyrics_resolved=resolved,
                                 n_unsure_utterances=sum(h["unsure_lyrics"] for h in health))
                update_analysis(sdir, f"align{suffix}", {
                    "aligner": "mfa",
                    "acoustic_model": cfg.acoustic_model,
                    "dictionary": cfg.dictionary,
                    "phone_set": cfg.phone_set,
                    **extra,
                    "n_phones": len(payload["phones"]),
                    "speech_sec": round(speech_sec, 3),
                    "align_score": score,
                    "align_score_method": "speech_sec / clean.silence_map.voiced_sec (v1)",
                    "date": _dt.date.today().isoformat(),
                })
                if variant is None:
                    rec.status.aligned = True
                    rec.quality.align_score = score
                    manifest.commit(rec)
                summary.aligned.append(rec.id)
            except Exception as exc:  # one bad song must not kill the batch
                summary.failed[rec.id] = str(exc)

    return summary
