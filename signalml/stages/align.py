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
from .common import song_dir, update_analysis

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
    tier_names = {name.lower(): name for name in tg.tierNames}
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


def align(
    data_root: str | Path,
    *,
    cfg: AlignConfig | None = None,
    force: bool = False,
    limit: int | None = None,
    runner: Runner | None = None,
) -> AlignSummary:
    data_root = Path(data_root)
    cfg = cfg or load_align_config()
    runner = runner or _default_runner
    manifest = Manifest.for_data_root(data_root)
    summary = AlignSummary()

    work: list[ManifestRecord] = []
    for rec in manifest.records:
        ready = rec.status.cleaned and (force or not rec.status.aligned)
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
        for rec in work:
            try:
                wav = song_dir(data_root, rec.id) / "clean" / "vocals.wav"
                if not wav.exists():
                    raise FileNotFoundError(f"clean vocal missing: {wav}")
                lyrics = (data_root / rec.meta.lyrics_path).read_text(encoding="utf-8")
                spk = corpus / rec.id  # one speaker dir per song
                spk.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(wav, spk / f"{rec.id}.wav")
                (spk / f"{rec.id}.lab").write_text(lyrics.strip() + "\n", encoding="utf-8")
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
                tg_dst = align_dir / "vocals.TextGrid"
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

                (align_dir / "phones.json").write_text(
                    dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                update_analysis(sdir, "align", {
                    "aligner": "mfa",
                    "acoustic_model": cfg.acoustic_model,
                    "dictionary": cfg.dictionary,
                    "phone_set": cfg.phone_set,
                    "n_phones": len(payload["phones"]),
                    "speech_sec": round(speech_sec, 3),
                    "align_score": score,
                    "align_score_method": "speech_sec / clean.silence_map.voiced_sec (v1)",
                    "date": _dt.date.today().isoformat(),
                })
                rec.status.aligned = True
                rec.quality.align_score = score
                manifest.upsert(rec)
                manifest.save()
                summary.aligned.append(rec.id)
            except Exception as exc:  # one bad song must not kill the batch
                summary.failed[rec.id] = str(exc)

    return summary
