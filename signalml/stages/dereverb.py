"""S4b dereverb - take baked-in reverb out of separated vocals, kept phrase by phrase.

Why: reverb is the most common fixable damage in the separated covers (a blind
listening survey, 2026-10-08, tagged it on ~45% of their phrases). Reverb also drags the
HNR gate down, so today the gate throws away phrases whose singing is clean but wet.
Studio stems are dry by construction and never go through this stage.

What: for every in-scope record (``source_quality`` in ``scope``, default
``["separated"]``):

1. The vocal -> a de-reverb model -> ``clean/vocals.dereverb.wav`` (same rate, same
   length as ``clean/vocals.wav``, so every alignment stays valid). By default the model
   hears the *stereo* separated stem (``stems/vocals.wav``): reverb is wide and the voice
   sits in the middle, and a stereo model leans on that difference - fed the mono clean
   vocal copied to both channels it barely touches room reverb (measured 2026-10-09).
   The output is mixed to mono and given the clean vocal's gain (``clean/vocals.wav`` is
   the stem's mono mix times one loudness gain, fitted here by least squares), so it
   lines up with the clean vocal sample for sample. The model is injectable; the real
   one is an MSST RoFormer (``signalml.restore.roformer``). Model, checkpoint hash,
   device, source and gain go to ``analysis.json`` under ``dereverb``.
2. Every phrase of each ``align/utterances*.json`` is judged on both versions
   (``judge_phrases``) and the verdict goes to ``align/dereverb<suffix>.json``. A phrase
   keeps the de-reverbed audio only when the reverb measurably went down (the tail or
   gap after the phrase got quieter, or HNR rose) AND the pitch track and loudness of
   the singing held. Anything else keeps the original. A model that rewrites the
   singing never reaches a dataset.

The dataset build (``DatasetRecipe.dereverb``) composes the two files per phrase
(``compose``) with short crossfades at phrase boundaries, which sit in rests. The phrase
gate then reads the HNR of the audio actually used. This stage does no augmentation; it
is preprocessing of a stem, like S4.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field

from ..config import CONFIGS_DIR
from ..manifest import Manifest
from .common import song_dir, update_analysis
from .phrase_quality import HOP_SEC, WIN_SEC, _frame_db, measure_phrases

DEREVERB_VERSION = 1
OUT_NAME = "vocals.dereverb.wav"
PITCH_STEP_SEC = 0.01

# backend(audio (T,) mono or (C, T), sr) -> de-reverbed mono audio (T,)
DereverbFn = Callable[[np.ndarray, int], np.ndarray]
# a stem whose mono mix explains the clean vocal worse than this is not its source
MAX_STEM_RESIDUAL_DB = -30.0


class DereverbGate(BaseModel):
    """When a phrase may use the de-reverbed audio. All must hold."""

    # pitch must survive: median |cents| between the two pitch tracks on frames
    # voiced in both, and the share of the original's voiced frames still voiced
    max_f0_cents: float = 30.0
    min_voiced_overlap: float = 0.85
    # the singing itself may not lose more than this (p90 frame level of the sung span)
    max_level_drop_db: float = 6.0
    # and something must have improved by at least this much
    min_tail_gain_db: float = 3.0   # tail_db or gap_db went down
    min_hnr_gain_db: float = 1.0    # or hnr_db went up


class DereverbConfig(BaseModel):
    model: str = "dereverb_bs_roformer_anvuew"
    # relative paths resolve against DATA_ROOT
    checkpoint: str = ("models/dereverb_bs_roformer_anvuew/"
                       "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt")
    model_yaml: str = "models/dereverb_bs_roformer_anvuew/config.yaml"  # training config
    device: str = "auto"
    batch_size: int = 2
    num_overlap: int | None = None
    arch: str = "bs_roformer"  # or mel_band_roformer (the checkpoint's architecture)
    # "stems": the stereo separated stem (falls back to the clean vocal when the stem is
    # missing, mono, or not the clean vocal's source); "clean": always the mono clean vocal
    source: Literal["stems", "clean"] = "stems"
    scope: list[str] = ["separated"]
    gate: DereverbGate = Field(default_factory=DereverbGate)


def load_dereverb_config(path: str | Path | None = None) -> DereverbConfig:
    path = Path(path) if path else CONFIGS_DIR / "dereverb.yaml"
    if path.exists():
        return DereverbConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return DereverbConfig()


def dereverb_name(utterances_name: str) -> str:
    """``utterances.phrase.json`` -> ``dereverb.phrase.json``."""
    return "dereverb" + utterances_name[len("utterances"):]


def make_backend(config: DereverbConfig, data_root: str | Path) -> tuple[DereverbFn, dict]:
    """The real backend: the configured RoFormer, loaded once (torch + the restore extra)."""
    from ..restore.roformer import RoformerRunner

    root = Path(data_root)

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else root / p

    runner = RoformerRunner(resolve(config.checkpoint), resolve(config.model_yaml),
                            arch=config.arch, device=config.device,
                            batch_size=config.batch_size, num_overlap=config.num_overlap)
    return runner, {"model": config.model, **runner.info}


# ---------------------------------------------------------------- judging


def _pitch_track(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    import parselmouth

    p = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr).to_pitch(
        time_step=PITCH_STEP_SEC, pitch_floor=75.0, pitch_ceiling=1100.0)
    return p.xs(), p.selected_array["frequency"]


def _span_level(db: np.ndarray, t: np.ndarray, s: float, e: float) -> float | None:
    m = (t >= s) & (t < e)
    return float(np.percentile(db[m], 90)) if m.any() else None


def judge_phrases(orig: np.ndarray, derev: np.ndarray, sr: int, utts: list[dict],
                  phones: list[dict], gate: DereverbGate | None = None) -> list[dict]:
    """One verdict per utterance (same order): before/after numbers + keep and why."""
    gate = gate or DereverbGate()
    orig = np.asarray(orig, dtype=np.float64)
    derev = np.asarray(derev, dtype=np.float64)
    before = measure_phrases(orig, sr, utts, phones)
    after = measure_phrases(derev, sr, utts, phones)
    pt, pf_o = _pitch_track(orig, sr)
    _, pf_d = _pitch_track(derev, sr)
    db_o, db_d = _frame_db(orig, sr), _frame_db(derev, sr)
    t_db = np.arange(len(db_o)) * HOP_SEC + WIN_SEC / 2

    out: list[dict] = []
    for b, a in zip(before, after):
        v = {"start": b["start"], "end": b["end"], "sung_start": b["sung_start"],
             "sung_end": b["sung_end"],
             "before": {k: b[k] for k in ("hnr_db", "tail_db", "gap_db")},
             "after": {k: a[k] for k in ("hnr_db", "tail_db", "gap_db")},
             "f0_cents": None, "voiced_overlap": None, "level_change_db": None,
             "keep": False, "reason": None}
        out.append(v)
        s, e = b["sung_start"], b["sung_end"]
        if s is None or b["hnr_db"] is None or a["hnr_db"] is None:
            v["reason"] = "unscored"
            continue
        lo, la = _span_level(db_o, t_db, s, e), _span_level(db_d, t_db, s, e)
        v["before"]["level_db"], v["after"]["level_db"] = lo, la
        if lo is not None and la is not None:
            v["level_change_db"] = round(la - lo, 2)
        m = (pt >= s) & (pt < e)
        vo, vd = pf_o[m] > 0, pf_d[m] > 0
        if vo.sum() >= 5:
            v["voiced_overlap"] = round(float((vo & vd).sum() / vo.sum()), 3)
            both = vo & vd
            if both.any():
                cents = 1200 * np.abs(np.log2(pf_d[m][both] / pf_o[m][both]))
                v["f0_cents"] = round(float(np.median(cents)), 1)

        if v["voiced_overlap"] is not None and v["voiced_overlap"] < gate.min_voiced_overlap:
            v["reason"] = "voicing"
        elif v["f0_cents"] is not None and v["f0_cents"] > gate.max_f0_cents:
            v["reason"] = "f0"
        elif v["level_change_db"] is not None and v["level_change_db"] < -gate.max_level_drop_db:
            v["reason"] = "level"
        else:
            gains = []
            for k in ("tail_db", "gap_db"):
                if b[k] is not None and a[k] is not None:
                    gains.append(b[k] - a[k] >= gate.min_tail_gain_db)
            gains.append(a["hnr_db"] - b["hnr_db"] >= gate.min_hnr_gain_db)
            v["keep"] = any(gains)
            v["reason"] = "improved" if v["keep"] else "no_gain"
    return out


def is_current(verdicts: dict, utts: list[dict], model: str | None = None) -> bool:
    phrases = verdicts.get("phrases") or []
    return (verdicts.get("version") == DEREVERB_VERSION and len(phrases) == len(utts)
            and (model is None or verdicts.get("model") == model)
            and all(p["start"] == u["start"] and p["end"] == u["end"]
                    for p, u in zip(phrases, utts)))


def load_verdicts(align_dir: Path, utterances_name: str, utts: list[dict],
                  model: str | None = None) -> list[dict] | None:
    """This alignment's verdicts, or None when missing or stale (re-aligned, re-run)."""
    try:
        data = json.loads((align_dir / dereverb_name(utterances_name)).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data["phrases"] if is_current(data, utts, model) else None


def compose(orig: np.ndarray, derev: np.ndarray, sr: int, verdicts: list[dict],
            fade_sec: float = 0.02) -> np.ndarray:
    """The original, with each kept phrase window taken from ``derev``.

    Phrase windows tile the song and meet in rests, so a short linear crossfade centred
    on each boundary is inaudible; neighbouring kept phrases join without one.
    """
    n = min(len(orig), len(derev))
    mask = np.zeros(n, dtype=np.float32)
    for v in verdicts:
        if v.get("keep"):
            mask[max(0, int(round(v["start"] * sr))):min(n, int(round(v["end"] * sr)))] = 1.0
    if not mask.any():
        return np.asarray(orig, dtype=np.float32).copy()
    k = max(1, int(round(fade_sec * sr)))
    if k > 1:
        mask = np.convolve(mask, np.ones(k, dtype=np.float32) / k, mode="same")
    out = np.asarray(orig, dtype=np.float32).copy()
    out[:n] = (1.0 - mask) * out[:n] + mask * np.asarray(derev[:n], dtype=np.float32)
    return out


# ---------------------------------------------------------------- stage


def stem_input(orig: np.ndarray, stem: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """(stereo model input (C, T), gain, residual dB) when ``stem`` (T, C) is the source of
    the mono clean vocal ``orig``: ``orig`` ~= gain * mono mix of the stem. None otherwise
    (mono stem, other length, or a mix that doesn't explain ``orig``)."""
    if stem.ndim != 2 or stem.shape[1] < 2 or abs(len(stem) - len(orig)) > 2:
        return None
    stem = stem[: len(orig)]
    if len(stem) < len(orig):
        stem = np.pad(stem, ((0, len(orig) - len(stem)), (0, 0)))
    mono = stem.mean(axis=1).astype(np.float64)
    o = np.asarray(orig, dtype=np.float64)
    energy = float(mono @ mono)
    if energy <= 0 or float(o @ o) <= 0:
        return None
    gain = float(o @ mono) / energy
    resid = o - gain * mono
    resid_db = 10 * np.log10(max(float(resid @ resid), 1e-20) / float(o @ o))
    if resid_db > MAX_STEM_RESIDUAL_DB:
        return None
    return stem.T.astype(np.float32), gain, round(float(resid_db), 1)



@dataclass
class DereverbSummary:
    processed: list[str] = field(default_factory=list)   # ids whose audio was written
    judged: list[str] = field(default_factory=list)      # "<id>/<verdict file>"
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    phrases: int = 0
    kept: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


def dereverb_songs(
    data_root: str | Path,
    *,
    config: DereverbConfig | None = None,
    backend: tuple[DereverbFn, dict] | None = None,
    force: bool = False,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> DereverbSummary:
    """De-reverb every in-scope cleaned record, then judge each of its phrase alignments.

    Manifest-driven and idempotent: audio is redone only when missing, made by another
    model, or with ``force``; verdicts only when stale. ``limit`` caps songs whose audio
    is (re)made this run. The backend loads on first use, so a run with nothing to do
    never touches torch.
    """
    import soundfile as sf

    data_root = Path(data_root)
    config = config or load_dereverb_config()
    manifest = Manifest.for_data_root(data_root)
    summary = DereverbSummary()
    made = 0
    for rec in manifest.records:
        if ids and rec.id not in ids:
            continue
        if (rec.meta.excluded or not rec.status.cleaned
                or rec.meta.source_quality not in config.scope):
            summary.skipped.append(rec.id)
            continue
        sdir = song_dir(data_root, rec.id)
        clean, out = sdir / "clean" / "vocals.wav", sdir / "clean" / OUT_NAME
        try:
            analysis = json.loads((sdir / "analysis.json").read_text(encoding="utf-8")) \
                if (sdir / "analysis.json").exists() else {}
            done = (analysis.get("dereverb") or {}).get("model") == config.model
            orig = derev = None
            sr = None
            if force or not (out.exists() and done):
                if limit is not None and made >= limit:
                    return summary
                if backend is None:
                    backend = make_backend(config, data_root)
                fn, info = backend
                orig, sr = sf.read(str(clean), dtype="float32")
                if orig.ndim > 1:
                    orig = orig.mean(axis=1)
                model_in, gain, resid_db, source = orig, 1.0, None, "clean/vocals.wav"
                stem_path = sdir / "stems" / "vocals.wav"
                if config.source == "stems" and stem_path.exists():
                    stem, stem_sr = sf.read(str(stem_path), dtype="float32", always_2d=True)
                    got = stem_input(orig, stem) if stem_sr == sr else None
                    if got is not None:
                        model_in, gain, resid_db = got
                        source = "stems/vocals.wav"
                started = _dt.datetime.now()
                derev = gain * np.asarray(fn(model_in, sr), dtype=np.float32)[:len(orig)]
                if len(derev) < len(orig):
                    derev = np.pad(derev, (0, len(orig) - len(derev)))
                derev = derev.astype(np.float32)
                sf.write(str(out), derev, sr, subtype="FLOAT")
                update_analysis(sdir, "dereverb", {
                    **info, "model": config.model, "version": DEREVERB_VERSION,
                    "source": source, "gain": round(gain, 6),
                    "stem_residual_db": resid_db, "output": f"clean/{OUT_NAME}",
                    "seconds": round(len(orig) / sr, 2),
                    "elapsed_sec": round((_dt.datetime.now() - started).total_seconds(), 2),
                    "date": _dt.date.today().isoformat(),
                })
                summary.processed.append(rec.id)
                made += 1
                force_judge = True
            else:
                force_judge = force

            align_dir = sdir / "align"
            for name in sorted(p.name for p in align_dir.glob("utterances*.json")) \
                    if align_dir.is_dir() else []:
                utts = json.loads((align_dir / name).read_text(encoding="utf-8"))
                if not force_judge and load_verdicts(align_dir, name, utts,
                                                     config.model) is not None:
                    continue
                phones_path = align_dir / ("phones" + name[len("utterances"):])
                phones = json.loads(phones_path.read_text(encoding="utf-8"))["phones"]
                if orig is None:
                    orig, sr = sf.read(str(clean), dtype="float32")
                    orig = orig.mean(axis=1) if orig.ndim > 1 else orig
                if derev is None:
                    derev, _ = sf.read(str(out), dtype="float32")
                verdicts = judge_phrases(orig, derev, sr, utts, phones, config.gate)
                (align_dir / dereverb_name(name)).write_text(json.dumps({
                    "version": DEREVERB_VERSION, "model": config.model,
                    "utterances": name, "audio": f"clean/{OUT_NAME}",
                    "gate": config.gate.model_dump(), "phrases": verdicts,
                }, ensure_ascii=False, indent=1), encoding="utf-8")
                summary.judged.append(f"{rec.id}/{dereverb_name(name)}")
                summary.phrases += len(verdicts)
                summary.kept += sum(v["keep"] for v in verdicts)
                for v in verdicts:
                    summary.reasons[v["reason"]] = summary.reasons.get(v["reason"], 0) + 1
        except Exception as exc:  # one bad song must not stop the batch
            summary.failed[rec.id] = f"{type(exc).__name__}: {exc}"
    return summary
