"""Per-phrase audio quality of the aligned vocal (``signalml quality``).

Why: a Demucs-separated vocal is not uniformly good or bad. Pop and EDM covers often
open clean, take on bleed, stacked harmonies or effects at the drop, and come back clean
later - phrase-to-phrase variation inside one song is close to half of all the
variation in phrase quality on the separated corpus (2026-10-08). A song-, singer- or
genre-level filter therefore throws away clean openings and keeps damaged choruses; the
dataset's phrase gate is the right granularity, and this stage gives it numbers.

What: for every phrase of an ``align/utterances*.json`` (``align --phrases`` output),
measured on ``clean/vocals.wav`` over the *sung span* - first to last aligned phone, since
the phrase windows tile the song and include the rests around the singing:

- ``hnr_db``: median harmonics-to-noise ratio (Praat cross-correlation, 10 ms) over the
  span's voiced frames. Bleed, separation warble, stacked voices and distortion all pull
  it down. The gate reads this one. A blind listening calibration put the cutoff at
  13 dB (agreeing with the listener on ~90% of the phrases they decided); studio stems
  sit around 22 dB.
- ``voiced``: share of the span's frames that have a pitch.
- ``tail_db`` / ``gap_db``: level just after the span (0.08-0.35 s) and deeper into the
  rest that follows, relative to the phrase's loud level - reverb, delay throws and
  bleed between lines. Diagnostic only: on the calibration they did not separate
  phrases the listener kept from ones they dropped (a held note the aligner cut short
  reads as a loud tail too).

Writes ``align/quality<suffix>.json`` beside each ``align/utterances<suffix>.json``,
one entry per utterance in the same order, with the utterance window copied in so a
re-alignment shows up as stale instead of silently gating the wrong phrases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..manifest import Manifest
from .common import song_dir

QUALITY_VERSION = 1
HOP_SEC = 0.01
WIN_SEC = 0.03
MIN_PITCH_HZ = 75.0
MIN_FRAMES = 10        # a sung span shorter than this is not scored
MIN_VOICED_FRAMES = 5  # fewer voiced frames: hnr_db is None (not measured, never gated)


def quality_name(utterances_name: str) -> str:
    """``utterances.phrase.json`` -> ``quality.phrase.json``."""
    return "quality" + utterances_name[len("utterances"):]


def _frame_db(y: np.ndarray, sr: int) -> np.ndarray:
    hop, win = max(1, int(sr * HOP_SEC)), max(1, int(sr * WIN_SEC))
    n = max(1, (len(y) - win) // hop + 1)
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = y[np.minimum(idx, len(y) - 1)]
    return 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)


def _power_mean_db(db: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(10 ** (db / 10)) + 1e-12))


def measure_phrases(y: np.ndarray, sr: int, utts: list[dict],
                    phones: list[dict]) -> list[dict]:
    """One quality entry per utterance (same order). Pure: audio + alignment in."""
    import parselmouth

    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    db = _frame_db(y, sr)
    t_db = np.arange(len(db)) * HOP_SEC + WIN_SEC / 2
    hn = parselmouth.Sound(y, sampling_frequency=sr).to_harmonicity_cc(
        time_step=HOP_SEC, minimum_pitch=MIN_PITCH_HZ)
    hv, ht = hn.values[0], hn.xs()
    duration = len(y) / sr

    spans: list[tuple[float, float] | None] = []
    for u in utts:
        ph = [p for p in phones if u["start"] <= (p["start"] + p["end"]) / 2 < u["end"]]
        spans.append((ph[0]["start"], ph[-1]["end"]) if ph else None)

    out: list[dict] = []
    for i, u in enumerate(utts):
        entry = {"start": u["start"], "end": u["end"], "sung_start": None, "sung_end": None,
                 "hnr_db": None, "voiced": None, "tail_db": None, "gap_db": None}
        out.append(entry)
        if spans[i] is None:
            continue
        s, e = spans[i]
        m = (t_db >= s) & (t_db < e)
        entry["sung_start"], entry["sung_end"] = round(s, 3), round(e, 3)
        if m.sum() < MIN_FRAMES:
            continue
        level = float(np.percentile(db[m], 90))
        h = hv[(ht >= s) & (ht < e)]
        v = h[h > -100]  # Praat marks unvoiced frames -200
        entry["voiced"] = round(len(v) / max(1, len(h)), 3)
        if len(v) >= MIN_VOICED_FRAMES:
            entry["hnr_db"] = round(float(np.median(v)), 2)
        nxt = next((sp[0] for sp in spans[i + 1:] if sp), duration)
        if nxt - e >= 0.5:
            tm = (t_db >= e + 0.08) & (t_db < e + 0.35)
            if tm.any():
                entry["tail_db"] = round(_power_mean_db(db[tm]) - level, 2)
        if nxt - e >= 0.8:
            gm = (t_db >= e + 0.5) & (t_db < nxt - 0.1)
            if gm.sum() >= 20:
                entry["gap_db"] = round(float(np.median(db[gm])) - level, 2)
    return out


def is_current(quality: dict, utts: list[dict]) -> bool:
    """True when ``quality`` was measured on exactly these utterance windows."""
    phrases = quality.get("phrases") or []
    return (quality.get("version") == QUALITY_VERSION and len(phrases) == len(utts)
            and all(q["start"] == u["start"] and q["end"] == u["end"]
                    for q, u in zip(phrases, utts)))


def load_quality(align_dir: Path, utterances_name: str, utts: list[dict]) -> list[dict] | None:
    """The quality entries for these utterances, or None when missing or stale."""
    path = align_dir / quality_name(utterances_name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data["phrases"] if is_current(data, utts) else None


@dataclass
class QualitySummary:
    measured: list[str] = field(default_factory=list)   # "<id>/<quality file>"
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def _phones_for(align_dir: Path, utterances_name: str) -> Path:
    return align_dir / ("phones" + utterances_name[len("utterances"):])


def measure_quality(
    data_root: str | Path,
    *,
    force: bool = False,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> QualitySummary:
    """Write ``align/quality*.json`` for every phrase alignment that lacks a current one.

    Manifest-driven: aligned, cleaned records; each of a song's ``utterances*.json``
    (main alignment and variants) gets its own quality file. Idempotent - an existing
    file is rewritten only when stale (the utterances changed) or with ``force``.
    """
    import soundfile as sf

    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    summary = QualitySummary()
    done = 0
    for rec in manifest.records:
        if ids and rec.id not in ids:
            continue
        sdir = song_dir(data_root, rec.id)
        align_dir = sdir / "align"
        names = sorted(p.name for p in align_dir.glob("utterances*.json")) \
            if align_dir.is_dir() else []
        if not (rec.status.cleaned and names):
            summary.skipped.append(rec.id)
            continue
        audio = None
        for name in names:
            label = f"{rec.id}/{quality_name(name)}"
            try:
                utts = json.loads((align_dir / name).read_text(encoding="utf-8"))
                if not force and load_quality(align_dir, name, utts) is not None:
                    summary.skipped.append(label)
                    continue
                if limit is not None and done >= limit:
                    return summary
                phones_path = _phones_for(align_dir, name)
                phones = json.loads(phones_path.read_text(encoding="utf-8"))["phones"]
                if audio is None:
                    wav = sdir / "clean" / "vocals.wav"
                    if not wav.exists():
                        raise FileNotFoundError(f"clean vocal missing: {wav}")
                    audio = sf.read(str(wav), dtype="float32")
                phrases = measure_phrases(audio[0], audio[1], utts, phones)
                (align_dir / quality_name(name)).write_text(json.dumps({
                    "version": QUALITY_VERSION, "source": "clean/vocals.wav",
                    "utterances": name, "phrases": phrases,
                }, ensure_ascii=False, indent=1), encoding="utf-8")
                summary.measured.append(label)
                done += 1
            except Exception as exc:  # one bad song must not stop the batch
                summary.failed[label] = f"{type(exc).__name__}: {exc}"
    return summary
