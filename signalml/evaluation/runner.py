"""Orchestration for the aligner eval — walks a reference set, scores each song,
writes a JSON report. The metrics themselves live in ``alignment``; this module only
handles finding files, bucketing songs and aggregating.

Usage is via ``signalml eval`` (see cli.py). Nothing here writes to the manifest or to
``songs/`` — an eval run is read-only over the pipeline's output by design, so it can
be re-run freely and can never contaminate a dataset.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .alignment import (
    OnsetMetrics,
    ScoreCalibration,
    WordOnset,
    calibrate_align_score,
    onset_metrics,
    pooled_onset_metrics,
    word_onsets,
)
from .refs import jamendolyrics_songs, load_jamendolyrics_onsets


@dataclass
class SongResult:
    song: str
    bucket: str
    hyp_path: str
    align_score: float | None = None
    metrics: dict | None = None
    error: str | None = None


@dataclass
class EvalReport:
    ref_set: str
    aligner: str
    generated: str
    n_songs: int
    n_scored: int
    pooled: dict[str, dict] = field(default_factory=dict)
    calibration: dict | None = None
    songs: list[SongResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return path

    def summary_lines(self) -> list[str]:
        out = [f"{self.ref_set} · aligner={self.aligner} · "
               f"{self.n_scored}/{self.n_songs} songs scored"]
        for bucket, metrics in sorted(self.pooled.items()):
            m = OnsetMetrics(**metrics)
            out.append(f"  [{bucket}] {m.summary()}")
        failed = [s for s in self.songs if s.error]
        if failed:
            out.append(f"  {len(failed)} failed:")
            out.extend(f"    {s.song}: {s.error}" for s in failed[:10])
            if len(failed) > 10:
                out.append(f"    ... and {len(failed) - 10} more")
        if self.calibration:
            cal = ScoreCalibration(**self.calibration)
            out.append(f"  align_score calibration: {cal.summary()}")
        return out


def phones_path(data_root: str | Path, song_id: str) -> Path:
    """Where S5 writes a song's alignment."""
    return Path(data_root) / "songs" / song_id / "align" / "phones.json"


def load_phones(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _identity_map(names: list[str]) -> dict[str, str]:
    return {n: n for n in names}


def load_song_map(path: str | Path) -> dict[str, str]:
    """CSV mapping reference song name -> manifest song id (columns ``ref,song_id``).

    Needed because a reference set names songs its own way and the pipeline names them
    ``sng_0042``. Without a map the two are assumed identical, which is only true if the
    songs were ingested under their reference names.
    """
    import csv

    mapping: dict[str, str] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"ref", "song_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        for row in reader:
            mapping[row["ref"].strip()] = row["song_id"].strip()
    return mapping


def run_jamendolyrics(
    ref_root: str | Path,
    data_root: str | Path,
    *,
    language: str = "English",
    song_map: dict[str, str] | None = None,
    aligner: str = "mfa",
    align_scores: dict[str, float] | None = None,
) -> EvalReport:
    """Score every JamendoLyrics song of ``language`` that has a ``phones.json``.

    Songs are split into two pooled buckets — ``clean`` and ``flagged`` (polyphonic
    vocals, lyric overlap, or non-lexical passages). The clean bucket is the headline
    number; the flagged one answers a different and also interesting question, which is
    how each aligner copes with the things that make *singing* hard. Mixing them would
    hide both answers.
    """
    songs = [s for s in jamendolyrics_songs(ref_root)
             if not language or s.language.casefold() == language.casefold()]
    mapping = song_map or _identity_map([s.name for s in songs])

    results: list[SongResult] = []
    per_bucket: dict[str, list[tuple[list[WordOnset], list[WordOnset]]]] = {}
    calibration_pairs: list[tuple[float, float]] = []

    for song in songs:
        bucket = "clean" if song.clean else "flagged"
        song_id = mapping.get(song.name, song.name)
        hyp_file = phones_path(data_root, song_id)
        result = SongResult(song=song.name, bucket=bucket, hyp_path=str(hyp_file))
        try:
            ref = load_jamendolyrics_onsets(ref_root, song.name)
            if not hyp_file.exists():
                raise FileNotFoundError(f"no alignment at {hyp_file} (song_id={song_id!r})")
            hyp = word_onsets(load_phones(hyp_file))
            metrics = onset_metrics(ref, hyp)
            result.metrics = metrics.to_dict()
            per_bucket.setdefault(bucket, []).append((ref, hyp))
            if align_scores and song_id in align_scores and metrics.aae is not None:
                score = align_scores[song_id]
                result.align_score = score
                calibration_pairs.append((score, metrics.aae))
        except Exception as exc:  # reported per song; one bad song must not kill the run
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)

    return EvalReport(
        ref_set="jamendolyrics",
        aligner=aligner,
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_songs=len(songs),
        n_scored=sum(1 for r in results if r.metrics is not None),
        pooled={b: pooled_onset_metrics(pairs).to_dict() for b, pairs in per_bucket.items()},
        calibration=(calibrate_align_score(calibration_pairs).to_dict()
                     if len(calibration_pairs) >= 3 else None),
        songs=results,
    )


def collect_align_scores(data_root: str | Path) -> dict[str, float]:
    """Pull ``quality.align_score`` for every aligned record in the manifest."""
    from ..manifest import Manifest

    scores: dict[str, float] = {}
    for rec in Manifest.for_data_root(data_root).records:
        score = getattr(rec.quality, "align_score", None)
        if score is not None:
            scores[rec.id] = float(score)
    return scores
