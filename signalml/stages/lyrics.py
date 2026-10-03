"""S5a lyrics — machine lyrics for songs that arrive without a ``lyrics.txt`` (Q14).

Alignment (S5) needs a transcript. The original corpus came with hand-corrected lyrics
sidecars; harvested songs (``signalml harvest``) arrive with none, and pasting lyrics by
hand does not scale past a few dozen songs. This stage transcribes the *cleaned* vocal
stem with Whisper and writes what was actually sung — ad-libs included, which published
lyrics miss — to ``songs/<id>/lyrics/lyrics.txt``.

Measured against Logan's hand-corrected lyrics on 15 well-aligned songs (2026-09-24,
large-v3): ~90% of reference words found; median word error ~19%, dominated not by
misheard words but by *extra* text — loops and "oh oh oh" over held notes — on a few
songs. So machine lyrics are provenance-tagged (``meta.lyrics_source = "asr:<model>"``)
and every song still has to clear the alignment gate before it reaches a dataset.

Hand-written lyrics are never overwritten: a record that already has lyrics without an
``asr:`` source is skipped even with ``force``.

Whisper's stock hallucinations ("you", "Thank you.", "Thanks for watching") are dropped
when they make up a whole segment: over silence, breaths and wordless singing they are
what Whisper emits, and alignment would pin each one to a breath or a held "ooh". The
dropped segments stay in ``asr.json`` (``dropped_segments``). ``scrub`` applies the same
filter to transcripts written before it existed.

Whisper is an external tool in its own venv (``scripts/whisper_lyrics.py``), invoked by
command like MFA and the trainer — one invocation per batch so the model loads once.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

from ..config import CONFIGS_DIR
from ..manifest import MANIFEST_NAME, Manifest, ManifestRecord, _manifest_lock
from .common import song_dir, update_analysis

LYRICS_NAME = "lyrics.txt"
ASR_NAME = "asr.json"
ASR_PREFIX = "asr:"


class LyricsConfig(BaseModel):
    """Machine-local wiring for the Whisper backend."""

    # argv; {jobs} is replaced by the batch file. Empty = refuse rather than guess.
    command: list[str] = []
    command_cwd: str | None = None
    stem: str = "vocals"
    language: str = "en"
    # below this many words the "lyrics" are noise, an instrumental or another language
    min_words: int = 8
    timeout_sec: float = 6 * 3600.0


def load_lyrics_config(path: str | Path | None = None) -> LyricsConfig:
    path = Path(path) if path else CONFIGS_DIR / "lyrics.yaml"
    if Path(path).exists():
        return LyricsConfig.model_validate(
            yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return LyricsConfig()


@dataclass
class LyricsSummary:
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    flagged: dict[str, str] = field(default_factory=dict)  # written, but looks looped


def _norm_line(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def lyrics_stats(segments: list[dict]) -> dict:
    """Numbers that expose Whisper's music failure mode (loops), for the record and
    for the flag — a chorus repeats, but the *same line five times running* is a loop."""
    lines = [_norm_line(s["text"]) for s in segments if _norm_line(s["text"])]
    words = [w for line in lines for w in line.split()]
    run = best = 0
    prev = None
    for line in lines:
        run = run + 1 if line == prev else 1
        best, prev = max(best, run), line
    ratios = [s.get("compression_ratio") or 0.0 for s in segments]
    return {"n_segments": len(segments), "n_words": len(words),
            "max_consecutive_repeat": best,
            "max_compression_ratio": round(max(ratios), 3) if ratios else None}


def loop_flag(stats: dict) -> str | None:
    if stats["max_consecutive_repeat"] >= 4:
        return f"same line {stats['max_consecutive_repeat']}x in a row (likely a loop)"
    return None


# Whole segments only - a sung "you" inside a line stays. On a first multitrack library
# (2026-10-03) such segments sat in 2 of 3 machine transcripts: 0.2 s long at the median
# against 3.5 s for the rest, no-speech probability 0.73 against 0.33.
HALLUCINATION = re.compile(
    r"(you|thank you( so much)?( for watching)?|thanks for watching|bye|"
    r"transcription by castingwords|subtitles by .*)[.!]*")


def drop_hallucinations(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """``(kept, dropped)``: a segment whose whole text is a known Whisper hallucination
    is dropped."""
    kept, dropped = [], []
    for seg in segments:
        text = seg["text"].strip().lower()
        (dropped if HALLUCINATION.fullmatch(text) else kept).append(seg)
    return kept, dropped


def _transcript(segments: list[dict]) -> str:
    return "\n".join(s["text"].strip() for s in segments if s["text"].strip())


def _default_runner(argv: list[str], cwd: str | None, timeout: float) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        raise RuntimeError(f"whisper backend failed (exit {proc.returncode}):\n{tail}")


def _wants_lyrics(rec: ManifestRecord, force: bool) -> bool:
    if rec.meta.domain != "sung" or not rec.status.cleaned:
        return False
    if not rec.meta.has_lyrics:
        return True
    machine = (rec.meta.lyrics_source or "").startswith(ASR_PREFIX)
    return force and machine  # hand-written lyrics are never replaced


def transcribe_batch(
    data_root: Path,
    records: list[ManifestRecord],
    cfg: LyricsConfig,
    runner: Callable[[list[str], str | None, float], None],
) -> dict[str, dict | str]:
    """One backend invocation over ``records``; returns id -> payload, or an error
    string for a song that could not be transcribed."""
    if not cfg.command:
        raise RuntimeError(
            "no lyrics command configured - copy configs/lyrics.yaml to "
            "configs/lyrics.local.yaml and point `command` at the whisper venv + "
            "scripts/whisper_lyrics.py")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_dir = data_root / "work" / f"lyrics_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict | str] = {}
    jobs, by_out = [], {}
    for rec in records:
        wav = song_dir(data_root, rec.id) / "clean" / f"{cfg.stem}.wav"
        if not wav.exists():
            results[rec.id] = f"no cleaned stem at {wav}"
            continue
        out = batch_dir / f"{rec.id}.json"
        jobs.append({"wav": str(wav), "out": str(out)})
        by_out[rec.id] = out
    jobs_path = batch_dir / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    if jobs:
        argv = [part.format(jobs=str(jobs_path)) for part in cfg.command]
        runner(argv, cfg.command_cwd, cfg.timeout_sec)
    for rid, out in by_out.items():
        if not out.exists():
            results[rid] = "backend wrote no result"
            continue
        payload = json.loads(out.read_text(encoding="utf-8"))
        results[rid] = payload["error"] if "error" in payload else payload
    return results


def run(
    data_root: str | Path,
    *,
    cfg: LyricsConfig | None = None,
    runner: Callable[[list[str], str | None, float], None] | None = None,
    force: bool = False,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> LyricsSummary:
    """Manifest-driven: transcribe every cleaned, sung song that has no lyrics yet."""
    data_root = Path(data_root)
    cfg = cfg or load_lyrics_config()
    runner = runner or _default_runner
    manifest = Manifest.for_data_root(data_root)
    summary = LyricsSummary()

    work: list[ManifestRecord] = []
    for rec in manifest.records:
        if ids and rec.id not in ids:
            continue
        if _wants_lyrics(rec, force):
            work.append(rec)
        else:
            summary.skipped.append(rec.id)
    if limit is not None:
        work = work[:limit]
    if not work:
        return summary
    results = transcribe_batch(data_root, work, cfg, runner)

    for rec in work:
        try:
            payload = results[rec.id]
            if isinstance(payload, str):
                raise RuntimeError(payload)
            segments, dropped = drop_hallucinations(payload.get("segments", []))
            payload = {**payload, "segments": segments, "dropped_segments": dropped}
            stats = lyrics_stats(segments)
            if stats["n_words"] < cfg.min_words:
                raise RuntimeError(f"only {stats['n_words']} words recognised "
                                   "(instrumental, noise or not English?)")
            sdir = song_dir(data_root, rec.id)
            ldir = sdir / "lyrics"
            ldir.mkdir(parents=True, exist_ok=True)
            (ldir / LYRICS_NAME).write_text(_transcript(segments) + "\n", encoding="utf-8")
            (ldir / ASR_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
            source = f"{ASR_PREFIX}{payload.get('model') or 'whisper'}"
            flag = loop_flag(stats)
            update_analysis(sdir, "lyrics", {
                "source": source, **stats, "flag": flag, "hallucinations_dropped": len(dropped),
                "date": _dt.date.today().isoformat()})
            rec.meta.has_lyrics = True
            rec.meta.lyrics_path = (ldir / LYRICS_NAME).relative_to(data_root).as_posix()
            rec.meta.lyrics_source = source
            manifest.commit(rec)
            summary.written.append(rec.id)
            if flag:
                summary.flagged[rec.id] = flag
        except Exception as exc:  # one bad song must not kill the batch
            summary.failed[rec.id] = str(exc)
    return summary


@dataclass
class ScrubSummary:
    cleaned: list[str] = field(default_factory=list)  # transcript rewritten: align again
    emptied: list[str] = field(default_factory=list)  # nothing real left: no lyrics now


def scrub(data_root: str | Path, *, cfg: LyricsConfig | None = None,
          ids: list[str] | None = None, dry_run: bool = False) -> ScrubSummary:
    """Drop hallucinated segments from machine lyrics written before the filter existed.

    A rewritten transcript must be aligned again, so ``aligned`` is cleared - a locked
    manifest edit, since stage commits only ever set flags. One left with fewer than
    ``min_words`` loses its lyrics, exactly like a transcription that failed (it was
    wordless singing). Hand-written lyrics are never touched.
    """
    data_root = Path(data_root)
    cfg = cfg or load_lyrics_config()
    summary = ScrubSummary()
    for rec in Manifest.for_data_root(data_root).records:
        if ids and rec.id not in ids:
            continue
        if not (rec.meta.lyrics_source or "").startswith(ASR_PREFIX):
            continue
        sdir = song_dir(data_root, rec.id)
        asr_path = sdir / "lyrics" / ASR_NAME
        if not asr_path.exists():
            continue
        payload = json.loads(asr_path.read_text(encoding="utf-8"))
        kept, dropped = drop_hallucinations(payload.get("segments", []))
        if not dropped:
            continue
        stats = lyrics_stats(kept)
        empty = stats["n_words"] < cfg.min_words
        (summary.emptied if empty else summary.cleaned).append(rec.id)
        if dry_run:
            continue
        payload = {**payload, "segments": kept,
                   "dropped_segments": payload.get("dropped_segments", []) + dropped}
        asr_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        if empty:
            (sdir / "lyrics" / LYRICS_NAME).unlink(missing_ok=True)
        else:
            (sdir / "lyrics" / LYRICS_NAME).write_text(_transcript(kept) + "\n",
                                                        encoding="utf-8")
        analysis = sdir / "analysis.json"
        prev = json.loads(analysis.read_text(encoding="utf-8")).get("lyrics", {}) \
            if analysis.exists() else {}
        update_analysis(sdir, "lyrics", {
            **prev, **stats, "flag": loop_flag(stats),
            "hallucinations_dropped": prev.get("hallucinations_dropped", 0) + len(dropped),
            "scrubbed": _dt.date.today().isoformat(),
            **({"failed": f"only {stats['n_words']} words after dropping hallucinations"}
               if empty else {})})
    if dry_run or not (summary.cleaned or summary.emptied):
        return summary
    path = data_root / MANIFEST_NAME
    with _manifest_lock(path):
        manifest = Manifest(path)
        for rid in summary.cleaned + summary.emptied:
            rec = manifest.get(rid)
            rec.status.aligned = False
            if rid in summary.emptied:
                rec.meta.has_lyrics = False
                rec.meta.lyrics_path = rec.meta.lyrics_source = None
        manifest.save()
    return summary
