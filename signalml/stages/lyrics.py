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
from ..manifest import Manifest, ManifestRecord
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
    if not cfg.command:
        raise RuntimeError(
            "no lyrics command configured - copy configs/lyrics.yaml to "
            "configs/lyrics.local.yaml and point `command` at the whisper venv + "
            "scripts/whisper_lyrics.py")

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = data_root / "work" / f"lyrics_{stamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    jobs, by_out = [], {}
    for rec in work:
        wav = song_dir(data_root, rec.id) / "clean" / f"{cfg.stem}.wav"
        if not wav.exists():
            summary.failed[rec.id] = f"no cleaned stem at {wav}"
            continue
        out = batch_dir / f"{rec.id}.json"
        jobs.append({"wav": str(wav), "out": str(out)})
        by_out[rec.id] = out
    jobs_path = batch_dir / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    if jobs:
        argv = [part.format(jobs=str(jobs_path)) for part in cfg.command]
        runner(argv, cfg.command_cwd, cfg.timeout_sec)

    for rec in work:
        out = by_out.get(rec.id)
        if out is None:
            continue
        try:
            if not out.exists():
                raise RuntimeError("backend wrote no result")
            payload = json.loads(out.read_text(encoding="utf-8"))
            if "error" in payload:
                raise RuntimeError(payload["error"])
            segments = payload.get("segments", [])
            stats = lyrics_stats(segments)
            if stats["n_words"] < cfg.min_words:
                raise RuntimeError(f"only {stats['n_words']} words recognised "
                                   "(instrumental, noise or not English?)")
            sdir = song_dir(data_root, rec.id)
            ldir = sdir / "lyrics"
            ldir.mkdir(parents=True, exist_ok=True)
            text = "\n".join(s["text"].strip() for s in segments if s["text"].strip())
            (ldir / LYRICS_NAME).write_text(text + "\n", encoding="utf-8")
            (ldir / ASR_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
            source = f"{ASR_PREFIX}{payload.get('model') or 'whisper'}"
            flag = loop_flag(stats)
            update_analysis(sdir, "lyrics", {
                "source": source, **stats, "flag": flag,
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
