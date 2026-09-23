"""S5b transcribe — aligned vocals to note labels (DECISION_POINTS D1).

The acoustic model trains on phonemes + durations, which P5 already provides. The
*variance* model additionally needs a score: which pitch, over which span. That is
the D1 gap, and it is what stands between a system that re-sings takes it has heard
and one that sings a score handed to it.

This stage is the manifest-driven half of D1, and it is deliberately backend-agnostic
because the backend question is Logan's to answer (see `docs/notes/note_transcription.md`):

- **SOME** (openvpi, MIT) — 9x realtime on *CPU*, outputs MIDI with float pitches
  explicitly for DiffSinger variance labelling. Superseded upstream.
- **GAME** (openvpi, MIT) — SOME's successor, diffusion-based, pins CUDA 12.9 and
  emits note CSV directly.

Both are external tools with their own pins, so they are invoked the way MFA and the
trainer are: a command in a config, run in its own environment, injectable for tests.
Nothing here imports them.

Output per song, `songs/<id>/transcribe/notes.json`::

    {"transcriber": "some/v0.0.1", "stem": "vocals", "notes": [
       {"start": 1.23, "end": 1.58, "midi": 67.4, "confidence": 0.9}, ...]}

`midi` stays *floating point* on purpose: the trainer's `note_seq` wants note names,
but rounding is a lossy decision (vibrato, slides, blue notes) that belongs to the
converter which builds variance transcriptions.csv, not to the record of what was
heard. Converting notes.json + phones.json into `note_seq` / `note_dur` / `ph_num`
is S6b's job once D1 picks a backend.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Protocol

import yaml
from pydantic import BaseModel

from ..config import CONFIGS_DIR
from ..manifest import Manifest, ManifestRecord
from .common import song_dir, update_analysis

NOTES_NAME = "notes.json"


class Note(BaseModel):
    start: float
    end: float
    midi: float  # float, not int: quantization is the converter's call
    confidence: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


class TranscribeConfig(BaseModel):
    """Machine-local wiring for whichever transcriber D1 settles on."""

    backend: Literal["some", "game", "external"] = "some"
    stem: str = "vocals"
    # Argv template run per song; {wav} and {out} are substituted. Empty = the backend
    # has no default command on this machine and the stage refuses rather than guesses.
    command: list[str] = []
    # Where the tool writes its result, relative to the temp dir ({out} above).
    output_name: str = "notes.mid"
    # Working directory for the command. SOME's infer.py does `import inference`, so
    # it only resolves from its own checkout — the same reason MFA gets a corpus dir.
    command_cwd: str | None = None
    # Notes shorter than this are dropped: transcribers emit slivers at consonant
    # onsets that are not notes anyone sang.
    min_note_sec: float = 0.05
    timeout_sec: float = 600.0


def load_transcribe_config(path: str | Path | None = None) -> TranscribeConfig:
    path = Path(path) if path else CONFIGS_DIR / "transcribe.yaml"
    if Path(path).exists():
        return TranscribeConfig.model_validate(
            yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        )
    return TranscribeConfig()


@dataclass
class TranscribeSummary:
    transcribed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    notes_written: int = 0


class Transcriber(Protocol):
    """A backend turns one vocal wav into notes. Implementations live outside."""

    name: str

    def transcribe(self, wav: Path, work_dir: Path) -> list[Note]:
        ...


class CommandTranscriber:
    """Runs an external transcriber per song and reads back its note file.

    Same posture as MFA and the trainer: their environment, their pins, our contract
    at the boundary. Accepts the two formats openvpi's tools emit - MIDI, or a CSV
    with onset/offset/pitch columns.
    """

    def __init__(self, cfg: TranscribeConfig, runner: Callable | None = None):
        self.cfg = cfg
        self.name = cfg.backend
        self._runner = runner or subprocess.run

    def transcribe(self, wav: Path, work_dir: Path) -> list[Note]:
        if not self.cfg.command:
            raise RuntimeError(
                "no transcriber command configured — set `command` in "
                "configs/transcribe.yaml (see docs/notes/note_transcription.md for the "
                "SOME and GAME invocations), or pass an explicit backend"
            )
        out = work_dir / self.cfg.output_name
        argv = [part.format(wav=str(wav), out=str(out)) for part in self.cfg.command]
        proc = self._runner(argv, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=self.cfg.timeout_sec,
                            cwd=self.cfg.command_cwd)
        if getattr(proc, "returncode", 0) != 0:
            tail = (getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "")
            raise RuntimeError(f"{self.name} failed: {tail.strip()[-400:]}")
        if not out.exists():
            raise RuntimeError(f"{self.name} wrote no {out.name} in {work_dir}")
        return read_notes(out)


def read_notes(path: Path) -> list[Note]:
    """Read a transcriber's output: MIDI, or CSV/TSV with onset/offset/pitch."""
    suffix = path.suffix.lower()
    if suffix in (".mid", ".midi"):
        return _read_midi(path)
    if suffix in (".csv", ".tsv", ".txt"):
        return _read_table(path)
    raise RuntimeError(f"unsupported transcriber output {path.name}")


def _read_midi(path: Path) -> list[Note]:
    try:
        import mido
    except ImportError as exc:  # pragma: no cover - dependency-gated path
        raise RuntimeError(
            "reading MIDI note files needs `mido` (python -m uv sync --extra train)"
        ) from exc

    midi = mido.MidiFile(str(path))
    notes: list[Note] = []
    open_notes: dict[int, float] = {}
    now = 0.0
    for msg in midi:  # mido yields wall-clock deltas when iterating the file
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes[msg.note] = now
        elif msg.type in ("note_off",) or (msg.type == "note_on" and msg.velocity == 0):
            start = open_notes.pop(msg.note, None)
            if start is not None:
                notes.append(Note(start=start, end=now, midi=float(msg.note)))
    return sorted(notes, key=lambda n: n.start)


def _read_table(path: Path) -> list[Note]:
    import csv

    text = path.read_text(encoding="utf-8").strip().splitlines()
    if not text:
        return []
    dialect = csv.Sniffer().sniff(text[0]) if "," in text[0] or "\t" in text[0] else None
    rows = list(csv.DictReader(text, dialect=dialect) if dialect
                else csv.DictReader(text))
    notes = []
    for row in rows:
        keys = {k.lower().strip(): (v or "").strip() for k, v in row.items() if k}
        start = keys.get("onset") or keys.get("start") or keys.get("start_time")
        end = keys.get("offset") or keys.get("end") or keys.get("end_time")
        midi = keys.get("pitch") or keys.get("midi") or keys.get("note")
        if start is None or end is None or midi is None:
            raise RuntimeError(
                f"{path.name}: expected onset/offset/pitch columns, got {sorted(keys)}")
        conf = keys.get("confidence")
        notes.append(Note(start=float(start), end=float(end), midi=float(midi),
                          confidence=float(conf) if conf else None))
    return sorted(notes, key=lambda n: n.start)


def coverage(notes: list[Note], phones: list[dict]) -> float | None:
    """Fraction of voiced (non-silence) aligned phone time covered by a note.

    The bake-off metric that needs no ground truth: a transcriber that misses half
    the melody still produces plausible-looking notes, and this is what catches it.
    """
    sung = [(p["start"], p["end"]) for p in phones
            if p.get("ph") not in ("SP", "AP", "sil", "sp", "") and not p.get("noise")]
    total = sum(end - start for start, end in sung)
    if total <= 0:
        return None
    covered = 0.0
    for start, end in sung:
        for note in notes:
            lo, hi = max(start, note.start), min(end, note.end)
            if hi > lo:
                covered += hi - lo
    return round(min(covered, total) / total, 4)


def transcribe_song(
    sdir: Path,
    cfg: TranscribeConfig,
    backend: Transcriber,
    work_dir: Path,
) -> tuple[list[Note], dict]:
    """Transcribe one song's clean stem; returns (notes, analysis payload)."""
    wav = sdir / "clean" / f"{cfg.stem}.wav"
    if not wav.exists():
        raise RuntimeError(f"no cleaned stem at {wav}")
    notes = [n for n in backend.transcribe(wav, work_dir)
             if n.duration >= cfg.min_note_sec]

    phones_path = sdir / "align" / "phones.json"
    cover = None
    if phones_path.exists():
        payload = json.loads(phones_path.read_text(encoding="utf-8"))
        cover = coverage(notes, payload.get("phones", []))

    analysis = {
        "transcriber": backend.name,
        "stem": cfg.stem,
        "n_notes": len(notes),
        "sung_sec": round(sum(n.duration for n in notes), 3),
        "note_coverage": cover,
        "min_note_sec": cfg.min_note_sec,
        "date": _dt.date.today().isoformat(),
    }
    return notes, analysis


def write_notes(sdir: Path, notes: list[Note], analysis: dict) -> Path:
    out_dir = sdir / "transcribe"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / NOTES_NAME
    path.write_text(json.dumps({
        "transcriber": analysis["transcriber"],
        "stem": analysis["stem"],
        "note_coverage": analysis["note_coverage"],
        "notes": [n.model_dump(exclude_none=True) for n in notes],
    }, indent=2), encoding="utf-8")
    return path


def run(
    data_root: str | Path,
    *,
    cfg: TranscribeConfig | None = None,
    backend: Transcriber | None = None,
    force: bool = False,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> TranscribeSummary:
    """Manifest-driven: transcribe every aligned song that has no notes yet."""
    data_root = Path(data_root)
    cfg = cfg or load_transcribe_config()
    backend = backend or CommandTranscriber(cfg)
    manifest = Manifest.for_data_root(data_root)
    summary = TranscribeSummary()

    work: list[ManifestRecord] = []
    for rec in manifest.records:
        if ids and rec.id not in ids:
            continue
        ready = rec.status.aligned and (force or not rec.status.transcribed)
        (work if ready else summary.skipped).append(rec if ready else rec.id)
    if limit is not None:
        work = work[:limit]

    for rec in work:
        sdir = song_dir(data_root, rec.id)
        try:
            work_dir = sdir / "transcribe" / "_work"
            work_dir.mkdir(parents=True, exist_ok=True)
            notes, analysis = transcribe_song(sdir, cfg, backend, work_dir)
            write_notes(sdir, notes, analysis)
            update_analysis(sdir, "transcribe", analysis)
            rec.status.transcribed = True
            manifest.commit(rec)
            summary.transcribed.append(rec.id)
            summary.notes_written += len(notes)
        except Exception as exc:  # one bad song must not kill the batch
            summary.failed[rec.id] = str(exc)
    return summary
