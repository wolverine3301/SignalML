"""Check human lyrics against what is actually sung (``signalml lyrics --check``).

Pasted lyrics are right almost everywhere, and wrong exactly where a performance
departs from the page: a repeated chorus, an ad-lib, a skipped verse, an improvised
line. Whisper alone is not trusted for timing labels (alignments from its lyrics
matched hand-lyric alignments on only ~58-78% of sung time), but it is good at
*noticing* such departures. So: transcribe each song, align Whisper's words to the
human text, and flag only the spans where they disagree - with timestamps - for a
person to listen to. The human lyrics file is never modified.

Short disagreements are Whisper's own noise (~15% word error on singing) and are not
flagged; the thresholds below keep the list to whole-line departures.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import Manifest, ManifestRecord
from .common import song_dir, update_analysis
from .lyrics import ASR_NAME, LyricsConfig, load_lyrics_config, transcribe_batch

# Tuned 2026-09-25 on 29 hand-verified songs with planted departures (a line removed
# from / duplicated in the text): these thresholds catch 98% of them while cutting false
# flags on correct lyrics to ~1.5 per song; tighter ones stop catching real departures.
MIN_EXTRA = 4    # sung words not in the text before it counts as a departure
MIN_MISSING = 5  # text words not heard
MIN_DIFFERS = 6  # a replaced span, counted on its longer side


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " ").replace("’", "'"))


def timed_words(segments: list[dict]) -> list[tuple[str, float]]:
    """(word, start) for every recognised word - word timestamps when the backend gave
    them, else the segment's time shared evenly across its words."""
    out: list[tuple[str, float]] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                out += [(n, float(w["start"])) for n in norm_words(w["w"])]
            continue
        toks = norm_words(seg.get("text", ""))
        span = max(0.0, float(seg["end"]) - float(seg["start"]))
        out += [(t, float(seg["start"]) + span * i / max(1, len(toks)))
                for i, t in enumerate(toks)]
    return out


@dataclass
class Flag:
    kind: str       # "extra" (sung, not in text) | "missing" (in text, not heard) | "differs"
    time: float     # seconds into the vocal, where to listen
    text: str       # the human lyrics side ("" for extra)
    sung: str       # what Whisper heard ("" for missing)


@dataclass
class CheckResult:
    agreement: float
    flags: list[Flag]
    ref_words: int
    heard_words: int


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int]]:
    """Minimum-edit word alignment (Levenshtein with backtrace) as a list of
    ("=" | "sub" | "del" | "ins", i, j). A true alignment, unlike difflib's greedy
    longest-block matching, which repeated choruses throw far off."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = ref[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (ri != hyp[j - 1]), prev[j] + 1, row[j - 1] + 1)
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    return ops[::-1]


def compare(ref_text: str, segments: list[dict]) -> CheckResult:
    ref = norm_words(ref_text)
    heard = timed_words(segments)
    hyp = [w for w, _ in heard]
    ops = align_words(ref, hyp)
    matched = sum(1 for op, _, _ in ops if op == "=")
    flags: list[Flag] = []

    def when(j: int) -> float:
        return heard[max(0, min(j, len(heard) - 1))][1] if heard else 0.0

    # group each maximal run of non-matching ops into one span, then classify it
    run: list[tuple[str, int, int]] = []
    for op in [*ops, ("=", -1, -1)]:
        if op[0] != "=":
            run.append(op)
            continue
        if run:
            r_idx = [i for o, i, _ in run if o in ("sub", "del")]
            h_idx = [j for o, _, j in run if o in ("sub", "ins")]
            text = " ".join(ref[i] for i in r_idx)
            sung = " ".join(hyp[j] for j in h_idx)
            at = when(h_idx[0] if h_idx else run[0][2] - 1)
            if not r_idx and len(h_idx) >= MIN_EXTRA:
                flags.append(Flag("extra", at, "", sung))
            elif not h_idx and len(r_idx) >= MIN_MISSING:
                flags.append(Flag("missing", at, text, ""))
            elif r_idx and h_idx and max(len(r_idx), len(h_idx)) >= MIN_DIFFERS:
                flags.append(Flag("differs", at, text, sung))
            run = []
    return CheckResult(agreement=round(matched / max(1, len(ref)), 3), flags=flags,
                       ref_words=len(ref), heard_words=len(hyp))


@dataclass
class CheckSummary:
    checked: dict[str, CheckResult] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    report: Path | None = None


def _wants_check(rec: ManifestRecord, data_root: Path, prefix: str | None, force: bool) -> bool:
    if prefix and not rec.file.path.startswith(prefix):
        return False
    human = rec.meta.has_lyrics and not (rec.meta.lyrics_source or "").startswith("asr:")
    if not (human and rec.status.cleaned and rec.meta.domain == "sung"):
        return False
    if force:
        return True
    analysis = song_dir(data_root, rec.id) / "analysis.json"
    try:
        return "lyrics_check" not in json.loads(analysis.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def write_report(path: Path, data_root: Path, manifest: Manifest,
                 results: dict[str, CheckResult], failed: dict[str, str]) -> Path:
    look = sorted((rid for rid, r in results.items() if r.flags),
                  key=lambda rid: results[rid].agreement)
    fine = sorted(rid for rid, r in results.items() if not r.flags)
    lines = [f"# Lyrics check - {_dt.date.today().isoformat()}", "",
             f"{len(results)} song(s): **{len(look)} to look at**, {len(fine)} clean "
             "(no departures). Times are into the song; the lyrics file is the one "
             "to edit.", "",
             "- **SUNG, NOT IN TEXT** - add it (a repeat, an ad-lib)",
             "- **IN TEXT, NOT HEARD** - remove it (a skipped or shortened part)",
             "- **DIFFERS** - listen and keep whichever is right", ""]
    if look:
        lines += ["## To look at (worst first)", ""]
    for rid in look:
        rec, r = manifest.get(rid), results[rid]
        lines += [f"### {rid} - {rec.meta.singer} - {rec.meta.song} "
                  f"({r.agreement:.0%} agreement)",
                  f"file: `{(data_root / rec.meta.lyrics_path).as_posix()}`", ""]
        for f in sorted(r.flags, key=lambda f: f.time):
            if f.kind == "extra":
                lines.append(f"- {_mmss(f.time)} **SUNG, NOT IN TEXT:** \"{f.sung}\"")
            elif f.kind == "missing":
                lines.append(f"- {_mmss(f.time)} **IN TEXT, NOT HEARD:** \"{f.text}\"")
            else:
                lines.append(f"- {_mmss(f.time)} **DIFFERS:** text \"{f.text}\" / "
                             f"heard \"{f.sung}\"")
        lines.append("")
    if fine:
        lines += ["## Clean", ""]
        lines += [f"- {rid} - {manifest.get(rid).meta.singer} - {manifest.get(rid).meta.song} "
                  f"({results[rid].agreement:.0%})" for rid in fine]
        lines.append("")
    if failed:
        lines += ["## Could not check", ""] + [f"- {rid}: {why}" for rid, why in failed.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_check(
    data_root: str | Path,
    *,
    cfg: LyricsConfig | None = None,
    runner: Callable[[list[str], str | None, float], None] | None = None,
    prefix: str | None = None,
    ids: list[str] | None = None,
    force: bool = False,
    report: str | Path | None = None,
) -> CheckSummary:
    """Transcribe cleaned songs that have human lyrics, compare, write a review file."""
    from .lyrics import _default_runner

    data_root = Path(data_root)
    cfg = cfg or load_lyrics_config()
    manifest = Manifest.for_data_root(data_root)
    summary = CheckSummary()
    work = []
    for rec in manifest.records:
        if ids and rec.id not in ids:
            continue
        (work if _wants_check(rec, data_root, prefix, force) else summary.skipped).append(
            rec if _wants_check(rec, data_root, prefix, force) else rec.id)
    if not work:
        return summary
    payloads = transcribe_batch(data_root, work, cfg, runner or _default_runner)
    for rec in work:
        try:
            payload = payloads[rec.id]
            if isinstance(payload, str):
                raise RuntimeError(payload)
            sdir = song_dir(data_root, rec.id)
            ref_text = (data_root / rec.meta.lyrics_path).read_text(encoding="utf-8",
                                                                   errors="replace")
            result = compare(ref_text, payload.get("segments", []))
            (sdir / "lyrics").mkdir(parents=True, exist_ok=True)
            (sdir / "lyrics" / ASR_NAME).write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            update_analysis(sdir, "lyrics_check", {
                "agreement": result.agreement, "n_flags": len(result.flags),
                "model": payload.get("model"), "date": _dt.date.today().isoformat()})
            summary.checked[rec.id] = result
        except Exception as exc:  # one bad song must not kill the batch
            summary.failed[rec.id] = str(exc)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary.report = write_report(
        Path(report) if report else data_root / "review" / f"lyrics_check_{stamp}.md",
        data_root, manifest, summary.checked, summary.failed)
    return summary
