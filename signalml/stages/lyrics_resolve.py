"""Resolve pasted lyrics against what is sung (``signalml lyrics --resolve``).

Official lyrics are right almost everywhere and wrong in predictable places, and a
cover is where they are wrong most: a parenthesised echo from the studio version that
the cover does or does not sing, a verse the cover skips, a section header. MFA makes
all of those worse - it turns a parenthesised group into one ``[bracketed]`` token
aligned as a ~30 ms ``spn``, so a sung echo's audio gets smeared across its neighbours
and an unsung one steals time from them.

``lyrics --check`` already sees these departures (Whisper's words aligned to the human
words) but only reports them for a person to listen to. This stage *decides*, from the
same alignment, and writes a resolved copy for the aligner to read:

- a ``(parenthesised group)`` Whisper heard is kept, brackets removed; one where nothing
  was sung is dropped; one where *something else* was sung is kept but marked unsure;
- ``[Chorus]``-style headers and ``(x2)``-style instructions are dropped;
- a run of whole lines nobody sang (a cut verse) is dropped;
- a line Whisper heard as something else entirely is kept but marked unsure.

Unsure lines are carried by line number into the phrase alignment, where the phrase
gate can leave their phrase out of a dataset. Singing that is *not in the text* is not
added (Whisper's words are too unreliable to be labels); the phrase planner keeps that
audio out of every utterance instead.

The human lyrics file is never modified. Output: ``lyrics/resolved.txt`` and
``lyrics/resolve.json`` (every decision, with its evidence).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..manifest import Manifest, ManifestRecord
from .common import song_dir, update_analysis
from .lyrics import ASR_NAME, ASR_PREFIX
from .lyrics_check import MIN_MISSING, norm_words

RESOLVED_NAME = "resolved.txt"
RESOLVE_JSON = "resolve.json"
SOURCE_NAME = "lyrics.source.txt"  # what import-batch keeps of the pasted lyrics
RESOLVE_VERSION = 2  # 2: aligner spelling (normalize_for_alignment), pasted source

# A parenthesised group counts as sung when at least this share of its words matched.
HEARD_FRAC = 0.5
# Below this whole-song agreement Whisper (or the pasted text) is too far off to decide
# anything line by line: every line is kept and marked unsure.
MIN_SONG_AGREEMENT = 0.4
# A line this badly matched, with at least this many words, is marked unsure.
UNSURE_LINE_FRAC = 0.25
UNSURE_LINE_WORDS = 3

# Edit costs, doubled so a parenthesised word's deletion can cost half: when the text
# says "I know (I know)" and one "I know" is sung, the echo is the one to call unsung.
_SUB, _INS, _DEL, _DEL_PAREN = 2, 2, 2, 1

_SECTION_LINE = re.compile(
    r"^\s*[\[(]?\s*(intro|outro|verse|pre-?chorus|chorus|post-?chorus|bridge|hook|"
    r"refrain|interlude|breakdown|instrumental)(\s*\d+)?\s*:?\s*[\])]?\s*:?\s*$", re.I)
_INSTRUCTION = re.compile(
    r"^\s*(x\s*\d+|\d+\s*x|repeat.*|chorus|verse.*|bridge|hook|outro|intro|"
    r"pre-?chorus|instrumental|spoken|whisper(ed)?|echo)\s*$", re.I)
_BRACKET = re.compile(r"\(([^()]*)\)?|\[[^\]]*\]?")


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _number_words(rest))
    return str(n)


def normalize_for_alignment(s: str) -> str:
    """The aligner's spelling of a lyric line: typographic quotes and dashes to plain
    ASCII, stutter/compound hyphens to spaces ("to-to-touch", "oh-oh"), ``&`` to "and",
    small numbers to words, stray symbols dropped. Every one of those otherwise reaches
    MFA as an out-of-dictionary token, which aligns as spn and costs the whole clip."""
    s = s.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    s = s.replace("`", "'").replace("´", "'")
    s = re.sub(r"[“”„\"]", " ", s)
    s = re.sub(r"[‐-―−]", " ", s)          # typographic dashes
    s = re.sub(r"(?<=[^\W\d_])-(?=[^\W\d_])", " ", s)       # to-to-touch, oh-oh
    s = s.replace("&", " and ").replace("…", " ")
    s = re.sub(r"\b\d{1,3}\b", lambda m: _number_words(int(m.group(0))), s)
    s = re.sub(r"[^\w' ,.!?;:]|_", " ", s)                  # letters (any script) stay
    return " ".join(s.split())


def _clean(s: str) -> str:
    """Stray bracket characters out, aligner spelling, whitespace collapsed."""
    return normalize_for_alignment(re.sub(r"[()\[\]{}]", " ", s))


@dataclass
class _Piece:
    text: str
    group: int | None  # paren group id, None = main text


@dataclass
class _Line:
    text: str
    pieces: list[_Piece]
    section: bool = False


def _parse(text: str) -> tuple[list[_Line], list[dict]]:
    """Lines as main-text / paren-group pieces. ``[...]`` and instruction parens
    are dropped here - they are never sung."""
    lines: list[_Line] = []
    groups: list[dict] = []
    for n, raw in enumerate(text.splitlines()):
        if not raw.strip():
            continue
        if _SECTION_LINE.match(raw):
            lines.append(_Line(raw, [], section=True))
            continue
        pieces: list[_Piece] = []
        pos = 0
        for m in _BRACKET.finditer(raw):
            if m.start() > pos:
                pieces.append(_Piece(raw[pos:m.start()], None))
            pos = m.end()
            if m.group(0).startswith("["):
                continue  # [annotation]
            inner = m.group(1) or ""
            if not norm_words(inner) or _INSTRUCTION.match(inner):
                groups.append({"line": n, "text": inner.strip(), "decision": "instruction"})
                continue
            groups.append({"line": n, "text": inner.strip(), "decision": None})
            pieces.append(_Piece(inner, len(groups) - 1))
        if pos < len(raw):
            pieces.append(_Piece(raw[pos:], None))
        lines.append(_Line(raw, pieces))
    return lines, groups


def plain_lyrics(text: str) -> str:
    """Brackets off, words kept; section headers and instructions gone. The text to hand
    an aligner when no resolution exists - never ``(...)``, which MFA turns into spn."""
    lines, _ = _parse(text)
    out = []
    for ln in lines:
        if ln.section:
            continue
        s = " ".join(_clean(p.text) for p in ln.pieces if norm_words(p.text))
        if norm_words(s):
            out.append(s)
    return "\n".join(out)


def _align_weighted(ref: list[str], del_cost: list[int], hyp: list[str]
                    ) -> list[tuple[str, int, int]]:
    """``lyrics_check.align_words`` with a per-reference-word deletion cost."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = d[i - 1][0] + del_cost[i - 1]
    for j in range(1, m + 1):
        d[0][j] = j * _INS
    for i in range(1, n + 1):
        ri, row, prev, dc = ref[i - 1], d[i], d[i - 1], del_cost[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (0 if ri == hyp[j - 1] else _SUB),
                         prev[j] + dc, row[j - 1] + _INS)
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else _SUB):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + del_cost[i - 1]:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    return ops[::-1]


def heard_words(segments: list[dict]) -> list[str]:
    out: list[str] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                out += norm_words(w["w"])
        else:
            out += norm_words(seg.get("text", ""))
    return out


@dataclass
class Resolution:
    text: str                      # resolved lyrics, one non-empty line per line
    unsure_lines: list[int]        # indices into text's lines
    agreement: float               # matched share of the (unresolved) text's words
    lines: list[dict] = field(default_factory=list)   # per input line: what happened
    groups: list[dict] = field(default_factory=list)  # per (...) group: decision + evidence

    def counts(self) -> dict:
        c: dict[str, int] = {}
        for g in self.groups:
            c[f"paren_{g['decision']}"] = c.get(f"paren_{g['decision']}", 0) + 1
        for ln in self.lines:
            c[f"line_{ln['status']}"] = c.get(f"line_{ln['status']}", 0) + 1
        return c


def resolve(text: str, segments: list[dict]) -> Resolution:
    lines, groups = _parse(text)
    ref: list[str] = []
    dcost: list[int] = []
    owner: list[tuple[int, int | None]] = []  # (line index in `lines`, group)
    for li, ln in enumerate(lines):
        for p in ln.pieces:
            for w in norm_words(p.text):
                ref.append(w)
                dcost.append(_DEL if p.group is None else _DEL_PAREN)
                owner.append((li, p.group))
    hyp = heard_words(segments)
    ops = _align_weighted(ref, dcost, hyp)

    status = ["del"] * len(ref)
    run_of = [-1] * len(ref)  # index of the non-matching run a word sits in
    runs: list[dict] = []
    cur: dict | None = None
    for op, i, _j in ops:
        if op == "=":
            status[i], cur = "=", None
            continue
        if cur is None:
            cur = {"ref": [], "hyp": 0}
            runs.append(cur)
        if op in ("sub", "del"):
            status[i] = op
            run_of[i] = len(runs) - 1
            cur["ref"].append(i)
        if op in ("sub", "ins"):
            cur["hyp"] += 1
    agreement = round(sum(s == "=" for s in status) / max(1, len(ref)), 3)
    trust = agreement >= MIN_SONG_AGREEMENT and bool(hyp)

    # parenthesised groups
    for gi, g in enumerate(groups):
        if g["decision"] == "instruction":
            continue
        idx = [i for i, (_, grp) in enumerate(owner) if grp == gi]
        matched = sum(status[i] == "=" for i in idx)
        # sung words in the same non-matching runs, minus the main-text words there
        # that could have claimed them
        rs = {run_of[i] for i in idx if run_of[i] >= 0}
        main_in_runs = sum(1 for r in rs for i in runs[r]["ref"] if owner[i][1] is None)
        unexplained = max(0, sum(runs[r]["hyp"] for r in rs) - main_in_runs)
        g.update(n_words=len(idx), matched=matched, unexplained_sung=unexplained)
        if not trust:
            g["decision"] = "unsure"
        elif matched >= HEARD_FRAC * len(idx):
            g["decision"] = "keep"
        elif unexplained >= max(1, (len(idx) - matched + 1) // 2):
            g["decision"] = "unsure"
        else:
            g["decision"] = "drop"

    # whole lines nobody sang: every main word deleted in runs with nothing sung
    silent_run = [not r["hyp"] and len(r["ref"]) >= MIN_MISSING for r in runs]
    out_lines: list[str] = []
    unsure: list[int] = []
    line_log: list[dict] = []
    for li, ln in enumerate(lines):
        if ln.section:
            line_log.append({"text": ln.text, "status": "section"})
            continue
        main = [i for i, (l2, grp) in enumerate(owner) if l2 == li and grp is None]
        m_frac = sum(status[i] == "=" for i in main) / len(main) if main else 1.0
        if trust and main and all(status[i] == "del" and silent_run[run_of[i]] for i in main):
            line_log.append({"text": ln.text, "status": "dropped_unheard"})
            continue
        parts, line_unsure = [], not trust
        for p in ln.pieces:
            if p.group is not None:
                dec = groups[p.group]["decision"]
                if dec == "drop":
                    continue
                line_unsure |= dec == "unsure"
            s = _clean(p.text)
            if norm_words(s):
                parts.append(s)
        out = " ".join(parts)
        if not norm_words(out):
            line_log.append({"text": ln.text, "status": "emptied"})
            continue
        line_unsure |= len(main) >= UNSURE_LINE_WORDS and m_frac < UNSURE_LINE_FRAC
        status_ = "unsure" if line_unsure else "kept"
        line_log.append({"text": ln.text, "status": status_, "out": out,
                         "matched_frac": round(m_frac, 3)})
        if line_unsure:
            unsure.append(len(out_lines))
        out_lines.append(out)
    return Resolution(text="\n".join(out_lines), unsure_lines=unsure, agreement=agreement,
                      lines=line_log, groups=groups)


def human_lyrics_source(lyrics_path: Path) -> Path:
    """The text to resolve: ``manifest import-batch`` tidies ``lyrics.txt`` (brackets
    stripped, words kept) and keeps what was pasted as ``lyrics.source.txt`` beside it;
    only the pasted text still says which words were in parentheses."""
    source = lyrics_path.with_name(SOURCE_NAME)
    return source if lyrics_path.name == "lyrics.txt" and source.exists() else lyrics_path


def load_resolution(sdir: Path) -> tuple[str, list[int]] | None:
    """(resolved text, unsure line indices) if this song has been resolved."""
    txt, meta = sdir / "lyrics" / RESOLVED_NAME, sdir / "lyrics" / RESOLVE_JSON
    if not (txt.exists() and meta.exists()):
        return None
    data = json.loads(meta.read_text(encoding="utf-8"))
    return txt.read_text(encoding="utf-8"), list(data.get("unsure_lines", []))


@dataclass
class ResolveSummary:
    resolved: dict[str, Resolution] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def _wants_resolve(rec: ManifestRecord, data_root: Path, force: bool) -> bool:
    human = rec.meta.has_lyrics and rec.meta.lyrics_path and \
        not (rec.meta.lyrics_source or "").startswith(ASR_PREFIX)
    if not (human and rec.meta.domain == "sung"):
        return False
    lyr = song_dir(data_root, rec.id) / "lyrics"
    if not (lyr / ASR_NAME).exists():
        return False
    return force or not (lyr / RESOLVE_JSON).exists()


def run_resolve(data_root: str | Path, *, ids: list[str] | None = None,
                force: bool = False) -> ResolveSummary:
    """Resolve every human-lyrics song that has a Whisper transcript (``asr.json``, from
    ``lyrics --check``). CPU only; no model runs here."""
    data_root = Path(data_root)
    manifest = Manifest.for_data_root(data_root)
    summary = ResolveSummary()
    for rec in manifest.records:
        if ids and rec.id not in ids:
            continue
        if not _wants_resolve(rec, data_root, force):
            summary.skipped.append(rec.id)
            continue
        try:
            sdir = song_dir(data_root, rec.id)
            asr = json.loads((sdir / "lyrics" / ASR_NAME).read_text(encoding="utf-8"))
            if "error" in asr:
                raise RuntimeError(f"asr.json holds an error: {asr['error']}")
            source = human_lyrics_source(data_root / rec.meta.lyrics_path)
            text = source.read_text(encoding="utf-8", errors="replace")
            if source.name == SOURCE_NAME:
                from ..ingest.batch import cut_note

                text = cut_note(text)[0]
            res = resolve(text, asr.get("segments", []))
            (sdir / "lyrics" / RESOLVED_NAME).write_text(res.text + "\n", encoding="utf-8")
            (sdir / "lyrics" / RESOLVE_JSON).write_text(json.dumps({
                "version": RESOLVE_VERSION,
                "source": source.relative_to(data_root).as_posix(),
                "asr_model": asr.get("model"), "agreement": res.agreement,
                "unsure_lines": res.unsure_lines, "counts": res.counts(),
                "lines": res.lines, "groups": res.groups,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            update_analysis(sdir, "lyrics_resolve", {
                "version": RESOLVE_VERSION, "agreement": res.agreement,
                "n_unsure_lines": len(res.unsure_lines), **res.counts(),
                "date": _dt.date.today().isoformat()})
            summary.resolved[rec.id] = res
        except Exception as exc:  # one bad song must not kill the batch
            summary.failed[rec.id] = str(exc)
    return summary
