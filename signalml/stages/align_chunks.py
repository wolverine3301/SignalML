"""Split a song into phrase-sized utterances for MFA, anchored by Whisper word times.

Why: MFA is an utterance aligner (tens of seconds). Handed a whole 3-4 minute song as
one utterance it degrades silently - on the 2026-09 corpus the median song had 45% of
its phones stacked at the 30 ms frame floor and single phones stretched over 80 s, and
``align_score`` (coverage) cannot see it. Aligning phrase by phrase fixes the search.

How: the human lyrics stay the only text MFA sees. Whisper is used for *timing only*:
its words are matched to the human words (the same minimum-edit alignment as
``lyrics --check``), each lyric line with confident matches gets a rough time span, and
the song is cut at the quietest point between two consecutive anchored lines - when
that point is a real dip (a breath or a rest), not a legato join. The S4 silence map is
too coarse for this: reverb and residual bleed keep a separated vocal above its -40 dB
threshold through most line breaks, while the dip relative to the singing around it is
unmistakable (>= 12 dB on ~75% of line breaks, measured 2026-09-26).

Lines with no confident anchor are never cut around - they ride inside the utterance of
their anchored neighbours, so an unsure line widens an utterance instead of being
misplaced. Singing outside every utterance (intros, outros after the last lyric) stays
unaligned: no phones there, so no training clip claims it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lyrics_check import align_words, norm_words

HOP_SEC = 0.01
# Whisper's word edges on sung audio drift by up to ~half a second; search that far
# past the anchors for the dip to cut at.
EDGE_TOL_SEC = 0.5
# A cut needs the smoothed level this far below the loud part of the singing around it.
MIN_DIP_DB = 12.0
SMOOTH_SEC = 0.08
CONTEXT_SEC = 3.0
# Margin kept before the first and after the last anchored word of the song.
EDGE_MARGIN_SEC = 1.0
# Leave the audio between two consecutive lines out of both utterances when at least
# this many heard words there belong to no line, or when the gap is this long.
MIN_EXTRA_WORDS = 3
MAX_GAP_SEC = 4.0
HOLD_SEC = 2.0  # how far past a line's last heard word its final note may still ring


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str
    lines: tuple[int, ...]  # indices into the non-empty lyric lines


def frame_level_db(y: np.ndarray, sr: int, hop_sec: float = HOP_SEC,
                   win_sec: float = 0.04) -> np.ndarray:
    """RMS level in dB per hop (mono float audio)."""
    hop, win = max(1, int(sr * hop_sec)), max(1, int(sr * win_sec))
    n = max(1, (len(y) - win) // hop + 1)
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = np.asarray(y, dtype=np.float32)[np.minimum(idx, len(y) - 1)]
    return 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)


def timed_words(segments: list[dict]) -> list[tuple[str, float, float]]:
    """(word, start, end) for every recognised word; segment time is shared evenly
    when the backend gave no word timestamps."""
    out: list[tuple[str, float, float]] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                s, e = float(w["start"]), float(w["end"])
                out += [(t, s, e) for t in norm_words(w["w"])]
            continue
        toks = norm_words(seg.get("text", ""))
        s0, span = float(seg["start"]), max(0.0, float(seg["end"]) - float(seg["start"]))
        step = span / max(1, len(toks))
        out += [(t, s0 + step * i, s0 + step * (i + 1)) for i, t in enumerate(toks)]
    return out


@dataclass(frozen=True)
class LineEvidence:
    spans: list[tuple[float, float] | None]   # confident span per line, else None
    claims: list[tuple[float, float] | None]  # span of every heard word the line took
    extra: list[float]                        # start times of heard words no line took


def line_evidence(lines: list[str], segments: list[dict]) -> LineEvidence:
    ref: list[str] = []
    line_of: list[int] = []
    for li, line in enumerate(lines):
        toks = norm_words(line)
        ref += toks
        line_of += [li] * len(toks)
    heard = timed_words(segments)
    ops = align_words(ref, [w for w, _, _ in heard])
    match = {i: j for op, i, j in ops if op == "="}
    spans: list[list[float] | None] = [None] * len(lines)
    claims: list[list[float] | None] = [None] * len(lines)

    def paired(i: int, j: int, d: int) -> bool:  # neighbour matched too, same line
        k = i + d
        return 0 <= k < len(ref) and line_of[k] == line_of[i] and match.get(k) == j + d

    def widen(store: list, li: int, s: float, e: float) -> None:
        if store[li] is None:
            store[li] = [s, e]
        else:
            store[li][0], store[li][1] = min(store[li][0], s), max(store[li][1], e)

    extra: list[float] = []
    for op, i, j in ops:
        if op == "ins":
            extra.append(heard[j][1])
        elif op in ("=", "sub"):
            widen(claims, line_of[i], heard[j][1], heard[j][2])
    for i, j in match.items():
        if paired(i, j, -1) or paired(i, j, 1):
            widen(spans, line_of[i], heard[j][1], heard[j][2])
    return LineEvidence([tuple(s) if s else None for s in spans],
                        [tuple(c) if c else None for c in claims], extra)


def line_spans(lines: list[str], segments: list[dict]) -> list[tuple[float, float] | None]:
    """Rough (start, end) per lyric line from its confidently matched words, else None.

    A match counts only when a neighbouring word matched too: a lone "you" or "the"
    lines up with the wrong occurrence often enough to drag a line across the song.
    """
    return line_evidence(lines, segments).spans


def find_dip(level_db: np.ndarray, lo: float, hi: float,
             hop_sec: float = HOP_SEC) -> float | None:
    """Time of the quietest point in [lo, hi] if it is a real dip, else None."""
    a, b = max(0, int(lo / hop_sec)), min(len(level_db), int(hi / hop_sec))
    k = max(1, int(SMOOTH_SEC / hop_sec))
    if b - a < k:
        return None
    smooth = np.convolve(level_db[a:b], np.ones(k) / k, mode="valid")
    c = int(CONTEXT_SEC / hop_sec)
    loud = np.percentile(level_db[max(0, a - c):min(len(level_db), b + c)], 75)
    i = int(np.argmin(smooth))
    if loud - smooth[i] < MIN_DIP_DB:
        return None
    return (a + i + k / 2) * hop_sec


def _excised_gap(ev: LineEvidence, a: int, b: int, level_db: np.ndarray,
                 hop_sec: float) -> tuple[float, float] | None:
    """(end of line a's utterance, start of line b's) when the audio between should
    belong to neither, else None. Both edges must be real dips near their line - a
    held last note can run past Whisper's word end, so the search reaches
    ``HOLD_SEC`` beyond it - and without two dips nothing is cut out: chopping a
    sustained vowel is worse than an utterance that is too wide."""
    a_end = max(ev.spans[a][1], (ev.claims[a] or ev.spans[a])[1])
    b_start = min(ev.spans[b][0], (ev.claims[b] or ev.spans[b])[0])
    n_extra = sum(1 for t in ev.extra if a_end < t < b_start)
    if b_start - a_end < MAX_GAP_SEC and n_extra < MIN_EXTRA_WORDS:
        return None
    mid = (a_end + b_start) / 2
    t1 = find_dip(level_db, a_end - EDGE_TOL_SEC, min(a_end + HOLD_SEC, mid), hop_sec)
    t2 = find_dip(level_db, max(b_start - HOLD_SEC, mid), b_start + EDGE_TOL_SEC, hop_sec)
    if t1 is None or t2 is None or t2 <= t1:
        return None
    return t1, t2


def plan_utterances(
    lyrics_text: str,
    segments: list[dict],
    level_db: np.ndarray,
    duration: float,
    hop_sec: float = HOP_SEC,
    excise_gaps: bool = True,
) -> list[Utterance]:
    """Cut the song into utterances at dips between consecutive anchored lyric lines.

    With ``excise_gaps``, the stretch between two lines is left out of both utterances
    when something was sung there that no line accounts for (an ad-lib, a repeat the
    text lacks, a line the resolve step dropped) or when it is long: no text covers
    that audio, and inside an utterance MFA would force neighbouring words over it."""
    lines = [ln.strip() for ln in lyrics_text.splitlines() if norm_words(ln)]
    if not lines:
        return []
    ev = line_evidence(lines, segments)
    spans = ev.spans
    anchored = [i for i, s in enumerate(spans) if s]
    if not anchored:
        return [Utterance(0.0, round(duration, 3), " ".join(lines), tuple(range(len(lines))))]

    # (first line after the cut, end of the utterance before, start of the one after);
    # the two times differ when the audio between is left out of both
    cuts: list[tuple[int, float, float]] = []
    for a, b in zip(anchored, anchored[1:]):
        if b != a + 1:  # unanchored lines between: keep them inside one utterance
            continue
        if excise_gaps:
            gap = _excised_gap(ev, a, b, level_db, hop_sec)
            if gap is not None and (not cuts or gap[0] > cuts[-1][2]):
                cuts.append((b, *gap))
                continue
        lo, hi = spans[a][1] - EDGE_TOL_SEC, spans[b][0] + EDGE_TOL_SEC
        t = find_dip(level_db, lo, hi, hop_sec) if hi > lo else None
        if t is not None and (not cuts or t > cuts[-1][2]):
            cuts.append((b, t, t))

    # song edges: a margin around the first / last anchored line, unless unanchored
    # lyric lines lie beyond it - then the whole edge, so they have audio to land in
    start = max(0.0, spans[anchored[0]][0] - EDGE_MARGIN_SEC) if anchored[0] == 0 else 0.0
    last = anchored[-1]
    end = (min(duration, spans[last][1] + EDGE_MARGIN_SEC)
           if last == len(lines) - 1 else duration)

    out: list[Utterance] = []
    line0, t0 = 0, start
    for b, t_end, t_next in cuts:
        out.append(Utterance(round(t0, 3), round(t_end, 3), " ".join(lines[line0:b]),
                             tuple(range(line0, b))))
        line0, t0 = b, t_next
    out.append(Utterance(round(t0, 3), round(end, 3), " ".join(lines[line0:]),
                         tuple(range(line0, len(lines)))))
    return [u for u in out if u.end > u.start]


FLOOR_SEC = 0.0305  # MFA's shortest phone: 3 frames x 10 ms


def utterance_health(utts: list[Utterance], phones: list[dict],
                     unsure_lines: set[int] | frozenset[int] = frozenset()) -> list[dict]:
    """Per-utterance alignment numbers - the ones that expose MFA losing the thread
    (docs/notes/chunked_alignment.md), which song-level ``align_score`` cannot see.

    A phone belongs to the utterance holding its midpoint. ``unsure_lyrics`` marks an
    utterance containing a line the resolve step could not vouch for."""
    out = []
    for u in utts:
        ds = [p["end"] - p["start"] for p in phones
              if u.start <= (p["start"] + p["end"]) / 2 < u.end]
        noise = sum(1 for p in phones
                    if u.start <= (p["start"] + p["end"]) / 2 < u.end and p.get("noise"))
        run = best = 0
        for d in ds:
            run = run + 1 if d <= FLOOR_SEC else 0
            best = max(best, run)
        out.append({
            "start": u.start, "end": u.end, "text": u.text, "lines": list(u.lines),
            "n_phones": len(ds),
            "floor_frac": round(sum(d <= FLOOR_SEC for d in ds) / len(ds), 3) if ds else None,
            "max_floor_run": best,
            "max_phone_sec": round(max(ds), 3) if ds else None,
            "n_noise": noise,
            "unsure_lyrics": any(li in unsure_lines for li in u.lines),
        })
    return out


def utterances_textgrid(utts: list[Utterance], duration: float, speaker: str) -> str:
    """A long-format Praat TextGrid with one tier (the speaker) - MFA's transcript
    input for a file with several utterances. Unlabelled intervals are ignored by MFA."""
    ivs: list[tuple[float, float, str]] = []
    t = 0.0
    for u in utts:
        if u.start > t:
            ivs.append((t, u.start, ""))
        ivs.append((u.start, u.end, u.text))
        t = u.end
    if duration > t:
        ivs.append((t, duration, ""))

    def q(s: str) -> str:
        return '"' + s.replace('"', '""') + '"'

    rows = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0", f"xmax = {duration}", "tiers? <exists>", "size = 1", "item []:",
        "    item [1]:", '        class = "IntervalTier"', f"        name = {q(speaker)}",
        "        xmin = 0", f"        xmax = {duration}",
        f"        intervals: size = {len(ivs)}",
    ]
    for k, (s, e, text) in enumerate(ivs, 1):
        rows += [f"        intervals [{k}]:", f"            xmin = {s}",
                 f"            xmax = {e}", f"            text = {q(text)}"]
    return "\n".join(rows) + "\n"
