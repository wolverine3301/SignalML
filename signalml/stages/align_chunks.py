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


def line_spans(lines: list[str], segments: list[dict]) -> list[tuple[float, float] | None]:
    """Rough (start, end) per lyric line from its confidently matched words, else None.

    A match counts only when a neighbouring word matched too: a lone "you" or "the"
    lines up with the wrong occurrence often enough to drag a line across the song.
    """
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
    def paired(i: int, j: int, d: int) -> bool:  # neighbour matched too, same line
        k = i + d
        return 0 <= k < len(ref) and line_of[k] == line_of[i] and match.get(k) == j + d

    for i, j in match.items():
        if not (paired(i, j, -1) or paired(i, j, 1)):
            continue
        li, (_, s, e) = line_of[i], heard[j]
        span = spans[li]
        if span is None:
            spans[li] = [s, e]
        else:
            span[0], span[1] = min(span[0], s), max(span[1], e)
    return [tuple(s) if s else None for s in spans]


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


def plan_utterances(
    lyrics_text: str,
    segments: list[dict],
    level_db: np.ndarray,
    duration: float,
    hop_sec: float = HOP_SEC,
) -> list[Utterance]:
    """Cut the song into utterances at dips between consecutive anchored lyric lines."""
    lines = [ln.strip() for ln in lyrics_text.splitlines() if norm_words(ln)]
    if not lines:
        return []
    spans = line_spans(lines, segments)
    anchored = [i for i, s in enumerate(spans) if s]
    if not anchored:
        return [Utterance(0.0, round(duration, 3), " ".join(lines), tuple(range(len(lines))))]

    cuts: list[tuple[int, float]] = []  # (first line after the cut, time)
    for a, b in zip(anchored, anchored[1:]):
        if b != a + 1:  # unanchored lines between: keep them inside one utterance
            continue
        lo, hi = spans[a][1] - EDGE_TOL_SEC, spans[b][0] + EDGE_TOL_SEC
        t = find_dip(level_db, lo, hi, hop_sec) if hi > lo else None
        if t is not None and (not cuts or t > cuts[-1][1]):
            cuts.append((b, t))

    # song edges: a margin around the first / last anchored line, unless unanchored
    # lyric lines lie beyond it - then the whole edge, so they have audio to land in
    start = max(0.0, spans[anchored[0]][0] - EDGE_MARGIN_SEC) if anchored[0] == 0 else 0.0
    last = anchored[-1]
    end = (min(duration, spans[last][1] + EDGE_MARGIN_SEC)
           if last == len(lines) - 1 else duration)

    out: list[Utterance] = []
    line0, t0 = 0, start
    for b, t in cuts:
        out.append(Utterance(round(t0, 3), round(t, 3), " ".join(lines[line0:b]),
                             tuple(range(line0, b))))
        line0, t0 = b, t
    out.append(Utterance(round(t0, 3), round(end, 3), " ".join(lines[line0:]),
                         tuple(range(line0, len(lines)))))
    return [u for u in out if u.end > u.start]


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
