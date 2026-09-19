"""Alignment metrics for the P5.4 aligner eval (docs/notes/aligner_eval.md).

Three measurements, all pure functions over ``phones.json`` payloads and reference
annotations:

* **word onsets** vs word-level ground truth (JamendoLyrics) — PCO / AAE / median AE.
  This is the headline number: it runs on separated vocals from polyphonic mixes, which
  is what the real pipeline sees.
* **phone boundaries** vs phone-level ground truth (CSD) — boundary error in ms. Studio
  a cappella, so it isolates the aligner from separation damage.
* **drift** between two alignments of the *same* lyrics (MedleyDB stem vs Demucs(mix)) —
  no annotation needed, because the reference is the same aligner on cleaner audio, so
  aligner bias cancels and what is left is the cost of separation.

Plus ``calibrate_align_score``: ``quality.align_score`` already gates training-set
inclusion but has never been checked against a real error measure. If it does not
correlate, it is excluding the wrong songs.

Deliberately dependency-light (stdlib + numpy) so the whole harness runs CPU-only and
offline, per the testing convention.
"""

from __future__ import annotations

import difflib
import math
import re
import statistics
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from typing import NamedTuple, Sequence

DEFAULT_ONSET_TOLERANCES = (0.1, 0.3)
"""Seconds. 0.3 s is the published JamendoLyrics benchmark cut, so our numbers stay
comparable to the literature; 0.1 s is the cut that actually matters for slicing
training clips."""

DEFAULT_BOUNDARY_TOLERANCES_MS = (20.0, 50.0)
"""Milliseconds. ~20 ms is roughly where boundary error starts being audible in a
sliced phoneme."""

_PUNCT = re.compile(r"[^\w']+", re.UNICODE)


class WordOnset(NamedTuple):
    """One word occurrence and the time it starts, in seconds."""

    word: str
    start: float


def normalize_word(word: str) -> str:
    """Casefold and strip punctuation, keeping apostrophes ("we're" stays one token).

    Reference lyrics and aligner word labels come from different sources — one is a
    human transcript, the other is whatever survived the pronunciation dictionary — so
    they are compared normalized or they will not match at all.
    """
    return _PUNCT.sub("", word.strip().casefold())


def word_onsets(payload: dict) -> list[WordOnset]:
    """``phones.json`` -> one onset per word *occurrence*, in time order.

    Consecutive phones carrying the same ``word`` are one occurrence, so a repeated
    lyric ("on and on and on") yields three onsets rather than one. Phones with a null
    ``word`` (noise marks, ``spn`` holes) break a run, which is the intended behaviour:
    an OOV hole genuinely separates two occurrences.
    """
    onsets: list[WordOnset] = []
    current: str | None = None
    for entry in payload.get("phones", []):
        word = entry.get("word")
        if word is None:
            current = None
            continue
        if word != current:
            onsets.append(WordOnset(word, float(entry["start"])))
            current = word
    return onsets


def phone_boundaries(payload: dict, *, include_ends: bool = True) -> list[float]:
    """``phones.json`` -> sorted boundary times in seconds.

    Every phone start is a boundary. Phone ends are included by default because gaps
    (dropped silence) mean an end is not always the next phone's start.
    """
    times: set[float] = set()
    for entry in payload.get("phones", []):
        times.add(round(float(entry["start"]), 6))
        if include_ends:
            times.add(round(float(entry["end"]), 6))
    return sorted(times)


@dataclass
class OnsetMetrics:
    """Word-onset agreement. ``pco`` maps tolerance (s) -> fraction within tolerance."""

    n_ref: int
    n_hyp: int
    n_matched: int
    coverage: float
    aae: float | None
    median_ae: float | None
    pco: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.n_matched == 0:
            return f"no word matches (ref {self.n_ref}, hyp {self.n_hyp})"
        pco = "  ".join(f"PCO@{t}s {v:.1%}" for t, v in sorted(self.pco.items()))
        return (f"AAE {self.aae:.3f}s  median {self.median_ae:.3f}s  {pco}  "
                f"coverage {self.coverage:.1%} ({self.n_matched}/{self.n_ref})")


def pair_onsets(
    ref: Sequence[WordOnset], hyp: Sequence[WordOnset]
) -> list[tuple[WordOnset, WordOnset]]:
    """Pair reference and hypothesis onsets by matching the *word sequences*.

    Index-based pairing is wrong here: the aligner drops OOV words, and a single
    dropped word shifts every subsequent comparison, turning a good alignment into a
    catastrophic score. Matching the token sequences first (``difflib``, longest
    matching blocks) keeps the comparison honest and makes the unmatched fraction a
    reported number — coverage — rather than silent damage.

    ``autojunk`` is off: it treats tokens appearing in >1% of a long sequence as junk,
    which in lyrics means exactly the common words ("the", "I", "you") that carry the
    alignment.
    """
    ref_tokens = [normalize_word(o.word) for o in ref]
    hyp_tokens = [normalize_word(o.word) for o in hyp]
    matcher = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens, autojunk=False)
    pairs: list[tuple[WordOnset, WordOnset]] = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            pairs.append((ref[block.a + k], hyp[block.b + k]))
    return pairs


def onset_metrics(
    ref: Sequence[WordOnset],
    hyp: Sequence[WordOnset],
    *,
    tolerances: Sequence[float] = DEFAULT_ONSET_TOLERANCES,
) -> OnsetMetrics:
    """Word-onset metrics between a reference and a hypothesis alignment.

    Reports mean *and* median absolute error deliberately: one derailed passage wrecks
    the mean while leaving the median intact, and that gap is the diagnostic — "usually
    fine, occasionally lost" is a different failure from "uniformly sloppy" and wants a
    different fix.
    """
    pairs = pair_onsets(ref, hyp)
    errors = [abs(r.start - h.start) for r, h in pairs]
    coverage = len(pairs) / len(ref) if ref else 0.0
    if not errors:
        return OnsetMetrics(len(ref), len(hyp), 0, coverage, None, None,
                            {_tol_key(t): 0.0 for t in tolerances})
    return OnsetMetrics(
        n_ref=len(ref),
        n_hyp=len(hyp),
        n_matched=len(pairs),
        coverage=coverage,
        aae=statistics.fmean(errors),
        median_ae=statistics.median(errors),
        # Denominator is the reference count, not the matched count: a word the aligner
        # never emitted is a miss, not an absence of evidence. Scoring only what matched
        # would reward an aligner for dropping the words it finds hard.
        pco={_tol_key(t): sum(e <= t for e in errors) / len(ref) for t in tolerances},
    )


def pooled_onset_metrics(
    per_song: Sequence[tuple[Sequence[WordOnset], Sequence[WordOnset]]],
    *,
    tolerances: Sequence[float] = DEFAULT_ONSET_TOLERANCES,
) -> OnsetMetrics:
    """Corpus-level onset metrics over many (ref, hyp) song pairs.

    Words are matched *within* each song and the resulting errors are then pooled, so a
    long song contributes proportionally more than a short one. That is the honest
    corpus figure; averaging per-song percentages would let a 40-word song outvote a
    400-word one. Report it alongside the per-song table, not instead of it.
    """
    errors: list[float] = []
    n_ref = n_hyp = n_matched = 0
    for ref, hyp in per_song:
        pairs = pair_onsets(ref, hyp)
        errors.extend(abs(r.start - h.start) for r, h in pairs)
        n_ref += len(ref)
        n_hyp += len(hyp)
        n_matched += len(pairs)
    coverage = n_matched / n_ref if n_ref else 0.0
    if not errors:
        return OnsetMetrics(n_ref, n_hyp, 0, coverage, None, None,
                            {_tol_key(t): 0.0 for t in tolerances})
    return OnsetMetrics(
        n_ref=n_ref,
        n_hyp=n_hyp,
        n_matched=n_matched,
        coverage=coverage,
        aae=statistics.fmean(errors),
        median_ae=statistics.median(errors),
        pco={_tol_key(t): sum(e <= t for e in errors) / n_ref for t in tolerances},
    )


@dataclass
class BoundaryMetrics:
    """Phone-boundary agreement. ``within`` maps tolerance (ms) -> fraction within."""

    n_ref: int
    n_hyp: int
    mean_ms: float | None
    median_ms: float | None
    within: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.mean_ms is None:
            return f"no boundaries (ref {self.n_ref}, hyp {self.n_hyp})"
        within = "  ".join(f"<={t}ms {v:.1%}" for t, v in sorted(
            self.within.items(), key=lambda kv: float(kv[0])))
        return (f"mean {self.mean_ms:.1f}ms  median {self.median_ms:.1f}ms  {within}  "
                f"({self.n_ref} ref boundaries)")


def boundary_metrics(
    ref: Sequence[float],
    hyp: Sequence[float],
    *,
    tolerances_ms: Sequence[float] = DEFAULT_BOUNDARY_TOLERANCES_MS,
) -> BoundaryMetrics:
    """Nearest-neighbour boundary error, in milliseconds.

    For each reference boundary, the distance to the closest hypothesis boundary. This
    is the standard forced-alignment measure and it is deliberately one-directional:
    spurious extra boundaries in the hypothesis are not punished here. Phone *labels*
    are ignored — reference phone sets (CSD's, SOFA's) do not share an inventory with
    ``mfa_ipa/en_v1``, and a label mapping would inject more error than it measures.
    """
    hyp_sorted = sorted(hyp)
    if not ref or not hyp_sorted:
        return BoundaryMetrics(len(ref), len(hyp_sorted), None, None,
                               {_tol_key(t): 0.0 for t in tolerances_ms})
    errors_ms = [abs(t - _nearest(hyp_sorted, t)) * 1000.0 for t in ref]
    return BoundaryMetrics(
        n_ref=len(ref),
        n_hyp=len(hyp_sorted),
        mean_ms=statistics.fmean(errors_ms),
        median_ms=statistics.median(errors_ms),
        within={_tol_key(t): sum(e <= t for e in errors_ms) / len(errors_ms)
                for t in tolerances_ms},
    )


def _nearest(sorted_times: Sequence[float], t: float) -> float:
    i = bisect_left(sorted_times, t)
    if i == 0:
        return sorted_times[0]
    if i == len(sorted_times):
        return sorted_times[-1]
    before, after = sorted_times[i - 1], sorted_times[i]
    return before if (t - before) <= (after - t) else after


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation, ties averaged. ``None`` if undefined.

    Rank-based rather than Pearson because ``align_score`` is a bounded heuristic ratio
    and the error distribution is long-tailed — we care whether the score *orders*
    songs correctly, not whether the relationship is linear.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 3:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


@dataclass
class ScoreCalibration:
    """Does ``quality.align_score`` actually predict alignment error?"""

    n: int
    rho: float | None
    buckets: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.rho is None:
            return f"n={self.n}: too few songs to correlate"
        # align_score is "higher is better" and error is "lower is better", so a working
        # score gives a NEGATIVE rho. Say so in words; the sign trips everyone up once.
        verdict = ("predictive" if self.rho <= -0.5 else
                   "weakly predictive" if self.rho <= -0.2 else
                   "NOT predictive — do not gate training on this")
        return f"n={self.n}  rho={self.rho:+.3f} ({verdict})"


def calibrate_align_score(
    pairs: Sequence[tuple[float, float]], *, n_buckets: int = 4
) -> ScoreCalibration:
    """Correlate ``align_score`` against measured error over a set of songs.

    ``pairs`` is ``(align_score, error)`` per song, where error is whichever headline
    metric the run used (AAE for word-level, mean ms for phone-level) — the correlation
    is scale-free, so it does not matter which, as long as it is consistent.

    Returns the rank correlation plus per-bucket error stats. No threshold is invented
    here on purpose: the buckets show where error actually climbs, and picking the cut
    is a judgement call that belongs in the note with the numbers in front of you.
    """
    pairs = [(float(s), float(e)) for s, e in pairs]
    rho = spearman([s for s, _ in pairs], [e for _, e in pairs]) if pairs else None
    buckets: list[dict] = []
    if pairs:
        ordered = sorted(pairs)
        size = max(1, len(ordered) // max(1, n_buckets))
        for i in range(0, len(ordered), size):
            chunk = ordered[i:i + size]
            if not chunk:
                continue
            errs = [e for _, e in chunk]
            buckets.append({
                "align_score_min": round(chunk[0][0], 4),
                "align_score_max": round(chunk[-1][0], 4),
                "n": len(chunk),
                "mean_error": round(statistics.fmean(errs), 4),
                "median_error": round(statistics.median(errs), 4),
            })
    return ScoreCalibration(n=len(pairs), rho=rho, buckets=buckets)


def drift(reference: dict, test: dict, *,
          tolerances: Sequence[float] = DEFAULT_ONSET_TOLERANCES) -> OnsetMetrics:
    """Separation-drift ablation: two ``phones.json`` of the same lyrics.

    ``reference`` is the alignment on cleaner audio (a MedleyDB isolated stem),
    ``test`` the alignment on the separated vocal from the same song's mix. Same
    aligner both sides, so aligner bias cancels and the residual is what Demucs cost.

    Small drift means separation is not the bottleneck and effort belongs in the
    aligner; large drift means no aligner swap will save it and the lever is S3/S4.
    """
    return onset_metrics(word_onsets(reference), word_onsets(test), tolerances=tolerances)


def _tol_key(t: float) -> str:
    """Tolerances key JSON dicts, so they must be strings — and stable ones."""
    return f"{t:g}"
