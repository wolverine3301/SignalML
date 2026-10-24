"""Phone-level alignment metrics against manual singing labels (GTSinger, NUS-48E).

``alignment.boundary_metrics`` is label-blind and one-directional, which is right for
reference sets in a foreign phone inventory but blind to the failure that hurts us:
a speech aligner on singing squeezes the sung vowel to its 30 ms floor and hands the
held note to a neighbouring consonant. The word onset can still be right while the
phone timing inside the word is wrong, so this module measures the inside of words.

Reference and hypothesis phone *strings* differ even when both are ``mfa_ipa/en_v1``:
``english_mfa`` picks pan-dialect variants (``ʋ`` for ``w``, ``ɛː`` for ``ɛ``, ``ɒ`` for
``ɔ``), while GTSinger's labels come from US ARPAbet. So words are paired by their text
(as ``alignment.pair_onsets`` does) and compared by structure:

* **word onset** - start of the word's first phone;
* **vowel onset** - start of the word's first nucleus. This is the DiffSinger
  ``vowel_onset`` anchor: the sung vowel sits on the note start and onset consonants
  are sung ahead of it, so this is the boundary the variance model learns from;
* **nucleus duration ratio** - the word's nucleus time, hypothesis over reference;
* **squeezed vowels** - reference words whose longest nucleus is a real sung vowel
  (>= ``long_vowel_sec``) but which the hypothesis crushed to the floor;
* **boundary F-measure** - label-blind, but in both directions, so an aligner cannot
  win by emitting extra boundaries.

Floor shares (the share of vowels / consonants at a duration floor) are reported for
both sides. They need no reference at all, which makes them the health check for
corpora without manual labels; the floor is a parameter because 30 ms is an MFA
artefact (3 HMM states x 10 ms frames), not a property of singing.

Stdlib only, CPU-only, offline - same convention as ``alignment``.
"""

from __future__ import annotations

import difflib
import statistics
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, NamedTuple, Sequence

from .alignment import _tol_key, normalize_word

MFA_FLOOR_SEC = 0.0305
"""MFA's minimum phone duration (3 states x 10 ms frames), with float slack."""

LONG_VOWEL_SEC = 0.1
"""A reference vowel at least this long is unmistakably sung, so a hypothesis that
puts it at the floor has squeezed it. 100 ms is about two thirds of a spoken vowel."""

DEFAULT_TOLERANCES_MS = (20.0, 50.0, 100.0)


class Phone(NamedTuple):
    ph: str
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Word:
    """One word occurrence: its normalized text and its phones in time order."""

    text: str
    phones: tuple[Phone, ...]

    @property
    def start(self) -> float:
        return self.phones[0].start

    @property
    def end(self) -> float:
        return self.phones[-1].end


def words_from_payload(payload: dict, *, gap_sec: float = 0.0) -> list[Word]:
    """``phones.json`` -> word occurrences.

    Consecutive phones carrying the same ``word`` are one occurrence, as in
    ``alignment.word_onsets``. A silence gap longer than ``gap_sec`` also splits a run
    when ``gap_sec`` > 0, which separates a repeated word ("la, la") that the label
    alone cannot. Phones without a word (noise marks, NUS-48E labels) are skipped.
    """
    words: list[Word] = []
    text: str | None = None
    run: list[Phone] = []

    def flush() -> None:
        if text and run:
            words.append(Word(text, tuple(run)))

    for entry in payload.get("phones", []):
        label = entry.get("word")
        phone = Phone(str(entry["ph"]), float(entry["start"]), float(entry["end"]))
        norm = normalize_word(label) if label else None
        split = norm != text or (gap_sec > 0 and run and phone.start - run[-1].end > gap_sec)
        if split:
            flush()
            text, run = norm, []
        if norm:
            run.append(phone)
    flush()
    return words


def pair_words(ref: Sequence[Word], hyp: Sequence[Word]) -> list[tuple[Word, Word]]:
    """Pair word occurrences by matching the text sequences (``difflib``, no autojunk),
    for the reasons given in ``alignment.pair_onsets``."""
    matcher = difflib.SequenceMatcher(a=[w.text for w in ref], b=[w.text for w in hyp],
                                      autojunk=False)
    pairs: list[tuple[Word, Word]] = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            pairs.append((ref[block.a + k], hyp[block.b + k]))
    return pairs


def _nuclei(word: Word, is_nucleus: Callable[[str], bool]) -> list[Phone]:
    return [p for p in word.phones if is_nucleus(p.ph)]


@dataclass
class ErrorStats:
    """Absolute errors in ms: mean, median, p90 and the share within each tolerance."""

    n: int
    mean_ms: float | None
    median_ms: float | None
    p90_ms: float | None
    within: dict[str, float] = field(default_factory=dict)

    @classmethod
    def of(cls, errors_sec: Sequence[float], tolerances_ms: Sequence[float]) -> "ErrorStats":
        if not errors_sec:
            return cls(0, None, None, None, {_tol_key(t): 0.0 for t in tolerances_ms})
        ms = sorted(e * 1000.0 for e in errors_sec)
        return cls(
            n=len(ms),
            mean_ms=round(statistics.fmean(ms), 2),
            median_ms=round(statistics.median(ms), 2),
            p90_ms=round(ms[min(len(ms) - 1, int(0.9 * len(ms)))], 2),
            within={_tol_key(t): round(sum(e <= t for e in ms) / len(ms), 4)
                    for t in tolerances_ms},
        )

    def summary(self) -> str:
        if not self.n:
            return "n=0"
        within = " ".join(f"<={k}ms {v:.0%}" for k, v in self.within.items())
        return f"median {self.median_ms:.0f}ms p90 {self.p90_ms:.0f}ms {within}"


@dataclass
class FloorShare:
    """Share of vowels (nuclei) and consonants at or below a duration floor."""

    floor_ms: float
    vowels: float | None
    consonants: float | None
    n_vowels: int
    n_consonants: int


def floor_share(payload: dict, is_nucleus: Callable[[str], bool], *,
                floor_sec: float = MFA_FLOOR_SEC) -> FloorShare:
    """The squeeze health check. Needs no reference.

    Count vowels and consonants separately: a floor-length coda plosive can be right
    (singers drop or clip them), a floor-length sung vowel cannot.
    """
    nv = nc = fv = fc = 0
    for entry in payload.get("phones", []):
        d = float(entry["end"]) - float(entry["start"])
        if is_nucleus(str(entry["ph"])):
            nv += 1
            fv += d <= floor_sec
        else:
            nc += 1
            fc += d <= floor_sec
    return FloorShare(round(floor_sec * 1000.0, 1),
                      round(fv / nv, 4) if nv else None,
                      round(fc / nc, 4) if nc else None, nv, nc)


@dataclass
class BoundaryF:
    """Bidirectional boundary agreement at one tolerance."""

    precision: float
    recall: float
    f1: float


def boundary_f(ref: Sequence[float], hyp: Sequence[float], tol_sec: float) -> BoundaryF:
    """Precision (hyp boundaries near a ref one), recall (ref near a hyp), F1.

    Greedy one-to-one matching in time order, so one hypothesis boundary cannot vouch
    for two reference boundaries 10 ms apart.
    """
    r, h = sorted(ref), sorted(hyp)
    used = [False] * len(h)
    hits = 0
    for t in r:
        i = bisect_left(h, t - tol_sec)
        while i < len(h) and h[i] <= t + tol_sec:
            if not used[i]:
                used[i] = True
                hits += 1
                break
            i += 1
    precision = hits / len(h) if h else 0.0
    recall = hits / len(r) if r else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BoundaryF(round(precision, 4), round(recall, 4), round(f1, 4))


def _boundaries(words: Iterable[Word]) -> list[float]:
    times: set[float] = set()
    for w in words:
        for p in w.phones:
            times.add(round(p.start, 4))
            times.add(round(p.end, 4))
    return sorted(times)


@dataclass
class PhoneMetrics:
    """Everything one comparison yields. Pool many takes with ``pool``."""

    n_ref_words: int
    n_matched_words: int
    coverage: float
    word_onset: ErrorStats
    vowel_onset: ErrorStats
    word_offset: ErrorStats
    nucleus_ratio_median: float | None
    n_long_vowels: int
    squeezed_vowels: float | None
    boundary: dict[str, BoundaryF]
    hyp_floor: FloorShare
    ref_floor: FloorShare

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        sq = f"{self.squeezed_vowels:.1%}" if self.squeezed_vowels is not None else "-"
        f = " ".join(f"F@{k}ms {v.f1:.2f}" for k, v in self.boundary.items())
        hv = self.hyp_floor.vowels
        rv = self.ref_floor.vowels
        return (
            f"words {self.n_matched_words}/{self.n_ref_words}\n"
            f"  word onset   {self.word_onset.summary()}\n"
            f"  vowel onset  {self.vowel_onset.summary()}\n"
            f"  nucleus dur  median ratio {self.nucleus_ratio_median}  "
            f"squeezed long vowels {sq} of {self.n_long_vowels}\n"
            f"  boundaries   {f}\n"
            f"  vowels at <= {self.hyp_floor.floor_ms:g}ms: hyp "
            f"{hv if hv is None else f'{hv:.1%}'} vs ref {rv if rv is None else f'{rv:.1%}'}"
        )


@dataclass
class _Raw:
    """Per-comparison raw material, kept so pooling is exact rather than averaged."""

    n_ref_words: int = 0
    pairs: list[tuple[Word, Word]] = field(default_factory=list)
    ref_bounds: list[list[float]] = field(default_factory=list)
    hyp_bounds: list[list[float]] = field(default_factory=list)
    ref_payloads: list[dict] = field(default_factory=list)
    hyp_payloads: list[dict] = field(default_factory=list)


def _metrics(raw: _Raw, is_nucleus: Callable[[str], bool], *, floor_sec: float,
             long_vowel_sec: float, tolerances_ms: Sequence[float]) -> PhoneMetrics:
    onset, voffset_err, offset, ratios = [], [], [], []
    n_long = squeezed = 0
    for r, h in raw.pairs:
        onset.append(abs(r.start - h.start))
        offset.append(abs(r.end - h.end))
        rn, hn = _nuclei(r, is_nucleus), _nuclei(h, is_nucleus)
        if not rn or not hn:
            continue
        voffset_err.append(abs(rn[0].start - hn[0].start))
        rdur = sum(p.dur for p in rn)
        if rdur > 0:
            ratios.append(sum(p.dur for p in hn) / rdur)
        if max(p.dur for p in rn) >= long_vowel_sec:
            n_long += 1
            squeezed += max(p.dur for p in hn) <= floor_sec
    boundary = {}
    for t in tolerances_ms:
        hits_p = hits_r = n_h = n_r = 0.0
        for rb, hb in zip(raw.ref_bounds, raw.hyp_bounds):
            bf = boundary_f(rb, hb, t / 1000.0)
            hits_p += bf.precision * len(hb)
            hits_r += bf.recall * len(rb)
            n_h += len(hb)
            n_r += len(rb)
        p = hits_p / n_h if n_h else 0.0
        rc = hits_r / n_r if n_r else 0.0
        boundary[_tol_key(t)] = BoundaryF(round(p, 4), round(rc, 4),
                                          round(2 * p * rc / (p + rc), 4) if p + rc else 0.0)
    merged_hyp = {"phones": [e for pl in raw.hyp_payloads for e in pl.get("phones", [])]}
    merged_ref = {"phones": [e for pl in raw.ref_payloads for e in pl.get("phones", [])]}
    return PhoneMetrics(
        n_ref_words=raw.n_ref_words,
        n_matched_words=len(raw.pairs),
        coverage=round(len(raw.pairs) / raw.n_ref_words, 4) if raw.n_ref_words else 0.0,
        word_onset=ErrorStats.of(onset, tolerances_ms),
        vowel_onset=ErrorStats.of(voffset_err, tolerances_ms),
        word_offset=ErrorStats.of(offset, tolerances_ms),
        nucleus_ratio_median=round(statistics.median(ratios), 3) if ratios else None,
        n_long_vowels=n_long,
        squeezed_vowels=round(squeezed / n_long, 4) if n_long else None,
        boundary=boundary,
        hyp_floor=floor_share(merged_hyp, is_nucleus, floor_sec=floor_sec),
        ref_floor=floor_share(merged_ref, is_nucleus, floor_sec=floor_sec),
    )


class PhoneComparison:
    """Accumulate (reference, hypothesis) ``phones.json`` pairs; read pooled metrics.

    Words are paired within each take and the raw errors pooled, so long takes weigh
    proportionally (the reasoning in ``alignment.pooled_onset_metrics``).
    """

    def __init__(self, is_nucleus: Callable[[str], bool], *,
                 floor_sec: float = MFA_FLOOR_SEC, long_vowel_sec: float = LONG_VOWEL_SEC,
                 tolerances_ms: Sequence[float] = DEFAULT_TOLERANCES_MS,
                 gap_sec: float = 0.0) -> None:
        self.is_nucleus = is_nucleus
        self.floor_sec = floor_sec
        self.long_vowel_sec = long_vowel_sec
        self.tolerances_ms = tuple(tolerances_ms)
        self.gap_sec = gap_sec
        self._raw = _Raw()

    def add(self, reference: dict, hypothesis: dict) -> PhoneMetrics:
        """Add one take and return that take's own metrics."""
        rw = words_from_payload(reference, gap_sec=self.gap_sec)
        hw = words_from_payload(hypothesis, gap_sec=self.gap_sec)
        one = _Raw(len(rw), pair_words(rw, hw), [_boundaries(rw)], [_boundaries(hw)],
                   [reference], [hypothesis])
        self._raw.n_ref_words += one.n_ref_words
        self._raw.pairs.extend(one.pairs)
        self._raw.ref_bounds.extend(one.ref_bounds)
        self._raw.hyp_bounds.extend(one.hyp_bounds)
        self._raw.ref_payloads.append(reference)
        self._raw.hyp_payloads.append(hypothesis)
        return self._compute(one)

    def pooled(self) -> PhoneMetrics:
        return self._compute(self._raw)

    def _compute(self, raw: _Raw) -> PhoneMetrics:
        return _metrics(raw, self.is_nucleus, floor_sec=self.floor_sec,
                        long_vowel_sec=self.long_vowel_sec, tolerances_ms=self.tolerances_ms)


def compare_phones(reference: dict, hypothesis: dict, is_nucleus: Callable[[str], bool],
                   **kwargs) -> PhoneMetrics:
    """One-shot comparison of two ``phones.json`` payloads."""
    return PhoneComparison(is_nucleus, **kwargs).add(reference, hypothesis)
