# Phrase-by-phrase alignment (2026-09-26)

**Status:** planner done (`signalml/stages/align_chunks.py`, tests in
`tests/test_align_chunks.py`); wiring into `signalml align` is next.

## The problem: whole-song MFA fails silently

S5 originally gave MFA each song as **one utterance**: the full 3-4 minute vocal and
the full lyrics as a single transcript. MFA is an utterance aligner built for clips of
tens of seconds. Over minutes of singing with instrumental gaps its search loses the
thread. It places whole lines seconds away from where they are sung, then fills the
leftover time by stacking phones at its 30 ms floor (3 frames x 10 ms) and stretching
single phones over the gap.

What that looked like in our training data:

- the median song had **~45% of its phones at the 30 ms floor**, no song was under 10%,
  and single phones ran to 80+ seconds;
- in one training set, **~63% of clips** had a run of >= 4 floor phones in a row or a
  phone longer than 3 s;
- a held-out phrase: "tell me why" at 30 ms per phone, then one vowel held 1.5 s; the
  next line crushed the same way and one vowel held 4 s.

The model trained on it kept every singer's timbre and learned pronunciation only
loosely. By ear it sounded like someone singing along to lyrics they don't really know.

**`align_score` could not see any of this.** It measures coverage (aligned seconds over
voiced seconds), and a phone stretched over 80 s covers the song perfectly. Every
affected song passed the `min_align_score` gate.

### Health checks that do catch it

Measure these on any alignment:

| symptom | healthy | whole-song MFA on singing |
|---|---|---|
| phones at the 30 ms floor | a few % | ~45% (median song) |
| runs of >= 4 floor phones | rare | tens per song |
| longest single phone | a held note, a few s | up to 80+ s |
| words > 1 s from where an ASR heard them | a few % | ~36% (median song) |

## The fix: align phrase by phrase

`plan_utterances()` turns one song into phrase-sized utterances. MFA still reads only the
human lyrics.

1. **Anchor lines with Whisper, for timing only.** The words Whisper transcribed
   (`lyrics --check` already stores them with word timestamps) are matched to the human
   lyric words with the minimum-edit alignment from `lyrics_check.align_words`. A lyric
   line gets a rough span from its matched words. A match counts only if a neighbouring
   word **in the same line** matched as well, because lone "you"/"the" matches latch onto
   the wrong occurrence.
2. **Cut at real dips between consecutive anchored lines.** In the window from the end of
   line *a* to the start of line *b* (widened by 0.5 s, since Whisper's word edges drift
   on singing), take the quietest 80 ms. Cut there only if it sits >= 12 dB below the
   loud part (75th percentile) of the singing within 3 s. The S4 silence map was tried
   first and is too coarse: reverb and residual bleed keep a separated vocal above its
   -40 dB threshold through most line breaks. The relative dip rule cuts ~75% of line
   breaks.
3. **Never cut around an unsure line.** A line with no confident anchor stays inside the
   utterance of its anchored neighbours. An uncertain line widens an utterance; it is
   never placed on its own guess. Singing outside every utterance (intro ad-libs, outros
   past the last lyric) stays unaligned, gets no phones, and so no training clip claims it.
4. **No audio cutting.** MFA gets the song wav plus a one-tier TextGrid of utterance
   intervals (`utterances_textgrid()`). Unlabelled intervals are ignored.

Typical utterances come out at ~4 s (max ~27 s).

## Result (pilot set, same MFA settings: beam 200, retry 1000)

| median song | whole song | phrase by phrase |
|---|---|---|
| words > 1 s off (vs Whisper) | 36% | **15%** |
| worst song: median word offset | 20 s | 0.7 s |
| longest phone | up to 81 s | < 9 s |
| vowels at the 30 ms floor | 43% | 41% |
| MFA wall time | 1x | **~17x faster** |

Songs that were already well aligned barely change. By ear, on the karaoke comparison
page (below), the phrase-by-phrase alignment tracks the singing far more closely.

## Not fixed: stacking *inside* words

~41% of vowels still sit at 30 ms, now inside correctly placed words. That comes from
MFA's speech-trained acoustic model on stretched sung vowels, not from utterance length,
so chunking cannot fix it. The next candidate is SOFA (singing-oriented aligner, the
documented S5 fallback), which also expects phrase-length input. The chunker supplies
that.

## Judging alignments by ear: `scripts/karaoke_compare.py`

This script builds a self-contained page: one recording, two alignments side by side,
the current word lit up in each column as the audio plays. Click a word to seek. It
also shows the health numbers above for each alignment.

```
python -m uv run python scripts/karaoke_compare.py --out <dir> \
    --a <TextGrids A> --b <TextGrids B> --audio <wav dir> \
    --label-a "Whole song" --label-b "Phrase by phrase" [--titles titles.json] [--flac]
```

Serve `<dir>` with `python -m http.server`; some viewers block audio on file:// pages.
The page works for any two alignments of the same audio: MFA vs SOFA, two beam settings,
an aligner vs hand-corrected labels.

To A/B two *models* by ear rather than two alignments, use the same method:

- put the same held-out clips in the same rows on two pages;
- give both models the same variance output (pitch curve and durations) and one fixed
  seed, so the only difference is the model under test.
