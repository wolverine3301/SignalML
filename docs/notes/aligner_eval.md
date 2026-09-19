# Aligner eval — MFA vs SOFA (protocol; results pending the rig)

MIGRATION P5.4 asks for this note: *"Evaluate SOFA on 3–5 real separated vocals vs MFA;
record verdict + samples."* This is the **design half**, written now because it needs no
corpus — the metrics, the ground truth, and the protocol are all decidable in advance. The
**Verdict** section is the part that waits for the corpus and the rig.

Alignment quality is risk #1 in the register: with N h in hand, it is the top quality lever
on the whole project. Worth doing properly rather than by ear.

## The problem this has to solve

`quality.align_score` today is a **v1 self-consistency heuristic** — aligned-speech seconds
over voiced seconds, honestly labelled as such in
[`align.py:12`](../../signalml/stages/align.py:12). It answers "did the aligner put phones
roughly where there was voice?" It cannot answer either question that matters:

1. **Is MFA or SOFA more accurate on our data?** The heuristic has no ground truth, so both
   aligners can score well while disagreeing with reality and with each other.
2. **Is `align_score` a usable training filter?** It is already wired to exclude bad
   alignments from training, and it has never been validated against a real error measure.
   If it does not correlate with boundary error, it is excluding the wrong songs.

So this eval produces two deliverables, not one: an aligner choice, **and** a calibration of
the confidence score that gates the training set.

## Contenders

| | MFA (current default) | SOFA |
|---|---|---|
| What | Kaldi-style speech forced aligner, `english_mfa` v3 IPA acoustic model + dictionary | [Singing-Oriented Forced Aligner](https://github.com/qiuqiao/SOFA) — built for singing, retrainable from mixed weak/full labels |
| Phone set | `mfa_ipa/en_v1` (our versioned subset of the english_us_mfa inventory) | its own; needs a converter to `phones.json` |
| Expected edge | mature, well-understood, already integrated | designed for the failure case that hurts us — held vowels and the wide phoneme-duration variance of singing |

The prior going in: a speech acoustic model has never seen a four-second sustained vowel, and
that is most of what a sung corpus is. SOFA is the plausible winner and MFA is the incumbent;
the eval exists to make that concrete rather than assumed.

Whatever wins, nothing downstream moves — `phones.json` is the contract and swapping aligners
is one converter plus one config (PIPELINE_AND_CONTRACTS §S5).

## Ground truth

No corpus we hold has reference phone boundaries, so the ground truth is external. Sourcing
and licensing detail in [`candidate_corpora.md`](candidate_corpora.md); both are used as
**evaluation only** — measured and discarded, nothing retained, nothing trained on. That
distinction is load-bearing for JamendoLyrics, which contains ND-licensed songs.

| Set | Level | Condition | Why it is here |
|---|---|---|---|
| **JamendoLyrics** (EN subset, 20 songs; ~15 after filtering) | word onsets | polyphonic mixes → Demucs → align | matches our real pipeline exactly |
| **CSD** (EN half, 50 songs, 1 singer) | phone boundaries | studio a cappella | upper bound; isolates aligner from separation |

These bracket the difficulty range deliberately. Evaluating only on clean a cappella would
flatter both aligners and tell us nothing about our actual failure mode — our input is
*separated* vocals with Demucs artifacts, reverb bleed and smeared consonants. The
JamendoLyrics condition is the one that decides the default.

CSD's phone labels will not be in `mfa_ipa/en_v1`. Do not attempt a label mapping — compare
**boundary times only**, which is what the metrics below need anyway.

## Metrics

Word level (JamendoLyrics). A word onset from `phones.json` is the `start` of the first phone
carrying that `word`.

- **PCO@0.3s** — percentage of correct onsets within 300 ms. The standard JamendoLyrics
  benchmark figure, so our numbers are comparable to published work.
- **PCO@0.1s** — the tighter cut; 300 ms is generous for training-data purposes.
- **AAE** — average absolute error, seconds.
- **Median AE** — report alongside AAE. A single catastrophic misalignment wrecks the mean,
  and the gap between the two is itself the signal: mean ≫ median means "usually fine,
  occasionally derailed", which is a very different problem from uniform sloppiness.

Phone level (CSD).

- **Mean / median absolute boundary error**, milliseconds.
- **% boundaries within 20 ms** and **within 50 ms**. 20 ms is roughly the point where
  boundary error starts being audible in sliced training clips.

Both.

- **Failure rate** — songs where the aligner errors, times out, or returns nothing. A tool
  that is 5 ms better on average and falls over on 10% of songs is the worse tool.
- **Wall-clock per song** on the rig. N h of corpus makes throughput a real constraint,
  not a footnote.

## The separation-drift ablation (free, no ground truth needed)

Worth running independently of the aligner choice, and it uses data already on disk.

**MedleyDB ships isolated stems *and* the corresponding mixes.** So for the same song:

1. Align the isolated `..._STEM_NN.wav` → treat as pseudo-reference.
2. Run the mix through Demucs, align the separated vocal → treat as test.
3. Measure boundary drift between the two.

This measures **what separation alone costs the alignment**, with no annotation required —
the reference is the same aligner on cleaner audio, so aligner bias cancels out. If drift is
small, S3 is not the bottleneck and alignment effort belongs in the aligner. If drift is
large, the lever is Demucs settings and S4 cleanup, and no aligner swap will save it.

MedleyDB has no lyrics, so this needs a lyrics sidecar for any song used — or run it on the
subset that gets transcripts during the backfill described in
[`medleydb.md`](medleydb.md). Four flagged `has_bleed: yes` tracks should be excluded or
reported separately.

## The harness

Implemented in `signalml/evaluation/` (branch `aligner-eval-harness`). Pure measurement —
it never writes to the manifest or to `songs/`, so a run cannot contaminate a dataset and
can be repeated freely.

```bash
signalml eval align --ref-root <jamendolyrics-checkout> --data-root <DATA_ROOT> --out report.json
```

```bash
signalml eval drift songs/<id>/align/phones.json songs/<id-separated>/align/phones.json
```

- `evaluation/alignment.py` — the metrics, as pure functions over `phones.json` payloads:
  `word_onsets`, `phone_boundaries`, `onset_metrics`, `pooled_onset_metrics`,
  `boundary_metrics`, `drift`, `spearman`, `calibrate_align_score`.
- `evaluation/refs.py` — ground-truth loaders. JamendoLyrics has a bespoke loader because
  its format was verified against the published data; everything else goes through the
  generic CSV/TextGrid readers, so plugging in CSD (or a hand-annotated Gaelic set) means
  writing one adapter, not touching the metrics.
- `evaluation/runner.py` — walks the reference set, buckets clean vs flagged, pools, and
  writes the JSON report. `--map` handles the case where reference song names differ from
  manifest ids.

Two implementation notes worth knowing before reading a report:

- **Words are paired by sequence matching, not by index.** The aligner drops OOV words, and
  a single dropped word would shift every later comparison and turn a good alignment into a
  catastrophic score. The unmatched fraction surfaces as `coverage` instead.
- **PCO's denominator is the reference word count, not the matched count.** A word the
  aligner never emitted is a miss. Scoring only what matched would reward an aligner for
  dropping the words it finds hard.

47 offline tests cover the metrics, including the cases where a naive implementation returns
a plausible wrong number.

## Protocol

1. Pick the eval subset and **freeze it** — record exact track ids in this note, same
   discipline as D8's benchmark scores, or reruns are not comparable.
   - **JamendoLyrics EN**: 20 songs available, of which **14** are clean (not
     polyphonic, no lyric overlap, no non-lexical passages). The harness buckets these
     automatically — overlapping vocal lines and ooh/ahh passages are their own failure
     mode and would dominate the error if mixed in. The other 6 are reported as a
     **separate `flagged` bucket**: how each aligner copes with non-lexical singing is a
     real question for a *singing* aligner, just not the headline number.
   - **CSD EN**: 10 songs.
   - **MedleyDB**: 5 stem/mix pairs for the ablation, excluding the four `has_bleed: yes`
     tracks.
   Licence note: all 20 JamendoLyrics songs are fine to *measure*; 13 are ND, so no stems,
   clips or derived artefacts from them may be retained after the run.
2. Stage both aligners behind the existing injectable-runner seam in `align.py`, so this is
   a config switch and not a fork.
3. Run each aligner over each condition. Keep the TextGrid/native intermediates for audit.
4. Compute metrics with `signalml eval align` (see above). Record the report JSON path
   alongside the verdict.
5. **Spot-check by listening** — slice phonemes at the predicted boundaries and listen, as
   P5's "Done when" requires. Metrics can agree while both aligners are wrong in the same
   direction; ears catch that.
6. Correlate `align_score` against measured error across all songs. Report the correlation
   and, if it holds, a defensible threshold for excluding songs from training. If it does
   not hold, that is a finding: `align_score` v1 needs replacing before it gates a dataset.
7. Record the verdict below, with sample clips.

## Verdict

**Pending** — needs the corpus, MFA installed per the README, and the rig.

Fill in on completion: chosen default aligner and why; the numbers table; the `align_score`
correlation and chosen threshold; the separation-drift figure; and links to the sample clips.
If the answer is "SOFA on separated vocals, MFA on clean a cappella", say so — a per-condition
default is a legitimate outcome and `align.yaml` can carry it.

## Notes for the Gaelic wave

None of this transfers to `ga`/`gd` directly: **MFA ships no Irish or Scottish Gaelic
acoustic model**, so wave-2 needs a trained-from-scratch aligner regardless of what wins here
(see [`candidate_corpora.md`](candidate_corpora.md) for the Common Voice + WikiPron route).
What *does* transfer is this protocol — the metrics and the ablation are language-agnostic,
and a Gaelic aligner will need exactly this treatment before its output is trusted. The
ground-truth problem will be harder; budget for hand-annotating a small reference set.
