# Note transcription — D1 working notes (2026-09-22)

The acoustic model trains on phonemes + durations, which P5 already gives us. The
**variance** model additionally needs a score — which pitch, over which span, bound to
which syllable. Without it the system re-sings takes it has already heard but cannot
sing a score handed to it. That gap is DECISION_POINTS **D1**.

## What changed since D1 was written

D1 named **SOME** and **ROSVOT** as the ecosystem's tools. SOME is now **superseded**:
its README points at **GAME** (Generative Adaptive MIDI Extractor), openvpi's successor,
last updated August 2026.

| | SOME | GAME |
|---|---|---|
| Code license | MIT | MIT |
| **Weights license** | **CC BY-NC-SA 4.0** (stated in the v0.0.1 release notes) | **not stated separately** |
| Runtime | 9x realtime on CPU, 300x on GPU | Python 3.12 / torch 2.8 / CUDA 12.9, D3PM diffusion, ~50M params |
| Output | MIDI with float pitches, explicitly for DiffSinger variance labelling; also a mode that writes `note_seq`/`note_dur` straight into a dataset's transcriptions.csv | MIDI plus `.txt`/`.csv` note tables |
| Status | superseded upstream | current |
| Checkpoints | `0119_continuous128_5spk` (415 MB, v1.0.0-baseline, expanded data) | releases/discussions |

Float pitch output matters: rounding to semitones is a lossy decision (vibrato, slides,
blue notes) that belongs to the converter which builds variance transcriptions.csv, not
to the record of what was heard. `songs/<id>/transcribe/notes.json` therefore stores
floats, and S6b rounds when it emits `note_seq`.

## Decision (Logan, 2026-09-22)

**Use the openvpi weights and move forward.** Rationale as given: where maintainers did
not deliberately attach a separate licence to the weights, treat them as carrying the
repository's licence.

Two things that follow, recorded so nobody has to re-derive them:

- That reasoning fits **GAME** (no separate statement). It does **not** fit **SOME**,
  whose release notes say verbatim: *"The model files apply CC BY-NC-SA 4.0 license."*
  If the commercial door of Q4 matters later, GAME is the cleaner of the two and SOME's
  labels are the ones with a claim attached.
- Labels are not renders. A CC BY-NC vocoder only colours a preview; note labels shape
  what the variance model learns, so their provenance rides along with the trained
  model. `notes.json` records which transcriber produced it for exactly this reason —
  relabelling later is a rerun, not an archaeology project.

## The bake-off

The stage (`signalml transcribe`, S5b) is backend-agnostic: backends are external
commands, injected, never imported — the same posture as MFA and the trainer.

Metric with no ground truth: **note coverage** — the fraction of *sung* aligned phone
time (silence excluded) that any note spans. A transcriber that drops half the melody
still emits plausible-looking notes, and coverage is what catches it. Beyond that,
judgement is by ear on the same clips, and eventually by whether a variance model
trained on the labels sings in tune.

## First bake-off run (2026-09-22, work PC, CPU only)

SOME `0119_continuous256_5spk` (v1.0.0-baseline, 415 MB) over five Davina Michelle
songs through `signalml transcribe`. ~20 s per 2-minute vocal on CPU, no GPU:

| id | notes | coverage | avg note | midi lo-hi |
|---|---|---|---|---|
| sng_0001 | 321 | 0.981 | 0.33 s | 43-83 |
| sng_0002 | 373 | 0.942 | 0.36 s | 55-76 |
| sng_0003 | 326 | 0.957 | 0.35 s | 44-74 |
| sng_0004 | 422 | 0.979 | 0.31 s | 51-72 |
| sng_0005 | 351 | 0.890 | 0.39 s | 45-82 |

**Mean coverage 0.950** — a note spans 95% of the time MFA says someone was singing.
Average note length (~0.33 s) is plausible for pop phrasing. The `43` and `44` floors
are suspicious for a female pop singer (F2/G#2) and are worth a look once pitch is
fractional: likely octave errors or breath/noise picked up as low notes.

Two findings that change how we integrate it:

1. **`infer.py --midi` quantizes to integer semitones.** 1793 notes came back with
   exactly zero cent deviation, and the file contains no pitch-bend messages. The
   floating-point pitch that makes SOME suitable for variance labelling only comes out
   of its *dataset* mode (`batch_infer.py`), which writes `note_seq` / `note_dur` into
   a DiffSinger `transcriptions.csv`. The MIDI path is fine for a coverage bake-off and
   wrong for producing training labels.
2. **`batch_infer.py` requires `ph_num` in transcriptions.csv**, alongside `name`,
   `ph_seq`, `ph_dur`. S6b does not emit it. `ph_num` is the phone count per word, and
   `phones.json` carries a `word` field on every phone, so it is computable exactly
   rather than guessed — that is the next code change, and it unblocks the whole
   variance path (their own table lists `ph_num` as required for duration prediction
   too, independent of pitch).

So the sequence to variance training is: S6b emits `ph_num` -> `dataset build` ->
`batch_infer.py` over that dataset -> `note_seq`/`note_dur` land in the same CSV ->
`trainer: variance` recipe stops being blocked.

## Proven end to end (2026-09-22)

S6b now emits `ph_num`, and SOME's dataset mode was run over six real clips from
`overfit_v1/runn-en`, CPU only:

```
columns: name, ph_seq, ph_dur, ph_num, note_seq, note_dur
ph_seq    SP aj n ew SP
ph_num    1 1 2 1
note_seq  rest rest F4+43 F4+41 F#4+11 F#4+11 F#4+11
note_dur  0.11 0.215079 0.719819 0.24381 0.171292 1.88 0.15
```

Pitch arrives as **note name plus cents** (`F4+43`, `G4-23`) - the fractional detail the
`--midi` path discards - and `rest` covers silence. Checked on every row: `note_seq`
and `note_dur` are the same length, and `note_dur` totals match `ph_dur` totals within
50 ms, so the note timeline and the phone timeline describe the same clip.

Throughput, CPU (no GPU): **6 clips in 11.5 s**, ~2 clips/s. That is ~7 minutes for
`overfit_v1` (873 clips) and ~40 minutes for `full_acoustic_v2` (4,525 clips) - small
enough to run on the work PC without touching a GPU, and ~300x faster on the rig if
we ever want it there.

What is still missing before `signalml train variance` runs: S6b generates an
*acoustic* config only. A variance run needs their `configs/variance.yaml` base with
`predict_dur` / `predict_pitch` flags, pointed at the same raw dataset directory. That
is the next code change; the data side is done.

## Own note transcriber (Logan's, roadmap)

Logan wants to build this rather than depend on openvpi's, and it is a better fit for
the project than it first looks — SOME's own README claims usable results from **3 h**
of labelled data, and its training code is MIT.

The honest obstacle: a note transcriber is *supervised*, so it needs audio with note
labels, which is the very thing we lack. Our 24 h of aligned vocals are unlabelled at
the note level. Paths that break the circle, roughly in order of cost:

1. **Bootstrap from openvpi labels.** Transcribe our corpus with SOME/GAME, hand-correct
   a few hours (SlurCutter or a small UI), train on the corrected subset. Cheapest, but
   the teacher's licence question rides along into the student unless the corrected
   subset is genuinely re-labelled rather than lightly touched.
2. **Openly-licensed labelled corpora.** Any singing dataset shipping MIDI or note
   annotations can train the transcriber cleanly; the corpus survey in
   `docs/notes/candidate_corpora.md` is the place to extend.
3. **Synthetic supervision.** Render known scores through a singing model (ours, once
   P7 lands) and train the transcriber on audio whose notes are known exactly.
   Circular, but the circularity is benign: the labels are ground truth by construction.
4. **F0 + alignment heuristic as the floor.** We already have per-phone spans and F0;
   quantizing pitch over syllable spans yields notes with no model at all. Crude, fully
   clean, and useful as the baseline every learned approach must beat — implementable
   as a `transcribe` backend in an afternoon.

Architecture-wise, SOME is the reference: a mel/F0 front end predicting note boundaries
plus continuous pitch. Our advantage is data volume at the *acoustic* level and an
alignment we trust, so the boundary-prediction half can lean on phone spans rather than
learning segmentation from scratch.
