# Candidate corpora — survey (2026-09-18)

A survey of external corpora *not yet onboarded*, written against the three gaps the
current collection has. Nothing here is downloaded or wired up; this is the shortlist and
the reasoning, so the decision is not re-derived later.

Companion notes: [`medleydb.md`](medleydb.md), [`vocalset.md`](vocalset.md) (onboarded),
and [`aligner_eval.md`](aligner_eval.md), which consumes the eval sources below.

## Where the collection actually stands

| Corpus | Female audio | Singers | Licence | Lyrics? |
|---|---|---|---|---|
| `own` | ~78 h en + ~12 h ga/gd | **unknown — census pending** | mixed / YouTube-sourced | yes, `.txt` sidecars |
| `medleydb` | 1.70 h | 11 artist-level identities | CC BY-NC-SA 4.0 | no |
| `vocalset` | 3.9 h | 9 | **CC BY 4.0** | no (`language=None` by design) |

Three gaps follow directly:

1. **Permissive + lyrics is empty.** VocalSet is the only permissively licensed corpus and
   it deliberately carries no language, so it cannot enter an `en` acoustic dataset. A
   shippable acoustic model (Q4's commercial door) currently has **no permissive alignable
   data at all** beyond `own`.
2. **Permissive singer identities = 9.** ARCHITECTURE §4 wants 20–50 female singers before
   the density model stops being "interpolation between a handful of real voices."
   MedleyDB's 11 are NC-encumbered, so they cannot close this for a shippable checkpoint.
3. **Gaelic alignment has no off-the-shelf path** — see below.

**Run the singer census (`manifest report`) before acting on gap 2.** If the 78 h already
holds 40 distinct singers, none of Tier 1's identity-boosting value applies and the priority
collapses to gap 1 alone.

## Tier 1 — permissive (CC BY / CC0)

Worth onboarding: these are the only sources that can feed a checkpoint intended to ship.

### vocadito — CC BY 4.0, best fit

40 short excerpts of solo monophonic singing, 7 languages, singers of varying training,
recorded on assorted devices. Annotations: frame-level f0, two independent note annotations,
**lyrics**, language tags, and **singer identifiers**.

The only permissive source found that has lyrics *and* singer IDs — it touches both gap 1
and gap 2. Tiny, and the "variety of devices" recording condition is the opposite of
`source_quality=studio`, so it is eval and diagnostic material rather than acoustic training
data. The per-track language tag means the English subset can be scoped cleanly.

[Zenodo 5578807](https://zenodo.org/records/5578807) · loadable via `mirdata`.

### ESMUC Choir Dataset — CC BY 4.0

12 undergraduate vocalists (5 soprano, plus altos; SATB overall), ~31 min total,
**44.1 kHz** — already at the `prod` profile rate. Close-mic'd per singer, so each track is
an isolated identity. F0 contour and note annotations per singer track; three choral pieces
plus warm-ups, in full / section / excerpt takes.

[Zenodo 5848990](https://zenodo.org/records/5848990).

### Dagstuhl ChoirSet — CC BY 4.0

Amateur vocal ensemble, two pieces, full-choir and quartet settings, close-up mics per
singer. Beat annotations, time-aligned sheet music, extracted F0 trajectories.

[AudioLabs page](https://www.audiolabs-erlangen.de/resources/MIR/2020-DagstuhlChoirSet) ·
licence confirmed CC BY 4.0 on Zenodo · loadable via `mirdata`.

**Caveat for both choir sets:** trained-choral timbre is a different region of the manifold
than pop, and ensemble singing is not solo performance. Good for *spread* in the embedding
space, bad if it drags the sampled-voice prior toward classical. Tag `corpus` distinctly so
a recipe can exclude them; treat as a voice-bank/ECAPA and vocoder contribution, not as
acoustic-model material.

### MTG-Jamendo — all-CC audio, Apache-2.0 metadata

55k+ full tracks from Jamendo, all Creative Commons, with genre/instrument/mood tags. The
only *scalable* permissive path found: filter to CC BY / CC BY-SA plus vocal tags, run S3
separation, ASR-draft the lyrics and human-fix — exactly the transcript-backfill workflow
[`medleydb.md`](medleydb.md) already describes.

Costs transcription effort rather than licensing risk, and per-track licence must be filtered
(Jamendo carries ND and NC variants too — see below). Real work, but it is the answer to
gap 1 at volume if `own`'s provenance ever proves insufficient.

[GitHub](https://github.com/MTG/mtg-jamendo-dataset).

## Eval-only

### JamendoLyrics — per-song CC, **includes ND**

79 songs, 4 languages (EN 20, FR 19, DE 20, ES 20), lyrics time-aligned at **word level**
and line level. Built as an automatic-lyrics-alignment benchmark.

**Eval only — do not retain derivatives.** Licence breakdown of the 20 English songs,
counted from `JamendoLyrics.csv` (2026-09-18):

| Licence | Songs |
|---|---|
| BY-NC-ND | 7 |
| BY-ND | 6 |
| BY-NC-SA | 3 |
| BY | 2 |
| BY-SA | 1 |
| BY-NC | 1 |

**13 of 20 are No-Derivatives.** For *measuring* an aligner and discarding the output that is
not a problem; for keeping stems, clips or a checkpoint it is, and only the remaining **7**
songs would qualify. Treat the set as a benchmark, never as a corpus.

The CSV also carries three columns that matter for eval design: `Polyphonic` (overlapping
vocal lines — 3 of the 20 English songs), `NonLexical` (ooh/ahh vocalisations — 3), and
`LyricOverlap` (1). Filtering on all three leaves a **14-song** monophonic, purely lexical
English core, which is the cleaner comparison; see [`aligner_eval.md`](aligner_eval.md).

Word-level annotations are **positional**: `annotations/words/<song>.csv` carries
`word_start,word_end,line_end` with no word text, matched row-for-row against
`lyrics/<song>.words.txt`. Verified against the published data (2026-09-18) — all 79 CSV
rows resolve to a words file, and the 6 English songs spot-checked load 147–322 words each.
`signalml.evaluation.refs` implements this.

This remains the highest-value item in this survey: word-level ground truth on real
polyphonic music is precisely what P5.4's MFA-vs-SOFA comparison lacks.

[HF dataset](https://huggingface.co/datasets/jamendolyrics/jamendolyrics) (the GitHub repo
is deprecated).

## Tier 2 — the NC bucket

Same standing as MedleyDB: fine for proving the pipeline, poisonous to anything that ships.
`exclude_corpora` already exists to keep them out of a permissive lineage.

### GTSinger — CC BY-NC-SA 4.0

80.59 h, 20 singers, 9 languages **including English**, all four vocal ranges, recorded in
professional studios. **Phoneme-level annotations and realistic music scores**, plus
controlled comparison of six techniques (mixed voice, falsetto, breathy, pharyngeal, vibrato,
glissando). NeurIPS 2024 spotlight.

The best-annotated singing corpus available, and the only Tier-2 entry with scores *and*
English *and* phoneme alignment — i.e. the natural reference run for proving the acoustic
model and for validating our own alignment against someone else's ground truth. Already
flagged as a later corpus in [`vocalset.md`](vocalset.md).

[GitHub](https://github.com/AaronZ345/GTSinger) ·
[HF](https://huggingface.co/datasets/GTSinger/GTSinger).

### OpenSinger — CC BY-NC-SA 4.0

Large multi-singer Chinese pop corpus. **Sources disagree on the numbers** — 50 h / 41 F +
25 M in some papers, 93 singers / 85 h in others; verify before planning around it.

**Blocker: 24 kHz.** That is below the `prod` profile's 44.1 kHz, and Q11 forbids mixing
profiles in a dataset. Upsampling buys nothing acoustically. It could only ever join a `dev`
profile experiment, which makes its large female singer count much less useful than it looks.

### CSD (Children's Song Dataset) — CC BY-NC-SA 4.0

100 songs (50 Korean / 50 English), **one** Korean female professional pop singer, each song
recorded in two keys → 200 recordings. MIDI transcription plus **grapheme- and phoneme-level
aligned lyrics**.

One singer, so no voice-bank value. The English half with phoneme-level alignment is a clean
aligner sanity-check set — studio-quality a cappella, unlike JamendoLyrics' polyphonic mixes,
so the two bracket the difficulty range nicely. (Note: NC, not CC BY — an easy thing to
assume wrong.)

[GitHub](https://github.com/equal-singer/CSD).

### NHSS — licence unclear

7 h, 10 singers (**5 F / 5 M**), English pop, 10 songs each. Parallel *sung and spoken*
lyrics from the same singers, with utterance- and word-level annotations.

Only 5 female singers, and access terms are not stated on the project page — obtaining it
likely means contacting NUS HLT. Low priority for the voice bank; the parallel speech/singing
structure is genuinely interesting for the D10 wave-3 speech plan, if that ever activates.

[Project page](https://hltnus.github.io/NHSSDatabase/).

### SingNet — **not a corpus; possibly a checkpoint source**

~3000 h of in-the-wild singing (2629 h from songs, 321 h from sample packs), assembled by an
automated extraction pipeline, with pretrained wav2vec2, BigVGAN and NSF-HiFiGAN checkpoints.
Reported vocoder MOS beats speech-trained baselines on in-the-wild audio (BigVGAN 3.55 vs
3.13; NSF-HiFiGAN 3.52 vs 3.39).

**The data is not obtainable.** The team states they cannot release the singing-voice data
because of copyright; what is released is the preprocessing pipeline and models. The project
page <https://singnet-dataset.github.io/> is demos only and 404s at the root. So SingNet is
struck from the corpus shortlist entirely.

What survives is the **checkpoint** question, which still bears on P7. Q4 budgets 1–2 weeks
to train our own NSF-HiFiGAN from scratch, "less if seeded from an existing checkpoint", and
a singing-trained NSF-HiFiGAN is exactly that seed. The related Amphion toolkit
(open-mmlab) is **MIT** and implements both vocoders, but its docs do not mention SingNet and
say nothing about checkpoint licensing as distinct from code licensing.

**Unresolved, and worth one more pass before P7 vocoder work starts:** whether a
SingNet-trained NSF-HiFiGAN checkpoint is actually downloadable and under what terms. Note
the awkward question underneath it — weights trained on undisclosed copyrighted audio carry
provenance risk that an MIT code licence does not wash out, which is the same reasoning that
put the community OpenVPI weights in the dev-preview-only bucket (Q4). Seeding from it may
be the wrong call even if the licence permits it.

[arXiv 2505.09325](https://arxiv.org/abs/2505.09325) ·
[Amphion](https://github.com/open-mmlab/Amphion).

## Gaelic (wave-2 — Q13)

**Hard finding: MFA ships no Irish and no Scottish Gaelic acoustic model or dictionary.**
The pretrained set covers ~30 languages; no Goidelic language is among them. So MIGRATION
P5.7's "per-language configs" is not a config change — it is train-your-own-aligner, and the
wave-2 deferral was correctly scoped.

Concrete inputs for when it is green-lit, better than espeak-ng bootstrapping alone:

- **Acoustic-model training data — Mozilla Common Voice**, which has both `ga-IE` and `gd`
  under **CC0**. Note Mozilla moved distribution to Mozilla Data Collective in Oct 2025, so
  older HuggingFace mirror instructions may be stale; check current terms.
- **Dictionary / G2P — WikiPron**, which scrapes Wiktionary IPA into MFA-shaped pronunciation
  dictionaries. Scottish Gaelic has ~4,145 IPA-tagged Wiktionary terms; Irish has an
  established transcription convention (Ní Chasaide, IPA Handbook). This is the same pipeline
  MFA's own 3.0 dictionaries were built from, so the path is well-trodden rather than novel.

Both feed the existing plan unchanged — `phones.json` is language-tagged IPA either way.

### Rejected: Tobar an Dualchais

40,000+ oral recordings of Scottish Gaelic and Scots traditional song from the 1930s onward
(School of Scottish Studies, BBC Scotland, the Canna Collection). Looks perfect and is not
usable: access is for research purposes with copyright cleared per informant and their
relations, so it is neither redistributable nor safe to train on for anything that ships.

Listed here so the next person does not spend an afternoon rediscovering it. Reference
listening and stylistic study only.

## Recommended order

1. **Singer census** — decides whether gap 2 is real before anything is downloaded.
2. **JamendoLyrics** (plus the CSD English half if a studio-quality contrast is wanted) —
   smallest effort, unblocks risk #1 (alignment quality). Eval only, nothing retained.
3. **vocadito** — CC BY, lyrics, singer IDs; small enough to onboard in an afternoon.
4. **ESMUC + Dagstuhl** — if the census says singer count is short.
5. **Chase SingNet's licence** — cheap to check, potentially reshapes P7's vocoder schedule.
6. **GTSinger** — only when an NC-bucket reference run is actually wanted.
7. **MTG-Jamendo** — the volume play, worth it only if `own`'s provenance proves insufficient.

## Open items

- SingNet: is a pretrained NSF-HiFiGAN checkpoint downloadable, under what licence, and is
  its training-data provenance acceptable? (Data itself: resolved — not released.)
- NHSS: access terms, and whether commercial use is permitted.
- OpenSinger: reconcile the conflicting singer/hour counts.
- vocadito: total duration, and how many of the 40 excerpts are English.
- ~~JamendoLyrics per-song licence breakdown~~ — resolved 2026-09-18, table above.
