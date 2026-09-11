# Decision Points Ahead — What Needs an Architect vs. What's Delegable

> Written 2026-07-07 at the end of the planning+P0–P4 session. Scope: everything
> between here and the goal — **"I give it music and lyrics, it outputs a vocal
> performance."** Each entry says why it's a planning-level decision (don't let an
> execution-focused model improvise it), what inputs the decision needs, and the
> current lean. Ordered by risk to the end goal.
>
> Rule of thumb for delegation: **cheaper models implement against a written
> contract; they don't get to invent contracts.** Everything in this file is a
> missing or unfinished contract.

---

## D1. 🔴 The note-annotation gap (NOT in the migration plan — biggest unplanned hole)

**The gap:** the singer trains on *score → audio*, so every training phrase needs not
just phoneme timings (P5/MFA gives those) but **note labels**: which pitch, over which
span, bound to which syllable, with slurs. The current pipeline extracts F0 curves
(P4) but nothing turns "78 h of aligned vocals" into **note-level training scores**.
Manual annotation at this scale is impossible.

**What must be decided:** the singing-voice **note transcription** approach —
options include quantizing our own RMVPE/pyin F0 against detected beats (cheap,
noisy), adopting the openvpi ecosystem's annotation tools (ROSVOT / SOME — built for
exactly this, PyTorch, but another vendored dependency), or a hybrid (auto-transcribe
+ spot-check UI). Choice interacts with D2 (the trainer's expected dataset format
already has conventions for this) and D3 (phone-set/slur representation).

**Inputs needed:** ~~D2's fork choice first~~ **D2 resolved (2026-07-12):** the trainer
expects `note_seq`/`note_dur` columns in transcriptions.csv; ecosystem tools are
openvpi **SOME** + MakeDiffSinger `variance-temp-solution`/`midi-recognition`, with
SlurCutter for manual fixes. Remaining input: the quality bake-off on 5–10 real
separated vocals (the TEST_RUN corpus is processed and ready for this).
**Current lean:** ROSVOT/SOME from the openvpi ecosystem, since D2 already leans
openvpi — but verify maintenance state at decision time. Note: the P7.5 overfit proof
can start acoustic-only (ground-truth durations) before D1 is fully solved.
**Delegable once decided:** the stage implementation (S5b "transcribe"), manifest
wiring, QC sampling scripts.

## D2. ✅ RESOLVED 2026-07-12 — Trainer integration boundary

**Decision:** vendor **openvpi/DiffSinger v2.5.1** (submodule `third_party/DiffSinger`,
commit `323a569`, Apache-2.0), posture **(a) adopt-their-world**, zero local patches,
trainer in its own venv (their pins conflict: librosa<0.10, numpy<2, lightning~=2.3).
Full spike notes + integration contract: `docs/notes/vendor_diffsinger.md`. Highlights:
their binarizer computes training mels from wavs (D5 dissolves — their spectral
defaults match our `prod` profile exactly); multi-speaker via `spk_id` table
(`use_spk_id: true`) is the D7 hook; v2.5 multilingual dictionaries are the Gaelic
mechanism; ⚠ phoneme naming is "ASCII preferred" — IPA smoke-test in the overfit run,
ASCII transliteration table as fallback (D3).

<details><summary>Original question (for the record)</summary>

**What must be decided:** which exact DiffSinger fork/tag to vendor, and **where our
pipeline ends and theirs begins.** Two coherent postures: (a) *adopt their world* —
produce their dataset format, use their binarizer/configs/phoneme-dictionary
conventions, wrap their CLI (fastest to first sound, but their conventions leak
upstream into S5/S6b and the score schema); or (b) *own the boundary* — write our own
binarizer to their tensor contract (cleaner long-term, more work, risks drift on
their updates). Also: submodule vs copied snapshot; patch policy (aim: zero local
patches).

**Inputs needed:** a half-day Fable-level read of the fork's dataset/binarizer/config
code paths. This is the single highest-leverage planning task remaining — D1, D3, D5,
D6 all shape themselves around it.
**Current lean:** (a) adopt-their-world for the MVP, with our manifest/score JSON as
the system of record and one converter into their format; revisit (b) only if their
conventions start distorting upstream stages.
**Delegable once decided:** the converter, config templating, run-dir bookkeeping.

</details>

## D3. 🔴 Phone-set reconciliation: MFA IPA ↔ trainer dictionary

**What must be decided:** the authoritative mapping from MFA's IPA phone inventory
(`mfa_ipa/en_v1`, per Q2) to the vendored trainer's phoneme-dictionary format
(openvpi uses its own dictionary conventions, incl. consonant/vowel structure and
slur handling), plus where stress lives, how word/syllable boundaries carry through,
and how the versioned "universal phoneme set" (Logan's multilingual plan, endorsed in
OPEN_QUESTIONS *Future direction*) layers on top without retraining churn. Getting
this wrong silently poisons every downstream model.

**Inputs needed:** D2's fork choice; MFA's actual emitted phone inventory on a sample
of the real corpus (run the P5 offline converter on real alignments and enumerate).
**Current lean:** single project-owned mapping table module (`score/phoneset.py`)
with golden-file tests; treat the trainer dictionary as *generated output* from it.
**Delegable once decided:** the table's implementation, converters, validators.

## D4. 🔴 The "music" input contract at inference (defines the product's front door)

**What must be decided:** when Logan "gives it music," what is *music*? (a) **MIDI
melody + backing track** — syllable-to-note assignment is deterministic-ish, quality
depends on the lyric-fitting algorithm; MVP-friendly; (b) **audio only** — requires
melody transcription of the backing/reference (same tech family as D1) before lyric
fitting; much harder, aligns with the long-term vision. And in both: the **lyric
fitting design** — syllable-to-note assignment, melisma/slur decisions, stress-to-
strong-beat preferences, breath placement. This is the "hold vowels, stress the right
phonemes" requirement from the brief, and it is a real algorithm-design task, not a
coding chore.

**Current lean:** MVP = (a) MIDI melody in, with the naive order-based assigner from
P6 upgraded by a designed lyric-fitting pass; (b) later, reusing D1's transcriber.
**Delegable once decided:** importer code, schema validation, test corpora.

> **D3 update (2026-07-12, from the D2 spike):** trainer dictionary format is
> tab-separated `syllable → phones` text, *generated* from `score/phoneset.py` as
> leaned; their phoneme-naming rule ("ASCII preferred", `/ - +` etc. forbidden) means
> IPA symbols need a smoke test in the overfit run — fallback is a deterministic
> ASCII transliteration in phoneset.py, invisible upstream. Their v2.5 multilingual
> system (language-prefixed phones, `merged_phoneme_groups`, `use_lang_id`) is the
> layering mechanism for the universal-set ambition.

## D5. 🟠 The mel contract lock (silent-failure zone)

> **Mostly dissolved by D2 (2026-07-12):** adopt-their-world means the vendored
> binarizer computes training mels *from the wavs* — our features NPZs are analysis
> artifacts, not trainer input. Their defaults equal our `prod` profile exactly
> (44100/2048/512/128/fmax16k/mel_base e). Residual D5 = keep `configs/audio.yaml`
> prod in lockstep with the vendored config templates (one assert in S6b).

**What must be decided:** one authoritative spectral spec shared by the features
stage, the acoustic model, and the vocoder — mel filterbank params, log base,
normalization/statistics, fmin/fmax, padding/centering. P4 chose ln-mel, 128 bins,
profile-driven hop; the vendored trainer and vocoder have their own conventions and
**any mismatch produces degraded audio with no error message**. Decide the canonical
spec (probably: adopt the vocoder's exactly), then add golden-file round-trip tests
(features NPZ → trainer input → vocoder input) as a hard gate.

**Inputs needed:** D2 + D6 choices.
**Delegable once decided:** the golden tests, any features-stage parameter changes.

## D6. 🟠 Vocoder training recipe (decided *that* we train our own; not *how*)

**What must be decided:** architecture variant (NSF-HiFiGAN vs PC-NSF-HiFiGAN vs
BigVGAN-class), training config for ~90 h (78 EN + 12 GA/GD) on one 5090, seeding
strategy (from-scratch vs architecture-only init), fine-tune cadence, and acceptance
bar (copy-synthesis ABX vs the NC community checkpoint — must be ≥ before it ships).
**Current lean:** PC-NSF-HiFiGAN-style from the vendored stack at prod profile.
**Delegable once decided:** launch scripts, monitoring, checkpoint hygiene.

## D7. 🟠 Voice-bank concrete mechanism inside the chosen trainer

Architecture doc §4 fixed the concept (embedding table → density fit → sample →
similarity guard → persisted profile). Once D2 lands, someone must map that onto the
fork's actual multi-speaker mechanics (spk_id table? mix embeddings? conditioning
points?), choose the embedding dimension, the density model (Gaussian first), the
ECAPA verifier wiring, and the re-projection procedure after retrains. Also the
**singer-count reality check**: census the 78 h corpus (how many distinct singers?)
— if it's < ~15 singers, novel-voice sampling quality is at risk and acquisition
priorities should shift (more singers > more hours; ARCHITECTURE §4).

**Delegable once decided:** `voice new` / `voice reproject` implementation, bank I/O.

## D8. 🟡 Evaluation harness design (decide once, run forever)

How to know a run improved: objective set (F0 RMSE / voicing F1 on held-out phrases,
MCD, speaker-similarity cosine for voice consistency, intelligibility spot checks) +
a small fixed listening protocol with a frozen benchmark score set. Design the
metric definitions and the held-out split policy (split **by singer**, never by
song) at Fable level; implementation is fully delegable. Without this, P7 tuning
becomes vibes-driven and unreproducible.

## D9. 🟡 Dataset curation & QC policy

Thresholds (align_score cutoffs, separation-artifact rejection, min phrase length),
per-singer balancing, language mixing ratio for the EN+GA/GD acoustic run, val/test
splits. Policy decisions ride on real corpus statistics (first manifest scan + P5
alignment scores) — set the policy when those numbers exist; executing it is
delegable (S6b already reserves the enforcement hooks).

## D10. 🟢 Wave-3: one voice that talks, acts, and sings (added 2026-07-13)

**The vision** (Logan's 2020 origin idea, endorsed in OPEN_QUESTIONS *Future
direction*): the same persisted sampled voice renders plain speech, expressive
voice-acting, and singing. Voice acting = the midpoint of the singing↔speech
continuum (expressive speech driven by an "emotional notes" prosody track), not a
third system. **Gate: do not start until the P7/P8 singing MVP proves the voice bank**
(a sampled voice sings consistently across two songs).

**What must be decided (the forks):**

1. **Identity mechanism** — how one voice spans domains:
   (a) *joint training*: one acoustic model on singing + speech corpora, shared
   learned `spk_id` table, domain flag. Simple; but every domain must train together,
   and adding a domain means retraining everything.
   (b) *shared external speaker encoder* (ECAPA/WavLM-class) conditioning all models;
   the voice bank samples in **encoder space**, and any model conditioned on that
   space renders the voice. **Current lean: (b)** — models train per-domain and
   asynchronously, unseen corpora still enrich the sampling space, and `voice
   reproject` generalizes to cross-domain projection natively. Known risk: encoder
   conditioning historically gives slightly weaker identity match than learned
   tables — mitigation is a hybrid (encoder-initialized table rows fine-tuned per
   model). ⚠ Evaluation must use a *different* verifier model than the conditioning
   encoder, or the system grades itself.
2. **Emotion/prosody representation** for the "emotional notes" track: categorical
   (≈6–8 classes + an intensity scalar — matches available labeled data; lean, start
   here) vs continuous (arousal/valence plane — richer, nearly unlabelable by hand).
   Lands as `signalml-score/0.3`: optional prosody events `{span, emotion,
   intensity, emphasis}` alongside (not replacing) note events. (0.2 was taken by
   D12's note ids, which shipped first; prosody spans will address notes *by id*,
   which is tidier than the index ranges this originally implied.)
3. **Speech corpora + licenses** (Logan's commercial-clean posture applies): decide
   at start time; candidates to evaluate then — LibriTTS-R-class audiobook corpora
   (clean, thousands of speakers, no emotion labels), emotion-labeled sets (ESD
   etc. — many are research-only; check), and SER pseudo-labeling of audiobook
   speech (the MFA/SOME auto-label pattern applied to emotion). License-check the
   speaker encoder checkpoint too (SpeechBrain ECAPA is Apache; WavLM is not).

**What it needs (requirements found 2026-07-13):**

- **Paired anchors** — speakers with BOTH speech and singing data are what teach
  "same timbre, different domain": NUS-48E is exactly paired read/sung (license:
  research — check before shipping anything); cheaper and cleaner: **our own cover
  artists' spoken content** (interviews/vlogs — same voices, same YouTube sourcing
  pipeline). ~10+ paired speakers at minutes each calibrates the mapping.
- **Pipeline deltas (small, by design):** `meta.domain: sung|spoken` manifest field;
  the import-as-stems path (speech skips Demucs — same gap the studio stems already
  need); an S5-emotion pseudo-label stage (SER model → `emotion.json` spans, human
  spot-check); S6b domain-tagged dataset recipes with per-domain sampling weights
  (so abundant speech doesn't swamp scarce singing in joint runs).
- **Model work:** a speech-prosody variance model (emotion-conditioned F0/duration —
  the monotone-vs-expressive difference lives here); domain conditioning in the
  acoustic model; the vocoder is shared (NSF-HiFiGAN handles speech as the easy
  case — add speech to its training data for robustness).
- **Evaluation:** cross-domain identity check = independent-verifier cosine between
  the same voice speaking vs singing (the D8 harness gains one metric); expressive
  range vs a monotone baseline (F0 variance stats + listening protocol).
- **Compute:** same class as everything else — speech acoustic runs are days on the
  5090; no new hardware implied.

**Top risk:** cross-domain identity drift (the voice sounds like a different person
speaking vs singing). The paired anchors + hybrid conditioning are the mitigation,
and the drift metric above is the early-warning gauge.
**Delegable once decided:** corpus onboarding, the SER pseudo-label stage, dataset
recipes, score-schema extension — all follow the established stage pattern.

## D11. 🟢 Studio: the voice-generation front-end (added 2026-07-19)

**The tool** (full design: `docs/STUDIO.md`): `signalml studio` — a local
FastAPI + browser app served from the rig, wrapping the S8 API. Two screens plus
a bank manager: **Voice Lab** (sample the timbre space with derived controls —
temperature / PCA axes / blend, never raw dims — audition candidates on a fixed
public-domain demo score set, compare via an evolutionary tray with
reroll/tweak/breed, save through the full `voice new` guard path) and
**Performance** (voice gallery with `ref/` playback + reproject staleness,
score picker / MIDI import / lyric retexting, per-render transpose/tempo/mix/
seed, render history). Hard rule: zero synthesis logic in the UI — every action
is a seeded, CLI-reproducible call into `signalml.voices`/`signalml.synth`.
**Gate: starts after P8.**

**What must be decided (the forks):**

1. **Frontend build** — Gradio prototype (fastest to first screen; fights the
   candidate-tray UX) vs custom single-page frontend (~2–3 extra days; no wall).
   **Current lean: custom SPA from the start**, optionally after a day-scale
   throwaway Gradio spike of the audition loop.
2. **Demo score set** — 2–3 short public-domain/original phrases (legato,
   syllabic, range-spanning); decide jointly with D8's frozen benchmark set so
   one curated set serves both.

**Upstream implication (the only one):** P8 builds S8 **API-first**, CLI as a
thin shell — free now, load-bearing later. Lyric retexting is D4's
lyric-fitting algorithm applied to a fixed melody — the Studio fronts D4, never
forks it.

**Editor screen (added same day, STUDIO.md §6):** a fourth screen — general
non-AI audio editing (trim/splice/fades, multitrack mixdown, LUFS normalize,
vocal-chain presets, cleanup) so small jobs skip the external-editor
round-trip. Non-destructive EDL-JSON edits, server-side CPU DSP over existing
deps, profile-aware export. The suite-specific wins: **take comping** across
seeds, open-from-render-history, send-back into `sing --mix`. Explicitly out
of scope forever: VST hosting, MIDI editing, recording. Phase E — model-
independent, interleaves any time after phase A.

**Delegable once decided:** the whole implementation — server, frontend, tray
state, render cache/queue, editor EDL + DSP ops — against STUDIO.md as the
contract.

---

## D12. 🟢 Targeted editing: re-render one line, keep the rest (added 2026-09-11)

**The want:** a verse or line is sung badly while the rest of the take is good —
fix that region without regenerating the song.

**Why it is cheap here:** the composed pipeline already decided this in our
favour. The vocal is a separate stem, so a region edit never touches the
instrumental; the singer is score-driven and non-autoregressive (DiffSinger's
own `.ds` is a *list of phrase segments with offsets*, rendered per phrase);
voice identity is a fixed vector, so a line re-rendered tomorrow is the same
singer; and the seed + resolved-config discipline makes a render a pure
function of its inputs. This is not a research problem — it is a **render-
addressing and seam problem**.

**Three user intents that feel like one** (the distinction drives everything):

| | Means | Mechanism |
|---|---|---|
| (a) new take | "that line is bad, roll again" | re-render segment, new seed — nearly free |
| (b) **repair** | "90% right, but 'shine' goes flat" | a reroll *discards the 90%*; needs explicit-variance editing or partial denoising |
| (c) re-direct | "verse 2 angrier" | conditioning change, then (a) |

(b) is the one users actually want most of the time and the one that decides
whether the feature feels like an editor or a slot machine.

**Method candidates, by confidence** — deliberately NOT decided yet (see the
gates below):

1. *Falls out of the architecture:* segment-addressed rendering + content-
   addressed cache; region re-seed; take comping (STUDIO.md §6) as the
   always-works fallback.
2. *Designed, confident:* render with a neighbour phrase of context and crop
   (the variance models see different edges in isolation); **splice at the mel,
   then vocode the whole song in one pass** — no waveform crossfade, no phase
   discontinuity, and vocoding a full song is seconds; **editable variance
   intermediates** (explicit F0 curve + durations in, same seed) — the answer to
   intent (b), and what OpenUtau-class editors already do because DiffSinger
   takes explicit F0/duration natively.
3. *Uncertain, highest payoff:* **mel inpainting via partial denoising** —
   RePaint-style: noise the target region to timestep *t*, denoise while
   re-clamping the surrounding mel each step. Seamless by construction, and *t*
   becomes a **"vary this line" strength slider** that spans (a) and (b) on one
   control. Risk: needs masked conditioning inside the vendored fork's sampler —
   a contained change, but it pokes D2's adopt-their-world boundary and becomes
   ours to maintain. Nothing else may depend on it.

**Effects are explicitly not on the critical path** (Logan, 2026-09-11): the
polish chain is non-destructive (EDL) and cheap to re-run wholesale, so
re-polishing the whole stem after a vocal fix is the expected flow. No
incremental-effects design is needed; no reverb tail crosses a splice. The hard
problem is the performance, which polish cannot fix.

**✅ RESOLVED 2026-09-11 — the three pre-commitments (implemented, not just
decided).** These are **addressing and provenance, not method**: every candidate
above needs the same two primitives — *name a region*, *know whether its audio
would change* — and differs only in what it does with a named region. Locking
them is what preserves the freedom to change methods after the ear tests.

1. **Stable note ids in the score** — `signalml-score/0.2`, `NoteEvent.id`.
   Ids are handles, excluded from the content hash; editors mint, never
   renumber. 0.1 files upgrade on load.
2. **The render unit is a segment, not a song** — `score/segment.py` splits at
   rests; `sing` is the composition of segment renders. Extends D11's API-first
   pre-commitment by one word.
3. **Render intermediates are artifacts, not temporaries** — `synth/render.py`
   persists per-segment provenance *and* the variance track (F0 + durations).
   Without them method 2's repair path is impossible; with them it is UI work.

Cost was near-zero because nothing consumed `score.json` yet (the training path
never reads it) and zero scores existed on disk. **Nothing in P2–P7 changed.**

**The gates — method stays open until these are heard** (this is the point):

- **E1. Seam test — needs no model, runnable today.** Cut a clean corpus stem at
  a breath (the S4 silence map already marks them) and splice it back. If a
  boundary at a real breath is inaudible, method 3 drops from "the magic
  version" to "nice to have" — a large scope reduction bought cheaply.
- **E2. Mel-splice test — needs only the vocoder**, which trains first anyway:
  audio → mel → swap a region → vocode whole. Answers "splice at the mel" well
  before P8.
- **E3. "Is a reroll usefully different?" — needs the acoustic model (P7).**
  Cannot be pulled forward, most likely to send the *method* back to the drawing
  board, and — by construction — none of the three pre-commitments depend on how
  it lands.

**Where it lives:** not a new screen — the convergence of STUDIO §4
(Performance) and §6 (Editor): a timeline of **live regions** (score-backed,
re-renderable) and **frozen regions** (audio, hand-edited), with freeze/unfreeze
as the DAW metaphor. Studio phase F; gated on P8.

---

## Explicitly delegable now (no architect needed)

- **P5 offline half:** praatio TextGrid→phones.json converter, `align.yaml`, stage
  plumbing, golden-file tests (fixtures already planned). MFA install + real-corpus
  alignment runs happen on the rig, following README/plan.
- **P6 entirely:** score JSON schema + validator, naive MIDI importer, espeak-ng/MFA
  G2P wiring (D3/D4 later *upgrade* it; the schema is already contract-fixed).
- **P8 plumbing:** CLI, mixdown, golden-path integration test (once P7 emits any
  checkpoint).
- All future stage code following the established pattern: manifest-driven job +
  injectable backend + config YAML + offline tests + per-song saves + analysis.json
  sections. The pattern is demonstrated four times (acquire/separate/clean/features);
  point the executing model at those files as the template.
- Corpus onboarding on the rig: `manifest scan` per folder with language/gender
  tags, then `separate`/`clean`/`features` batch runs (README workflow section).

## Suggested order of the remaining planning sessions

1. **D2 spike** (choose fork, set the boundary) → unblocks D1, D3, D5, D6, D7.
2. **D1 + D3 together** (annotation + phone mapping — they share the dataset spec).
3. **D5 + D6** (spectral contract + vocoder recipe) — can start as soon as D2 lands,
   in parallel with 2.
4. **D4** (music-input contract + lyric fitting) — needed before P8 is meaningful.
5. **D8/D9** as P7 spins up.
