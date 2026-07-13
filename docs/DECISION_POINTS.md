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
(P4) but nothing turns "N h of aligned vocals" into **note-level training scores**.
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
separated vocals (the test_corpus corpus is processed and ready for this).
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
**singer-count reality check**: census the N h corpus (how many distinct singers?)
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
