# Studio — the voice-generation front-end (post-P8)

> Designed 2026-07-19 (planning session). **Gate: starts after P8** — the Studio is
> a shell over S8; it renders nothing P8's CLI can't already render. Decision-point
> entry: DECISION_POINTS.md D11. Wave-3 (D10) extends this tool; it does not
> replace it.

## 1. What it is

`signalml studio` — a local web app for the two loops the CLI is bad at:

1. **Voice Lab** — sample the timbre space, audition candidates side by side,
   refine, and persist the winner as a voice profile (`voice new` with a face).
2. **Performance** — pick a saved voice, pick/import/retext a score, render songs
   and voice lines (`sing` with a face).

**Hard rule: the Studio contains zero synthesis logic.** Every button calls the
same `signalml.voices` / `signalml.synth` Python API the CLI calls; every action
records its seed and resolved config, so anything done in the UI is reproducible
from the command line. The UI layer is disposable by construction.

**The one upstream implication (pre-commitment for P8):** build S8 as an
importable API first, CLI as a thin shell over it. Costs nothing extra; the
Studio (and tests) then get the engine for free.

## 2. Form factor

- **FastAPI server + browser UI**, package `signalml/studio/` (server, static
  frontend, no model code). Launched as `signalml studio --host 0.0.0.0`.
- Why web: sliders + audio playback + side-by-side grids need a GUI, and the
  two-machine setup (laptop dev, 5090 rig renders) makes browser-over-LAN the
  natural transport. Batch jobs keep running on the rig with the laptop closed.
- Renders execute through a small in-process job queue (one GPU consumer);
  results are content-addressed and cached (§3) so replay/compare never
  re-renders.
- Frontend fork (D11): Gradio prototype vs plain single-page frontend. Lean:
  **custom SPA from the start** (vanilla or lightest-possible framework) — the
  candidate-tray UX in §3 is exactly what Gradio fights. A day-scale Gradio
  spike to feel out the audition loop first is acceptable; it may not grow
  features.

## 3. Screen 1 — Voice Lab

### Sampling controls

The space is a d≈256 speaker-embedding Gaussian (ARCHITECTURE.md §4). Raw
per-dimension sliders are meaningless; expose derived controls only:

- **Temperature** — scale of deviation from the density model's mean. The
  day-one slider: low = safe/averaged, high = distinctive/risky.
- **Principal-component axes** — PCA over the training-singer embeddings; expose
  the top ~6–10 components as sliders. Axis meanings (brightness, weight, age…)
  are discovered by ear; axes are **renameable in the UI** and the labels persist
  (`studio/pca_labels.json` beside the checkpoint). PCA basis is recomputed per
  checkpoint and cached.
- **Blend** — interpolation slider between any two candidates. The most reliable
  operation in a tight female-only manifold; powers *breed* below.
- **Seed** — visible field + reroll button, always recorded.

Sliders can steer *onto* a training singer, so the ECAPA similarity guard is not
just a save-time gate: every rendered candidate shows its **nearest-training-
singer cosine as a live green/amber/red readout** (threshold from the profile
schema, PIPELINE_AND_CONTRACTS.md §5).

### Demo phrases

Auditioning uses a **fixed demo score set**: 2–3 short phrases (~5–8 s) — one
sustained/legato line, one fast syllabic line, one range-spanning phrase.
Constraints:

- **Public-domain or original melodies only** (licensing hygiene: a recognizable
  modern melody baked into every voice's demo renders is a rights problem).
- Shared with D8's frozen benchmark score set where possible — one curated set,
  two consumers.
- Short by design: seconds-long phrases keep the render loop interactive on the
  5090 and become the `ref/` phrases when a candidate is saved.

Every render is cached keyed on `(embedding_hash, score_id, checkpoint_hash,
seed)`; the tray never re-renders on replay or re-sort.

### Candidate tray (the compare loop)

The generate→compare→regenerate loop is interactive evolutionary search; the UI
leans into that. A session holds a tray of **candidates** (embedding + sampler
params + seed + cached demo renders) under a scratch dir — disposable until
saved. Per-candidate actions:

- **Reroll** — new seed, same slider settings.
- **Tweak** — open the sampling controls pre-loaded at this candidate's position.
- **Breed** — interpolate two pinned candidates + small jitter → 3–4 children.
- **Pin / discard**; keyboard-driven triage (space = play, arrows = navigate,
  digits = rate).
- **Blind A/B mode** — hide candidate identity while comparing (anchoring bias
  is real when one card is "yours").
- **Batch sampling** — queue N candidates at varied temperatures (overnight on
  the rig), triage in the morning.

### Save

Promotes a candidate through the full `voice new` path (PIPELINE_AND_CONTRACTS.md
§3/S8): render the full validation-phrase set, similarity guard as a **hard
gate**, write `voices/<name>/` with checkpoint pin, seed, and slider/sampler
provenance in `profile.json` notes. Unsaved candidates die with their session.

## 4. Screen 2 — Performance

### Voice picker

Gallery of saved voices; clicking plays `ref/` phrases (already defined as the
"this is what aurora sounds like" demo, PIPELINE_AND_CONTRACTS.md §5). Each card
shows the checkpoint pin and a **stale badge** when the active checkpoint has
moved, with a *reproject* action that runs Migration P7's re-projection and
presents before/after ref-phrase A/B for acceptance.

### Score sources (ascending effort; the lyrics box is scoped honestly)

The model sings `score.json`, not raw lyrics — freeform "type lyrics, get a
song" is melody generation, which stays deferred (P9). What the screen offers:

1. **Score picker** — browse existing `songs/<id>/score.json` from the corpus,
   render any of them with any voice. Works day one.
2. **MIDI + lyrics import** — fronts the existing P6 importer (naive assigner
   now, upgraded by D4's lyric-fitting pass when that lands).
3. **Lyric retexting** — pick an existing melody, type *new* words; the tool
   re-syllabifies, runs G2P into `mfa_ipa/en_v1`, and re-binds syllables to the
   existing notes. This is **D4's lyric-fitting algorithm applied to a fixed
   melody** — the Studio fronts it, does not define it. Retexting a template
   flat melody is the stopgap for spoken-ish "voice lines" until D10's real
   speech domain.

### Per-render controls & history

- Transpose (with an auto-fit suggestion once a voice has an observed
  comfortable range), tempo scale, backing-track upload → S8 `--mix` gain
  staging, explicit seed.
- **Render history**: every render row stores voice, score, checkpoint hash,
  seed, resolved config; replayable and re-runnable (the reproducibility
  convention surfacing in the UI — accidental great takes stay recoverable).

## 5. Screen 3 — Bank manager

List of saved voices: created date, checkpoint pin + staleness, license/
provenance notes, ref playback, rename/archive. Small on purpose; exists so the
other two screens don't accrete management clutter. When D10 lands, per-voice
domain coverage (sings/speaks/acts) surfaces here.

## 6. Phasing

- **A (right after P8):** S8 API layer, Voice Lab with temperature + seed +
  tray + save, demo score set defined. No PCA sliders yet — random sampling
  with reroll proves the loop.
- **B:** breed/blend, PCA axes, blind A/B, batch sampling.
- **C:** Performance screen — score picker and MIDI import first; retexting
  when D4's lyric-fitting pass exists.
- **D:** rides D10 — speech/acting domains, emotional-notes track editing
  (`signalml-score/0.2` prosody events).

Nothing here blocks or changes P7. The only pre-commitment is §1's API-first
shape for P8.
