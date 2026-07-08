# Project Brief — Singing / Audio Synthesis Pipeline

> Purpose of this document: seed a **planning session** (no code written yet). Read this,
> survey the existing code in the repo, ask clarifying questions, then propose a target
> architecture and a phased migration plan from the current experimental state to a
> polished, production-ready pipeline.

---

## 1. Vision & End Goal

The origin of the project was a **singing AI**: compose songs and sing them. The scope has since
broadened in two directions:

1. **Generalize the toolset** so the same machinery serves many kinds of audio and signal work,
   not just singing (e.g. synthesizing a *novel bird song* that is not a copy of any real bird).
2. **Parametric, controllable generation** rather than a single end-to-end black box.

The desired end-state inference capability:

- Generate **instrumental music** with controllable parameters: **key**, **BPM**, and a
  **swappable instrument** for the melody/parts.
- A **voice generator** that can produce a voice.
- Provide **lyrics** + **select a voice** → the system **sings the lyrics to the provided music**,
  matching notes, holding vowels, and stressing the right phonemes.

Note: a Suno-style single end-to-end model is explicitly *not* required and is likely the wrong
target for a solo effort — it needs massive data/compute and gives up the fine control this
project actually wants. The modular, parametric pipeline below is both more achievable and better
aligned with the control requirements. (Open to being challenged on this.)

---

## 2. Core Requirements (non-negotiable design constraints)

- **Unique, reusable voice — not a clone.** The system must be able to create a *novel* voice
  and reuse that exact voice consistently across different songs. It is NOT simply cloning a real
  person's voice. The intended mechanism: learn a **latent / timbre space**, then **sample a point
  in that space and persist it** as a "voice profile" so it can be re-injected for consistency.
  (This same idea generalizes to the bird-song example — sample a novel-but-consistent timbre.)
- **Multi-domain.** The preprocessing/analysis/generation tooling should be reusable for general
  audio and signals, not hard-wired to human singing.
- **Production quality.** The goal of this effort is a clean, reproducible, modular pipeline with
  clear stage boundaries and I/O contracts. (Note: the messy experimental training code is *not*
  in this repo — see Stage 7.)

---

## 2a. Compute & Dataset Constraints (hard facts for architecture decisions)

- **Local training hardware:** a single **RTX 5090** (32 GB VRAM), **AMD Threadripper**, **64 GB RAM**,
  on **Windows 11**. The full model must be **trainable end-to-end on this local machine** — I want a
  working model built locally, not one that requires a cluster.
- **Scalability:** the design should **scale up** cleanly if/when I get access to more powerful
  machines (multi-GPU, more VRAM), but single-5090 training is the baseline target, not a stretch.
- **OS is flexible: I can boot either Windows 10 or Windows 11 easily.** Most of the existing work was
  built on **Windows 10**. *(Flag for Fable: several standard audio-ML tools — forced aligners like MFA,
  older TensorFlow-based Spleeter, some vocoder repos — are painful or unsupported natively on Windows.
  Weigh whether to standardize on **WSL2** (works on either Win10 or Win11) for the training/tooling
  environment, and factor OS-compatibility into every tool recommendation, not just quality.)*
- **Framework:** the existing code is **TensorFlow** (Spleeter is TF-based, and my old training
  experiments were TF). I am **open to PyTorch if it's the better direction.** *(Flag for Fable: the
  modern audio-ML / singing-synthesis ecosystem — Demucs, torchaudio, most SVS repos (DiffSinger,
  VISinger2, NNSVS), neural vocoders (HiFi-GAN/NSF/BigVGAN), CREPE/RMVPE, RVC — is overwhelmingly
  PyTorch. Since the training stage is greenfield anyway (Stage 7), recommend a framework decisively
  with that ecosystem reality in mind, and note what it costs to port the existing TF preprocessing.)*
- **Dataset scope:** deliberately limited to **female singers only** for the first model. This
  narrows the timbre distribution and should make learning the voice/timbre space more tractable —
  factor this into how the "unique reusable voice" latent space is designed and sampled.

---

## 3. Current Pipeline (stage by stage)

Each stage lists: what it does, current tooling, the input/output contract, and status.

### Stage 1 — Data acquisition
- Downloads audio from a list of YouTube URLs (convenience for gathering training data).
- Current tool: a YouTube downloader. *(Modernization candidate: `yt-dlp`.)*
- Output: files land in `raw/`.
- Status: exists.

### Stage 2 — Raw storage + metadata
- `raw/` holds raw song files; should handle **most audio file types**.
- Should contain a **metadata document** describing each file: **singer**, **gender**, **song name**.
- Status: partially exists. *(Open question: is the metadata format structured — CSV/JSON — or freeform text? A structured manifest is strongly preferred for a production pipeline.)*

### Stage 3 — Stem separation + directory scaffolding
- Separates each song into stems and creates a per-song folder, placing each stem in a
  **uniformly labeled subdirectory**.
- Current tool: **Spleeter** — works, runs locally, and I'm **inclined to keep it**. Open to a
  replacement only if it's **free**, runs locally, and is clearly better.
  *(Flag for Fable: **Demucs / HT-Demucs** is free, local, PyTorch-based, and substantially higher
  quality than Spleeter. Beyond quality, Spleeter's older TensorFlow dependency can be a maintenance/
  install headache on modern Python + Windows, whereas Demucs fits a PyTorch stack cleanly and runs
  comfortably on a 5090. Recommend for/against keeping Spleeter with that full picture — not just the
  quality delta.)*
- Status: exists.

### Stage 4 — Optional basic preprocessing (cleaning)
- Generic, reusable audio/signal cleaning: **normalization**, **filtering**, etc.
- Status: intended, should be modular and optional per-run.

### Stage 5 — Phonemization / alignment
- Turn lyrics into **tokenized phonemes with durations**, for mapping to song notes.
- Vision: given a **MIDI file or sheet music + lyrics**, adapt the lyrics to the notes —
  hold vowels, decide which phonemes to stress, etc.
- Output location: **top level of the song's stem directory**.
- Format history: originally **TextGrid** (from an older **MAUS**-based method). I have **since
  changed the phonemization method**, so the format is **not fixed** — whatever the new method and the
  singing step want is fine.
  *(Flag for Fable: **ask me what phonemization method I'm using now** before designing this stage —
  it's changed and the answer determines the format. Then recommend a representation that carries what
  the singer actually needs: likely a score-aligned per-phoneme record of {start, end, pitch/note,
  stress}, not just phoneme + timing. Don't assume TextGrid.)*
- Status: exists; method recently changed, format open.

### Stage 6 — Analysis + more preprocessing
- Chunk songs into smaller, digestible pieces.
- Generate **Mel spectrograms** and other spectrograms.
- Analysis tools: **BPM**, **key**, and other features.
- Likely tooling: `librosa` / `torchaudio`.
- **Likely gap to flag:** for singing, **per-frame fundamental frequency (F0 / pitch contour)**
  is arguably the single most important feature (it carries the melody) and isn't listed here.
  Extraction options for Fable to consider: CREPE, WORLD/pyworld, or RMVPE (strong for singing).
- Status: intended/partial.

### Stage 7 — Training (greenfield in this repo)
- **The messy experimental training code is NOT in this repo.** This repo currently holds only the
  cleaned-up preprocessing/pipeline stages above. The plan is to **build the training stage from
  scratch** here, cleanly, and possibly port useful pieces from the old experiments later.
- Original architecture idea (from the old experiments): a **stacked autoencoder, somewhat like a
  U-Net**, with a **sample of the latent space saved to preserve a consistent, unique voice profile**.
- **User is open to a different architecture** if there's a better fit — since this is a from-scratch
  build, there's no legacy code to preserve, so choose the best design for the requirements and the
  single-5090 compute budget.
- Status: not started in this repo; design it here.

---

## 4. Open Architecture Decisions (for this planning session to resolve)

These are the real forks. Do not assume — reason through them and recommend, then confirm with me.

1. **Framework: TensorFlow vs PyTorch.** The existing code is TensorFlow; the training stage is
   greenfield. Recommend which to standardize on, accounting for the audio-ML ecosystem, the
   single-5090 + Windows/WSL2 environment, and the porting cost of the existing TF preprocessing.
2. **Build-your-own vs. build-on-existing.** Is the U-Net/stacked-autoencoder path the right call,
   or should this build on an established singing-voice-synthesis (SVS) approach/framework
   (e.g. DiffSinger, VISinger2, NNSVS) and/or a neural vocoder (HiFi-GAN / NSF-HiFiGAN / BigVGAN)?
   Consider the "novel reusable voice" requirement specifically — how each option supports a
   **persisted timbre/speaker embedding** you can sample and reuse.
3. **How to realize "unique reusable voice."** Map the latent-sampling idea onto a concrete
   conditioning mechanism (speaker/timbre embedding, disentangled latent, etc.), including how the
   profile is stored ("voice bank") and re-injected at inference for consistency.
4. **Separation of concerns for the three inference capabilities** (instrumental generation,
   voice generation, lyric-to-song singing) — are these one model or a composed pipeline?
5. **Score representation.** What format best carries lyrics + notes + timing + stress into the
   singer? Keep TextGrid, or move to something richer?
6. **Compute reality check.** Given a **single RTX 5090 (32 GB VRAM), able to boot Win10 or Win11**
   as the training baseline (scalable later), what training approach fits? Model sizes, batch/precision
   strategy, whether each pipeline component trains separately, and whether to standardize the
   environment on WSL2. The full model must train locally on this machine.

---

## 5. Known Gaps / Modernization Candidates (my flags, not decisions)

- `yt-dlp` in place of an older YouTube downloader.
- **Demucs / HT-Demucs** as a higher-quality alternative to Spleeter.
- **F0 / pitch extraction** appears to be missing and is critical for singing.
- **Structured metadata manifest** (CSV/JSON) instead of freeform text.
- **Config management** (e.g. Hydra/YAML) so pipeline stages are reproducible and parameterized.
- **Clear I/O contracts + a manifest/dataset spec** so stages are independently runnable and testable.

---

## 6. Assumptions I've Made (correct me if wrong)

- Primary language is **Python** (spleeter, librosa, ML tooling all point this way).
- This is a **solo, long-running** project moving from research-grade experiments to a maintainable
  pipeline — so maintainability, reproducibility, and clear module boundaries matter as much as raw
  model quality.
- `README.md` and `pyproject.toml` currently exist but are **empty** — packaging/metadata is greenfield.
- The repo holds only the **cleaned-up preprocessing pipeline**; the **training stage is a
  from-scratch build** (no legacy training code to accommodate here).

---

## 7. What This Planning Session Should Produce

**IMPORTANT — you are running mostly UNATTENDED. I will check in intermittently (roughly once an
hour), but I am NOT at the keyboard continuously and cannot answer questions in real time.**
Do **not** stop and wait for me. Work autonomously through everything below. When you hit something
that genuinely needs my input, **do not block and do not guess silently** — instead:

- Append the question to a file called **`OPEN_QUESTIONS.md`** *as you go* (keep it current throughout,
  not just at the end), with enough context that I can answer it quickly, and your own recommended
  default for each. That way there's always something useful for me to resolve when I check in.
- Where a question is a **hard fork that later design depends on** (framework TF-vs-PyTorch,
  phonemization method, score representation), lay out the options with a clear recommendation, pick
  the most likely answer as a **stated working assumption**, and continue — but flag clearly anywhere
  that downstream design would change if I choose differently. Don't build a huge amount of detail on
  top of an unconfirmed fork; sketch the alternative path briefly instead.
- All output stays **local** to this project directory — write files here; there is no remote to push to.

Deliverables to produce as files before you finish:

1. **`OPEN_QUESTIONS.md`** — every question/decision needing my input, each with your recommended
   default, so I can resolve them fast when I return.
2. A **survey of the existing code** — an honest read of what to keep, refactor, or discard.
3. A **recommended target architecture** for the model(s), with tradeoffs of the main options laid
   out (including a clear recommendation on the U-Net idea vs. alternatives).
4. A proposed **module / directory structure** and **stage I/O contracts** for the production pipeline.
5. A **phased migration plan** (mess → production) with sequenced milestones, so the actual code can
   be executed later against this plan by a cheaper model.
6. A first draft of **`CLAUDE.md`** capturing the stack, conventions, and stage map.

Write these as clearly-named markdown files in the repo (e.g. `docs/ARCHITECTURE.md`,
`docs/MIGRATION_PLAN.md`, `OPEN_QUESTIONS.md`, `CLAUDE.md`).

**Do not write implementation code.** Plan, research, and design only — the output is documents,
not a working pipeline.
