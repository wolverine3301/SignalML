# Target Architecture — Singing / Audio Synthesis Pipeline

> Planning output, no code. Working assumptions marked **[WA-Qn]** trace back to
> `OPEN_QUESTIONS.md`; if you answer a question differently, the sections flagged with
> that tag are what change.
>
> **Updated 2026-07-07 after Logan answered Q1–Q12.** Confirmed decisions lost their WA
> tags; the changed ones (Windows-first environment, MFA IPA phone set, adjustable
> sample-rate profiles, own vocoder, English+Gaelic, 78 h corpus) are folded in below.
> Remaining open: Q13 (which Gaelic + sequencing), Q14 (lyrics transcript coverage).

---

## 0. Decisions at a glance

| Decision | Recommendation | Confidence |
|---|---|---|
| Framework | **PyTorch** (≥2.7, CUDA 12.8 wheels) | ✅ Decided (Q1) |
| Environment | **Native Windows first** (Logan's call, Q5); WSL2 documented as fallback/upgrade path | ✅ Decided (Q5) — see §2 |
| Stem separation | **Demucs `htdemucs_ft`**, retire Spleeter | ✅ Decided (Q6) |
| System shape | **Composed pipeline of 3 separately-trained components**, not one model | Very high |
| Acoustic model | **Adapt OpenVPI DiffSinger** (diffusion/rectified-flow acoustic + variance models), *not* a from-scratch stacked autoencoder | High — see §3 |
| Vocoder | **Train our own NSF-HiFiGAN-class vocoder** (Q4); community NC checkpoint only as a temporary dev preview | ✅ Decided (Q4) — see §3.3 |
| Unique voice | **Speaker-embedding timbre space + density model + persisted voice profiles** ("voice bank") | High — see §4 |
| F0 extraction | **RMVPE** (singing-tuned); pyworld for auxiliary analysis | High |
| Alignment | **MFA with the MFA IPA phone set** → pipeline-native JSON; SOFA kept as fallback aligner | ✅ Decided (Q2) — see §6 |
| Score format | **JSON score file** modeled on DiffSinger `.ds`; MIDI/MusicXML importers | ✅ Decided (Q3) |
| Instrumental generation | **Symbolic-first** (MIDI gen + rendered instruments), sequenced last | ✅ Decided (Q8) |
| Audio standard | **Config profiles**: 22.05 kHz `dev` for fast pipeline testing, 44.1 kHz `prod` for real training (Q11); mono float32 either way | ✅ Decided (Q11) — see §3.3 note |
| Languages | **Language-agnostic pipeline; English first, Gaelic second wave** (78 h female-vocal corpus exists) | Q13 pending (which Gaelic, sequencing) |

---

## 1. Framework: PyTorch

The brief framed this as a real fork ("existing code is TensorFlow"). The code survey
(docs/CODE_SURVEY.md) dissolved it: **the repo contains no TensorFlow code at all** — the
only TF artifact is Spleeter run as a subprocess. Porting cost of "switching" is zero.

Independent reasons PyTorch is the only sensible choice:

- **Hardware.** The RTX 5090 (Blackwell, sm_120) is supported by stable PyTorch wheels
  since 2.7 with CUDA 12.8. TensorFlow dropped native-Windows GPU support after TF 2.10
  and its Blackwell story remains poor even under WSL2.
- **Ecosystem.** Every tool this project will touch is PyTorch: Demucs, torchaudio,
  DiffSinger and all maintained forks, NNSVS, VISinger2, HiFi-GAN/NSF/BigVGAN vocoders,
  RMVPE, torchcrepe, RVC. Building the one greenfield stage (training) in TF would mean
  translating every reference implementation by hand.

There is no flagged place where a TF answer would be salvageable; if you choose TF anyway
(Q1), the honest consequence is: no established SVS codebase to build on, vocoder ports by
hand, and a GPU support fight before the first training step. The docs do not sketch that
path further.

## 2. Environment: native Windows first (decided, Q5)

Logan's call: prioritize Windows right now. Consequences and guardrails:

- **Everything except MFA is first-class on native Windows**: PyTorch CUDA 12.8 wheels,
  Demucs, yt-dlp, librosa/soundfile, RMVPE, the vendored DiffSinger trainer. No blocker.
- **MFA is the one fragile piece.** Plan: conda-forge MFA in its own conda env on
  Windows (`conda create -n aligner -c conda-forge montreal-forced-aligner`); it works
  but is historically the most install-fragile tool in the stack. Two escape hatches,
  in order: (1) **SOFA** — a PyTorch singing-oriented aligner, runs anywhere PyTorch
  does and may beat speech-trained MFA on sung vowels anyway (P5 includes a head-to-head
  eval); (2) a minimal WSL2 Ubuntu used *only* for MFA runs, reading/writing a shared
  data directory.
- **Code must stay environment-portable** (this was already a convention): `pathlib`
  everywhere, no shell-string assembly, `DATA_ROOT` configurable — so a later move to
  WSL2/Linux (or a bigger Linux box, per the scale-up requirement) is a config change,
  not a refactor. Treat WSL2 as the documented upgrade path if Windows friction
  accumulates, not as a parallel environment to maintain.
- Win10 vs Win11 no longer matters much without WSL2 in the loop; either is fine for
  native training. (If the WSL2 fallback is ever used: prefer Win11, keep data on the
  Linux filesystem side.)

## 3. System decomposition and the acoustic-model choice

### 3.1 Three capabilities, three components (not one model)

The desired end state (instrumental generation, voice generation, sing-the-lyrics) is a
**composed pipeline**:

```
                        ┌──────────────────────┐
     MIDI/params ─────► │ A. Instrumental gen  │ ─────► backing track (audio)
                        │  (symbolic-first)    │            │
                        └──────────────────────┘            │
                        ┌──────────────────────┐            ▼
     sample/pick ─────► │ B. Voice bank        │ ─► voice profile ─┐
                        │  (timbre space)      │                   │
                        └──────────────────────┘                   ▼
                        ┌────────────────────────────────────────────────┐
 lyrics + score ──────► │ C. Singer: score→phonemes→variance→acoustic    │ ─► vocal audio
                        │    model→mel→vocoder                           │        │
                        └────────────────────────────────────────────────┘        ▼
                                                                          mixdown ─► song
```

Why composed, not end-to-end: it matches the control requirements (key/BPM/instrument are
*inputs*, not prayers to a latent space), each piece trains separately within the 32 GB
budget, failures are debuggable per stage, and component B is exactly the "persist a
sampled voice" requirement — which end-to-end models make hard. This confirms the brief's
own instinct against the Suno-style monolith.

### 3.2 The core fork: from-scratch U-Net autoencoder vs. building on SVS work

Options considered for component C (the singer), which is the heart of the project:

| Option | What it is | For | Against |
|---|---|---|---|
| **(a) From-scratch stacked AE / U-Net on mels** (original idea) | Autoencode mel spectrograms; sample & persist a latent as the voice | Full control; educational; matches original experiments | Reconstruction AEs produce blurry, over-smoothed mels (audible as muffled/robotic); a single latent entangles timbre with content/pitch, so a "sampled voice" won't stay consistent across songs; no duration model, no explicit F0 control — you'd re-derive years of SVS research alone |
| **(b) Adapt OpenVPI DiffSinger** ✅ | Maintained fork of DiffSinger: phoneme+duration+F0-conditioned **diffusion/rectified-flow acoustic model** producing mels, plus **variance models** (duration, pitch) and NSF-HiFiGAN vocoder; multi-speaker via `spk_embed` | Active community (v2.3.x, 2025-26 releases), designed for exactly this task, explicit speaker-embedding conditioning = clean hook for the voice bank, proven trainable on single consumer GPUs, tooling ecosystem (OpenUtau, SlurCutter, dataset formats) | Chinese-first community/docs; codebase carries UTAU-editor baggage; English phonemization path needs assembling (Q2/Q4) |
| **(c) NNSVS** | Modular Kaldi-style SVS toolkit | Very modular, research-friendly | Smaller community; overall quality ceiling below DiffSinger-family; more assembly required for the same result |
| **(d) VISinger2** | End-to-end VITS-style SVS (score→waveform directly) | One model, fast inference | Less modular (against the project's stage-boundary requirement), harder to bolt a sampled-timbre bank on, weaker community tooling |
| **(e) RVC-style conversion** | Generate/sing with anything, then voice-convert | Cheap timbre swapping | It's cloning-shaped, not synthesis; doesn't satisfy "sing these lyrics to these notes" natively |

**Recommendation: (b), decisively.** The stacked-AE intuition isn't wasted — the diffusion
denoiser *is* an encoder-decoder with skip connections doing iterative refinement, and
"sample a latent, persist it" survives as "sample a speaker embedding, persist it" (§4) —
relocated to the place in the architecture where it actually works. Build the pipeline to
*produce DiffSinger-format training data*, vendor the openvpi trainer behind our own
config/CLI, and keep the model swappable behind the Stage-7 contract.

**If you insist on from-scratch (Q-fork on decision #2):** the honest version is
"reimplement a small rectified-flow acoustic model with phoneme/F0/speaker conditioning" —
a 2-3 month detour to reach roughly where option (b) starts, worth it only if the goal is
learning-by-building rather than the working system. Flagged; not designed further.

### 3.3 Vocoder (decided, Q4: we train our own)

- **Decision:** train our own NSF-HiFiGAN-class vocoder on the own corpus (~78 h is
  ample; vocoder training doesn't need alignments, just clean audio — it can start as
  soon as S3/S4 produce clean vocal stems, well before the acoustic model is ready).
  ~14M params; roughly 1–2 weeks from scratch on the 5090, less if seeded from an
  architecture-only (unweighted) config.
- **Bootstrap only:** the community **PC-NSF-HiFiGAN** checkpoint (CC BY-NC-SA 4.0) may
  be used as a *temporary dev preview* while our vocoder trains — never in any artifact
  meant to outlive development, and always recorded in the run config so nothing NC
  leaks forward. Logan has kept the dataset commercial-clean deliberately; the vocoder
  decision completes that posture.
- **Mel parameterization is dictated by the vocoder contract and the active audio
  profile (Q11):** `prod` = 44.1 kHz / 128 mels / hop 512 / win 2048 / fmax 16k;
  `dev` = 22.05 kHz scaled equivalents (hop 256 / win 1024) for fast pipeline testing.
  **Models and vocoders do not transfer across profiles** — a production run means
  production-profile features, vocoder, and acoustic model end-to-end. The dev profile
  exists to validate plumbing cheaply, not to produce keepable checkpoints.

### 3.4 F0 and analysis

- **RMVPE** for per-frame F0 on singing (state of the art on vocals, robust to residual
  accompaniment bleed); **pyworld** as a cheap cross-check and for voicing/aperiodicity
  analysis; torchcrepe as an alternative if RMVPE integration is awkward.
- BPM/key/beats for the analysis stage: librosa is sufficient to start (`beat_track`,
  chroma-based key estimation); upgrade path is essentia if accuracy disappoints.

## 4. "Unique reusable voice": the voice bank

The core requirement: create a **novel** voice (not a clone), then reuse it consistently.

Mechanism (three layers, from data to artifact):

1. **Learn a timbre space.** Train the acoustic model multi-speaker over N female singers
   **[WA-Q4]** with a **learned speaker-embedding table** (one d-dim vector per training
   singer, d ≈ 256). The female-only scope narrows the manifold exactly as the brief
   hopes — embeddings occupy a tighter, better-interpolable region.
2. **Model the space.** After training, fit a light density model over the N embeddings —
   start with a full-covariance Gaussian (or a small GMM if N is large enough); a
   normalizing flow is the upgrade if samples sound "averaged." A **small VAE over the
   embeddings** (not over audio — see the option-(a) rejection in §3.2, which does not
   apply at this scale) is the flow's sibling here, and is worth a try specifically if the
   Studio's PCA axes prove too entangled to label: a learned latent may disentangle the
   timbre axes that linear PCA smears together. Either upgrade is minutes to fit — it
   trains on N vectors, not hours of audio. This is what turns the embedding table into a
   *space you can sample*.
3. **Sample → validate → persist.** Draw a novel embedding; render a standard test phrase
   set; auto-check it isn't a near-clone (cosine similarity to every training embedding
   below a threshold, e.g. < 0.85, using an independent speaker-verification embedder like
   ECAPA-TDNN on the rendered audio); persist as a **voice profile** — a small artifact
   (`voices/<name>/profile.json` + `embedding.npy`, schema in PIPELINE_AND_CONTRACTS.md)
   recording the vector, the model checkpoint hash it belongs to, creation params, and
   license/provenance notes. Re-injection at inference = load vector, condition the
   acoustic model. Consistency across songs is then trivially exact — it's the same vector.

Notes and risks:

- **Checkpoint coupling:** an embedding is only meaningful w.r.t. the checkpoint that
  learned the space. The profile pins the checkpoint hash; retraining requires a
  re-projection step (render the voice's reference phrases, re-fit the embedding under
  the new model) — planned in Migration P7.
- **How many singers is enough?** For a usable sampling space: the more *singers* the
  better, even at few minutes each; 20–50 female singers is a reasonable first target.
  Fewer singers → the Gaussian collapses toward interpolation between a handful of real
  voices (closer to "blend" than "novel"). This shapes data acquisition priorities (P1).
- This mechanism generalizes unchanged to the **bird-song ambition**: same architecture,
  bird vocalization corpus, timbre space of birds → sample a novel-but-consistent bird.
  Nothing in the core is human-specific except the phoneme conditioning (birds would use
  a learned or unit-based "phoneme" inventory) — deferred, but the door stays open.

## 5. Component A: instrumental generation (symbolic-first) **[WA-Q8]**

Requirement: controllable **key**, **BPM**, **swappable instrument**. Two families:

- **Neural audio generation** (MusicGen-class): key/BPM control is soft (prompt-level),
  instrument swapping means regenerating, training your own is data/compute-prohibitive,
  and fine control was the whole point of avoiding the Suno path.
- **Symbolic-first** ✅: generate/arrange **MIDI**, where key, BPM, and structure are
  *exact free parameters*; render audio with swappable instruments (FluidSynth/soundfonts
  to start; VST rendering or neural timbre later). A melody line is exportable straight
  into the singer's score (§6) — the two components share the score format.

Sequencing: **last** (Migration P9). The singing path needs only *a* backing track and a
score; both can come from existing MIDI files or a DAW for the entire development period.
The symbolic generator itself has cheap options when its turn comes (train a small
transformer on Lakh MIDI; or integrate an existing symbolic model) — decision deferred
until then, intentionally.

## 6. Component C: the singer, end to end

Inference-time dataflow (formats in PIPELINE_AND_CONTRACTS.md):

```
lyrics (text) ──► G2P/phonemizer ──┐
MIDI / MusicXML ──► score importer ─┴─► score JSON  {notes, syllables, phonemes, stress}
                                            │
                                            ▼
                              variance models (duration → per-phoneme timing;
                              pitch → F0 curve from notes + expressiveness)
                                            │        ▲
                                            ▼        │ voice profile (embedding)
                              acoustic model (rectified-flow/diffusion) ──► mel
                                            │
                                            ▼
                              NSF-HiFiGAN vocoder ──► vocal WAV ──► mixdown with backing
```

The "hold vowels, stress the right phonemes" requirement lives in two places: the score
JSON carries syllable→note bindings + stress flags (importer's job, from MusicXML lyric
syllabification or MIDI+lyrics heuristics), and the duration/pitch variance models learn
how singers actually stretch vowels and shape note transitions from the aligned training
data. TextGrid is not the score format (Q3, decided: JSON score).

**Phone set (decided, Q2): MFA IPA.** Training-side alignment uses MFA's IPA-based
English models; inference-side G2P for new lyrics must emit the *same* IPA set
(espeak-ng/`phonemizer` mapped through a project-owned normalization table — MFA also
ships G2P models, which keeps train/inference phone inventories identical and is the
preferred route). IPA is also what makes the multilingual ambition (English + Gaelic
now, "universal phoneme set" later — see OPEN_QUESTIONS Q13 and *Future direction*)
tractable: one shared inventory, per-language dictionaries.

## 7. Compute reality check (single RTX 5090, 32 GB)

| Component | Params (approx) | VRAM (train, bf16) | Wall-clock estimate on 5090 |
|---|---|---|---|
| Acoustic model (DiffSinger-class) | 40–70 M | well under 16 GB at batch 32–64 | days (≈2–5) per full run |
| Variance models (duration, pitch) | 10–30 M | trivial | hours to a day |
| Vocoder fine-tune (NSF-HiFiGAN) | ~14 M | ~10–16 GB (GAN: two nets + features) | 1–3 days fine-tune; ~1–2 weeks from scratch |
| Instrumental symbolic model (later) | 20–100 M transformer | modest | days |
| Voice-bank density model | tiny (fit, not trained) | — | minutes |

Verdict: **comfortable.** These models were routinely trained on 12–24 GB cards; the 5090
adds headroom for larger batches and bf16 throughput. Practices to bake in from day one:
bf16 autocast, gradient accumulation as the batch-size escape hatch, single-GPU-first but
DDP-compatible training loops (so multi-GPU later is a launch-flag change, not a rewrite),
and checkpoint/EMA discipline. The real constraint is **data quality, not compute or
volume** — Logan already holds **~78 h of curated female vocals** (English, plus **~12 h
of Irish + Scottish Gaelic in separate folders, all with lyrics `.txt` sidecars** —
Q13/Q14), which is *more* than the corpora behind most published SVS systems
(Opencpop ≈ 5 h, M4Singer ≈ 30 h). At that volume the bottleneck shifts entirely to the
pipeline: separation cleanliness, alignment accuracy, and per-singer labeling (the
timbre space needs to know *which* singer each segment is — singer identity in the
manifest is load-bearing, see §4). Vocoder training (own vocoder, §3.3) is data-hungry
but **alignment-free**, so the full corpus — Gaelic included, even before Gaelic
alignment exists (wave-2) — feeds it from day one.

## 8. Risk register (top 5)

1. **Alignment quality on separated vocals** is now the #1 quality lever (78 h exists;
   Q4 resolved the scarcity fear) → invest in P2–P5 quality checks, per-song alignment
   confidence scores, and the MFA-vs-SOFA eval.
2. ~~Lyrics transcript coverage unknown~~ **Resolved (Q14):** every song ships with a
   lyrics `.txt` — P5's coverage scan is a verification pass, not a backfill hunt.
3. **License hygiene** — resolved in principle (own vocoder, Q4); residual risk is only
   discipline: NC dev-preview artifacts must never leak into keepable outputs
   (mitigation: license fields in manifest/run configs, dataset-card roll-ups).
4. **Voice-bank novelty vs. quality tension** — sampled embeddings may sound averaged with
   few singers; mitigated by prioritizing singer *count* (how many distinct singers are
   in the 78 h? — worth adding to the manifest early) and the flow-upgrade path (§4).
   External corpora that could raise the *permissively licensed* singer count are surveyed
   in `docs/notes/candidate_corpora.md` — but run the singer census first; it may show the
   gap is already closed.
5. **MFA on native Windows** is the stack's most fragile install (Q5 decision) —
   mitigated by SOFA as first fallback and an MFA-only WSL2 env as second (§2).
6. **Gaelic alignment path** (Q13, resolved: wave-2 confirmed; both Irish `ga` and
   Scottish `gd` exist, ~12 h, lyrics included) — no pretrained MFA model for either;
   custom dictionary/acoustic training happens only after the English model proves out.
   Until then Gaelic audio still earns its keep in vocoder training (alignment-free).
