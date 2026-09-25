# CLAUDE.md — SignalML

Parametric singing/audio synthesis pipeline. Solo project (Logan), moving from research
experiments to a production-grade modular pipeline. **Current state: all major
decisions resolved (see the Resolved tables in OPEN_QUESTIONS.md); Migration P0–P4
completed 2026-07-07; P5 (align) code + P6 (score format) completed 2026-07-12 —
P5's corpus work (MFA install, real alignment runs, SOFA eval) still needs the
dataset/rig; P7.3 (`signalml train` wrapper + run records) done 2026-09-20, with the
rig recipes `configs/dataset.overfit.yaml` (P7.5 sanity run, amended to the **prod**
profile) and `configs/dataset.full_v2.yaml` (first multi-singer run).** The design
docs below are the source of truth. **Rig session 1 (2026-09-20) is done**: overfit_v1
trained to 20k steps, acoustic resynthesis works end to end. Next actions: **D1** (note
labels via SOME — without them there is no variance model, so the system re-sings
existing takes but cannot sing a new score), then rig session 2 =
`full_acoustic_v2` (speaker floor off since 2026-09-24: N singers, ~14.7 h; ship `corpus`, rebuild the dataset *there* since
generated configs carry absolute paths), and **P7.4** (openvpi/SingingVocoders is
**not yet vendored**, so every render is still NC-licensed). NOTE:
development happens on Logan's laptop (GTX 1650) and the 2080S work PC; heavy/GPU work
runs on the rig (borrowed RTX 4090, everything under `G:\SIGNAL_AI`, reached by SSH —
`docs/notes/rig_session_2026-09-20.md`). First training run landed 2026-09-20:
overfit_v1, 20k steps, resynthesis works. F0 note: RMVPE is
not on PyPI; backends are pyin (default) / torchcrepe, with rmvpe vendored in P7.

Dev loop: `python -m uv sync` · `python -m uv run pytest` · `python -m uv run ruff
check .` — all must pass before a phase is called done.

## Read these before changing anything

- `docs/ARCHITECTURE.md` — model/tooling decisions and why (framework, DiffSinger,
  voice bank, compute budget)
- `docs/PIPELINE_AND_CONTRACTS.md` — target repo/data layout, manifest schema, per-stage
  I/O contracts, score/voice-profile formats
- `docs/MIGRATION_PLAN.md` — phased execution plan (P0–P9); implement phases in order,
  gate on each phase's "Done when"
- `OPEN_QUESTIONS.md` — **check first**: unresolved forks with working assumptions.
  An answered question there overrides anything in the other docs.

## Stack (decided)

- **Python ≥3.11, PyTorch ≥2.7 (CUDA 12.8 wheels)**. The rig is a borrowed **RTX 4090**
  (sm_89, 24 GB), not the 5090 long assumed here; cu128 wheels list no sm_89 and run on
  it anyway (cubins are compatible within a generation). No TensorFlow —
  do not reintroduce it (Spleeter is retired in favor of Demucs).
- Environment: **native Windows first** (Logan's decision, Q5); `uv` for the Python env;
  conda only for MFA (its own `aligner` env). Keep all code path-portable (`pathlib`,
  `DATA_ROOT` config) — WSL2/Linux is the documented fallback (MFA) and upgrade path.
- Key tools: yt-dlp (S1), Demucs `htdemucs_ft` (S3), pyloudnorm (S4), MFA with **IPA
  phone set** / SOFA fallback (S5), librosa + RMVPE (S6), vendored OpenVPI DiffSinger
  (S7/S8) + **our own-trained NSF-HiFiGAN-class vocoder** (community NC checkpoint is
  dev-preview only).

## Stage map

acquire (S1) → manifest (S2, JSONL spine) → separate (S3) → clean (S4) → align (S5) →
features (S6) → dataset (S6b) → train (S7: acoustic/variance/vocoder) → synth (S8:
`sing`, `voice new`). Instrumental generation (P9) is symbolic-first MIDI and deferred.

## Hard conventions

- **Audio profiles (Q11):** mono float32 WAV at the active profile's rate at every
  inter-stage boundary — `dev` = 22.05 kHz (fast pipeline testing), `prod` = 44.1 kHz
  (real training). All audio/mel parameters come from `configs/audio.yaml` — never
  literal sample rates in code (the legacy hardcoded 22050s are a known bug class).
  Artifacts record their profile; never mix profiles in a dataset or checkpoint;
  dev-profile checkpoints are throwaway.
- **Phone set: MFA IPA** (`mfa_ipa/en_v1`, versioned). All phoneme fields
  (`phones.json`, `score.json`) are IPA; stress is a separate field, never a phone
  suffix. Languages: English first, Gaelic wave-2 (OPEN_QUESTIONS Q13).
- **Targeted editing (D12):** `score.json` is `signalml-score/0.2` — notes carry stable
  `id`s. **Mint new ids, never renumber**, or saved references retarget silently. The
  render unit is a **segment** (phrase, split at rests), not a song; a segment's content
  hash drives the render cache, so anything that changes the audio must be in
  `synth.render.RenderInputs` (bump `RENDER_KEY_VERSION` when you add a field) — an input
  missing from the key serves stale audio with no error. Variance output (F0 +
  durations) is a persisted artifact, not a temporary: it's what makes repairing a take
  possible instead of rerolling it.
- Stages are manifest-driven: query `manifest.jsonl` for work, write artifacts under
  `songs/<id>/`, update `status.*`. Never directory-scan for work.
- Every stochastic step takes an explicit seed. Every stage is idempotent and has a
  contract test against `tests/fixtures/` (CPU-only, offline).
- `pathlib` everywhere; paths must work on Windows *and* WSL2.
- Augmentation (pitch shift, mixing) never happens at ingest — only in dataset recipes
  or `signalml/tasks/masking/`.
- Licensing hygiene: every external artifact (dataset, checkpoint) gets its license
  recorded in the manifest / dataset card / run config. Community NSF-HiFiGAN weights
  are **CC BY-NC** — dev preview only; the production vocoder is trained in-house
  (decided Q4), and nothing rendered through NC weights ships.

## Voice bank (the project's signature feature)

A "voice" = a sampled speaker-embedding vector + the checkpoint hash it belongs to,
persisted under `voices/<name>/` with rendered reference phrases. Novelty is enforced by
an ECAPA cosine-similarity guard against training singers. Never treat a voice profile
as portable across checkpoints without running `voice reproject`.

## Legacy code notes

`archive/` (post-P0) holds the 2023 scripts — reference only, never import. Known
legacy bugs documented in `docs/CODE_SURVEY.md` (TextGrid parser tier logic, O(n²)
re-decode chunking, hardcoded sample rates); don't copy those patterns forward.
