# Migration Plan — Experimental Mess → Production Pipeline

> Written to be executed later, phase by phase, by a cheaper model with limited context.
> Each phase: goal, ordered tasks, acceptance criteria ("done when"), and what it does
> NOT depend on. Follow `docs/PIPELINE_AND_CONTRACTS.md` for every format/path named
> here; follow `docs/ARCHITECTURE.md` for every tool choice. If a task conflicts with an
> answered question in `OPEN_QUESTIONS.md`, the answer wins — check that file first.
>
> Phases are sequential by default; ⫽ marks phases that can proceed in parallel.
> **Rule for the executing model: do not start a phase until the previous phase's
> acceptance criteria pass. Do not redesign — implement the referenced contract.**
>
> **Updated 2026-07-07 after Q1–Q12 were answered:** native **Windows-first** (Q5), MFA
> **IPA** phone set (Q2), audio **profiles** dev/prod (Q11), **own vocoder** is a
> first-class milestone (Q4), Gaelic is wave-2 pending Q13. Still open: Q13, Q14
> (lyrics coverage) — P5 has a contingency for Q14.

---

## Phase 0 — Repo hygiene & packaging ✅ DONE 2026-07-07

> Completed as specified (all acceptance criteria verified: `uv sync` + 34 tests
> passing, ruff clean, target tree in place, legacy in `archive/`). Notes: `uv` is
> installed via `python -m pip install uv` and invoked as `python -m uv ...`; a minimal
> README.md was added; the CLI stub routes every stage name to its migration phase.

Goal: a modern, installable, linted, testable skeleton with the legacy code archived.

1. Fill `pyproject.toml`: package `signalml`, Python ≥3.11, deps split into
   `[project.dependencies]` = core (numpy, librosa, soundfile, pydantic, pyyaml, tqdm),
   extras `train` (torch pinned per CUDA 12.8 wheels, demucs, etc. — added in later
   phases as actually needed), `dev` (pytest, ruff). Use `uv` for env management.
2. Add `.gitignore` (data dirs, checkpoints, `__pycache__`, `.venv`, npz/wav outputs).
3. Create `archive/` and move: `signalml/pipeline/PreprocessRaw.py`,
   `signalml/ingest/youtubeDataScrape.py`, `signalml/analysis/untitled0.py`
   (verdicts in CODE_SURVEY.md). Fix any imports that referenced them (none expected —
   verify with grep).
4. Create `tests/` + `tests/fixtures/` with a generated 2-second sine WAV fixture
   (generate in a fixture-making script, don't commit logic-free binaries beyond a few KB
   each). Smoke tests: `signalml` imports; `audio.io.load` round-trips the fixture.
5. Add `ruff` config (defaults + line length 100); run and fix.
6. Create `configs/audio.yaml` with **named profiles** (Q11): `dev` = 22050 Hz / hop 256
   / win 1024 / 128 mels; `prod` = 44100 Hz / hop 512 / win 2048 / 128 mels (vocoder
   contract). `signalml/config.py` pydantic loader exposes the *active* profile
   (selected per-run via CLI/env); port `AudioConfig`/`SpectrogramConfig` to read from
   it. Remove the 22050 literals — the number now exists only inside the `dev` profile.
7. Restructure package dirs to the PIPELINE_AND_CONTRACTS.md §1 tree (create empty
   stage modules with docstring stubs; move `audio_loader.py`+`io.py` → `audio/io.py`,
   `chunking.py` → `audio/segment.py` (rewrite lands in P4), `mixing.py`+`masking.py` →
   `tasks/masking/`).

**Done when:** `uv sync && uv run pytest` passes; `uv run ruff check .` clean; repo tree
matches the target layout; legacy files live only in `archive/`.

## Phase 1 — Manifest + acquisition ✅ DONE 2026-07-07

> Completed: `signalml/manifest.py` (pydantic JSONL records, atomic save, status
> queries, sha256/url dedupe, `scan_directory` backfill with lyrics-sidecar detection
> per Q14 and `--language/--gender/--singer` tagging per Q13), `stages/acquire.py`
> (injectable downloader; yt-dlp native-container backend, no ffmpeg dependency), CLI
> subcommands (`manifest scan`, `acquire`), 17 new offline tests (51 total). Real-
> network smoke against actual YouTube URLs is left for Logan's first corpus run.

Goal: the manifest is the single source of truth; downloads are reproducible.

1. Implement `signalml/manifest.py`: JSONL read/write, record schema (pydantic, §2 of
   contracts doc), query helper (`select(status_filters)`), atomic append/update
   (write-temp-rename).
2. Implement `signalml manifest scan`: propose records for files already in `raw/`
   (checksum, duration via soundfile; meta fields null → flagged for human backfill).
3. Implement S1 `signalml acquire`: yt-dlp **Python API**, URL list → `raw/<id>.<ext>`,
   manifest records with source URL + retrieval date + checksum. Skip-if-checksum-known.
4. CLI scaffold `signalml/cli.py` (argparse or typer) with `acquire`, `manifest scan`.
5. Contract tests with a local fixture file (no network in tests; mock yt-dlp call).

**Done when:** running `acquire` on a 2-URL list produces 2 raw files + 2 valid manifest
records; re-running is a no-op; tests pass offline.

## Phase 2 — Stem separation with Demucs ✅ DONE 2026-07-07

> Completed: `stages/separate.py` (manifest-driven, injectable separator, uniform stem
> names, analysis.json model/version/device record, per-song manifest saves for
> resumability), `stages/common.py` (analysis.json helper), `configs/separate.yaml`,
> CLI `signalml separate` (idle runs never import torch), 9 offline tests (59 total).
> **Verified end-to-end with real Demucs 4.0.1 (htdemucs) on CPU** — the dev box is a
> GTX 1650/old-driver machine, not the 5090 rig, so the GPU run + htdemucs_ft default
> happens on the rig (install path documented in README "GPU install"). CPU torch
> wheels are what the `train` extra resolves by default on Windows.

1. Add `demucs` to the `train` extra (it pulls torch; document the Windows CUDA 12.8
   wheel install line — `--index-url .../cu128` — in README).
2. Implement S3 `signalml separate` per contract: manifest-driven selection, Demucs
   `htdemucs_ft` via its Python API, outputs renamed to uniform
   `songs/<id>/stems/{vocals,drums,bass,other}.wav`, model+version recorded in
   `analysis.json`, `status.separated=true`.
3. GPU used when available, CPU fallback works (tests use CPU + a 3-second fixture).
4. Spleeter: no port. It simply has no successor stage — note in README changelog.

**Done when:** one real song separates end-to-end on the 5090; fixture test passes on
CPU; manifest reflects status; stems are uniformly named.

## Phase 3 — Cleaning stage ✅ DONE 2026-07-07

> Completed: `stages/clean.py` (profile resample+mono, pyloudnorm BS.1770 loudness to
> target with peak ceiling, optional Butterworth high-pass, non-destructive silence map
> into analysis.json; every op an independent flag; pure `process_audio` split from I/O
> for property tests), `configs/clean.yaml`, CLI `signalml clean`, 11 offline tests
> (70 total) incl. the ±1 LU acceptance check. De-click intentionally not implemented
> (would need a real algorithm, not a placeholder) — documented as a future optional op.
> Verified on the real P2 smoke song; analysis.json accumulates separate+clean sections.

1. Implement S4 `signalml clean` per contract: resample→44.1k mono, pyloudnorm
   normalization (target LUFS in `clean.yaml`), optional high-pass, silence-map into
   `analysis.json`. Every op an independent flag (brief: "modular and optional").
2. Property tests: output sr/channels/loudness within tolerance; ops toggle off cleanly.

**Done when:** `clean` runs manifest-driven on separated songs; loudness verified within
±1 LU of target on fixtures.

## Phase 4 — Features & analysis ✅ DONE 2026-07-07

> Completed: `audio/segment.py` rewritten (single-decode chunking — verified by a
> decode-count test — plus gap-based silence-aware phrase segmentation with
> min/max/pad handling), `stages/features.py` (named-key NPZ per contract with
> profile/sr/hop/f0_method stamps and a profile-mismatch guard; log-mel base e;
> frame-aligned f0/voicing/energy; KS key estimate + BPM/beat grid + phrase map into
> analysis.json), `configs/features.yaml`, CLI `signalml features`, 17 new tests
> (87 total). **Deviation from plan, documented:** RMVPE has no standalone PyPI
> package (only heavy RVC bundles), so F0 is a backend abstraction — `pyin` default
> (torch-free), `torchcrepe` in the train extra (both verified finding 220 Hz on
> fixtures; torchcrepe verified on the real smoke song), `rmvpe` slot raises until
> vendored with third_party/ in P7. Every artifact records its f0_method.

1. Rewrite chunking as in-memory slicing in `audio/segment.py` (single decode; the O(n²)
   re-decode pattern in old `chunking.py`/`phonemes.py` must not survive). Add
   silence-aware phrase segmentation using the S4 silence map.
2. Implement S6 `signalml features` per contract: mel (vocoder params from
   `configs/audio.yaml`), **F0 via RMVPE** (vendored weights path in `features.yaml`;
   torchcrepe fallback flag), voicing, energy — all frame-aligned, named NPZ keys.
3. BPM + key + beat grid via librosa into `analysis.json`.
4. Tests: frame-count consistency (mel vs f0 vs voicing same T), F0 of a 220 Hz sine
   fixture ≈ 220 Hz, named keys present.

**Done when:** features exist for all cleaned fixture+real songs; frame-alignment test
passes; no anonymous `arr_0` keys anywhere.

## Phase 5 — Alignment / phonemization ✅ CODE DONE 2026-07-12 (corpus runs pending)

> Completed: `stages/align.py` (manifest-driven, one batched MFA invocation via
> injectable runner, corpus staged per-song-as-speaker, TextGrid audit copy kept,
> praatio converter → `align/phones.json` with strict phone-set validation, v1
> alignment-confidence heuristic = speech_sec/voiced_sec recorded to
> `quality.align_score`), `configs/align.yaml`, `score/phoneset.py` (versioned
> `mfa_ipa/en_v1` seeded from the english_us_mfa v3 inventory + `signalml score
> phoneset --dict` verifier), CLI `signalml align`, README MFA-install section,
> legacy `ingest/textgrid.py`+`phonemes.py` (+ their `phoneme_jobs` consumers)
> archived, `manifest report` census command + `meta.source_quality` field added.
> **Still needs the real corpus/rig:** MFA conda install + phone-set verify (README
> steps), first real alignment runs + spot-check listening, the SOFA eval (P5.4),
> and the lyrics-coverage verification pass over the 78 h corpus (P5.2).

1. Document **native-Windows MFA install** (conda env `aligner`, conda-forge package) in
   README, including the known-fragility note and both fallbacks (SOFA; MFA-only WSL2
   env) per ARCHITECTURE §2. Add `align.yaml` (language, acoustic model, dictionary,
   beam params) — English uses the **`english_mfa` IPA acoustic model + dictionary**.
2. **Lyrics coverage pass (Q14 resolved — verification, not backfill):** every song is
   expected to have a `.txt` lyrics sidecar next to the raw audio; add `meta.has_lyrics`
   to the manifest, scan, and report any stragglers. (A lyrics-*acquisition* step —
   fetch/Whisper-assist + human verify — is a future S1 enhancement for new data, per
   Logan; not needed for the existing corpus.)
3. Implement S5 `signalml align`: drive MFA CLI on `clean/vocals.wav` + lyrics sidecar;
   keep TextGrid intermediate; convert → `align/phones.json` (schema per contracts §S5,
   `phone_set: "mfa_ipa/en_v1"`) using `praatio` (not the buggy hand parser — see
   CODE_SURVEY.md on `textgrid.py`). Record aligner version; compute/record alignment
   confidence.
4. Evaluate **SOFA** on 3–5 real separated vocals vs MFA; record verdict + samples in
   `docs/notes/aligner_eval.md`. (Singing-oriented aligner may beat speech-model MFA on
   held vowels; cheap eval, decides the default aligner going forward.)
5. Delete `signalml/ingest/textgrid.py` + `phonemes.py` after their consumers are ported
   (segment extraction by phoneme moves into `stages/dataset.py` if still needed).
6. Tests: TextGrid fixture → phones.json golden file; converter rejects phones outside
   the declared phone set.
7. **Gaelic (wave-2 confirmed — Q13; do not build now):** corpus has both Irish (`ga`)
   and Scottish Gaelic (`gd`) in separate folders (~12 h, lyrics included) — tag
   `meta.language` per folder during P1's manifest scan so the split is preserved. When
   green-lit, this phase gains per-language configs: espeak-ng-bootstrapped `ga`/`gd`
   dictionaries → MFA custom acoustic-model training on the transcribed subsets.
   Nothing downstream changes (`phones.json` is language-tagged IPA).

**Done when:** ≥3 real songs have `phones.json` with plausible boundaries (spot-check by
listening to sliced phonemes); lyrics-coverage report exists; aligner eval note written;
old parser deleted.

## Phase 6 — Score format + importers ✅ DONE 2026-07-12

> Completed: `score/schema.py` (pydantic `signalml-score/0.2` since D12; slur convention =
> continuation notes carry same syllable + empty phonemes; validators for overlap/
> sort/slur/key), `score/g2p.py` (backend abstraction: LexiconG2P → MfaG2P
> (`mfa g2p`, preferred, batch + cache, injectable runner) → EspeakG2P (phonemizer,
> experimental, normalization table); max-onset `syllabify`; stress as separate
> field), `score/from_midi.py` (tempo-map-accurate tick→sec, monophonic enforcement,
> hyphen/`-`-melisma lyric tokens, per-occurrence syllable-count check),
> `from_musicxml.py` stub with implementer notes, CLI `signalml score
> validate|from-midi|phoneset`, golden MIDI→score test + 40 new tests overall.

1. Implement `score/schema.py` (pydantic, `signalml-score/0.2` per contracts §4) +
   validation CLI `signalml score validate`.
2. Implement `from_midi.py`: monophonic melody track + lyrics list → score JSON
   (syllable-to-note by order; melisma → `slur` continuation). Document limitations.
3. Implement `g2p.py`: English G2P emitting **MFA IPA** (Q2) — preferred: MFA's own G2P
   model (keeps train/inference phone inventories identical); fallback: espeak-ng via
   `phonemizer` + a normalization table into `mfa_ipa/en_v1`. Stress carried as the
   separate `stress` field, not phone suffixes.
4. `from_musicxml.py`: stub with clear NotImplemented + issue notes (MusicXML carries
   syllabification natively; implement when a real need appears).
5. Tests: golden MIDI fixture → golden score JSON.

**Done when:** a hand-made MIDI + lyrics for one verse produces a valid, human-readable
`score.json` that round-trips the validator.

## Phase 7 — Training stage (1–2 weeks calendar, mostly compute)

> Everything here per ARCHITECTURE.md §3–4; this is the greenfield stage.

1. ✅ DONE 2026-07-12 (D2 spike): **openvpi/DiffSinger v2.5.1** vendored as submodule
   at `third_party/DiffSinger` (`323a569`, Apache-2.0), posture adopt-their-world,
   zero patches, dedicated trainer venv (their pins conflict with the pipeline env).
   Full integration contract in `docs/notes/vendor_diffsinger.md` — S6b emits their
   `wavs/ + transcriptions.csv` per (singer, language); their binarizer owns training
   mels (matches `prod` profile exactly).
2. Implement S6b `signalml dataset build`: manifest query → trainer's binarized format;
   `dataset_card.md` with hours/singers/license roll-up. Refuse records with null gender
   or below alignment-confidence threshold.
3. Implement `signalml train acoustic|variance` wrappers: our YAML → vendored configs;
   run dir = config snapshot + git hash + dataset name. bf16 on; batch/grad-accum in YAML.
4. **Own-vocoder training (first-class milestone — Q4 decision).** Vocoder training
   needs only clean vocal audio (no alignments), so **start it as soon as P3/P4 output
   exists**, in parallel with P5/P6: NSF-HiFiGAN-class architecture from the vendored
   stack, trained on the cleaned own corpus at the **prod profile** (44.1 kHz) —
   **including the ~12 h Gaelic audio** (vocoder training is alignment-free, so wave-2
   sequencing doesn't idle that data; Q13). Expect 1–2 weeks wall-clock on the 5090 — it can run while alignment
   and score work proceeds. The community PC-NSF-HiFiGAN checkpoint may be used
   *only* as a temporary dev preview meanwhile (CC BY-NC — record it in run configs;
   never ship artifacts rendered with it).
5. **First training milestone — overfit sanity run:** tiny dataset (even 30 min, 2–3
   singers), **dev profile** (22.05 kHz, Q11) for fast iteration, train until it can
   resing a training snippet recognizably. This validates the entire data path before
   burning days on real runs. (Dev-profile checkpoints are throwaway by definition.)
6. **Multi-singer run:** full female dataset, **prod profile**, own vocoder; monitor
   per-singer quality. English-only first; Gaelic data joins per Q13 (wave-2 —
   phoneme-level IPA conditioning means added Gaelic data extends, not restructures,
   this run).
7. Implement voice bank (`signalml voice new` per contracts §5): embedding extraction
   from checkpoint, Gaussian fit, sampling, ECAPA similarity guard, profile persistence,
   ref-phrase rendering. Include `voice reproject` (re-fit an existing profile under a
   new checkpoint using its ref phrases).

**Done when:** a sampled (novel) voice profile sings a held-out score snippet **through
the own-trained vocoder** (no NC artifacts in the chain); the same profile produces the
*same voice* on a second, different snippet (informal consistency check + ECAPA cosine
between the two renders > 0.85).

## Phase 8 — Inference CLI & mixdown (2–3 days)

1. `signalml sing --score score.json --voice aurora [--mix backing.wav] -o out.wav`:
   full chain per contracts §S8.
2. Simple mixdown: time-align at 0, gain staging from config, soft-limit output.
3. End-to-end example in README with a public-domain melody.
4. Golden-path integration test (CPU, tiny checkpoint or mocked model, asserts pipeline
   plumbing not audio quality).

**Done when:** one command turns (score.json, voice, backing track) into a mixed WAV.

## Phase 9 — Instrumental generation (symbolic-first; deferred by design **[WA-Q8]**)

Scope intentionally thin until reached: 1. renderer first (`signalml render`:
MIDI + soundfont → stems via FluidSynth; key/BPM transforms on MIDI are exact and free);
2. then melody/accompaniment *generation* (evaluate small symbolic transformers vs.
existing checkpoints at that time — do not pre-decide now); 3. shared score JSON is
already the bridge to the singer.

**Done when:** `render` produces a backing track in a chosen key/BPM/instrument that
`sing --mix` consumes.

---

## Sequencing summary & first check-in targets

```
P0 → P1 → P2 → P3 → P4 → P5 ─┐
                    └── P6 ⫽ ┴─→ P7 → P8 → P9
```

Rough calendar (solo, part-time): P0–P4 ≈ one focused week; P5–P6 ≈ one more; P7 is
dominated by training wall-clock (own vocoder starts early and runs in parallel — P7.4);
P8 days; P9 open-ended. With ~78 h of vocals already curated (Q4), the highest-leverage
early activity is **not more audio**: it's *metadata* — per-song singer identity and
gender in the manifest (the timbre space depends on singer labels, ARCHITECTURE.md §4),
lyrics-transcript coverage (Q14), and a singer-count census, all doable during P0–P4.
