# Pipeline Structure & Stage I/O Contracts

> Companion to ARCHITECTURE.md. Defines the module layout, the data layout, and the
> contract (inputs → outputs, config, invariants) for every stage, so stages are
> independently runnable and testable.
>
> **Updated 2026-07-07 for the answered questions:** phone set is **MFA IPA** (Q2),
> audio parameters are **named profiles** (`dev` 22.05 kHz / `prod` 44.1 kHz, Q11),
> environment is **native Windows first** (Q5), score JSON confirmed (Q3), JSONL
> manifest confirmed (Q9).

---

## 1. Repository layout (target)

```
SignalML/
├── pyproject.toml            # uv-managed; deps split: [core], [train], [dev]
├── CLAUDE.md
├── README.md
├── OPEN_QUESTIONS.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PIPELINE_AND_CONTRACTS.md
│   ├── MIGRATION_PLAN.md
│   └── CODE_SURVEY.md
├── configs/                  # YAML, validated by pydantic models [WA-Q7]
│   ├── audio.yaml            # THE sample-rate/mel contract (one place only):
│   │                         #   named profiles — dev: 22050 Hz/hop 256/win 1024
│   │                         #   prod: 44100 Hz/128 mels/hop 512/win 2048 (Q11);
│   │                         #   every stage + artifact records which profile made it
│   ├── acquire.yaml
│   ├── separate.yaml
│   ├── clean.yaml
│   ├── align.yaml
│   ├── features.yaml
│   ├── dataset.yaml
│   └── train/
│       ├── acoustic.yaml
│       ├── variance.yaml
│       └── vocoder.yaml
├── signalml/
│   ├── config.py             # pydantic models mirroring configs/*.yaml
│   ├── manifest.py           # manifest read/write/query + status transitions
│   ├── audio/                # shared audio primitives (from ingest/)
│   │   ├── io.py             # load/save (absorbs audio_loader.py, io.py)
│   │   └── segment.py        # in-memory chunking/slicing (chunking.py rewrite)
│   ├── stages/               # one module per pipeline stage; uniform job shape
│   │   ├── acquire.py        # S1: yt-dlp wrapper
│   │   ├── separate.py       # S3: Demucs
│   │   ├── clean.py          # S4: resample/loudness/filters
│   │   ├── align.py          # S5: MFA/SOFA driver + converters
│   │   ├── features.py       # S6: mel/F0/BPM/key (absorbs features.py + phoneme_features.py)
│   │   └── dataset.py        # S6b: binarize training data
│   ├── score/                # score JSON: schema, MIDI/MusicXML importers, G2P
│   │   ├── schema.py
│   │   ├── from_midi.py
│   │   ├── from_musicxml.py
│   │   └── g2p.py
│   ├── voices/               # voice bank: profile schema, sampler, validators
│   ├── train/                # S7: thin wrappers around vendored trainer
│   ├── synth/                # S8: inference pipeline (score+profile → wav)
│   ├── tasks/
│   │   └── masking/          # legacy general-signals track (mixing.py, masking.py)
│   └── cli.py                # `signalml <stage> --config ...` entrypoints
├── third_party/              # vendored/pinned DiffSinger fork (git submodule or copy)
├── scripts/                  # thin drivers & one-off utilities only
├── tests/
│   └── fixtures/             # 2–3 s WAVs, tiny TextGrid, tiny score JSON
└── archive/                  # frozen legacy files (PreprocessRaw.py, etc.)
```

Data lives **outside the repo** on a local fast drive (Windows-first per Q5; keep
`DATA_ROOT` path-portable so a later Linux/WSL2 move is config-only), rooted at a single
configurable `DATA_ROOT`:

```
DATA_ROOT/
├── manifest.jsonl            # the spine of everything (§2)
├── raw/<song_id>.<ext>       # as-downloaded/as-provided audio, never modified
├── songs/<song_id>/          # per-song working directory
│   ├── stems/
│   │   ├── vocals.wav        # uniform stem names, always
│   │   ├── drums.wav  bass.wav  other.wav
│   ├── clean/vocals.wav      # post-S4 normalized vocal
│   ├── align/
│   │   ├── vocals.TextGrid   # aligner-native intermediate (kept for audit)
│   │   └── phones.json       # pipeline-native alignment (§S5)
│   ├── features/
│   │   └── vocals.npz        # named-key features (§S6)
│   ├── score.json            # when a score exists for this song (§4)
│   └── analysis.json         # BPM, key, loudness stats, chunk map
├── datasets/<dataset_name>/  # binarized training sets (S6b output)
├── voices/<voice_name>/      # voice bank (§5)
└── checkpoints/<run_name>/   # training outputs + config snapshot + git hash
```

## 2. The manifest (`manifest.jsonl`) — Stage 2's answer **[WA-Q9]**

One JSON object per source recording; append-only log discipline (stages update their own
namespaced block). Example record:

```json
{
  "id": "sng_0042",
  "source": {"kind": "youtube", "url": "https://...", "retrieved": "2026-07-03"},
  "file": {"path": "raw/sng_0042.mp3", "sha256": "...", "duration_sec": 213.4,
            "sample_rate": 44100, "channels": 2},
  "meta": {"singer": "artist-name", "gender": "F", "song": "title",
            "language": "en", "license_note": "personal research use",
            "source_quality": "separated", "processing": "produced",
            "domain": "sung", "genre": "edm"},
  "status": {"separated": true, "cleaned": true, "aligned": false, "featurized": false},
  "quality": {"separation_snr_est": null, "align_score": null, "notes": ""}
}
```

Rules: `id` is assigned once and is the join key for everything under `songs/<id>/`;
`gender` is required (dataset scope filter); `license_note`/`source` are required (the
license-hygiene mechanism from ARCHITECTURE.md §8); stages **never** rescan directories —
they query the manifest for `status.<prev_stage> == true && status.<this_stage> == false`.

Tag semantics (dataset recipes filter on these; tags never delete data): `source_quality`
= stem provenance (`studio` skips Demucs, `separated` = Demucs output); `processing`
= production baked into the *voice* (`dry`/`produced`/`heavy` — orthogonal to
provenance; human-tagged); `domain` = `sung` (default) / `spoken` (wave-3, D10);
`genre` = free text. Human source of truth is the per-song `META.txt`
(`SINGER:`/`SONG:`/`GENRE:`/`PROCESSING:`/`DOMAIN:`); `manifest scan` reads it for new
records and `manifest retag` refreshes existing records (additive tags only by
default, so hand-repaired manifest fields aren't clobbered by sidecar typos).

## 3. Stage contracts

Uniform stage-job shape (generalizing `phoneme_jobs.py`, the one good pattern in the
existing code): *read manifest → select work → process per song → write artifacts +
update manifest status → emit summary stats*. Every stage: idempotent (re-running skips
or overwrites deterministically per a `--force` flag), config-driven (one YAML each), and
runnable standalone via `signalml <stage>`.

### S1 · acquire — `signalml acquire`
| | |
|---|---|
| Input | URL list file or CLI args; `configs/acquire.yaml` |
| Tool | **yt-dlp** (library API, not shell strings) |
| Output | `raw/<id>.<ext>` + new manifest records (source, checksum, duration) |
| Invariants | never re-download an existing checksum; metadata (singer/gender/song) may be filled interactively later — record lands with `meta.gender: null` and S6b refuses null-gender records |

### S2 · manifest — not a stage, a library (`signalml/manifest.py`)
Backfill command exists for pre-existing files: `signalml manifest scan` proposes records
for audit. All other stages consume/update it.

### S3 · separate — `signalml separate`
| | |
|---|---|
| Input | manifest records with unseparated raw files |
| Tool | **Demucs `htdemucs_ft`** (4-stem), GPU **[WA-Q6]** |
| Output | `songs/<id>/stems/{vocals,drums,bass,other}.wav` at source sample rate (profile resampling happens in S4) |
| Invariants | uniform stem filenames (no model-named subdirs leaking through); store demucs model+version in `analysis.json` for reproducibility |

### S4 · clean — `signalml clean`
| | |
|---|---|
| Input | `stems/vocals.wav` (and optionally others) |
| Ops | resample→active-profile rate, mono; loudness normalize (**pyloudnorm**, target ≈ −23 LUFS); optional high-pass (~50 Hz) & de-click; silence-map computed and stored (not destructively trimmed) |
| Output | `clean/vocals.wav` + loudness/silence entries in `analysis.json` |
| Invariants | modular & optional per-run (brief requirement): each op is a flag in `clean.yaml`; raw and stems never modified |

### S5 · align — `signalml align` **(Q2 decided: MFA, IPA phone set)**
| | |
|---|---|
| Input | `clean/vocals.wav` + lyrics text (from manifest/meta or sidecar `.txt`; coverage of the N h corpus = OPEN_QUESTIONS Q14) |
| Tool | **MFA** (conda env, native Windows per Q5) with the **`english_mfa` IPA acoustic model + dictionary**; **SOFA** as singing-tuned fallback (P5 runs a head-to-head eval). Gaelic: no pretrained MFA model — custom dictionary/acoustic training, wave-2 (Q13) |
| Output | aligner-native intermediate (`align/vocals.TextGrid`, kept for audit) → converted to **`align/phones.json`**: `[{"ph": "aɪ", "start": 1.02, "end": 1.19, "word": "I", "stress": 1}, ...]` with header `{"phone_set": "mfa_ipa/en_v1", "aligner": "mfa-3.x", "language": "en"}` |
| Invariants | downstream stages read **only** `phones.json` — swapping aligners (or adding Gaelic) = one new converter/config, nothing downstream changes; alignment confidence recorded to `quality.align_score` so bad alignments can be excluded from training; `phone_set` is versioned to leave room for the future project-owned universal IPA subset (OPEN_QUESTIONS *Future direction*) |

### S6 · features — `signalml features`
| | |
|---|---|
| Input | `clean/vocals.wav` (+ `phones.json` if present) |
| Output | `features/vocals.npz` with **named** keys: `mel` (T×n_mels per active profile), `f0` (T, Hz, RMVPE), `voicing` (T, bool), `energy` (T); plus `analysis.json`: `bpm`, `key`, beat grid |
| Invariants | mel params come from `configs/audio.yaml` **only** (kills the five hardcoded 22050s found in the survey); the **profile name is stored in the NPZ** (`profile`, `sr`, `frame_hop` keys) and S6b refuses to mix profiles in one dataset; frame rate of every per-frame array is identical; no `arr_0`-style anonymous arrays |

### S6b · dataset — `signalml dataset build`
| | |
|---|---|
| Input | manifest query (e.g. `gender==F && aligned && quality.align_score > θ`) |
| Output | `datasets/<name>/` in the **vendored trainer's expected format** (DiffSinger binarized format under WA) + `dataset_card.md` (recipe: query, counts, hours, singer list, license roll-up) |
| Invariants | dataset is a *pure function of the manifest + config* — rebuildable byte-for-byte; the license roll-up makes NC-contamination visible at a glance |

### S7 · train — `signalml train {acoustic|variance|vocoder}`
Thin wrapper over the vendored DiffSinger fork: our YAML → their config; run dir gets
config snapshot + git hash + dataset name. Contract with the rest of the system is only:
*a checkpoint directory whose hash voice profiles can pin* (ARCHITECTURE.md §4).

### S8 · synth — `signalml sing` / `signalml voice new`
| | |
|---|---|
| `voice new` | samples timbre space → renders validation phrases → similarity guard → writes voice profile (§5) |
| `sing` | inputs: `score.json` + voice name (+ optional backing track) → phoneme timing (variance) → mel (acoustic, conditioned on profile embedding) → vocoder → `out/<song>_<voice>.wav`; `--mix` overlays backing track |

## 4. Score JSON (`score.json`) **[WA-Q3]**

Per song; array of note events bound to syllables and phonemes:

```json
{
  "format": "signalml-score/0.1",
  "bpm": 96, "key": "G:major", "language": "en", "phone_set": "mfa_ipa/en_v1",
  "notes": [
    {"start": 4.125, "end": 4.875, "midi": 67,
     "syllable": "shine", "phonemes": ["ʃ", "aɪ", "n"], "stress": 1,
     "slur": false}
  ]
}
```

- Held vowels: one syllable spanning multiple notes = consecutive note events with
  `"slur": true` continuation (DiffSinger convention, enables melisma).
- Importers lower **MIDI+lyrics** and **MusicXML** into this; hand-editing is viable
  (it's small JSON). TextGrid is *not* a score — it appears only as the S5 intermediate.
- Deliberately close to DiffSinger `.ds` so a mechanical converter to the trainer's
  format is trivial and community editors remain usable.

## 5. Voice profile (`voices/<name>/`)

```
voices/aurora/
├── profile.json    # {"name": "aurora", "created": "...", "embedding_dim": 256,
│                   #  "checkpoint": "ckpt_sha256...", "sampler": "gaussian_v1",
│                   #  "similarity_guard": {"max_cos_train": 0.79, "verifier": "ecapa"},
│                   #  "notes": "sampled 2026-07-03, seed 1234"}
├── embedding.npy   # the voice — a single d-dim vector
└── ref/            # rendered reference phrases (regression set for re-projection
                    # after retraining; also the "this is what aurora sounds like" demo)
```

The checkpoint pin + `ref/` phrases are what make voices survive model retraining
(re-projection procedure: Migration P7).

## 6. Conventions

- **Audio:** mono float32 WAV at the **active profile's** sample rate at every
  inter-stage boundary (Q11: `dev` = 22.05 kHz for fast pipeline testing, `prod` =
  44.1 kHz for real training runs). Artifacts record their profile; profiles never mix
  within a dataset or checkpoint. Only S1 raw files keep their native format.
- **Paths:** `pathlib` everywhere; no `os.getcwd()` anchoring; `DATA_ROOT` from config/env.
- **Config:** every tunable lives in `configs/*.yaml`, validated by pydantic at stage
  start; stages log their resolved config into their outputs.
- **Randomness:** every stochastic step takes an explicit seed (the existing
  `build_dataset_example.py` already does this — keep it universal).
- **Testing:** each stage ships a contract test against `tests/fixtures/` (seconds-long
  audio) that runs CPU-only and offline.
- **Augmentation** (pitch-shift, mixing/masking) lives only in `tasks/masking/` or inside
  S6b dataset recipes — never in ingest (survey finding).
