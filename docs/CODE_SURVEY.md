# Code Survey — What Exists, What to Keep

> Honest read of every file in the repo (all 15 Python files were read in full).
> Verdicts: **KEEP** (use as-is or with light edits) · **REFACTOR** (right idea, wrong
> shape) · **DISCARD** (superseded or exploratory; archive, don't maintain).

---

## Headline findings

1. **There is no TensorFlow in this repo.** The brief assumes "the existing code is
   TensorFlow," but the cleaned `signalml/` package imports only librosa, numpy,
   soundfile, tqdm. The single TF touchpoint is **Spleeter, invoked as a subprocess**
   from a 2023-era legacy script. Switching to PyTorch + Demucs costs *nothing* in
   porting effort. This defuses Open Architecture Decision #1 almost entirely.
2. **The repo is already mid-refactor.** `signalml/` is clearly a cleanup pass over the
   2023 scripts (`PreprocessRaw.py`, `youtubeDataScrape.py`) — docstrings even reference
   the original function names (`genElements`, `makeClassSamples`, `synthesis2/3/4/5`).
   The right move is to *finish* that refactor, not restart it.
3. **The code serves the old classification/masking experiments**, not singing synthesis.
   Chunking, class-mixing, and masking datasets (voice+vehicle, singing+music) are a
   source-separation/denoising training set generator. Useful as a general-signals side
   capability; almost none of it is on the singing-synthesis critical path.
4. **What's missing for singing:** F0 extraction, loudness normalization, a metadata
   manifest, note/score handling, any training or inference code, tests, packaging
   (`pyproject.toml` and `README.md` are empty), and a real CLI.

---

## File-by-file verdicts

### Legacy scripts (2023)

| File | Verdict | Notes |
|---|---|---|
| `signalml/ingest/youtubeDataScrape.py` | **DISCARD** (salvage intent only) | Already calls `yt-dlp` internally, so "modernize to yt-dlp" is half-done. But: module-level side effects (`toWav(...)` runs on import), hardcoded `raw/birds` paths, dead youtube-dl code, shell-string building, and the Spleeter subprocess call. Rewrite as a thin `yt-dlp` wrapper stage reading URLs from the manifest (Migration Plan P1). |
| `signalml/pipeline/PreprocessRaw.py` | **DISCARD** | Explicitly superseded by the `signalml` package refactor. Also broken on modern librosa (`librosa.audio.get_duration(x, sr1)` positional args, `librosa.output.write_wav` removed). Executes work at import time (lines 489–513). Archive for reference. |
| `signalml/analysis/untitled0.py` | **DISCARD** | Exploratory PCA/KMeans/DBSCAN latent-space poking; 80% commented out; loads from a hardcoded `extract/` dir. The *intent* (visualizing timbre clusters) returns in Stage 7 as proper embedding-space diagnostics. Archive. |

### Cleaned package — `signalml/ingest/`

| File | Verdict | Notes |
|---|---|---|
| `config.py` | **KEEP / extend** | Frozen dataclasses are the right pattern. But defaults encode the old experiments: `sample_rate=22050` and `n_mels=256` conflict with the singing target (44.1 kHz / 128 mels to match modern NSF-HiFiGAN vocoders — see ARCHITECTURE.md). Config values must move to YAML with pydantic validation; dataclasses stay as the typed carrier. |
| `audio_loader.py` | **KEEP** | Thin, correct librosa wrapper. Fine as the single audio-in choke point. |
| `io.py` | **KEEP (short-term)** | `save_wav` fine. The `arr_0, arr_1, …` NPZ pattern loses all metadata (which song, which offset, what sr) — replaced by per-song feature files with named keys (PIPELINE_AND_CONTRACTS.md §Stage 6). |
| `chunking.py` | **REFACTOR** | Correctness OK, performance bad: it re-opens and re-decodes the file from disk **once per chunk** (`load_audio(path, offset=…)` in a loop) — O(n²) decode cost on long files, and resampling each time. Load once, slice the array. Also: fixed-size chunking with remainder-clipping is the old classification scheme; singing training wants phrase/silence-aware segmentation (P4). |
| `folder.py` | **REFACTOR** | Two design smells: (a) **augmentation is baked into ingest** (`augment_with_pitch` at chunk time) — augmentation belongs at dataset/training time, or you silently double your data and can't turn it off per-experiment; (b) `save_wavs` writes chunk WAVs back into the *raw input folder*, polluting the source dir. Note: random ±6-semitone pitch shift is actively harmful for singing training once F0 becomes a conditioning signal (it changes formants and detunes against any score); keep it only for the general/masking track. |
| `textgrid.py` | **REFACTOR → replace** | Hand-rolled TextGrid parser with two real bugs it faithfully inherited: `tier = +1` (line 57) *assigns 1*, doesn't increment — so everything after the first `MAU` marker is treated as phone tier; and the `pop(0)` heuristic silently misaligns if tier headers also match `xmin/xmax`. Verdict: keep only until alignment stage is rebuilt; then parse TextGrids with `praatio` (or consume MFA/SOFA JSON directly) and convert to the pipeline's score/alignment JSON. |
| `phonemes.py` | **REFACTOR** | Segment extraction logic is sound (and again re-decodes the file per phoneme — same O(n²) I/O as chunking). `scale_to_constant_timeframe` uses an arbitrary inherited formula (`factor = dur * 0.75`) and time-stretching phoneme audio destroys the temporal structure a duration-aware acoustic model needs — drop it for the singing path. `phoneme_safe_name` maps MAUS/SAMPA tokens (`<p:>`, `h\`) — replace with the new phone set's mapping once Q2 is answered. |

### Cleaned package — `signalml/pipeline/`

| File | Verdict | Notes |
|---|---|---|
| `features.py` | **KEEP / merge** | Correct librosa mel/STFT wrappers. Merge with `phoneme_features.py` into one `features` module; add the missing critical features: **F0 (RMVPE)**, loudness, voicing flags. |
| `phoneme_features.py` | **REFACTOR / merge** | Same as above; `sr=22050` hardcoded inside the MFCC branch (line 37) — a latent bug the moment the sample rate changes. `n_chroma=48` inherited without rationale. |
| `mixing.py` | **KEEP (general track)** | Fine utilities for the masking/denoising side quest. `mix_mel_npz_roundtrip` (mel → Griffin-Lim audio → mix) is very lossy — document as legacy-parity only. Not on the singing critical path. |
| `masking.py` | **KEEP (general track)** | Clean version of the old masking-dataset generator (even fixed the tqdm `total` bug the original had). Same status as `mixing.py`. |
| `phoneme_jobs.py` | **KEEP as template** | The one file that already looks like a proper *stage job*: iterate manifest-ish inputs → extract → featurize → write artifacts → return stats. Its shape (not its specifics) is the pattern for every production stage. Numeric-folder dataset layout (`dataset/{1..106}/`) gets replaced by the manifest. |
| `__init__.py` files | **KEEP** | Fine; exports will track the reorganization. |

### Scripts

| File | Verdict | Notes |
|---|---|---|
| `scripts/build_dataset_example.py` | **KEEP as example** | Reproducible (seeded rng), readable driver for the masking track. Becomes a documented example once a real CLI exists. |
| `scripts/build_phoneme_dataset.py` | **KEEP as example** | Same; hardcoded `range(1, 107)` sample IDs illustrate exactly why the manifest is needed. |

---

## Cross-cutting observations

- **No tests anywhere.** The refactored package is testable (pure functions, dataclasses) —
  it just has no tests. Migration P0 adds pytest + a tiny fixture WAV; every later stage
  lands with contract tests.
- **Sample rate 22050 is baked in five places** (config default, masking default,
  `phoneme_features` hardcode, `PreprocessRaw`, `savewav`). Singing synthesis should be
  44.1 kHz end-to-end (vocoder-dictated). One config, zero literals — see OPEN_QUESTIONS
  Q11.
- **Windows-only path habits** in legacy code (`'\\'` joins, `os.getcwd()` anchoring).
  The cleaned package already uses `pathlib` — keep that standard, since the environment
  recommendation is WSL2 (paths must work on both).
- **Two products are tangled together**: (a) the singing-synthesis pipeline (the goal) and
  (b) a general audio classification/masking dataset factory (birds/vehicles/voice
  experiments). They share ingest/features but diverge after. The proposed layout keeps
  (b) as a thin, honest side module (`signalml/tasks/masking/`) instead of letting it
  shape the core pipeline.
- **Nothing here handles notes, scores, or MIDI** — the entire musical dimension of the
  project is greenfield, which is why the score format (OPEN_QUESTIONS Q3) is a hard fork
  worth deciding early.

## What survives into the production pipeline

Concretely: `config.py` (pattern), `audio_loader.py`, `io.save_wav`, the *shape* of
`phoneme_jobs.py`, `features.py`'s wrappers, and the masking/mixing pair as an optional
task module. Everything else is either replaced by better-suited tools (Demucs, MFA/SOFA,
RMVPE, yt-dlp wrapper) or rewritten against the stage contracts in
PIPELINE_AND_CONTRACTS.md.
