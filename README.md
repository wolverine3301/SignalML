# SignalML

Parametric, controllable singing/audio synthesis pipeline: acquire → separate → clean →
align → featurize → train → sing, with a persistable "voice bank" of novel sampled
voices. Design docs live in `docs/` (start with `docs/ARCHITECTURE.md`); decisions and
their history in `OPEN_QUESTIONS.md`; execution plan in `docs/MIGRATION_PLAN.md`.

**Status:** Migration P0–P6 done (packaging, manifest/acquire, separation, cleaning,
features, alignment, score format); next: P7 (training + own vocoder).

## Setup (Windows-first)

```powershell
python -m pip install uv          # once
python -m uv sync                 # creates .venv, installs core + dev deps
python -m uv run pytest           # run the test suite
python -m uv run ruff check .     # lint
python -m uv run signalml --help  # stage CLI (stages arrive per migration phase)
```

Audio parameters come from `configs/audio.yaml` profiles (`dev` 22.05 kHz for fast
testing, `prod` 44.1 kHz for real training) — select with `SIGNALML_AUDIO_PROFILE` or a
`--profile` flag on stages. Never hardcode sample rates.

## GPU install (the training rig)

The `train` extra installs Demucs + CPU torch wheels (PyPI default on Windows). On a
GPU box, swap in CUDA 12.8 wheels after syncing — Blackwell (sm_120) needs cu128, and
the same wheels cover Ada (sm_89, the RTX 4090 the rig actually has) through
generation-level cubin compatibility:

```powershell
python -m uv sync --extra train
python -m uv pip install --python .venv/Scripts/python.exe --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then `signalml separate` picks the GPU automatically (`device: auto` in
`configs/separate.yaml`). Requires a current NVIDIA driver (570+).

Both flags are load-bearing, and both failure modes are silent:

- **`--python`** — without an activated venv, `uv pip install` resolves to the
  *system* interpreter, so the CUDA wheels land somewhere the project never imports.
- **`--reinstall`** — `--upgrade` alone treats an equal-or-newer CPU wheel as already
  satisfying the requirement and does nothing.

**A plain `python -m uv sync` undoes this.** `train` is an optional extra, so a bare
sync removes torch/torchaudio/torchcrepe/demucs from the venv entirely; re-syncing
with `--extra train` then pulls CPU wheels from PyPI. The dev loop in `CLAUDE.md` uses
plain `uv sync`, so after running it on a GPU box, check
`python -m uv run python -c "import torch; print(torch.cuda.is_available())"` and
redo the swap above if it prints `False`.

## Alignment (MFA) install

MFA is the one tool that does not live in the uv env — it gets its own conda env
(historically the most install-fragile piece of the stack; ARCHITECTURE §2):

```powershell
# 1. install Miniconda (or Miniforge), then:
conda create -n aligner -c conda-forge montreal-forced-aligner
conda run -n aligner mfa model download acoustic english_mfa
conda run -n aligner mfa model download dictionary english_mfa
conda run -n aligner mfa model download g2p english_us_mfa
# 2. verify the pipeline's phone set matches the installed dictionary:
python -m uv run signalml score phoneset --dict "$env:USERPROFILE\Documents\MFA\pretrained_models\dictionary\english_mfa.dict"
```

`signalml align` drives MFA through `conda run -n aligner mfa ...` (configurable via
`mfa_command` in `configs/align.yaml`), keeps the aligner-native TextGrid for audit,
and converts to the pipeline-native `align/phones.json` (MFA IPA phone set, Q2).

Detached runs inherit a pre-Miniforge `PATH`, so bare `conda` may not resolve. Copy
`configs/align.local.example.yaml` to `configs/align.local.yaml` (gitignored) and put
the absolute conda path in `mfa_command`; `scripts/onboard_full.ps1` uses that file
when it exists and `configs/align.yaml` otherwise.

If MFA won't install or aligns sung vowels poorly, the fallbacks are (in order):
**SOFA** (PyTorch singing-oriented aligner — P5.4 runs a head-to-head eval) and an
MFA-only WSL2 Ubuntu env sharing the data directory (ARCHITECTURE §2).

## Typical corpus workflow (so far)

```powershell
# onboard existing audio (tag language/gender/source per corpus folder, Q13):
python -m uv run signalml manifest scan --data-root D:\data --path raw\english --language en --gender F --source-quality separated
# onboard MedleyDB vocal stems (gender/singer come from its per-stem instrument
# taxonomy, not from a flag; --dry-run first — see docs/notes/medleydb.md):
python -m uv run signalml manifest import-medleydb --data-root D:\data --overrides configs\medleydb_overrides.yaml --dry-run
# onboard VocalSet (CC BY 4.0; gender comes from the filename's singer id):
python -m uv run signalml manifest import-vocalset --data-root D:\data --dry-run
# tag pre-existing records with the corpus they came from (recipes scope on it):
python -m uv run signalml manifest set-corpus own --data-root D:\data
# corpus census: singers (voice-bank census), hours, lyrics coverage, stage status:
python -m uv run signalml manifest report --data-root D:\data
# after editing META.txt tags (PROCESSING: dry|produced|heavy, DOMAIN: sung|spoken,
# GENRE: ...), refresh existing records without rescanning:
python -m uv run signalml manifest retag --data-root D:\data
# download new audio:
python -m uv run signalml acquire --urls urls.txt --data-root D:\data
# separate stems (Demucs; resumable, idempotent):
python -m uv run signalml separate --data-root D:\data
# clean + featurize (profile-aware):
python -m uv run signalml clean --data-root D:\data
python -m uv run signalml features --data-root D:\data
# align (needs the MFA conda env, see above):
python -m uv run signalml align --data-root D:\data
# build a score from MIDI + syllabified lyrics ("shin-ing - star"; '-' holds a note):
python -m uv run signalml score from-midi verse.mid --lyrics verse.txt --out score.json
python -m uv run signalml score validate score.json
```

## Moving a corpus to the training rig

Dev happens on the laptop / work PC; prod training happens on the rig (currently a
borrowed RTX 4090 — see `docs/notes/rig_session_2026-09-20.md`).
`signalml ship` transfers a *selected* corpus plus the exact commit that produced it
over the LAN, hash-verified and resumable; `signalml doctor` preflights the receiving
machine. Full design and gotchas: `docs/notes/transfer.md`.

A rig with nothing on it takes the code first — `ship pull` runs *through* signalml,
so it cannot be what delivers it:

```powershell
# on the rig, from nothing
git clone --recurse-submodules https://github.com/wolverine3301/SignalML.git D:\SignalML
powershell -ExecutionPolicy Bypass -File D:\SignalML\scripts\bootstrap_rig.ps1 -DataRoot D:\DATA_ROOT -SetDataRootEnv

# sender (holds DATA_ROOT) — plan a selection, then serve it
signalml ship plan  --data-root Y:\DATA_ROOT --what rebuildable --name rig01
signalml ship serve --data-root Y:\DATA_ROOT --name rig01

# receiver (the rig) — `serve` prints this line with the real address and token
signalml ship pull http://10.0.0.144:8770/<token> --data-root D:\DATA_ROOT --no-code

# before committing the GPU to a multi-day run
signalml ship verify --data-root D:\DATA_ROOT --name rig01
signalml doctor      --data-root D:\DATA_ROOT
```

`--what dataset` ships only `datasets/<name>/` (the trainer's input); `rebuildable`
(default) ships `clean/` + `align/` for the songs the recipe selects, so the rig can
rebuild datasets under new recipes without another transfer; `full` adds the stems.
**Prefer `rebuildable` and rebuild the dataset on the rig**: a generated trainer config
carries absolute paths from the machine that built it, and `signalml train` refuses a
config whose dataset folders are not there.

Shipping refuses a dirty worktree, so checkpoint git hashes stay honest. The
`--with-code` git bundle remains the code path for a rig that cannot reach GitHub
(clone from the bundle, then fetch it on later shipments).

## Training (P7)

```powershell
# 1. build the dataset ON the rig, so its config carries rig-local paths
signalml dataset build --recipe configs/dataset.overfit.yaml   # 3 singers, ~1.8 h
# 2. the vocoder used for validation playback (CC BY-NC, dev preview only - Q4)
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_dev_vocoder.ps1
# 3. preflight, then binarize + train with a run record
signalml train acoustic --dataset overfit_v1 --dry-run         # look before leaping
signalml train acoustic --dataset overfit_v1
```

`train` wraps the vendored DiffSinger scripts in their own venv (`configs/train.yaml`
points at it) and refuses runs that would otherwise fail hours in: missing trainer
venv, unbuilt dataset, a config from another machine, mel parameters matching no audio
profile (D5), or `val_with_vocoder` with no vocoder checkpoint on disk. It binarizes
when binary data is missing, then trains, and writes
`<DATA_ROOT>/runs/<exp>/<timestamp>/run.json` — git hash, dataset-card hash, config +
dictionary snapshot, the exact command line. Checkpoints land where the vendored
trainer puts them: `third_party/DiffSinger/checkpoints/<exp>/`.

Recipes: `configs/dataset.overfit.yaml` is the P7.5 sanity run (three singers, trained
until it resings a training snippet); `configs/dataset.full_v2.yaml` is the first real
multi-singer run (a floor of 5 clipped minutes per speaker). A recipe's `trainer_opts`
sizes batches for the box — defaults fit 8 GB, the rig recipes are set for the 5090.
`train variance` and `train vocoder` name their blockers (D1 note labels; P7.4
SingingVocoders vendoring) instead of pretending.

Watch a run — from the rig, over SSH, or from a phone:

```powershell
signalml train status --exp overfit_v1                 # steps, losses, checkpoints
signalml train status --exp overfit_v1 --audio .\out   # + newest validation renders
```

It reads the trainer's TensorBoard event files (through the trainer's own venv) and
reports the latest step and loss, the step where validation actually **bottomed**, and
whether a checkpoint still exists at that step — `num_ckpt_keep` is a rolling window,
so a run that peaks early silently deletes its best result. Set
`trainer_opts.permanent_ckpt_start` / `_interval` in the recipe to keep a ladder.

## Layout

- `signalml/` — the package: `audio/` primitives, `stages/` pipeline stages, `score/`,
  `voices/`, `train/`, `synth/`, `net/` (LAN shipping), `tasks/masking/` (quarantined
  side tool), `doctor.py`, `cli.py`
- `configs/` — YAML configs (pydantic-validated)
- `docs/` — architecture, contracts, migration plan, code survey
- `archive/` — frozen 2023 experimental scripts (reference only, never import)
- Spleeter was retired in favor of Demucs (removes the last TensorFlow dependency).
