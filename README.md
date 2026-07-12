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

## GPU install (RTX 5090 training rig)

The `train` extra installs Demucs + CPU torch wheels (PyPI default on Windows). On the
5090 rig, swap in CUDA 12.8 wheels after syncing — Blackwell (sm_120) needs cu128:

```powershell
python -m uv sync --extra train
python -m uv pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then `signalml separate` picks the GPU automatically (`device: auto` in
`configs/separate.yaml`). Requires a current NVIDIA driver (570+).

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

If MFA won't install or aligns sung vowels poorly, the fallbacks are (in order):
**SOFA** (PyTorch singing-oriented aligner — P5.4 runs a head-to-head eval) and an
MFA-only WSL2 Ubuntu env sharing the data directory (ARCHITECTURE §2).

## Typical corpus workflow (so far)

```powershell
# onboard existing audio (tag language/gender/source per corpus folder, Q13):
python -m uv run signalml manifest scan --data-root D:\data --path raw\english --language en --gender F --source-quality separated
# corpus census: singers (voice-bank census), hours, lyrics coverage, stage status:
python -m uv run signalml manifest report --data-root D:\data
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

## Layout

- `signalml/` — the package: `audio/` primitives, `stages/` pipeline stages, `score/`,
  `voices/`, `train/`, `synth/`, `tasks/masking/` (quarantined side tool), `cli.py`
- `configs/` — YAML configs (pydantic-validated)
- `docs/` — architecture, contracts, migration plan, code survey
- `archive/` — frozen 2023 experimental scripts (reference only, never import)
- Spleeter was retired in favor of Demucs (removes the last TensorFlow dependency).
