# SignalML

Parametric, controllable singing/audio synthesis pipeline: acquire → separate → clean →
align → featurize → train → sing, with a persistable "voice bank" of novel sampled
voices. Design docs live in `docs/` (start with `docs/ARCHITECTURE.md`); decisions and
their history in `OPEN_QUESTIONS.md`; execution plan in `docs/MIGRATION_PLAN.md`.

**Status:** Migration P0 (packaging/skeleton) done; pipeline stages land phase by phase.

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

## Typical corpus workflow (so far)

```powershell
# onboard existing audio (tag language/gender per corpus folder, Q13):
python -m uv run signalml manifest scan --data-root D:\data --path raw\english --language en --gender F
# download new audio:
python -m uv run signalml acquire --urls urls.txt --data-root D:\data
# separate stems (Demucs; resumable, idempotent):
python -m uv run signalml separate --data-root D:\data
```

## Layout

- `signalml/` — the package: `audio/` primitives, `stages/` pipeline stages, `score/`,
  `voices/`, `train/`, `synth/`, `tasks/masking/` (quarantined side tool), `cli.py`
- `configs/` — YAML configs (pydantic-validated)
- `docs/` — architecture, contracts, migration plan, code survey
- `archive/` — frozen 2023 experimental scripts (reference only, never import)
- Spleeter was retired in favor of Demucs (removes the last TensorFlow dependency).
