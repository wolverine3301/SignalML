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

GPU deps (PyTorch/Demucs) join in Migration P2; RTX 5090 needs the CUDA 12.8 wheel
index (`https://download.pytorch.org/whl/cu128`).

## Layout

- `signalml/` — the package: `audio/` primitives, `stages/` pipeline stages, `score/`,
  `voices/`, `train/`, `synth/`, `tasks/masking/` (quarantined side tool), `cli.py`
- `configs/` — YAML configs (pydantic-validated)
- `docs/` — architecture, contracts, migration plan, code survey
- `archive/` — frozen 2023 experimental scripts (reference only, never import)
- Spleeter was retired in favor of Demucs (removes the last TensorFlow dependency).
