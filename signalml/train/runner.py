"""S7 training runs — wrap the vendored DiffSinger scripts, with provenance.

Contract: docs/PIPELINE_AND_CONTRACTS.md §S7; integration: docs/notes/vendor_diffsinger.md
(D2 "adopt-their-world", zero patches). Nothing here trains anything itself. It
resolves a built dataset to the config `signalml dataset build` generated for it,
*refuses* runs whose preconditions would only blow up hours in, snapshots what
produced the run, and hands off to their `binarize.py` / `train.py` in their own venv.

Their conventions, adopted as-is:

``trainer="variance"`` looks for ``config_variance.yaml``, which
``signalml dataset variance-config`` writes beside the acoustic one (D1, 2026-09-22);
preflight reports its absence like any other missing config.

- the trainer runs with ``cwd = third_party/DiffSinger`` (``base_config`` and
  ``vocoder_ckpt`` paths in their configs are relative to it), and
- checkpoints land in ``checkpoints/<exp_name>`` *inside the submodule*.

What is ours is the run record under ``<data_root>/runs/<exp_name>/<timestamp>/``:
the config + dictionary snapshot, the git hash, the dataset card hash, the exact
command line. A checkpoint whose inputs cannot be named afterwards is not a result,
and by the time a run finishes the recipe may well have moved on.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import yaml
from pydantic import BaseModel

from ..config import CONFIGS_DIR, load_audio_profiles
from ..manifest import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_YAML = CONFIGS_DIR / "train.yaml"
RUN_RECORD_NAME = "run.json"

Trainer = Literal["acoustic", "variance", "vocoder"]

# Marker the vendored binarizer leaves behind; its absence means "not binarized yet",
# which is a 20-minute step, not a failure.
BINARY_MARKER = "train.data"


class TrainConfig(BaseModel):
    """Machine-local trainer wiring (``configs/train.yaml``).

    Every path is injectable for the same reason ``mfa_command`` is: the trainer
    lives outside our env, and rigs disagree about where interpreters are.
    """

    trainer_dir: str = "third_party/DiffSinger"
    trainer_python: str | None = None  # None = autodetect the submodule's own venv
    runs_dirname: str = "runs"
    # The trainer reads/writes IPA phoneme names; Windows' cp1252 default mangles
    # them on the way to the dictionary and the phone set, which surfaces much later
    # as "unknown phoneme". UTF-8 is not optional here.
    env: dict[str, str] = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    min_free_gb: float = 50.0  # checkpoints accumulate on the repo's drive

    def resolved_trainer_dir(self) -> Path:
        p = Path(self.trainer_dir)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def resolved_trainer_python(self) -> Path:
        if self.trainer_python:
            p = Path(self.trainer_python)
            return p if p.is_absolute() else (REPO_ROOT / p)
        venv = self.resolved_trainer_dir() / ".venv"
        win, posix = venv / "Scripts" / "python.exe", venv / "bin" / "python"
        return win if win.exists() or not posix.exists() else posix


def load_train_config(path: str | Path | None = None) -> TrainConfig:
    path = Path(path) if path else DEFAULT_TRAIN_YAML
    if not Path(path).exists():
        return TrainConfig()
    return TrainConfig.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    )


@dataclass
class TrainPlan:
    trainer: Trainer
    exp_name: str
    dataset: str
    dataset_dir: Path
    config_path: Path
    binary_dir: Path
    ckpt_dir: Path
    run_dir: Path
    cwd: Path
    python: Path
    train_cmd: list[str]
    binarize_cmd: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"exp_name   {self.exp_name}",
                 f"dataset    {self.dataset_dir}",
                 f"config     {self.config_path}",
                 f"binary     {self.binary_dir}",
                 f"checkpoints {self.ckpt_dir}",
                 f"run record {self.run_dir}",
                 f"cwd        {self.cwd}"]
        if self.binarize_cmd:
            lines.append("binarize   " + " ".join(self.binarize_cmd))
        else:
            lines.append("binarize   (skipped — already binarized)")
        lines.append("train      " + " ".join(self.train_cmd))
        return "\n".join(lines)


def load_generated_config(config_path: Path) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}


def binary_is_built(binary_dir: Path) -> bool:
    return (binary_dir / BINARY_MARKER).exists()


def plan_run(
    data_root: str | Path,
    dataset: str,
    *,
    trainer: Trainer = "acoustic",
    exp_name: str | None = None,
    cfg: TrainConfig | None = None,
    binarize: bool | None = None,  # None = only when the binary dir is missing
    reset: bool = False,
    hparams: str | None = None,
    now: _dt.datetime | None = None,
) -> TrainPlan:
    """Resolve a built dataset into the two commands that train it."""
    if trainer == "vocoder":
        raise NotImplementedError(
            "own-vocoder training lives in openvpi/SingingVocoders, not yet vendored "
            "(MIGRATION_PLAN P7.4). The community NSF-HiFiGAN checkpoint is CC BY-NC, "
            "dev preview only (Q4)."
        )
    cfg = cfg or load_train_config()
    data_root = Path(data_root)
    now = now or _dt.datetime.now()

    dataset_dir = data_root / "datasets" / dataset
    config_path = dataset_dir / f"config_{trainer}.yaml"
    generated = load_generated_config(config_path) if config_path.exists() else {}
    binary_dir = Path(generated.get("binary_data_dir") or (dataset_dir / "binary"))

    # the trainer's work dir is checkpoints/<exp_name>, so acoustic and variance runs
    # over one dataset must not share a name: different binarizers, different tensors,
    # and their config.yaml lands in that directory too
    exp_name = exp_name or (dataset if trainer == "acoustic" else f"{dataset}_{trainer}")
    trainer_dir = cfg.resolved_trainer_dir()
    python = cfg.resolved_trainer_python()
    stamp = now.strftime("%Y%m%d_%H%M%S")

    want_binarize = (not binary_is_built(binary_dir)) if binarize is None else binarize
    binarize_cmd = None
    if want_binarize:
        binarize_cmd = [str(python), "scripts/binarize.py", "--config", str(config_path)]
    train_cmd = [str(python), "scripts/train.py", "--config", str(config_path),
                 "--exp_name", exp_name]
    if reset:
        train_cmd.append("--reset")
    if hparams:
        train_cmd += ["--hparams", hparams]

    notes: list[str] = []
    if binarize is False and not binary_is_built(binary_dir):
        notes.append(f"binarization skipped by request but {binary_dir} has no "
                     f"{BINARY_MARKER} — the trainer will fail immediately")
    if not want_binarize:
        notes.append(f"reusing existing binary data in {binary_dir}")

    return TrainPlan(
        trainer=trainer,
        exp_name=exp_name,
        dataset=dataset,
        dataset_dir=dataset_dir,
        config_path=config_path,
        binary_dir=binary_dir,
        ckpt_dir=trainer_dir / "checkpoints" / exp_name,
        run_dir=data_root / cfg.runs_dirname / exp_name / stamp,
        cwd=trainer_dir,
        python=python,
        train_cmd=train_cmd,
        binarize_cmd=binarize_cmd,
        env=dict(cfg.env),
        notes=notes,
    )


def preflight(plan: TrainPlan, cfg: TrainConfig | None = None) -> list[str]:
    """Everything that would otherwise fail hours in, checked in milliseconds.

    Returns hard problems (empty = go). Soft observations live in ``plan.notes``.
    """
    cfg = cfg or load_train_config()
    problems: list[str] = []

    if not (plan.cwd / "scripts" / "train.py").exists():
        problems.append(
            f"{plan.cwd} has no scripts/train.py — the submodule is not checked out: "
            f"git submodule update --init --recursive")
    if not plan.python.exists():
        problems.append(
            f"trainer interpreter {plan.python} is missing — create the vendored "
            f"trainer's venv: scripts/bootstrap_rig.ps1 (docs/notes/vendor_diffsinger.md)")
    if not plan.config_path.exists():
        problems.append(
            f"{plan.config_path} is missing — build the dataset first: "
            f"signalml dataset build --recipe configs/dataset.<recipe>.yaml")
        return problems  # nothing below can be checked without it

    generated = load_generated_config(plan.config_path)

    for lang, dict_path in (generated.get("dictionaries") or {}).items():
        if not Path(dict_path).exists():
            problems.append(f"dictionary for {lang!r} is missing: {dict_path}")

    missing_data = [d.get("raw_data_dir") for d in (generated.get("datasets") or [])
                    if not Path(d.get("raw_data_dir", "")).exists()]
    if missing_data:
        problems.append(
            f"{len(missing_data)} dataset folder(s) in the config do not exist here "
            f"(first: {missing_data[0]}) — the config carries absolute paths from the "
            f"machine that built it. Rebuild the dataset on this machine.")

    if generated.get("val_with_vocoder"):
        ckpt = Path(generated.get("vocoder_ckpt", ""))
        if not ckpt.is_absolute():
            ckpt = plan.cwd / ckpt
        if not ckpt.exists():
            problems.append(
                f"val_with_vocoder is on but {ckpt} is missing — validation would die "
                f"at the first val step. Fetch it with "
                f"scripts/fetch_dev_vocoder.ps1 (community weights are CC BY-NC: dev "
                f"preview only, nothing rendered with them ships) or set "
                f"trainer_opts.val_with_vocoder: false in the recipe.")

    problems += _profile_problems(generated)

    free_gb = _free_gb(plan.cwd)
    if free_gb is not None and free_gb < cfg.min_free_gb:
        plan.notes.append(
            f"only {free_gb:.0f} GB free on the drive holding {plan.ckpt_dir} "
            f"(checkpoints land inside the submodule, min_free_gb={cfg.min_free_gb:.0f})")
    return problems


def _profile_problems(generated: dict) -> list[str]:
    """Refuse a config whose audio contract matches no profile in audio.yaml (D5).

    The dataset was binarized under these numbers; training under different ones
    corrupts silently. Cheap to check, invisible when it goes wrong.
    """
    keys = ("audio_sample_rate", "audio_num_mel_bins", "hop_size", "fft_size",
            "win_size")
    if not all(k in generated for k in keys):
        return []
    for prof in load_audio_profiles().profiles.values():
        if (generated["audio_sample_rate"] == prof.sample_rate
                and generated["audio_num_mel_bins"] == prof.n_mels
                and generated["hop_size"] == prof.hop_length
                and generated["fft_size"] == prof.n_fft
                and generated["win_size"] == prof.win_length):
            return []
    got = {k: generated[k] for k in keys}
    return [f"the dataset's audio contract {got} matches no profile in "
            f"configs/audio.yaml — a dataset built under different mel parameters "
            f"than the trainer uses is the D5 silent-failure case. Rebuild it."]


def _free_gb(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return None


def _git_provenance() -> dict:
    try:
        from ..net.plan import git_provenance

        return git_provenance(REPO_ROOT).model_dump()
    except Exception as exc:  # git missing, or not a checkout (shipped tarball)
        return {"error": f"{type(exc).__name__}: {exc}"}


def _torch_probe(python: Path) -> dict:
    """Ask the *trainer's* interpreter what it will actually run on."""
    code = ("import json,torch;print(json.dumps({'torch':torch.__version__,"
            "'cuda':torch.version.cuda,'available':torch.cuda.is_available(),"
            "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() "
            "else None,'arch_list':torch.cuda.get_arch_list()}))")
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=120)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def write_run_record(plan: TrainPlan, *, probe: bool = True) -> Path:
    """Snapshot everything needed to say later what this checkpoint came from."""
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    generated = load_generated_config(plan.config_path) \
        if plan.config_path.exists() else {}

    snapshots: dict[str, str] = {}
    for src in [plan.config_path, plan.dataset_dir / "dataset_card.md",
                *[Path(p) for p in (generated.get("dictionaries") or {}).values()]]:
        if Path(src).exists():
            dst = plan.run_dir / Path(src).name
            shutil.copy2(src, dst)
            snapshots[Path(src).name] = sha256_file(dst)

    record = {
        "version": 1,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "trainer": plan.trainer,
        "exp_name": plan.exp_name,
        "dataset": plan.dataset,
        "dataset_dir": str(plan.dataset_dir),
        "binary_dir": str(plan.binary_dir),
        "checkpoints_dir": str(plan.ckpt_dir),
        "audio": {k: generated.get(k) for k in
                  ("audio_sample_rate", "audio_num_mel_bins", "hop_size", "fft_size",
                   "win_size", "fmax", "mel_base")},
        "speakers": generated.get("num_spk"),
        "vocoder": {"name": generated.get("vocoder"),
                    "ckpt": generated.get("vocoder_ckpt"),
                    "val_with_vocoder": generated.get("val_with_vocoder"),
                    "license": "community pc-nsf-hifigan is CC BY-NC — dev preview "
                               "only (Q4); nothing rendered with it ships"},
        "snapshots_sha256": snapshots,
        "git": _git_provenance(),
        "commands": {"binarize": plan.binarize_cmd, "train": plan.train_cmd},
        "cwd": str(plan.cwd),
        "env": plan.env,
        "notes": plan.notes,
        "torch": _torch_probe(plan.python) if probe and plan.python.exists() else None,
    }
    path = plan.run_dir / RUN_RECORD_NAME
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


Runner = Callable[..., subprocess.CompletedProcess]


def execute(plan: TrainPlan, *, runner: Runner | None = None) -> int:
    """Run binarize (when planned) then train, in the trainer's own world."""
    runner = runner or subprocess.run
    env = {**os.environ, **plan.env}
    for cmd in ([plan.binarize_cmd] if plan.binarize_cmd else []) + [plan.train_cmd]:
        print(f"$ {' '.join(cmd)}   (cwd={plan.cwd})", flush=True)
        out = runner(cmd, cwd=str(plan.cwd), env=env)
        code = getattr(out, "returncode", 0)
        if code != 0:
            return code
    return 0
