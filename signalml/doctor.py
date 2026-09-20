"""``signalml doctor`` — preflight for a machine that is about to run the pipeline.

Written for the moment a shipment lands on the training rig: the code and corpus are
there, and the question is whether *this* box can actually run them. Every check maps
to a failure we have already paid for at least once — CPU torch wheels silently
surviving a ``uv sync``, an empty submodule directory, a mistyped ``DATA_ROOT``, a disk
that is 20 GB short. Each one prints its own fix.

Exit code is 0 when nothing FAILs (warnings are informational), 1 otherwise, so it can
gate a bootstrap script or a scheduled run.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import ENV_PROFILE
from .manifest import ENV_DATA_ROOT, MANIFEST_NAME, resolve_data_root

REPO_ROOT = Path(__file__).resolve().parents[1]
Status = Literal["ok", "warn", "fail"]

# Blackwell (RTX 5090). Anything older still trains; we only insist that torch can
# actually address whatever GPU is present.
BLACKWELL = (12, 0)


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""


def _run(*cmd: str, timeout: float = 60.0) -> tuple[int, str]:
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=timeout)
        return out.returncode, (out.stdout + out.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


# ------------------------------------------------------------------ environment


def check_python() -> Check:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 11):
        return Check("python", "ok", f"{got} ({sys.executable})")
    return Check("python", "fail", f"{got} — need >= 3.11",
                 "install Python 3.11+ and recreate the venv")


def _base_python() -> Path:
    """The interpreter the venv was built from — `uv` is commonly installed --user
    there rather than inside the venv, so `python -m uv` works from a plain shell even
    when it does not from `uv run`."""
    exe = Path(sys.base_prefix) / ("python.exe" if sys.platform == "win32" else "bin/python3")
    return exe


def check_uv() -> Check:
    candidates: list[tuple[str, list[str]]] = [
        (f"{Path(sys.executable).name} -m uv", [sys.executable, "-m", "uv", "--version"]),
    ]
    on_path = shutil.which("uv")
    if on_path:
        candidates.append(("uv (on PATH)", [on_path, "--version"]))
    base = _base_python()
    if base.exists() and base != Path(sys.executable):
        candidates.append((f"{base} -m uv", [str(base), "-m", "uv", "--version"]))

    for label, cmd in candidates:
        code, out = _run(*cmd)
        if code == 0:
            version = out.splitlines()[0] if out else "present"
            return Check("uv", "ok", f"{version} via {label}")
    return Check("uv", "fail", "uv not runnable from this interpreter or PATH",
                 f"{Path(sys.executable).name} -m pip install --user uv")


def check_git() -> Check:
    code, out = _run("git", "--version")
    if code == 0:
        return Check("git", "ok", out.splitlines()[0])
    return Check("git", "fail", "git not on PATH",
                 "install Git for Windows (ship pull needs it to apply the bundle)")


def check_ffmpeg() -> Check:
    if shutil.which("ffmpeg"):
        return Check("ffmpeg", "ok", shutil.which("ffmpeg") or "")
    return Check("ffmpeg", "warn", "not on PATH — yt-dlp/librosa decoding will be limited",
                 "winget install Gyan.FFmpeg (only needed for acquire/decode)")


# ------------------------------------------------------------------------ torch


def check_torch() -> list[Check]:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return [Check("torch", "fail", "not installed",
                      "python -m uv sync --extra train")]
    version = getattr(torch, "__version__", "?")
    checks = [Check("torch", "ok", version)]

    cuda_build = getattr(torch.version, "cuda", None)
    if cuda_build is None:
        checks.append(Check(
            "torch cuda build", "fail", f"{version} is a CPU-only wheel",
            "python -m uv pip install --python .venv/Scripts/python.exe --reinstall "
            "torch torchaudio --index-url https://download.pytorch.org/whl/cu128"))
        return checks
    checks.append(Check("torch cuda build", "ok", f"cu{cuda_build.replace('.', '')}"))

    if not torch.cuda.is_available():
        checks.append(Check(
            "cuda available", "fail", "torch.cuda.is_available() is False",
            "check the NVIDIA driver (570+ for Blackwell) and that no other process "
            "holds the GPU; nvidia-smi should list the card"))
        return checks

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    checks.append(Check("cuda available", "ok",
                        f"{name} (sm_{cap[0]}{cap[1]}), {torch.cuda.device_count()} device(s)"))

    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if arch_list and f"sm_{cap[0]}{cap[1]}" not in arch_list:
        checks.append(Check(
            "gpu arch supported", "fail",
            f"this torch was built for {', '.join(arch_list)} — it cannot emit code "
            f"for sm_{cap[0]}{cap[1]}",
            "swap in the cu128 wheels (README 'GPU install'); a plain `uv sync` "
            "reverts them, so re-run the swap after every sync"))
    else:
        checks.append(Check("gpu arch supported", "ok",
                            f"sm_{cap[0]}{cap[1]} in {len(arch_list) or '?'} built archs"))

    if cap < BLACKWELL:
        checks.append(Check(
            "gpu class", "warn",
            f"sm_{cap[0]}{cap[1]} is pre-Blackwell — fine for dev-profile runs, "
            f"slower than the 5090 for prod training"))
    return checks


def check_imports() -> list[Check]:
    out: list[Check] = []
    for mod, status, why in [
        ("soundfile", "fail", "python -m uv sync"),
        ("librosa", "fail", "python -m uv sync"),
        ("torchaudio", "fail", "python -m uv sync --extra train"),
        ("demucs", "warn", "python -m uv sync --extra train (only needed for separate)"),
        ("torchcrepe", "warn", "python -m uv sync --extra train (pyin is the default F0)"),
    ]:
        try:
            m = importlib.import_module(mod)
            out.append(Check(mod, "ok", getattr(m, "__version__", "present")))
        except ImportError:
            out.append(Check(mod, status, "not importable", why))
    return out


# ------------------------------------------------------------------------- repo


def _pinned_submodule_sha(repo: Path, path: str) -> str | None:
    code, out = _run("git", "-C", str(repo), "ls-tree", "HEAD", path)
    if code != 0 or not out:
        return None
    parts = out.split()
    return parts[2] if len(parts) >= 3 else None


def check_submodule(repo_root: Path = REPO_ROOT) -> list[Check]:
    rel = "third_party/DiffSinger"
    sub = repo_root / rel
    if not sub.is_dir() or not any(sub.iterdir()):
        return [Check("DiffSinger submodule", "fail", f"{sub} is missing or empty",
                      "git submodule update --init --recursive")]
    code, head = _run("git", "-C", str(sub), "rev-parse", "HEAD")
    pinned = _pinned_submodule_sha(repo_root, rel)
    checks = []
    if code == 0 and pinned and head != pinned:
        checks.append(Check("DiffSinger submodule", "warn",
                            f"at {head[:8]}, pinned at {pinned[:8]}",
                            "git submodule update --init --recursive"))
    else:
        checks.append(Check("DiffSinger submodule", "ok",
                            f"{head[:8] if code == 0 else 'present'}"))
    venv = sub / ".venv"
    if venv.is_dir():
        checks.append(Check("DiffSinger venv", "ok", str(venv)))
        checks.append(check_trainer_torch(venv))
    else:
        checks.append(Check(
            "DiffSinger venv", "warn", f"{venv} missing (the trainer needs py3.10)",
            "scripts/bootstrap_rig.ps1 creates it; see docs/notes/vendor_diffsinger.md"))
    return checks


_TRAINER_PROBE = (
    "import json,torch;"
    "a=torch.cuda.is_available();"
    "print(json.dumps({'v':torch.__version__,'cuda':torch.version.cuda,'avail':a,"
    "'name':torch.cuda.get_device_name(0) if a else None,"
    "'cap':list(torch.cuda.get_device_capability(0)) if a else None,"
    "'archs':torch.cuda.get_arch_list()}))"
)


def check_trainer_torch(venv: Path) -> Check:
    """Probe torch *inside the trainer venv* — the one that runs the GPU for days.

    ``requirements.txt`` deliberately leaves torch unpinned ("install PyTorch
    manually"), so this venv is exactly where a CPU wheel hides: it imports fine and
    dies at the first kernel launch, which on a multi-day run is an expensive way to
    find out. Separate from ``check_torch``, which sees only the pipeline venv.
    """
    name = "DiffSinger torch"
    python = venv / "Scripts" / "python.exe"
    if not python.exists():
        python = venv / "bin" / "python"
    if not python.exists():
        return Check(name, "warn", f"no interpreter in {venv}",
                     "scripts/bootstrap_rig.ps1 recreates the trainer venv")
    fix = ("python -m uv pip install --python "
           f"{python} --reinstall torch torchaudio "
           "--index-url https://download.pytorch.org/whl/cu128")
    code, out = _run(str(python), "-c", _TRAINER_PROBE, timeout=180.0)
    if code != 0:
        return Check(name, "fail", f"probe failed: {out.splitlines()[-1] if out else code}",
                     fix)
    try:
        import json as _json
        info = _json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return Check(name, "warn", f"unreadable probe output: {out[:80]}", fix)
    if not info["cuda"] or not info["avail"]:
        return Check(name, "fail",
                     f"{info['v']} in the trainer venv has no usable CUDA "
                     f"(cuda={info['cuda']}, available={info['avail']})", fix)
    cap = f"sm_{info['cap'][0]}{info['cap'][1]}"
    if info["archs"] and cap not in info["archs"]:
        return Check(name, "fail",
                     f"{info['v']} was built for {', '.join(info['archs'])} — no {cap} "
                     f"kernels for {info['name']}", fix)
    return Check(name, "ok", f"{info['v']} / {info['name']} ({cap})")


def check_audio_profiles() -> Check:
    try:
        from .config import active_profile
        p = active_profile()
        return Check("audio profile", "ok",
                     f"{p.name} @ {p.sample_rate} Hz (${ENV_PROFILE} overrides)")
    except Exception as exc:  # config is the one thing every stage touches
        return Check("audio profile", "fail", str(exc),
                     "check configs/audio.yaml")


# ------------------------------------------------------------------------- data


def check_data_root(data_root: str | Path | None, need_bytes: int = 0) -> list[Check]:
    root = resolve_data_root(data_root)
    if not root.is_dir():
        return [Check("DATA_ROOT", "fail", f"{root} does not exist",
                      f"set {ENV_DATA_ROOT} or pass --data-root")]
    checks = [Check("DATA_ROOT", "ok", str(root))]

    manifest = root / MANIFEST_NAME
    if manifest.exists():
        n = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip())
        checks.append(Check("manifest", "ok", f"{n} record(s) in {manifest.name}"))
    else:
        checks.append(Check("manifest", "warn", f"no {MANIFEST_NAME} yet",
                            "it arrives with the first `signalml ship pull`"))

    free = shutil.disk_usage(root).free
    detail = f"{free / (1 << 30):.1f} GB free on {Path(root).anchor or root}"
    if need_bytes and free < need_bytes:
        checks.append(Check("disk space", "fail",
                            f"{detail} — shipment needs {need_bytes / (1 << 30):.1f} GB",
                            "free space or point --data-root at a bigger volume"))
    elif need_bytes:
        checks.append(Check("disk space", "ok",
                            f"{detail} (shipment needs {need_bytes / (1 << 30):.1f} GB)"))
    else:
        checks.append(Check("disk space", "ok", detail))
    return checks


def check_mfa() -> Check:
    """Alignment only — a training rig never needs it, so this is warn-at-worst."""
    code, out = _run("conda", "run", "-n", "aligner", "mfa", "version", timeout=120.0)
    if code == 0 and out:
        return Check("MFA (aligner env)", "ok", out.splitlines()[-1])
    return Check("MFA (aligner env)", "warn", "conda env 'aligner' not usable",
                 "only needed to run `signalml align` here — see README 'Alignment (MFA)'")


# ------------------------------------------------------------------------- run


def run_checks(
    *,
    data_root: str | Path | None = None,
    need_bytes: int = 0,
    repo_root: str | Path = REPO_ROOT,
    with_mfa: bool = False,
) -> list[Check]:
    repo_root = Path(repo_root)
    checks: list[Check] = [check_python(), check_uv(), check_git(), check_ffmpeg()]
    checks += check_torch()
    checks += check_imports()
    checks.append(check_audio_profiles())
    checks += check_submodule(repo_root)
    checks += check_data_root(data_root, need_bytes)
    if with_mfa:
        checks.append(check_mfa())
    return checks


def format_report(checks: list[Check]) -> str:
    marks: dict[Status, str] = {"ok": "  OK ", "warn": "WARN ", "fail": "FAIL "}
    width = max((len(c.name) for c in checks), default=0)
    lines = []
    for c in checks:
        lines.append(f"{marks[c.status]}{c.name.ljust(width)}  {c.detail}")
        if c.fix and c.status != "ok":
            lines.append(f"      {' ' * width}  -> {c.fix}")
    fails = sum(1 for c in checks if c.status == "fail")
    warns = sum(1 for c in checks if c.status == "warn")
    lines.append("")
    lines.append(f"{len(checks)} check(s): {len(checks) - fails - warns} ok, "
                 f"{warns} warning(s), {fails} failure(s)")
    if fails:
        lines.append("NOT ready — fix the FAIL lines above, then re-run `signalml doctor`.")
    else:
        lines.append("Ready. Next: signalml dataset build  ->  training (see README).")
    return "\n".join(lines)


def has_failures(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)
