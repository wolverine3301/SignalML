# Bring a fresh training rig up to "can run the pipeline and the trainer".
#
# On a bare rig (nothing installed yet) the order is:
#
#   1. install Python 3.11+, git and an NVIDIA driver (570+ for Blackwell/sm_120)
#   2. git clone --recurse-submodules https://github.com/wolverine3301/SignalML.git
#   3. powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_rig.ps1 -DataRoot D:\DATA_ROOT
#   4. signalml ship pull <url> --data-root D:\DATA_ROOT --no-code   (corpus comes last)
#
# `ship pull` needs a working signalml, so the repo cannot arrive by shipment on a
# machine that has nothing  -  clone first, ship the *data* afterwards. (A git bundle
# from `ship plan` still works as the update path for a rig with no GitHub access.)
#
# Idempotent and safe to re-run  -  in particular after any `uv sync`, which reverts the
# CUDA wheels back to the CPU ones PyPI resolves on Windows (the recurring footgun this
# script exists to close). Ends by running `signalml doctor`, whose exit code is this
# script's exit code, so a green run really does mean ready.

param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [string]$PythonExe = "python",
    [switch]$SkipTorchSwap,      # already on cu128, or deliberately CPU-only
    [switch]$SkipDiffSinger,     # skip the vendored trainer's own py3.10 venv
    [switch]$SetDataRootEnv,     # persist SIGNALML_DATA_ROOT for this user
    [switch]$CheckMfa            # also probe the conda 'aligner' env (alignment only)
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
$trainerDir = Join-Path $repo "third_party\DiffSinger"
$trainerPy = Join-Path $trainerDir ".venv\Scripts\python.exe"
$cudaIndex = "https://download.pytorch.org/whl/cu128"
$failed = @()

function Step([string]$Name, [scriptblock]$Block) {
    Write-Host ""
    Write-Host "=== $Name" -ForegroundColor Cyan
    try {
        & $Block
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            throw "exit code $LASTEXITCODE"
        }
    } catch {
        Write-Host "    FAILED: $_" -ForegroundColor Red
        $script:failed += $Name
    }
}

# torch.cuda.is_available() is the only honest test that the CUDA wheels landed in
# the venv we think they did: a CPU wheel imports and runs fine right up to the first
# kernel launch, hours into a run. sm_120 (5090) additionally needs the arch compiled
# into the wheel, which a cu126 build does not have.
function Assert-TorchCuda([string]$Interpreter, [string]$Label) {
    $probe = @"
import torch
caps = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('arch_list', ' '.join(caps))
if not torch.cuda.is_available():
    raise SystemExit('torch.cuda.is_available() is False')
print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
"@
    $out = & $Interpreter -c $probe
    $out | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) { throw "$Label torch has no CUDA  -  see README 'GPU install'" }
    $cap = & $Interpreter -c "import torch;print('%d%d' % torch.cuda.get_device_capability(0))"
    $archs = & $Interpreter -c "import torch;print(' '.join(torch.cuda.get_arch_list()))"
    if ($archs -notmatch "sm_$cap") {
        Write-Host ("    WARNING: device is sm_$cap but this wheel only has: $archs") -ForegroundColor Yellow
        Write-Host ("             kernels will fail at launch; install from $cudaIndex") -ForegroundColor Yellow
    }
}

Write-Host "repo      $repo"
Write-Host "DATA_ROOT $DataRoot"

Step "python >= 3.11" {
    $v = & $PythonExe -c "import sys; print('%d.%d' % sys.version_info[:2])"
    Write-Host "    python $v"
    if ([version]$v -lt [version]"3.11") { throw "need Python 3.11+, found $v" }
}

Step "uv" {
    & $PythonExe -m uv --version
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    installing uv --user"
        & $PythonExe -m pip install --user uv
    }
}

Step "uv sync --extra train" {
    Push-Location $repo
    try { & $PythonExe -m uv sync --extra train } finally { Pop-Location }
}

if (-not $SkipTorchSwap) {
    Step "CUDA 12.8 torch wheels (sm_120)" {
        Push-Location $repo
        try {
            # --python: without it uv resolves to the *system* interpreter and the
            #           CUDA wheels land where the project never imports them.
            # --reinstall: --upgrade alone treats the equal-or-newer CPU wheel as
            #           already satisfying the requirement and does nothing.
            & $PythonExe -m uv pip install --python $venvPy --reinstall `
                torch torchaudio --index-url $cudaIndex
            if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
            Assert-TorchCuda $venvPy "pipeline venv"
        } finally { Pop-Location }
    }
}

Step "submodules (third_party/DiffSinger)" {
    & git -C $repo submodule update --init --recursive
}

if (-not $SkipDiffSinger) {
    Step "DiffSinger py3.10 venv" {
        if (Test-Path $trainerPy) {
            Write-Host "    already present: $(Split-Path $trainerPy -Parent)"
        } else {
            # the vendored trainer pins py3.10 (pyworld 0.3.4 has no cp312 wheel);
            # uv fetches an interpreter if the rig lacks one
            & $PythonExe -m uv venv --python 3.10 (Join-Path $trainerDir ".venv")
            if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
            & $PythonExe -m uv pip install --python $trainerPy `
                -r (Join-Path $trainerDir "requirements.txt")
            if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
        }
    }

    if (-not $SkipTorchSwap) {
        # requirements.txt deliberately does not pin torch ("install PyTorch
        # manually"), so whatever a dependency drags in is a CPU wheel. The trainer
        # venv is the one that actually runs the GPU for days  -  it needs the same
        # cu128 treatment as the pipeline venv, every time, not a printed reminder.
        Step "DiffSinger venv: CUDA 12.8 torch wheels" {
            & $PythonExe -m uv pip install --python $trainerPy --reinstall `
                torch torchaudio --index-url $cudaIndex
            if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
            Assert-TorchCuda $trainerPy "trainer venv"
        }
    }
}

if ($SetDataRootEnv) {
    Step "persist SIGNALML_DATA_ROOT" {
        [Environment]::SetEnvironmentVariable("SIGNALML_DATA_ROOT", $DataRoot, "User")
        Write-Host "    set for this user (new shells only)"
    }
} else {
    Write-Host ""
    Write-Host "To avoid passing --data-root every time, either re-run with" -ForegroundColor Yellow
    Write-Host "  -SetDataRootEnv   or:  setx SIGNALML_DATA_ROOT `"$DataRoot`"" -ForegroundColor Yellow
}
$env:SIGNALML_DATA_ROOT = $DataRoot

Write-Host ""
Write-Host "=== signalml doctor" -ForegroundColor Cyan
Push-Location $repo
try {
    $doctorArgs = @("-m", "uv", "run", "signalml", "doctor", "--data-root", $DataRoot)
    if ($CheckMfa) { $doctorArgs += "--check-mfa" }
    & $PythonExe @doctorArgs
    $doctorExit = $LASTEXITCODE
} finally { Pop-Location }

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "bootstrap steps that failed: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
exit $doctorExit
