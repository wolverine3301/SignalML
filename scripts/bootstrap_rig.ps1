# Bring a fresh training rig up to "can run the pipeline" after `signalml ship pull`
# has landed the repo + corpus. Run it from inside the pulled repo:
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_rig.ps1 -DataRoot D:\DATA_ROOT
#
# Idempotent and safe to re-run — in particular after any `uv sync`, which reverts the
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
            & $PythonExe -m uv pip install --upgrade torch torchaudio `
                --index-url https://download.pytorch.org/whl/cu128
        } finally { Pop-Location }
    }
}

Step "submodules (third_party/DiffSinger)" {
    & git -C $repo submodule update --init --recursive
}

if (-not $SkipDiffSinger) {
    Step "DiffSinger py3.10 venv" {
        $sub = Join-Path $repo "third_party\DiffSinger"
        $venv = Join-Path $sub ".venv"
        if (Test-Path (Join-Path $venv "Scripts\python.exe")) {
            Write-Host "    already present: $venv"
        } else {
            # the vendored trainer pins py3.10; uv fetches it if the rig lacks one
            & $PythonExe -m uv venv --python 3.10 $venv
            & $PythonExe -m uv pip install --python (Join-Path $venv "Scripts\python.exe") `
                -r (Join-Path $sub "requirements.txt")
            Write-Host "    NOTE: torch inside this venv also needs the cu128 index —"
            Write-Host "          see docs/notes/vendor_diffsinger.md"
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
