# Unattended corpus onboarding: RAW/Full (pre-separated legacy stems) -> fully
# processed + first acoustic dataset + binarized trainer input.
#
# Designed to run DETACHED (survives the Claude session / a logout):
#   Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
#     '-File','Y:\SignalAI\SignalML-main\scripts\onboard_full.ps1' -WindowStyle Hidden
#
# Every step logs to DATA_ROOT\logs\onboard_<timestamp>.log and a failed step does
# not stop later independent steps (each stage is idempotent/resumable, so simply
# re-running this script continues where things left off).
#
# Deliberately NOT included: launching training. The log ends with the exact
# command — a multi-day GPU run should be started by a human.

param(
    [string]$DataRoot = "Y:\SignalAI\DATA_ROOT",
    [string]$ImportPath = "RAW/Full"
)

$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$cli = Join-Path $repo ".venv\Scripts\signalml.exe"
$logDir = Join-Path $DataRoot "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("onboard_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Step([string]$Name, [scriptblock]$Block) {
    "`n=== $Name === $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
    try {
        & $Block 2>&1 | Out-File $log -Append -Encoding utf8
        "=== $Name done (exit $LASTEXITCODE)" | Out-File $log -Append -Encoding utf8
    } catch {
        "=== $Name FAILED: $_" | Out-File $log -Append -Encoding utf8
    }
}

Step "import-stems $ImportPath" {
    & $cli manifest import-stems --data-root $DataRoot --path $ImportPath --language en --gender F
}

Step "retag (META.txt tags incl. QUALITY letter grades)" {
    & $cli manifest retag --data-root $DataRoot
}

Step "clean at PROD profile (force: everything moves to 44.1k)" {
    & $cli clean --data-root $DataRoot --profile prod --force
}

Step "align (MFA, new songs only)" {
    & $cli align --data-root $DataRoot --config (Join-Path $repo "configs\align.workpc.yaml")
}

Step "census report" {
    & $cli manifest report --data-root $DataRoot
}

Step "dataset build (full_acoustic_v1)" {
    & $cli dataset build --data-root $DataRoot --force
}

Step "community vocoder download (CC BY-NC - DEV PREVIEW ONLY, Q4)" {
    $ckptDir = Join-Path $repo "third_party\DiffSinger\checkpoints"
    $target = Join-Path $ckptDir "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    if (Test-Path (Join-Path $target "model.ckpt")) { "already present"; return }
    New-Item -ItemType Directory -Force $ckptDir | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $release = Invoke-RestMethod ("https://api.github.com/repos/openvpi/vocoders/" +
        "releases/tags/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02")
    $asset = $release.assets | Where-Object { $_.name -match 'zip$' -and
        $_.name -notmatch 'onnx|openutau|oudep' } | Select-Object -First 1
    if (-not $asset) { "no suitable asset found: $($release.assets.name -join ', ')"; return }
    "downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)"
    $zip = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip
    Expand-Archive $zip -DestinationPath $ckptDir -Force
    Remove-Item $zip
    Get-ChildItem $ckptDir -Recurse -Filter model.ckpt | Select-Object FullName
}

Step "binarize (vendored trainer venv - validates IPA dictionary end-to-end, D3)" {
    Push-Location (Join-Path $repo "third_party\DiffSinger")
    & .\.venv\Scripts\python.exe scripts\binarize.py --config (
        Join-Path $DataRoot "datasets\full_acoustic_v1\config_acoustic.yaml")
    Pop-Location
}

Step "features (slow CPU tail: BPM/key/F0 for the dashboard)" {
    & $cli features --data-root $DataRoot --force
}

@"

=== ALL DONE $(Get-Date -Format o)
Next (human-launched):
  cd $repo\third_party\DiffSinger
  .\.venv\Scripts\python.exe scripts\train.py --config $DataRoot\datasets\full_acoustic_v1\config_acoustic.yaml --exp_name full_acoustic_v1 --reset
Monitor: signalml dash --data-root $DataRoot  (and TensorBoard in checkpoints\full_acoustic_v1)
Reminder: the community vocoder is CC BY-NC - dev preview only; nothing rendered with it ships.
"@ | Out-File $log -Append -Encoding utf8
