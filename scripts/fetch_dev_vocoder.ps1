# Download the community PC-NSF-HiFiGAN checkpoint into the vendored trainer's
# checkpoints/ directory.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\fetch_dev_vocoder.ps1
#
# LICENSE: openvpi's released vocoder weights are CC BY-NC. They are a DEV PREVIEW
# ONLY (OPEN_QUESTIONS Q4): they make `val_with_vocoder` validation audible while the
# acoustic model trains, and nothing rendered through them ships. The production
# vocoder is trained in-house (MIGRATION_PLAN P7.4).
#
# `signalml train` refuses to start a run whose config has val_with_vocoder on while
# this checkpoint is missing - validation would otherwise die at the first val step,
# hours in. Run this, or set trainer_opts.val_with_vocoder: false in the recipe.

param(
    [string]$Tag = "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$ckptDir = Join-Path $repo "third_party\DiffSinger\checkpoints"
$expected = Join-Path $ckptDir "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt"

if ((Test-Path $expected) -and (-not $Force)) {
    Write-Host "already present: $expected"
    exit 0
}

New-Item -ItemType Directory -Force $ckptDir | Out-Null
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

Write-Host "querying openvpi/vocoders release $Tag"
$release = Invoke-RestMethod "https://api.github.com/repos/openvpi/vocoders/releases/tags/$Tag"
# the plain PyTorch checkpoint, not the ONNX / OpenUtau / dependency bundles
$asset = $release.assets |
    Where-Object { $_.name -match 'zip$' -and $_.name -notmatch 'onnx|openutau|oudep' } |
    Select-Object -First 1
if (-not $asset) {
    throw "no suitable asset in $Tag (saw: $($release.assets.name -join ', '))"
}

$zip = Join-Path $env:TEMP $asset.name
Write-Host "downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)"
Invoke-WebRequest $asset.browser_download_url -OutFile $zip
Expand-Archive $zip -DestinationPath $ckptDir -Force
Remove-Item $zip

$found = Get-ChildItem $ckptDir -Recurse -Filter model.ckpt | Select-Object -First 1
if (-not $found) { throw "archive expanded but no model.ckpt under $ckptDir" }
Write-Host "checkpoint: $($found.FullName)"
if ($found.FullName -ne $expected) {
    Write-Host "NOTE: generated configs point at $expected" -ForegroundColor Yellow
    Write-Host "      set trainer_opts.vocoder_ckpt in the recipe to match." -ForegroundColor Yellow
}
Write-Host "CC BY-NC: dev preview only - nothing rendered with this ships (Q4)."
