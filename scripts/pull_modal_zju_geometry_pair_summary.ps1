param(
    [string]$ModalExe = "",
    [string]$RemotePairRoot = "",
    [string]$LocalOutputDir = "",
    [string]$BaselineLabel = "baseline",
    [string]$CandidateLabel = "unproject_geometry"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Resolve-ModalExe([string]$Preferred) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }

    $candidates = @(
        ".venv5080\\Scripts\\modal.exe",
        ".venv\\Scripts\\modal.exe",
        "venv\\Scripts\\modal.exe",
        "D:\\anaconda\\Scripts\\modal.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return "modal"
}

function Resolve-PythonExe([string]$RepoRoot) {
    $candidates = @(
        (Join-Path $RepoRoot ".venv5080\\Scripts\\python.exe"),
        (Join-Path $RepoRoot ".venv\\Scripts\\python.exe"),
        (Join-Path $RepoRoot "venv\\Scripts\\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return "python"
}

function Invoke-ModalVolumeGetForce([string]$Modal, [string]$Volume, [string]$RemotePath, [string]$LocalPath) {
    $parent = Split-Path -Parent $LocalPath
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $quotedModal = '"' + $Modal + '"'
    $quotedLocal = '"' + $LocalPath + '"'
    $cmd = "chcp 65001>nul & $quotedModal volume get --force $Volume $RemotePath $quotedLocal & exit /b 0"
    cmd /v:on /c $cmd | Out-Null

    if (-not (Test-Path $LocalPath)) {
        throw "Failed to materialize local file from Modal volume: $RemotePath -> $LocalPath"
    }
}

if ([string]::IsNullOrWhiteSpace($RemotePairRoot)) {
    throw "Please pass -RemotePairRoot, for example /geometry_pairs/20260322_xxx_pair_name"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$pythonExe = Resolve-PythonExe $repoRoot
$pairName = Split-Path -Leaf $RemotePairRoot.TrimEnd("/")

if ([string]::IsNullOrWhiteSpace($LocalOutputDir)) {
    $LocalOutputDir = Join-Path $repoRoot "output\\geometry_pairs_cloud\\$pairName"
}

New-Item -ItemType Directory -Force -Path $LocalOutputDir | Out-Null

$baselineLog = Join-Path $LocalOutputDir "baseline.log.txt"
$candidateLog = Join-Path $LocalOutputDir "unproject.log.txt"
$pairStatus = Join-Path $LocalOutputDir "pair_status.json"

Write-Host "[pull-modal-zju-pair] modal=$modal"
Write-Host "[pull-modal-zju-pair] remote_pair_root=$RemotePairRoot"
Write-Host "[pull-modal-zju-pair] local_output_dir=$LocalOutputDir"

Invoke-ModalVolumeGetForce -Modal $modal -Volume "vggt-out" -RemotePath "$RemotePairRoot/pair_status.json" -LocalPath $pairStatus
Invoke-ModalVolumeGetForce -Modal $modal -Volume "vggt-out" -RemotePath "$RemotePairRoot/baseline/logs/log.txt" -LocalPath $baselineLog
Invoke-ModalVolumeGetForce -Modal $modal -Volume "vggt-out" -RemotePath "$RemotePairRoot/unproject/logs/log.txt" -LocalPath $candidateLog

$compareScript = Join-Path $repoRoot "scripts\\compare_zju_finetune_runs.py"
$title = "Modal ZJU baseline vs unproject_geometry"

& $pythonExe $compareScript `
    --baseline-log $baselineLog `
    --candidate-log $candidateLog `
    --baseline-label $BaselineLabel `
    --candidate-label $CandidateLabel `
    --output-dir $LocalOutputDir `
    --title $title

if ($LASTEXITCODE -ne 0) {
    throw "compare_zju_finetune_runs.py failed with exit code $LASTEXITCODE."
}

Write-Host "[pull-modal-zju-pair] summary_md=$(Join-Path $LocalOutputDir 'summary.md')"
Write-Host "[pull-modal-zju-pair] summary_json=$(Join-Path $LocalOutputDir 'summary.json')"
