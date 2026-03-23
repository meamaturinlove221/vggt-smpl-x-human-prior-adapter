param(
    [string]$ModalExe = "",
    [string]$Co3dSubdir = "",
    [string]$Co3dAnnotationSubdir = "",
    [string]$CheckpointSubpath = "checkpoints/model.pt",
    [string]$LocalCheckpoint = "",
    [string]$Config = "default",
    [string]$ExpName = "geometry_minimal_modal",
    [string]$OutputSubdir = "",
    [int]$MaxImgPerGpu = 8,
    [int]$AccumSteps = 2,
    [int]$MaxEpochs = 5,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 200,
    [int]$LimitValBatches = 100,
    [string]$ExtraOverrides = "",
    [switch]$NoFreezeAggregator,
    [switch]$DryRun
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

if ([string]::IsNullOrWhiteSpace($Co3dSubdir)) {
    throw "Co3dSubdir is required."
}
if ([string]::IsNullOrWhiteSpace($Co3dAnnotationSubdir)) {
    throw "Co3dAnnotationSubdir is required."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$entryScript = Join-Path $repoRoot "modal_geometry_minimal_finetune.py"

$argList = @(
    "run",
    "$entryScript::run_geometry_finetune",
    "--co3d-subdir", $Co3dSubdir,
    "--co3d-annotation-subdir", $Co3dAnnotationSubdir,
    "--checkpoint-subpath", $CheckpointSubpath,
    "--config", $Config,
    "--exp-name", $ExpName,
    "--max-img-per-gpu", $MaxImgPerGpu,
    "--accum-steps", $AccumSteps,
    "--max-epochs", $MaxEpochs,
    "--learning-rate", $LearningRate,
    "--limit-train-batches", $LimitTrainBatches,
    "--limit-val-batches", $LimitValBatches
)

if ($NoFreezeAggregator) {
    $argList += "--no-freeze-aggregator"
} else {
    $argList += "--freeze-aggregator"
}

if (-not [string]::IsNullOrWhiteSpace($LocalCheckpoint)) {
    $argList += @("--local-checkpoint", $LocalCheckpoint)
}
if (-not [string]::IsNullOrWhiteSpace($OutputSubdir)) {
    $argList += @("--output-subdir", $OutputSubdir)
}
if (-not [string]::IsNullOrWhiteSpace($ExtraOverrides)) {
    $argList += @("--extra-overrides", $ExtraOverrides)
}

Write-Host "[modal-geometry] repo_root=$repoRoot"
Write-Host "[modal-geometry] modal=$modal"
Write-Host "[modal-geometry] entry=$entryScript"

if ($DryRun) {
    Write-Host "[modal-geometry] dry run command:"
    Write-Host "$modal $($argList -join ' ')"
    return
}

Push-Location $repoRoot
try {
    & $modal @argList
} finally {
    Pop-Location
}
