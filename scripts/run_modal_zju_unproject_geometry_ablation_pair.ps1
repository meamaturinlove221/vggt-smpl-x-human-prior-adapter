param(
    [string]$ModalExe = "",
    [string]$ZjuSubdir = "zju_mocap",
    [string]$SeqNames = "CoreView_390",
    [string]$GeomSubdir = "vggt_geom",
    [string]$CheckpointSubpath = "checkpoints/model.pt",
    [string]$LocalCheckpoint = "",
    [string]$BaselineConfig = "zju_vggt_geom_minimal",
    [string]$CandidateConfig = "zju_vggt_geom_unproject_minimal",
    [string]$ExpPrefix = "zju_geom_modal_pair",
    [string]$OutputSubdirBase = "",
    [int]$NumImages = 4,
    [int]$MaxImgPerGpu = 0,
    [int]$AccumSteps = 0,
    [int]$MaxEpochs = 1,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 500,
    [int]$LimitValBatches = 20,
    [int]$NumWorkers = 0,
    [int]$HoldoutStride = 10,
    [string]$CameraSource = "gt",
    [string]$MaskSource = "mask",
    [double]$MinDepthConf = 0.0,
    [string[]]$ExtraOverrides = @(),
    [string]$ThroughputProfile = "a10080_balanced",
    [string]$ModalGpu = "",
    [double]$ModalCpu = 0.0,
    [int]$ModalMemoryMb = 0,
    [int]$ModalTimeoutSec = 0,
    [string]$DataVolume = "vggt-zju-data",
    [string]$OutputVolume = "vggt-out",
    [switch]$NoDetach,
    [switch]$SkipPreflight,
    [switch]$StopExistingApps,
    [switch]$AllowLargeLocalUpload,
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

function Invoke-CheckedCommand([string[]]$CommandParts, [string]$StepName) {
    & $CommandParts[0] $CommandParts[1..($CommandParts.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

function Get-ProfileSettings([string]$ProfileName) {
    switch ($ProfileName) {
        "strict_compare" {
            return [pscustomobject]@{
                ModalGpu = "A100-40GB"
                ModalCpu = 8
                ModalMemoryMb = 65536
                MaxImgPerGpu = 4
                AccumSteps = 1
                NumWorkers = 4
            }
        }
        "a10040_balanced" {
            return [pscustomobject]@{
                ModalGpu = "A100-40GB"
                ModalCpu = 8
                ModalMemoryMb = 65536
                MaxImgPerGpu = 8
                AccumSteps = 1
                NumWorkers = 8
            }
        }
        "a10080_balanced" {
            return [pscustomobject]@{
                ModalGpu = "A100-80GB"
                ModalCpu = 10
                ModalMemoryMb = 98304
                MaxImgPerGpu = 8
                AccumSteps = 1
                NumWorkers = 8
            }
        }
        "a10080_fast" {
            return [pscustomobject]@{
                ModalGpu = "A100-80GB"
                ModalCpu = 12
                ModalMemoryMb = 131072
                MaxImgPerGpu = 12
                AccumSteps = 1
                NumWorkers = 12
            }
        }
        default {
            throw "Unsupported ThroughputProfile: $ProfileName"
        }
    }
}

$profileSettings = Get-ProfileSettings $ThroughputProfile
if ([string]::IsNullOrWhiteSpace($ModalGpu)) {
    $ModalGpu = $profileSettings.ModalGpu
}
if ($ModalCpu -le 0) {
    $ModalCpu = $profileSettings.ModalCpu
}
if ($ModalMemoryMb -le 0) {
    $ModalMemoryMb = $profileSettings.ModalMemoryMb
}
if ($MaxImgPerGpu -le 0) {
    $MaxImgPerGpu = $profileSettings.MaxImgPerGpu
}
if ($AccumSteps -le 0) {
    $AccumSteps = $profileSettings.AccumSteps
}
if ($NumWorkers -le 0) {
    $NumWorkers = $profileSettings.NumWorkers
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$modalEntry = Join-Path $repoRoot "modal_zju_geometry_minimal_finetune.py"
$preflightScript = Join-Path $repoRoot "scripts\\invoke_modal_zju_preflight.ps1"
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputSubdirBase)) {
    $OutputSubdirBase = "geometry_pairs/${runStamp}_$ExpPrefix"
}

$pairCommand = @(
    $modal,
    "run"
)
if (-not $NoDetach) {
    $pairCommand += "--detach"
}

Write-Host "[modal-zju-ablation-pair] repo_root=$repoRoot"
Write-Host "[modal-zju-ablation-pair] throughput_profile=$ThroughputProfile"
Write-Host "[modal-zju-ablation-pair] modal_gpu=$ModalGpu modal_cpu=$ModalCpu modal_memory_mb=$ModalMemoryMb"
Write-Host "[modal-zju-ablation-pair] num_images=$NumImages max_img_per_gpu=$MaxImgPerGpu accum_steps=$AccumSteps num_workers=$NumWorkers"
Write-Host "[modal-zju-ablation-pair] data_volume=$DataVolume output_volume=$OutputVolume"
Write-Host "[modal-zju-ablation-pair] zju_subdir=$ZjuSubdir geom_subdir=$GeomSubdir checkpoint_subpath=$CheckpointSubpath"
Write-Host "[modal-zju-ablation-pair] output_subdir_base=$OutputSubdirBase"
Write-Host "[modal-zju-ablation-pair] detach=$(-not $NoDetach)"
if ($ExtraOverrides.Count -gt 0) {
    Write-Host "[modal-zju-ablation-pair] extra_overrides=$($ExtraOverrides -join ' | ')"
}

if ($MaxImgPerGpu -ne 4) {
    Write-Host "[modal-zju-ablation-pair] note: max_img_per_gpu=$MaxImgPerGpu changes effective batch size."
    Write-Host "[modal-zju-ablation-pair] note: keep NumImages fixed for view-comparable experiments; compare runs within the same paired schedule."
}

if ([string]::IsNullOrWhiteSpace($LocalCheckpoint)) {
    Write-Host "[modal-zju-ablation-pair] local_checkpoint is empty; will use remote checkpoint path or output-volume fallback"
} else {
    Write-Host "[modal-zju-ablation-pair] local_checkpoint=$LocalCheckpoint"
}

if (-not $SkipPreflight) {
    $preflightArgs = @(
        "-ExecutionPolicy", "Bypass", "-File", $preflightScript,
        "-ModalExe", $modal,
        "-DataVolume", $DataVolume,
        "-OutputVolume", $OutputVolume
    )
    if (-not [string]::IsNullOrWhiteSpace($LocalCheckpoint)) {
        $preflightArgs += @("-LocalCheckpoint", $LocalCheckpoint)
    }
    if (-not $NoDetach) {
        $preflightArgs += "-Detach"
    }
    $preflightArgs += "-StopRepoProcesses"
    if ($StopExistingApps) {
        $preflightArgs += "-StopExistingApps"
    }
    if ($AllowLargeLocalUpload) {
        $preflightArgs += "-AllowLargeLocalUpload"
    }
    & powershell @preflightArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Modal preflight failed with exit code $LASTEXITCODE."
    }
}

if (-not [string]::IsNullOrWhiteSpace($ModalGpu)) {
    $env:VGGT_ZJU_MODAL_GPU = $ModalGpu
}
if ($ModalCpu -gt 0) {
    $env:VGGT_ZJU_MODAL_CPU = [string]$ModalCpu
}
if ($ModalMemoryMb -gt 0) {
    $env:VGGT_ZJU_MODAL_MEMORY_MB = [string]$ModalMemoryMb
}
if ($ModalTimeoutSec -gt 0) {
    $env:VGGT_ZJU_MODAL_TIMEOUT_SEC = [string]$ModalTimeoutSec
}
if (-not [string]::IsNullOrWhiteSpace($DataVolume)) {
    $env:VGGT_ZJU_MODAL_DATA_VOLUME = $DataVolume
}
if (-not [string]::IsNullOrWhiteSpace($OutputVolume)) {
    $env:VGGT_ZJU_MODAL_OUTPUT_VOLUME = $OutputVolume
}

$resolvedCheckpointSubpath = $CheckpointSubpath
if (-not [string]::IsNullOrWhiteSpace($LocalCheckpoint) -and -not $DryRun) {
    $uploadCommand = @(
        $modal,
        "run",
        "$modalEntry::upload_checkpoint",
        "--local-path", $LocalCheckpoint,
        "--remote-subpath", $CheckpointSubpath
    )
    Write-Host "[modal-zju-ablation-pair] uploading local checkpoint before remote launch"
    Push-Location $repoRoot
    try {
        Invoke-CheckedCommand -CommandParts $uploadCommand -StepName "checkpoint upload"
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint upload failed with exit code $LASTEXITCODE."
    }
}

$pairCfg = [ordered]@{
    zju_subdir = $ZjuSubdir
    seq_names = $SeqNames
    geom_subdir = $GeomSubdir
    checkpoint_subpath = $resolvedCheckpointSubpath
    baseline_config = $BaselineConfig
    candidate_config = $CandidateConfig
    exp_prefix = $ExpPrefix
    output_subdir_base = $OutputSubdirBase
    num_images = $NumImages
    max_img_per_gpu = $MaxImgPerGpu
    accum_steps = $AccumSteps
    max_epochs = $MaxEpochs
    learning_rate = $LearningRate
    limit_train_batches = $LimitTrainBatches
    limit_val_batches = $LimitValBatches
    num_workers = $NumWorkers
    holdout_stride = $HoldoutStride
    camera_source = $CameraSource
    mask_source = $MaskSource
    min_depth_conf = $MinDepthConf
    freeze_aggregator = (-not $NoFreezeAggregator)
    extra_overrides = ($ExtraOverrides -join " ")
}
$pairCfgJsonRaw = $pairCfg | ConvertTo-Json -Compress
$pairCfgJson = "base64:" + [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($pairCfgJsonRaw))

$pairCommand += @(
    "$modalEntry::run_remote_zju_geometry_ablation_pair",
    "--cfg-json", $pairCfgJson
)

if ($DryRun) {
    Write-Host "[modal-zju-ablation-pair] pair command:"
    Write-Host ($pairCommand -join " ")
    Write-Host "[modal-zju-ablation-pair] cfg_json=$pairCfgJson"
    return
}

Push-Location $repoRoot
try {
    Invoke-CheckedCommand -CommandParts $pairCommand -StepName "paired modal run"
} finally {
    Pop-Location
}

Write-Host "[modal-zju-ablation-pair] done"
