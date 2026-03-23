param(
    [string]$ModalExe = "",
    [string]$ZjuSubdir = "zju_mocap",
    [string]$SeqNames = "CoreView_390",
    [string]$GeomSubdir = "vggt_geom",
    [string]$CheckpointSubpath = "checkpoints/model.pt",
    [string]$LocalCheckpoint = "",
    [string]$Config = "zju_vggt_geom_minimal",
    [string]$ExpName = "zju_geometry_minimal_modal",
    [string]$OutputSubdir = "",
    [int]$NumImages = 4,
    [int]$MaxImgPerGpu = 4,
    [int]$AccumSteps = 1,
    [int]$MaxEpochs = 1,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 100,
    [int]$LimitValBatches = 20,
    [int]$NumWorkers = 4,
    [int]$HoldoutStride = 10,
    [string]$CameraSource = "gt",
    [string]$MaskSource = "mask",
    [double]$MinDepthConf = 0.0,
    [string]$ModalGpu = "",
    [double]$ModalCpu = 0.0,
    [int]$ModalMemoryMb = 0,
    [int]$ModalTimeoutSec = 0,
    [string]$DataVolume = "",
    [string]$OutputVolume = "",
    [string]$ExtraOverrides = "",
    [switch]$Detach,
    [switch]$SkipPreflight,
    [switch]$StopExistingApps,
    [switch]$AllowLargeLocalUpload,
    [switch]$AllowAttachedLongRun,
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$entryScript = Join-Path $repoRoot "modal_zju_geometry_minimal_finetune.py"
$preflightScript = Join-Path $repoRoot "scripts\\invoke_modal_zju_preflight.ps1"

if (-not $Detach -and -not $AllowAttachedLongRun) {
    if ($LimitTrainBatches -gt 20 -or $MaxEpochs -gt 1) {
        throw "Refusing a long non-detached Modal run. Use -Detach or override with -AllowAttachedLongRun."
    }
}

if (-not $SkipPreflight) {
    $preflightArgs = @(
        "-ExecutionPolicy", "Bypass", "-File", $preflightScript,
        "-ModalExe", $modal
    )
    if (-not [string]::IsNullOrWhiteSpace($DataVolume)) {
        $preflightArgs += @("-DataVolume", $DataVolume)
    }
    if (-not [string]::IsNullOrWhiteSpace($OutputVolume)) {
        $preflightArgs += @("-OutputVolume", $OutputVolume)
    }
    if (-not [string]::IsNullOrWhiteSpace($LocalCheckpoint)) {
        $preflightArgs += @("-LocalCheckpoint", $LocalCheckpoint)
    }
    if ($Detach) {
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

$argList = @(
    "run"
)

if ($Detach) {
    $argList += "--detach"
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

Write-Host "[modal-zju-geometry] repo_root=$repoRoot"
Write-Host "[modal-zju-geometry] modal=$modal"
Write-Host "[modal-zju-geometry] entry=$entryScript"
Write-Host "[modal-zju-geometry] zju_subdir=$ZjuSubdir geom_subdir=$GeomSubdir checkpoint_subpath=$CheckpointSubpath"
if (-not [string]::IsNullOrWhiteSpace($LocalCheckpoint)) {
    Write-Host "[modal-zju-geometry] local_checkpoint=$LocalCheckpoint"
} else {
    Write-Host "[modal-zju-geometry] local_checkpoint=<empty>; will use remote checkpoint path or output-volume fallback"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_ZJU_MODAL_GPU)) {
    Write-Host "[modal-zju-geometry] modal_gpu=$env:VGGT_ZJU_MODAL_GPU"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_ZJU_MODAL_CPU)) {
    Write-Host "[modal-zju-geometry] modal_cpu=$env:VGGT_ZJU_MODAL_CPU"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_ZJU_MODAL_MEMORY_MB)) {
    Write-Host "[modal-zju-geometry] modal_memory_mb=$env:VGGT_ZJU_MODAL_MEMORY_MB"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_ZJU_MODAL_TIMEOUT_SEC)) {
    Write-Host "[modal-zju-geometry] modal_timeout_sec=$env:VGGT_ZJU_MODAL_TIMEOUT_SEC"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_ZJU_MODAL_DATA_VOLUME)) {
    Write-Host "[modal-zju-geometry] data_volume=$env:VGGT_ZJU_MODAL_DATA_VOLUME"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_ZJU_MODAL_OUTPUT_VOLUME)) {
    Write-Host "[modal-zju-geometry] output_volume=$env:VGGT_ZJU_MODAL_OUTPUT_VOLUME"
}
Write-Host "[modal-zju-geometry] detach=$Detach"

$resolvedCheckpointSubpath = $CheckpointSubpath
if (-not [string]::IsNullOrWhiteSpace($LocalCheckpoint) -and -not $DryRun) {
    $uploadArgs = @(
        "run",
        "$entryScript::upload_checkpoint",
        "--local-path", $LocalCheckpoint,
        "--remote-subpath", $CheckpointSubpath
    )
    Write-Host "[modal-zju-geometry] uploading local checkpoint before remote launch"
    Push-Location $repoRoot
    try {
        & $modal @uploadArgs
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint upload failed with exit code $LASTEXITCODE."
    }
}

$cfg = [ordered]@{
    zju_subdir = $ZjuSubdir
    seq_names = $SeqNames
    geom_subdir = $GeomSubdir
    checkpoint_subpath = $resolvedCheckpointSubpath
    config = $Config
    exp_name = $ExpName
    output_subdir = $OutputSubdir
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
    extra_overrides = $ExtraOverrides
}
$cfgJsonRaw = $cfg | ConvertTo-Json -Compress
$cfgJson = "base64:" + [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($cfgJsonRaw))

$argList += @(
    "$entryScript::run_remote_zju_geometry_finetune",
    "--cfg-json", $cfgJson
)

if ($DryRun) {
    Write-Host "[modal-zju-geometry] dry run command:"
    Write-Host "$modal $($argList -join ' ')"
    Write-Host "[modal-zju-geometry] cfg_json=$cfgJson"
    return
}

Push-Location $repoRoot
try {
    & $modal @argList
} finally {
    Pop-Location
}
