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
    [int]$MaxImgPerGpu = 16,
    [int]$AccumSteps = 1,
    [int]$MaxEpochs = 1,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 100,
    [int]$LimitValBatches = 20,
    [int]$NumWorkers = 16,
    [int]$TrainPrefetchFactor = 8,
    [int]$ValPrefetchFactor = 4,
    [int]$ProbeAggregateSamples = 4,
    [int]$ProbeAggregateStride = 50,
    [string]$ProbeReferenceConfig = "",
    [string]$ProbeLabelCurrent = "",
    [string]$ProbeLabelReference = "",
    [int]$HoldoutStride = 10,
    [string]$CameraSource = "gt",
    [string]$MaskSource = "mask",
    [double]$MinDepthConf = 0.0,
    [string]$ModalGpu = "A100-80GB",
    [double]$ModalCpu = 24.0,
    [int]$ModalMemoryMb = 196608,
    [int]$ModalTimeoutSec = 0,
    [string]$DataVolume = "",
    [string]$OutputVolume = "",
    [string]$ExtraOverrides = "",
    [double]$PreflightMinFreeMemoryGb = 4.0,
    [switch]$Detach,
    [switch]$SkipPreflight,
    [switch]$StopExistingApps,
    [switch]$AllowLargeLocalUpload,
    [switch]$AllowAttachedLongRun,
    [switch]$EnableCompile,
    [switch]$SkipActiveAppCheck,
    [switch]$NoProbeContractDiff,
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

function Get-ActiveModalApps([string]$ModalCmd, [string]$DescriptionFilter) {
    $raw = & $ModalCmd app list --json
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query Modal app list."
    }
    $items = @()
    if (-not [string]::IsNullOrWhiteSpace($raw)) {
        $parsed = $raw | ConvertFrom-Json
        if ($parsed -is [System.Array]) {
            $items = $parsed
        } elseif ($null -ne $parsed) {
            $items = @($parsed)
        }
    }

    return @(
        $items | Where-Object {
            $state = "$($_.'State')".ToLowerInvariant()
            $description = "$($_.'Description')"
            $isActive = $state -notin @("stopped", "stopping", "completed", "failed")
            $matchesDescription = [string]::IsNullOrWhiteSpace($DescriptionFilter) -or $description -eq $DescriptionFilter
            $isActive -and $matchesDescription
        }
    )
}

function Stop-ModalApps([string]$ModalCmd, [object[]]$Apps) {
    foreach ($app in $Apps) {
        $appId = "$($app.'App ID')"
        if ([string]::IsNullOrWhiteSpace($appId)) {
            continue
        }
        Write-Host "[modal-zju-geometry] stopping active Modal app $appId"
        & $ModalCmd app stop $appId
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to stop Modal app $appId."
        }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$entryScript = Join-Path $repoRoot "modal_zju_geometry_minimal_finetune.py"
$preflightScript = Join-Path $repoRoot "scripts\\invoke_modal_zju_preflight.ps1"
$modalAppDescription = "vggt-zju-geometry-minimal-finetune"

if (-not $Detach -and -not $AllowAttachedLongRun) {
    if ($LimitTrainBatches -gt 20 -or $MaxEpochs -gt 1) {
        throw "Refusing a long non-detached Modal run. Use -Detach or override with -AllowAttachedLongRun."
    }
}

if (-not $SkipPreflight) {
    $preflightArgs = @(
        "-ExecutionPolicy", "Bypass", "-File", $preflightScript,
        "-ModalExe", $modal,
        "-MinFreeMemoryGb", $PreflightMinFreeMemoryGb
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

if (-not $SkipActiveAppCheck) {
    $activeApps = Get-ActiveModalApps -ModalCmd $modal -DescriptionFilter $modalAppDescription
    if ($activeApps.Count -gt 0) {
        $appSummary = ($activeApps | ForEach-Object { "$($_.'App ID'):$($_.'State')" }) -join ", "
        if ($StopExistingApps) {
            Write-Host "[modal-zju-geometry] found active Modal apps: $appSummary"
            Stop-ModalApps -ModalCmd $modal -Apps $activeApps
            $remainingApps = Get-ActiveModalApps -ModalCmd $modal -DescriptionFilter $modalAppDescription
            if ($remainingApps.Count -gt 0) {
                $remainingSummary = ($remainingApps | ForEach-Object { "$($_.'App ID'):$($_.'State')" }) -join ", "
                throw "Refusing launch because active Modal apps still remain after stop attempt: $remainingSummary"
            }
        } else {
            throw "Refusing launch because active Modal apps already exist: $appSummary. Use -StopExistingApps or wait for the existing run to finish."
        }
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
Write-Host "[modal-zju-geometry] max_img_per_gpu=$MaxImgPerGpu num_workers=$NumWorkers train_prefetch=$TrainPrefetchFactor val_prefetch=$ValPrefetchFactor"

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
    train_prefetch_factor = $TrainPrefetchFactor
    val_prefetch_factor = $ValPrefetchFactor
    emit_dataset_probe_contract_diff = (-not $NoProbeContractDiff)
    probe_aggregate_samples = $ProbeAggregateSamples
    probe_aggregate_stride = $ProbeAggregateStride
    probe_reference_config = $ProbeReferenceConfig
    probe_label_current = $ProbeLabelCurrent
    probe_label_reference = $ProbeLabelReference
    holdout_stride = $HoldoutStride
    camera_source = $CameraSource
    mask_source = $MaskSource
    min_depth_conf = $MinDepthConf
    freeze_aggregator = (-not $NoFreezeAggregator)
    enable_compile = $EnableCompile.IsPresent
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
