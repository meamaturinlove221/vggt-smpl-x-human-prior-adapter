param(
    [string]$ModalExe = "",
    [string]$ModalAppName = "vggt-zju-geometry-smplxsurfaceposealign-headhair",
    [string]$ZjuSubdir = "zju_mocap",
    [string]$SeqNames = "CoreView_390",
    [string]$GeomSubdir = "vggt_geom",
    [string]$CheckpointSubpath = "checkpoints/model.pt",
    [string]$ExpName = "zju_smplxsurfaceposealign_headhair_a100_longrun",
    [string]$OutputSubdir = "",
    [int]$NumImages = 6,
    [int]$MaxImgPerGpu = 24,
    [int]$AccumSteps = 1,
    [int]$MaxEpochs = 9,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 800,
    [int]$LimitValBatches = 200,
    [int]$NumWorkers = 4,
    [int]$TrainPrefetchFactor = 2,
    [int]$ValPrefetchFactor = 2,
    [int]$HoldoutStride = 10,
    [double]$MinDepthConf = 0.0,
    [string]$ModalGpu = "A100-80GB",
    [double]$ModalCpu = 24.0,
    [int]$ModalMemoryMb = 196608,
    [int]$ModalTimeoutSec = 86400,
    [string]$DataVolume = "",
    [string]$OutputVolume = "",
    [string]$ExtraOverrides = "",
    [switch]$Attach,
    [switch]$DisableCompile,
    [switch]$StopExistingApps,
    [switch]$SkipPreflight,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$baseScript = Join-Path $PSScriptRoot "run_modal_zju_geometry_minimal_finetune.ps1"
$configName = "zju_vggt_geom_unproject_source_policy_nearestplusuniformtail_rawpool_confdepth_dropworst_gradconfmask_smplxsurfaceposealign_headhair_longrun"

$argList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $baseScript,
    "-ModalAppName", $ModalAppName,
    "-ZjuSubdir", $ZjuSubdir,
    "-SeqNames", $SeqNames,
    "-GeomSubdir", $GeomSubdir,
    "-CheckpointSubpath", $CheckpointSubpath,
    "-Config", $configName,
    "-ExpName", $ExpName,
    "-NumImages", $NumImages,
    "-MaxImgPerGpu", $MaxImgPerGpu,
    "-AccumSteps", $AccumSteps,
    "-MaxEpochs", $MaxEpochs,
    "-LearningRate", $LearningRate,
    "-LimitTrainBatches", $LimitTrainBatches,
    "-LimitValBatches", $LimitValBatches,
    "-NumWorkers", $NumWorkers,
    "-TrainPrefetchFactor", $TrainPrefetchFactor,
    "-ValPrefetchFactor", $ValPrefetchFactor,
    "-HoldoutStride", $HoldoutStride,
    "-MinDepthConf", $MinDepthConf,
    "-ModalGpu", $ModalGpu,
    "-ModalCpu", $ModalCpu,
    "-ModalMemoryMb", $ModalMemoryMb,
    "-ModalTimeoutSec", $ModalTimeoutSec
)

if (-not [string]::IsNullOrWhiteSpace($ModalExe)) {
    $argList += @("-ModalExe", $ModalExe)
}
if (-not [string]::IsNullOrWhiteSpace($OutputSubdir)) {
    $argList += @("-OutputSubdir", $OutputSubdir)
}
if (-not [string]::IsNullOrWhiteSpace($DataVolume)) {
    $argList += @("-DataVolume", $DataVolume)
}
if (-not [string]::IsNullOrWhiteSpace($OutputVolume)) {
    $argList += @("-OutputVolume", $OutputVolume)
}
if (-not [string]::IsNullOrWhiteSpace($ExtraOverrides)) {
    $argList += @("-ExtraOverrides", $ExtraOverrides)
}
if (-not $Attach) {
    $argList += "-Detach"
} else {
    $argList += "-AllowAttachedLongRun"
}
if ($DisableCompile) {
    $argList += "-DisableCompile"
} else {
    $argList += "-EnableCompile"
}
if ($StopExistingApps) {
    $argList += "-StopExistingApps"
}
if ($SkipPreflight) {
    $argList += "-SkipPreflight"
}
if ($DryRun) {
    $argList += "-DryRun"
}

& powershell @argList
exit $LASTEXITCODE
