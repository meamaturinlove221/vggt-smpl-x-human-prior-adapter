param(
    [string]$PythonExe = "",
    [string]$Co3dDir = "",
    [string]$Co3dAnnotationDir = "",
    [string]$Checkpoint = "",
    [string]$Config = "default",
    [string]$ExpName = "geometry_minimal_local",
    [int]$MaxImgPerGpu = 4,
    [int]$AccumSteps = 4,
    [int]$MaxEpochs = 5,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 100,
    [int]$LimitValBatches = 50,
    [switch]$NoFreezeAggregator,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Resolve-PythonExe([string]$Preferred) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }

    $candidates = @(
        ".venv5080\\Scripts\\python.exe",
        ".venv\\Scripts\\python.exe",
        "venv\\Scripts\\python.exe",
        "D:\anaconda\envs\vggt-colmap\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return "python"
}

if ([string]::IsNullOrWhiteSpace($Co3dDir)) {
    throw "Co3dDir is required."
}
if ([string]::IsNullOrWhiteSpace($Co3dAnnotationDir)) {
    throw "Co3dAnnotationDir is required."
}
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    throw "Checkpoint is required."
}

$python = Resolve-PythonExe $PythonExe
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launchPath = Join-Path $repoRoot "training\\launch.py"
$existingPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
    $env:PYTHONPATH = $repoRoot
} else {
    $env:PYTHONPATH = "$repoRoot;$existingPythonPath"
}

$overrides = @(
    "exp_name=$ExpName",
    "data.train.dataset.dataset_configs[0].CO3D_DIR=$Co3dDir",
    "data.train.dataset.dataset_configs[0].CO3D_ANNOTATION_DIR=$Co3dAnnotationDir",
    "data.val.dataset.dataset_configs[0].CO3D_DIR=$Co3dDir",
    "data.val.dataset.dataset_configs[0].CO3D_ANNOTATION_DIR=$Co3dAnnotationDir",
    "checkpoint.resume_checkpoint_path=$Checkpoint",
    "max_img_per_gpu=$MaxImgPerGpu",
    "accum_steps=$AccumSteps",
    "max_epochs=$MaxEpochs",
    "optim.optimizer.lr=$LearningRate",
    "limit_train_batches=$LimitTrainBatches",
    "limit_val_batches=$LimitValBatches",
    "model.enable_camera=True",
    "model.enable_depth=True",
    "model.enable_point=False",
    "model.enable_track=False",
    "loss.point=null",
    "loss.track=null"
)

if ($NoFreezeAggregator) {
    $overrides += "optim.frozen_module_names=[]"
}

$argList = @($launchPath, "--config", $Config) + $overrides

Write-Host "[geometry-finetune] repo_root=$repoRoot"
Write-Host "[geometry-finetune] python=$python"
Write-Host "[geometry-finetune] config=$Config"
Write-Host "[geometry-finetune] exp_name=$ExpName"
Write-Host "[geometry-finetune] max_img_per_gpu=$MaxImgPerGpu accum_steps=$AccumSteps max_epochs=$MaxEpochs"
Write-Host "[geometry-finetune] PYTHONPATH=$env:PYTHONPATH"

if ($DryRun) {
    Write-Host "[geometry-finetune] dry run command:"
    Write-Host "$python $($argList -join ' ')"
    return
}

Push-Location (Join-Path $repoRoot "training")
try {
    & $python @argList
} finally {
    Pop-Location
}
