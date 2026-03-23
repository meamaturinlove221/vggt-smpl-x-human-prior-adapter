param(
    [string]$PythonExe = "",
    [string]$BaselineConfig = "zju_vggt_geom_minimal",
    [string]$CandidateConfig = "zju_vggt_geom_minimal",
    [string]$ExpPrefix = "zju_depth_target_reliability_pair",
    [double]$CandidateMinDepthConf = 0.0,
    [int]$NumImages = 4,
    [int]$MaxImgPerGpu = 4,
    [int]$AccumSteps = 1,
    [int]$MaxEpochs = 1,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 5,
    [int]$LimitValBatches = 2,
    [int]$NumWorkers = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Resolve-PythonExe([string]$Preferred, [string]$RepoRoot) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }

    $candidates = @(
        (Join-Path $RepoRoot ".venv5080\\Scripts\\python.exe"),
        (Join-Path $RepoRoot ".venv\\Scripts\\python.exe"),
        (Join-Path $RepoRoot "venv\\Scripts\\python.exe"),
        "D:\anaconda\envs\vggt-colmap\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return "python"
}

function Invoke-CheckedCommand([string[]]$CommandParts, [string]$StepName) {
    & $CommandParts[0] $CommandParts[1..($CommandParts.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$trainingRoot = Join-Path $repoRoot "training"
$python = Resolve-PythonExe -Preferred $PythonExe -RepoRoot $repoRoot
$baselineExp = "${ExpPrefix}_baseline"
$candidateExp = "${ExpPrefix}_minconf"
$reportDir = Join-Path $repoRoot "output\\zju_training_ablation\\$ExpPrefix"

$commonArgs = @(
    "-NumImages", $NumImages,
    "-MaxImgPerGpu", $MaxImgPerGpu,
    "-AccumSteps", $AccumSteps,
    "-MaxEpochs", $MaxEpochs,
    "-LearningRate", $LearningRate,
    "-LimitTrainBatches", $LimitTrainBatches,
    "-LimitValBatches", $LimitValBatches,
    "-NumWorkers", $NumWorkers
)

if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    $commonArgs = @("-PythonExe", $PythonExe) + $commonArgs
}

$baselineCommand = @(
    "powershell", "-ExecutionPolicy", "Bypass", "-File",
    (Join-Path $repoRoot "scripts\\run_zju_vggt_geom_minimal_finetune.ps1"),
    "-Config", $BaselineConfig,
    "-ExpName", $baselineExp,
    "-MinDepthConf", 0.0
) + $commonArgs

$candidateCommand = @(
    "powershell", "-ExecutionPolicy", "Bypass", "-File",
    (Join-Path $repoRoot "scripts\\run_zju_vggt_geom_minimal_finetune.ps1"),
    "-Config", $CandidateConfig,
    "-ExpName", $candidateExp,
    "-MinDepthConf", $CandidateMinDepthConf
) + $commonArgs

$compareCommand = @(
    $python,
    (Join-Path $repoRoot "scripts\\compare_zju_finetune_runs.py"),
    "--baseline-log", (Join-Path $trainingRoot "logs\\$baselineExp\\log.txt"),
    "--candidate-log", (Join-Path $trainingRoot "logs\\$candidateExp\\log.txt"),
    "--baseline-label", "baseline_raw_target",
    "--candidate-label", "baseline_min_depth_conf",
    "--output-dir", $reportDir,
    "--title", "ZJU baseline vs reliable cached-depth target filtering"
)

Write-Host "[zju-depth-target-pair] repo_root=$repoRoot"
Write-Host "[zju-depth-target-pair] baseline_exp=$baselineExp"
Write-Host "[zju-depth-target-pair] candidate_exp=$candidateExp"
Write-Host "[zju-depth-target-pair] candidate_min_depth_conf=$CandidateMinDepthConf"
Write-Host "[zju-depth-target-pair] report_dir=$reportDir"

if ($DryRun) {
    Write-Host "[zju-depth-target-pair] baseline command:"
    Write-Host ($baselineCommand -join " ")
    Write-Host "[zju-depth-target-pair] candidate command:"
    Write-Host ($candidateCommand -join " ")
    Write-Host "[zju-depth-target-pair] compare command:"
    Write-Host ($compareCommand -join " ")
    return
}

Invoke-CheckedCommand -CommandParts $baselineCommand -StepName "baseline raw-target run"
Invoke-CheckedCommand -CommandParts $candidateCommand -StepName "candidate min-depth-conf run"
Invoke-CheckedCommand -CommandParts $compareCommand -StepName "comparison report"

Write-Host "[zju-depth-target-pair] done"
