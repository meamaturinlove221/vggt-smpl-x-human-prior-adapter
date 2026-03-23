param(
    [string]$PythonExe = "",
    [string]$ImageFolder = "examples\kitchen\images",
    [string]$OutputDir = "",
    [string]$Checkpoint = "",
    [ValidateSet("crop", "pad")]
    [string]$PreprocessMode = "crop",
    [int]$MaxImages = 8,
    [string]$TargetFrames = "0",
    [switch]$IncludeTargetFrame,
    [switch]$SkipSavePredictions
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
        ".venv\Scripts\python.exe",
        "venv\Scripts\python.exe",
        "D:\anaconda\envs\vggt-colmap\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return "python"
}

$python = Resolve-PythonExe $PythonExe
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptPath = Join-Path $repoRoot "scripts\compare_geometry_branches.py"

$argList = @(
    $scriptPath,
    "--image_folder", $ImageFolder,
    "--preprocess_mode", $PreprocessMode,
    "--target_frames", $TargetFrames
)

if ($MaxImages -gt 0) {
    $argList += @("--max_images", [string]$MaxImages)
}

if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $argList += @("--output_dir", $OutputDir)
}

if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
    $argList += @("--checkpoint", $Checkpoint)
}

if ($IncludeTargetFrame) {
    $argList += "--include_target_frame"
}

if ($SkipSavePredictions) {
    $argList += "--skip_save_predictions"
}

Write-Host "[geometry-baseline] repo_root=$repoRoot"
Write-Host "[geometry-baseline] python=$python"
Write-Host "[geometry-baseline] image_folder=$ImageFolder"
Write-Host "[geometry-baseline] target_frames=$TargetFrames"

Push-Location $repoRoot
try {
    & $python @argList
} finally {
    Pop-Location
}
