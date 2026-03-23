param(
    [string]$PythonExe = "",
    [string]$ReportJson = "",
    [string]$LocalZjuRoot = "",
    [string]$Checkpoint = "",
    [string]$OutputDir = "",
    [string]$Device = "auto",
    [string]$DType = "auto",
    [double]$ConfPercentile = 25.0,
    [ValidateSet("depth_unproject", "point_map", "auto")]
    [string]$PrimaryBranch = "depth_unproject"
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
        "python"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -eq "python") {
            return $candidate
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
}

function Resolve-DefaultZjuRoot() {
    $gData = "G:\" + [char]0x6570 + [char]0x636E + [char]0x96C6
    $gProjects = "G:\" + [char]0x9879 + [char]0x76EE + [char]0x5907 + [char]0x4EFD
    $candidates = @(
        (Join-Path $gData "datasets\\ZJU_MoCap\\data\\zju_mocap"),
        (Join-Path $gProjects "Redo_viewpoints_at_60_intervals_add_random_perturbations_vggt\\datasets\\ZJU_MoCap\\data\\zju_mocap")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Resolve-DefaultCheckpoint() {
    $driveRoot = "G:\"
    foreach ($topDir in @(Get-ChildItem -Path $driveRoot -Directory -ErrorAction SilentlyContinue)) {
        foreach ($childDir in @(Get-ChildItem -Path $topDir.FullName -Directory -ErrorAction SilentlyContinue)) {
            $candidate = Join-Path $childDir.FullName "vggt\\model.pt"
            if (Test-Path $candidate) {
                return (Resolve-Path $candidate).Path
            }
        }
    }
    return ""
}

if ([string]::IsNullOrWhiteSpace($ReportJson)) {
    throw "ReportJson is required."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Resolve-PythonExe $PythonExe
if ([string]::IsNullOrWhiteSpace($LocalZjuRoot)) {
    $LocalZjuRoot = Resolve-DefaultZjuRoot
}
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Resolve-DefaultCheckpoint
}

$argList = @(
    ".\\scripts\\compare_geometry_branches_zju_report.py",
    "--report_json", $ReportJson,
    "--device", $Device,
    "--dtype", $DType,
    "--conf_percentile", $ConfPercentile,
    "--primary_branch", $PrimaryBranch
)
if (-not [string]::IsNullOrWhiteSpace($LocalZjuRoot)) {
    $argList += @("--local_zju_root", $LocalZjuRoot)
}
if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
    $argList += @("--checkpoint", $Checkpoint)
}
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $argList += @("--output_dir", $OutputDir)
}

Write-Host "[zju-geometry-primary] repo_root=$repoRoot"
Write-Host "[zju-geometry-primary] python=$python"
Write-Host "[zju-geometry-primary] report_json=$ReportJson"
Write-Host "[zju-geometry-primary] local_zju_root=$LocalZjuRoot"
Write-Host "[zju-geometry-primary] checkpoint=$Checkpoint"
Write-Host "[zju-geometry-primary] primary_branch=$PrimaryBranch"

Push-Location $repoRoot
try {
    & $python @argList
} finally {
    Pop-Location
}
