param(
    [string]$DatasetRoot = "",
    [string]$Seq = "0012_11",
    [int]$Frame = 0,
    [string]$TargetCamera = "00",
    [int]$AutoSources = 6,
    [string]$OutputBase = "",
    [string]$PythonExe = "",
    [string]$ModalExe = "",
    [switch]$FullViews,
    [switch]$PullResults,
    [string]$LocalResultsDir = "",
    [switch]$OverwriteScene,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Resolve-PythonExe([string]$Preferred, [string]$RepoRoot) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }

    $candidates = @(
        (Join-Path $RepoRoot ".venv5080\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot "venv\Scripts\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return "python"
}

function Invoke-ModalVolumeGetWithRetry(
    [string]$ModalPath,
    [string]$VolumeName,
    [string]$RemotePath,
    [string]$LocalDestination,
    [int]$MaxAttempts = 3
) {
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUTF8 = "1"
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            & $ModalPath volume get --force $VolumeName $RemotePath $LocalDestination
            return
        } catch {
            if ($attempt -eq $MaxAttempts) {
                throw
            }
            Write-Host "[4k4d-vggt] modal volume get failed (attempt $attempt/$MaxAttempts): $RemotePath"
            Start-Sleep -Seconds ([Math]::Min(10 * $attempt, 30))
        }
    }
}

function Test-AsciiPlyValid(
    [string]$PythonPath,
    [string]$RepoRoot,
    [string]$PlyPath
) {
    $validator = Join-Path $RepoRoot "tools\validate_ascii_ply.py"
    & $PythonPath $validator $PlyPath
    return ($LASTEXITCODE -eq 0)
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Resolve-PythonExe $PythonExe $repoRoot
$modalScript = Join-Path $repoRoot "scripts\run_modal_4k4d_vggt_infer.ps1"
$modal = if (-not [string]::IsNullOrWhiteSpace($ModalExe)) {
    $ModalExe
} elseif (Test-Path (Join-Path $repoRoot ".venv5080\Scripts\modal.exe")) {
    (Resolve-Path (Join-Path $repoRoot ".venv5080\Scripts\modal.exe")).Path
} elseif (Test-Path (Join-Path $repoRoot ".venv\Scripts\modal.exe")) {
    (Resolve-Path (Join-Path $repoRoot ".venv\Scripts\modal.exe")).Path
} else {
    "modal"
}

if ([string]::IsNullOrWhiteSpace($OutputBase)) {
    $OutputBase = Join-Path $repoRoot "output"
}

if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $preferredRoot = "G:\数据集\datasets\data_used_in_4K4D"
    if (Test-Path $preferredRoot) {
        $detected = (Resolve-Path $preferredRoot).Path
    } else {
        $detected = Get-ChildItem 'G:\' -Directory -Recurse -Depth 3 -Filter 'data_used_in_4K4D' -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if ([string]::IsNullOrWhiteSpace($detected)) {
        throw "DatasetRoot was not provided and no data_used_in_4K4D directory could be auto-detected under G:\\."
    }
    $DatasetRoot = $detected
}

$sceneName = if ($FullViews) {
    "{0}_frame{1:D4}_fullviews" -f $Seq, $Frame
} else {
    "{0}_frame{1:D4}_{2}views" -f $Seq, $Frame, ($AutoSources + 1)
}
$sceneDir = Join-Path $OutputBase ("4k4d_scenes\" + $sceneName)
$modalOutputSubdir = "vggt_4k4d_infer/$sceneName"
if ([string]::IsNullOrWhiteSpace($LocalResultsDir)) {
    $LocalResultsDir = Join-Path $OutputBase ("modal_results\" + $sceneName)
}

$exportArgs = @(
    "tools/export_4k4d_scene.py",
    "--dataset-root", $DatasetRoot,
    "--seq", $Seq,
    "--frame", $Frame,
    "--target-camera", $TargetCamera,
    "--output-dir", $sceneDir
)
if ($FullViews) {
    $exportArgs += "--all-cameras"
} else {
    $exportArgs += @("--auto-sources", $AutoSources)
}
if ($OverwriteScene) {
    $exportArgs += "--overwrite"
}

$modalArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $modalScript,
    "-LocalSceneDir", $sceneDir,
    "-OutputSubdir", $modalOutputSubdir,
    "-ModalExe", $modal
)
if ($DryRun) {
    $modalArgs += "-DryRun"
}

Write-Host "[4k4d-vggt] repo_root=$repoRoot"
Write-Host "[4k4d-vggt] python=$python"
Write-Host "[4k4d-vggt] modal=$modal"
Write-Host "[4k4d-vggt] dataset_root=$DatasetRoot"
Write-Host "[4k4d-vggt] scene_name=$sceneName"
Write-Host "[4k4d-vggt] scene_dir=$sceneDir"
Write-Host "[4k4d-vggt] modal_output_subdir=$modalOutputSubdir"
Write-Host "[4k4d-vggt] local_results_dir=$LocalResultsDir"

Push-Location $repoRoot
try {
    if ($DryRun) {
        Write-Host "[4k4d-vggt] dry run export command:"
        Write-Host ($python + " " + ($exportArgs -join " "))
        & powershell @modalArgs
    } else {
        & $python @exportArgs
        & powershell @modalArgs
        if ($PullResults) {
            New-Item -ItemType Directory -Force -Path $LocalResultsDir | Out-Null
            Invoke-ModalVolumeGetWithRetry $modal "vggt-4k4d-output" "/$modalOutputSubdir/summary.json" (Join-Path $LocalResultsDir "summary.json")
            Invoke-ModalVolumeGetWithRetry $modal "vggt-4k4d-output" "/$modalOutputSubdir/previews" $LocalResultsDir
            Invoke-ModalVolumeGetWithRetry $modal "vggt-4k4d-output" "/$modalOutputSubdir/pointcloud" $LocalResultsDir
            Invoke-ModalVolumeGetWithRetry $modal "vggt-4k4d-output" "/$modalOutputSubdir/pointcloud_depth_unprojection" $LocalResultsDir

            $criticalPlys = @(
                @{
                    Remote = "/$modalOutputSubdir/pointcloud/fused_pointcloud_raw.ply"
                    Local = (Join-Path $LocalResultsDir "pointcloud\fused_pointcloud_raw.ply")
                },
                @{
                    Remote = "/$modalOutputSubdir/pointcloud/fused_pointcloud_masked.ply"
                    Local = (Join-Path $LocalResultsDir "pointcloud\fused_pointcloud_masked.ply")
                },
                @{
                    Remote = "/$modalOutputSubdir/pointcloud_depth_unprojection/fused_pointcloud_raw.ply"
                    Local = (Join-Path $LocalResultsDir "pointcloud_depth_unprojection\fused_pointcloud_raw.ply")
                },
                @{
                    Remote = "/$modalOutputSubdir/pointcloud_depth_unprojection/fused_pointcloud_masked.ply"
                    Local = (Join-Path $LocalResultsDir "pointcloud_depth_unprojection\fused_pointcloud_masked.ply")
                }
            )

            foreach ($ply in $criticalPlys) {
                $valid = $false
                for ($attempt = 1; $attempt -le 3; $attempt++) {
                    Invoke-ModalVolumeGetWithRetry $modal "vggt-4k4d-output" $ply.Remote $ply.Local
                    if (Test-AsciiPlyValid $python $repoRoot $ply.Local) {
                        $valid = $true
                        break
                    }
                    Write-Host "[4k4d-vggt] invalid PLY after pull (attempt $attempt/3): $($ply.Local)"
                }
                if (-not $valid) {
                    throw "Failed to pull a complete ASCII PLY after retries: $($ply.Local)"
                }
            }
        }
    }
} finally {
    Pop-Location
}
