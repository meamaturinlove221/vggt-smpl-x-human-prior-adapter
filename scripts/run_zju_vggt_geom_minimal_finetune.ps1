param(
    [string]$PythonExe = "",
    [string]$ZjuDir = "",
    [string]$SeqNames = "CoreView_390",
    [string]$GeomSubdir = "vggt_geom",
    [string]$Checkpoint = "",
    [string]$Config = "zju_vggt_geom_minimal",
    [string]$ExpName = "zju_vggt_geom_minimal_local",
    [int]$NumImages = 4,
    [int]$MaxImgPerGpu = 4,
    [int]$AccumSteps = 1,
    [int]$MaxEpochs = 1,
    [double]$LearningRate = 5e-5,
    [int]$LimitTrainBatches = 1,
    [int]$LimitValBatches = 1,
    [int]$NumWorkers = 0,
    [int]$HoldoutStride = 10,
    [string]$CameraSource = "gt",
    [string]$MaskSource = "mask",
    [string]$SourceViewPool = "",
    [double]$MinDepthConf = 0.0,
    [string[]]$ExtraOverrides = @(),
    [switch]$EnableUnprojectGeometry,
    [double]$UnprojectGeometryWeight = 0.2,
    [string]$UnprojectGeometryLossType = "l2",
    [double]$UnprojectGeometryValidRange = 0.98,
    [int]$UnprojectGeometryMinValidPoints = 100,
    [switch]$NoFreezeAggregator,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:USE_LIBUV = "0"
$env:MASTER_ADDR = "127.0.0.1"
$env:MASTER_PORT = "29501"
$env:RANK = "0"
$env:LOCAL_RANK = "0"
$env:WORLD_SIZE = "1"

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

function Resolve-DefaultZjuDir() {
    $gDatasets = "G:\" + [char]0x6570 + [char]0x636E + [char]0x96C6
    $candidates = @(
        (Join-Path $gDatasets "datasets\\ZJU_MoCap\\data\\zju_mocap"),
        "F:\\datasets\\ZJU_MoCap\\data\\zju_mocap"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return ""
}

function Split-SeqNames([string]$SeqNamesText) {
    return $SeqNamesText.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

function Split-GeomSubdirs([string]$GeomSubdirText) {
    return $GeomSubdirText.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

function Get-ZjuGeomRootScore([string]$Root, [string[]]$SeqNamesList, [string[]]$GeomSubdirsList) {
    if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path $Root)) {
        return [pscustomobject]@{
            ValidSubdirCount = 0
            TotalFrameCount = 0
        }
    }
    $validSubdirCount = 0
    $totalFrameCount = 0
    foreach ($seqName in $SeqNamesList) {
        foreach ($geomSubdir in $GeomSubdirsList) {
            $geomDir = Join-Path (Join-Path $Root $seqName) $geomSubdir
            if (-not (Test-Path $geomDir)) {
                continue
            }
            $frameCount = (Get-ChildItem $geomDir -Filter "frame_*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
            if ($frameCount -gt 0) {
                $validSubdirCount += 1
                $totalFrameCount += $frameCount
            }
        }
    }
    return [pscustomobject]@{
        ValidSubdirCount = $validSubdirCount
        TotalFrameCount = $totalFrameCount
    }
}

function Resolve-ZjuGeomRoot([string]$Requested, [string]$SeqNamesText, [string]$GeomSubdir) {
    $seqNamesList = Split-SeqNames $SeqNamesText
    $geomSubdirsList = Split-GeomSubdirs $GeomSubdir
    if ($seqNamesList.Count -eq 0) {
        throw "SeqNames must contain at least one sequence."
    }
    if ($geomSubdirsList.Count -eq 0) {
        throw "GeomSubdir must contain at least one subdir."
    }

    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($Requested)) {
        $candidates += $Requested
    }

    $defaultRoot = Resolve-DefaultZjuDir
    if (-not [string]::IsNullOrWhiteSpace($defaultRoot)) {
        $candidates += $defaultRoot
    }

    $gDatasets = "G:\" + [char]0x6570 + [char]0x636E + [char]0x96C6
    $fallbacks = @(
        (Join-Path $gDatasets "datasets\\ZJU_MoCap\\data\\zju_mocap"),
        "F:\\datasets\\ZJU_MoCap\\data\\zju_mocap"
    )
    $seen = @{}
    $bestCandidate = $null
    $bestScore = $null
    foreach ($candidate in ($candidates + $fallbacks)) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        $key = $candidate.ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true
        $score = Get-ZjuGeomRootScore $candidate $seqNamesList $geomSubdirsList
        if ($score.ValidSubdirCount -le 0) {
            continue
        }
        if ($null -eq $bestCandidate -or
            $score.ValidSubdirCount -gt $bestScore.ValidSubdirCount -or
            ($score.ValidSubdirCount -eq $bestScore.ValidSubdirCount -and $score.TotalFrameCount -gt $bestScore.TotalFrameCount)) {
            $bestCandidate = $candidate
            $bestScore = $score
        }
    }

    if ($null -ne $bestCandidate) {
        return (Resolve-Path $bestCandidate).Path
    }

    throw "Unable to resolve a ZJU root containing frame_*.npz under geom_subdir='$GeomSubdir' for seq_names='$SeqNamesText'."
}

function Resolve-DefaultCheckpoint() {
    $gProjects = "G:\" + [char]0x9879 + [char]0x76EE + [char]0x5907 + [char]0x4EFD
    $projectName = "vggt_" + [char]0x5C0F + [char]0x611F + [char]0x5EA6 + [char]0x4E0D + [char]0x8D77 + [char]0x4F5C + [char]0x7528
    $candidate = Join-Path $gProjects "$projectName\\vggt\\model.pt"
    if (Test-Path $candidate) {
        return (Resolve-Path $candidate).Path
    }

    if (Test-Path $gProjects) {
        $fallback = Get-ChildItem -Path $gProjects -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like "vggt*" } |
            ForEach-Object { Join-Path $_.FullName "vggt\\model.pt" } |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1
        if ($fallback) {
            return (Resolve-Path $fallback).Path
        }
    }

    return ""
}

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Resolve-DefaultCheckpoint
}
$ZjuDir = Resolve-ZjuGeomRoot $ZjuDir $SeqNames $GeomSubdir
if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    throw "Checkpoint is required."
}

$python = Resolve-PythonExe $PythonExe
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launchPath = Join-Path $repoRoot "training\\launch.py"
$zjuDirHydra = $ZjuDir.Replace("\", "/")
$checkpointHydra = $Checkpoint.Replace("\", "/")
$existingPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
    $env:PYTHONPATH = $repoRoot
} else {
    $env:PYTHONPATH = "$repoRoot;$existingPythonPath"
}

$overrides = @(
    "exp_name='$ExpName'",
    "logging.log_dir='logs/$ExpName'",
    "zju_dir='$zjuDirHydra'",
    "zju_seq_names='$SeqNames'",
    "zju_geom_subdir='$GeomSubdir'",
    "zju_camera_source='$CameraSource'",
    "zju_mask_source='$MaskSource'",
    "zju_min_depth_conf=$MinDepthConf",
    "zju_holdout_stride=$HoldoutStride",
    "data.train.common_config.fix_img_num=$NumImages",
    "data.val.common_config.fix_img_num=$NumImages",
    "data.train.common_config.fix_aspect_ratio=1.0",
    "data.val.common_config.fix_aspect_ratio=1.0",
    "data.train.common_config.allow_duplicate_img=False",
    "data.val.common_config.allow_duplicate_img=False",
    "data.train.common_config.load_depth=True",
    "data.val.common_config.load_depth=True",
    "data.train.num_workers=$NumWorkers",
    "data.val.num_workers=$NumWorkers",
    "num_workers=$NumWorkers",
    "max_img_per_gpu=$MaxImgPerGpu",
    "accum_steps=$AccumSteps",
    "max_epochs=$MaxEpochs",
    "optim.optimizer.lr=$LearningRate",
    "limit_train_batches=$LimitTrainBatches",
    "limit_val_batches=$LimitValBatches",
    "checkpoint.resume_checkpoint_path='$checkpointHydra'",
    "distributed.backend='gloo'",
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

if (-not [string]::IsNullOrWhiteSpace($SourceViewPool)) {
    $overrides += "zju_source_view_pool='$SourceViewPool'"
}

if ($EnableUnprojectGeometry) {
    $overrides += "++loss.unproject_geometry.weight=$UnprojectGeometryWeight"
    $overrides += "++loss.unproject_geometry.loss_type='$UnprojectGeometryLossType'"
    $overrides += "++loss.unproject_geometry.valid_range=$UnprojectGeometryValidRange"
    $overrides += "++loss.unproject_geometry.min_valid_points=$UnprojectGeometryMinValidPoints"
}

if ($ExtraOverrides.Count -gt 0) {
    $overrides += $ExtraOverrides
}

$argList = @(
    $launchPath,
    "--config", $Config
) + $overrides

Write-Host "[zju-geom-finetune] repo_root=$repoRoot"
Write-Host "[zju-geom-finetune] python=$python"
Write-Host "[zju-geom-finetune] zju_dir=$ZjuDir"
Write-Host "[zju-geom-finetune] seq_names=$SeqNames"
Write-Host "[zju-geom-finetune] geom_subdir=$GeomSubdir"
if (-not [string]::IsNullOrWhiteSpace($SourceViewPool)) {
    Write-Host "[zju-geom-finetune] source_view_pool=$SourceViewPool"
} else {
    Write-Host "[zju-geom-finetune] source_view_pool=(use config default)"
}
Write-Host "[zju-geom-finetune] checkpoint=$Checkpoint"
Write-Host "[zju-geom-finetune] config=$Config"
Write-Host "[zju-geom-finetune] num_images=$NumImages max_img_per_gpu=$MaxImgPerGpu accum_steps=$AccumSteps"
if ($EnableUnprojectGeometry) {
    Write-Host "[zju-geom-finetune] unproject_geometry=on weight=$UnprojectGeometryWeight loss_type=$UnprojectGeometryLossType valid_range=$UnprojectGeometryValidRange min_valid_points=$UnprojectGeometryMinValidPoints"
}
if ($ExtraOverrides.Count -gt 0) {
    Write-Host "[zju-geom-finetune] extra_overrides=$($ExtraOverrides -join ' | ')"
}
Write-Host "[zju-geom-finetune] distributed.backend=gloo"
Write-Host "[zju-geom-finetune] USE_LIBUV=$env:USE_LIBUV"
Write-Host "[zju-geom-finetune] MASTER_ADDR=$env:MASTER_ADDR MASTER_PORT=$env:MASTER_PORT RANK=$env:RANK LOCAL_RANK=$env:LOCAL_RANK WORLD_SIZE=$env:WORLD_SIZE"
Write-Host "[zju-geom-finetune] PYTHONPATH=$env:PYTHONPATH"

if ($DryRun) {
    Write-Host "[zju-geom-finetune] dry run command:"
    Write-Host "$python $($argList -join ' ')"
    return
}

Push-Location (Join-Path $repoRoot "training")
try {
    & $python @argList
} finally {
    Pop-Location
}
