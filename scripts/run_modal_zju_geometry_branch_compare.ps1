param(
    [string]$ModalExe = "",
    [string[]]$ReportJsons = @(),
    [string]$CheckpointSubpath = "checkpoints/model.pt",
    [string]$ExpName = "zju_geometry_branch_compare_batch",
    [string]$OutputSubdir = "",
    [string]$ZjuSubdir = "zju_mocap",
    [ValidateSet("depth_unproject", "point_map", "auto")]
    [string]$PrimaryBranch = "depth_unproject",
    [string]$Device = "cuda",
    [string]$DType = "bfloat16",
    [double]$ConfPercentile = 25.0,
    [int]$ExportMaxPoints = 100000,
    [int]$RenderMaxPoints = 500000,
    [double]$ZTolerance = 0.02,
    [double]$MinConf = 1e-6,
    [string]$ModalGpu = "A100-40GB",
    [double]$ModalCpu = 8,
    [int]$ModalMemoryMb = 49152,
    [int]$ModalTimeoutSec = 28800,
    [string]$DataVolume = "vggt-zju-data",
    [string]$OutputVolume = "vggt-out",
    [switch]$NoDetach,
    [switch]$SkipPreflight,
    [switch]$StopExistingApps,
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

function Resolve-DefaultReportJsons() {
    $profiles = @("6src_hist", "12src_nested", "23cam_fullset")
    $results = @()
    $driveRoot = "G:\"
    $projectRoots = @()

    foreach ($topDir in @(Get-ChildItem -Path $driveRoot -Directory -ErrorAction SilentlyContinue)) {
        foreach ($childDir in @(Get-ChildItem -Path $topDir.FullName -Directory -ErrorAction SilentlyContinue)) {
            $candidate = Join-Path $childDir.FullName "vggt"
            if (Test-Path (Join-Path $candidate "infer_out")) {
                $projectRoots += (Resolve-Path $candidate).Path
            }
        }
    }

    foreach ($profile in $profiles) {
        $match = $null
        foreach ($root in $projectRoots) {
            $profileDir = Join-Path $root "infer_out\\vggt_raw_viewcount\\$profile\\CoreView_390\\frame_001080_Camera_B5"
            if (-not (Test-Path $profileDir)) {
                continue
            }
            $match = Get-ChildItem -Path $profileDir -Recurse -Filter report.json -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1 -ExpandProperty FullName
            if ($match) {
                break
            }
        }
        if (-not $match) {
            throw "Could not resolve a default report.json for profile $profile."
        }
        $results += $match
    }

    return $results
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$modalEntry = Join-Path $repoRoot "modal_zju_geometry_branch_compare.py"
$preflightScript = Join-Path $repoRoot "scripts\\invoke_modal_zju_preflight.ps1"

if ($ReportJsons.Count -eq 0) {
    $ReportJsons = @(Resolve-DefaultReportJsons)
}

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($OutputSubdir)) {
    $OutputSubdir = "geometry_compare/${runStamp}_${ExpName}"
}

Write-Host "[modal-zju-geom-compare] repo_root=$repoRoot"
Write-Host "[modal-zju-geom-compare] modal=$modal"
Write-Host "[modal-zju-geom-compare] exp_name=$ExpName"
Write-Host "[modal-zju-geom-compare] output_subdir=$OutputSubdir"
Write-Host "[modal-zju-geom-compare] report_count=$($ReportJsons.Count)"
foreach ($reportPath in $ReportJsons) {
    Write-Host "[modal-zju-geom-compare] report_json=$reportPath"
}

if (-not $SkipPreflight) {
    $preflightArgs = @(
        "-ExecutionPolicy", "Bypass", "-File", $preflightScript,
        "-ModalExe", $modal,
        "-AppDescription", "vggt-zju-geometry-branch-compare",
        "-DataVolume", $DataVolume,
        "-OutputVolume", $OutputVolume,
        "-StopRepoProcesses"
    )
    if (-not $NoDetach) {
        $preflightArgs += "-Detach"
    }
    if ($StopExistingApps) {
        $preflightArgs += "-StopExistingApps"
    }
    & powershell @preflightArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Modal preflight failed with exit code $LASTEXITCODE."
    }
}

$env:VGGT_ZJU_GEOM_COMPARE_MODAL_APP_NAME = "vggt-zju-geometry-branch-compare"
$env:VGGT_ZJU_GEOM_COMPARE_DATA_VOLUME = $DataVolume
$env:VGGT_ZJU_GEOM_COMPARE_OUTPUT_VOLUME = $OutputVolume
$env:VGGT_ZJU_GEOM_COMPARE_GPU = $ModalGpu
$env:VGGT_ZJU_GEOM_COMPARE_CPU = [string]$ModalCpu
$env:VGGT_ZJU_GEOM_COMPARE_MEMORY_MB = [string]$ModalMemoryMb
$env:VGGT_ZJU_GEOM_COMPARE_TIMEOUT_SEC = [string]$ModalTimeoutSec

$casePayloads = @()
foreach ($reportPath in $ReportJsons) {
    $resolved = (Resolve-Path $reportPath).Path
    $payloadText = [System.IO.File]::ReadAllText($resolved, [System.Text.Encoding]::UTF8)
    $reportObj = $payloadText | ConvertFrom-Json
    $meta = $reportObj.meta
    $profile = [string]$meta.view_profile
    if ([string]::IsNullOrWhiteSpace($profile)) {
        $profile = "unknown_profile"
    }
    $caseId = "{0}_frame_{1:D6}_{2}_{3}" -f [string]$meta.seq_name, [int]$meta.frame_id, [string]$meta.tgt_camera, $profile
    $payloadBytes = [System.Text.Encoding]::UTF8.GetBytes($payloadText)
    $payloadB64 = [Convert]::ToBase64String($payloadBytes)
    $casePayloads += [ordered]@{
        case_id = $caseId
        report_json_b64 = $payloadB64
    }
}

$cfg = [ordered]@{
    cases = $casePayloads
    zju_subdir = $ZjuSubdir
    checkpoint_subpath = $CheckpointSubpath
    exp_name = $ExpName
    output_subdir = $OutputSubdir
    device = $Device
    dtype = $DType
    conf_percentile = $ConfPercentile
    export_max_points = $ExportMaxPoints
    render_max_points = $RenderMaxPoints
    z_tolerance = $ZTolerance
    min_conf = $MinConf
    primary_branch = $PrimaryBranch
    skip_save_predictions = $true
}
$cfgJsonRaw = $cfg | ConvertTo-Json -Compress -Depth 8
$cfgLocalDir = Join-Path $repoRoot "output\\modal_geom_compare_cfg"
New-Item -ItemType Directory -Force -Path $cfgLocalDir | Out-Null
$cfgLocalPath = Join-Path $cfgLocalDir "${runStamp}_${ExpName}.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($cfgLocalPath, $cfgJsonRaw, $utf8NoBom)
$cfgSubpath = "geometry_compare_cfg/${runStamp}_${ExpName}.json"

$uploadCommand = @(
    $modal,
    "volume",
    "put",
    $OutputVolume,
    $cfgLocalPath,
    $cfgSubpath,
    "--force"
)

$command = @(
    $modal,
    "run"
)
if (-not $NoDetach) {
    $command += "--detach"
}
$command += @(
    "$modalEntry::run_remote_zju_geometry_branch_compare_batch_from_cfg_path",
    "--cfg-subpath", $cfgSubpath
)

if ($DryRun) {
    Write-Host "[modal-zju-geom-compare] dry run command:"
    Write-Host ($uploadCommand -join " ")
    Write-Host ($command -join " ")
    return
}

Push-Location $repoRoot
try {
    Invoke-CheckedCommand -CommandParts $uploadCommand -StepName "geometry compare config upload"
    Invoke-CheckedCommand -CommandParts $command -StepName "geometry compare batch"
} finally {
    Pop-Location
}

Write-Host "[modal-zju-geom-compare] done"
