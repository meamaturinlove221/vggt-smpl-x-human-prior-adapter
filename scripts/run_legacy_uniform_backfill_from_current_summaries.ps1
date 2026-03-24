param(
    [string]$CurrentSweepRoot = "F:\vggt\vggt-main\output\geometry_view_sweep_zju\round3_12src_uniform_v1",
    [string]$OldRepoRoot = "G:\项目备份\vggt_小感度不起作用\vggt",
    [string]$PythonExe = "",
    [string]$SeqName = "CoreView_390",
    [int]$FrameId = 1080,
    [string]$ViewProfile = "12src_nested",
    [string]$TargetCameras = "Camera_B3,Camera_B5,Camera_B8,Camera_B19",
    [string]$ZjuRoot = "G:\数据集\datasets\ZJU_MoCap\data\zju_mocap",
    [string]$Ckpt = "G:\项目备份\vggt_小感度不起作用\vggt\model.pt",
    [string]$OutRoot = "F:\vggt\vggt-main\output\legacy_uniform_backfill_from_current",
    [string]$LegacyViewProfileTag = "12src_uniform_from_current",
    [double]$FallbackMaxSim3RmseAfter = 0.25,
    [switch]$DryRunOnly
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Resolve-PythonExe([string]$Preferred) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }
    $candidates = @(
        "D:\anaconda\envs\vggt-colmap\python.exe",
        "C:\Python313\python.exe",
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
    return "python"
}

function Resolve-ExistingPath([string[]]$Candidates, [string]$Label) {
    foreach ($candidate in @($Candidates)) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "Could not resolve $Label from candidates: $($Candidates -join ' | ')"
}

function Resolve-OldRepoRootAuto() {
    foreach ($drive in @("G:\", "F:\")) {
        if (-not (Test-Path $drive)) {
            continue
        }
        foreach ($dir in @(Get-ChildItem $drive -Directory -ErrorAction SilentlyContinue)) {
            $directRepoRoot = Join-Path $dir.FullName "vggt"
            foreach ($repoRoot in @($directRepoRoot)) {
                $scriptPath = Join-Path $repoRoot "scripts\orig_vggt_viewcount\render_raw_compare.py"
                if (Test-Path $scriptPath) {
                    return (Resolve-Path $repoRoot).Path
                }
            }
            foreach ($subdir in @(Get-ChildItem $dir.FullName -Directory -ErrorAction SilentlyContinue)) {
                $repoRoot = Join-Path $subdir.FullName "vggt"
                $scriptPath = Join-Path $repoRoot "scripts\orig_vggt_viewcount\render_raw_compare.py"
                if (Test-Path $scriptPath) {
                    return (Resolve-Path $repoRoot).Path
                }
            }
        }
    }
    return ""
}

function Resolve-ZjuRootAuto() {
    foreach ($drive in @("F:\", "G:\")) {
        if (-not (Test-Path $drive)) {
            continue
        }
        $directCandidate = Join-Path $drive "datasets\ZJU_MoCap\data\zju_mocap"
        $directProbe = Join-Path $directCandidate "CoreView_390\annots.npy"
        if (Test-Path $directProbe) {
            return (Resolve-Path $directCandidate).Path
        }
        foreach ($dir in @(Get-ChildItem $drive -Directory -ErrorAction SilentlyContinue)) {
            $zjuRoot = Join-Path $dir.FullName "datasets\ZJU_MoCap\data\zju_mocap"
            $probe = Join-Path $zjuRoot "CoreView_390\annots.npy"
            if (Test-Path $probe) {
                return (Resolve-Path $zjuRoot).Path
            }
        }
    }
    return ""
}

function Ensure-Dir([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        Ensure-Dir $dir
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($Path, $json, $enc)
}

function Write-TextNoBom([string]$Path, [string]$Text) {
    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        Ensure-Dir $dir
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $enc)
}

function Resolve-PatchedRenderScript([string]$OriginalScript, [string]$OutRootPath, [double]$RmseThreshold) {
    $patchedDir = Split-Path -Parent $OriginalScript
    $patchedName = "render_raw_compare_rmse_{0}.py" -f (($RmseThreshold.ToString("0.00")).Replace(".", "p"))
    $patchedPath = Join-Path $patchedDir $patchedName
    if (Test-Path $patchedPath) {
        return (Resolve-Path $patchedPath).Path
    }
    $text = [System.IO.File]::ReadAllText($OriginalScript)
    $needle = "if rmse_after > 0.15:"
    if (-not $text.Contains($needle)) {
        throw "Could not find sim3 rmse guard in $OriginalScript"
    }
    $replacement = "if rmse_after > ${RmseThreshold}:"
    $text = $text.Replace($needle, $replacement)
    Write-TextNoBom $patchedPath $text
    return (Resolve-Path $patchedPath).Path
}

function Extract-RmseAfter([string]$Text) {
    $match = [regex]::Match($Text, "sim3 alignment rmse_after too high: ([0-9.]+)")
    if (-not $match.Success) {
        return $null
    }
    return [double]$match.Groups[1].Value
}

function Invoke-NativeCapture([string]$Exe, [string[]]$ArgList, [string]$WorkingDir, [string]$LogRoot, [string]$Tag) {
    Ensure-Dir $LogRoot
    $safeTag = ($Tag -replace "[^A-Za-z0-9_.-]", "_")
    $stdoutPath = Join-Path $LogRoot "${safeTag}_stdout.log"
    $stderrPath = Join-Path $LogRoot "${safeTag}_stderr.log"
    foreach ($path in @($stdoutPath, $stderrPath)) {
        if (Test-Path $path) {
            Remove-Item -Force $path
        }
    }
    $proc = Start-Process `
        -FilePath $Exe `
        -ArgumentList $ArgList `
        -WorkingDirectory $WorkingDir `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
    $chunks = @()
    foreach ($path in @($stdoutPath, $stderrPath)) {
        if (Test-Path $path) {
            $text = (Get-Content $path -Raw)
            if (-not [string]::IsNullOrWhiteSpace($text)) {
                $chunks += $text.TrimEnd()
            }
        }
    }
    $combined = [string]::Join("`n", $chunks)
    if (-not [string]::IsNullOrWhiteSpace($combined)) {
        Write-Host $combined
    }
    return [pscustomobject]@{
        ExitCode = [int]$proc.ExitCode
        Output = [string]$combined
    }
}

$pythonResolved = Resolve-PythonExe $PythonExe
$oldRepoAuto = Resolve-OldRepoRootAuto
$zjuRootAuto = Resolve-ZjuRootAuto
$oldRepoRootResolved = Resolve-ExistingPath @(
    $OldRepoRoot,
    $oldRepoAuto
) "old repo root"
$zjuRootResolved = Resolve-ExistingPath @(
    "F:\datasets\ZJU_MoCap\data\zju_mocap",
    $zjuRootAuto,
    $ZjuRoot
) "ZJU root"
$ckptResolved = Resolve-ExistingPath @(
    $Ckpt,
    (Join-Path $oldRepoRootResolved "model.pt")
) "legacy checkpoint"
$oldRenderScript = Resolve-ExistingPath @(
    (Join-Path $oldRepoRootResolved "scripts\orig_vggt_viewcount\render_raw_compare.py")
) "old render script"

Ensure-Dir $OutRoot
$targets = @($TargetCameras -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$rows = @()

foreach ($target in $targets) {
    $frameTag = ("frame_{0:D6}_{1}" -f $FrameId, $target)
    $summaryPath = Join-Path $CurrentSweepRoot (Join-Path $ViewProfile (Join-Path $frameTag "summary.json"))
    if (-not (Test-Path $summaryPath)) {
        throw "Missing current summary: $summaryPath"
    }
    $summaryMiniJson = & $pythonResolved -c "import json,sys; p=json.load(open(sys.argv[1], 'r', encoding='utf-8')); print(json.dumps({'source_cameras': p['case']['source_cameras']}, ensure_ascii=False))" $summaryPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract source cameras from $summaryPath"
    }
    $summaryMini = $summaryMiniJson | ConvertFrom-Json
    $srcCameras = @($summaryMini.source_cameras | ForEach-Object { [string]$_ })
    if ($srcCameras.Count -le 0) {
        throw "No source cameras found in $summaryPath"
    }
    $srcCsv = [string]::Join(",", $srcCameras)
    $caseOutRoot = Join-Path $OutRoot $LegacyViewProfileTag
    $cmdArgs = @(
        $oldRenderScript,
        "--seq_name", $SeqName,
        "--frame_id", "$FrameId",
        "--tgt_camera", $target,
        "--view_profile", $LegacyViewProfileTag,
        "--src_cameras", $srcCsv,
        "--zju_root", $zjuRootResolved,
        "--ckpt", $ckptResolved,
        "--out_dir", $caseOutRoot
    )
    $row = [ordered]@{
        target_camera = $target
        current_summary_json = $summaryPath
        source_cameras = $srcCameras
        src_cameras_csv = $srcCsv
        command_preview = (@($pythonResolved) + $cmdArgs) -join " "
        legacy_view_profile_tag = $LegacyViewProfileTag
    }
    $rows += [pscustomobject]$row

    if (-not $DryRunOnly) {
        Push-Location $oldRepoRootResolved
        try {
            $logRoot = Join-Path $OutRoot "_native_logs"
            $result = Invoke-NativeCapture $pythonResolved $cmdArgs $oldRepoRootResolved $logRoot "${target}_initial"
            if ($result.ExitCode -ne 0) {
                $rmseAfter = Extract-RmseAfter $result.Output
                if (($null -ne $rmseAfter) -and ($rmseAfter -le $FallbackMaxSim3RmseAfter)) {
                    $patchedScript = Resolve-PatchedRenderScript $oldRenderScript $OutRoot $FallbackMaxSim3RmseAfter
                    $cmdArgs[0] = $patchedScript
                    Write-Host "[legacy-backfill] retry $target with relaxed sim3 rmse_after <= $FallbackMaxSim3RmseAfter (observed $rmseAfter)"
                    $retryResult = Invoke-NativeCapture $pythonResolved $cmdArgs $oldRepoRootResolved $logRoot "${target}_retry"
                    if ($retryResult.ExitCode -ne 0) {
                        throw "Old render command failed for $target after relaxed sim3 retry with exit code $($retryResult.ExitCode)"
                    }
                } else {
                    throw "Old render command failed for $target with exit code $($result.ExitCode)"
                }
            }
        } finally {
            Pop-Location
        }
    }
}

$manifest = [ordered]@{
    generated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    dry_run_only = [bool]$DryRunOnly
    current_sweep_root = $CurrentSweepRoot
    old_repo_root = $oldRepoRootResolved
    python = $pythonResolved
    seq_name = $SeqName
    frame_id = $FrameId
    view_profile = $ViewProfile
    legacy_view_profile_tag = $LegacyViewProfileTag
    zju_root = $zjuRootResolved
    ckpt = $ckptResolved
    out_root = $OutRoot
    targets = $targets
    rows = $rows
}

$manifestPath = Join-Path $OutRoot "backfill_manifest.json"
Write-JsonNoBom $manifestPath $manifest

$md = @(
    "# Legacy Uniform Backfill Manifest",
    "",
    "- dry_run_only: ``$([bool]$DryRunOnly)``",
    "- current_sweep_root: ``$CurrentSweepRoot``",
    "- old_repo_root: ``$oldRepoRootResolved``",
    "- seq_name: ``$SeqName``",
    "- frame_id: ``$FrameId``",
    "- view_profile: ``$ViewProfile``",
    "- legacy_view_profile_tag: ``$LegacyViewProfileTag``",
    "- targets: ``$([string]::Join(',', $targets))``",
    "",
    "## Cases",
    ""
)
foreach ($row in $rows) {
    $md += "- $($row.target_camera): ``$($row.src_cameras_csv)``"
}
$md += ""
$md += "- manifest_json: ``$manifestPath``"
Write-TextNoBom (Join-Path $OutRoot "backfill_manifest.md") ($md -join "`n")

Write-Host "BACKFILL_MANIFEST_JSON: $manifestPath"
Write-Host "BACKFILL_MANIFEST_MD: $(Join-Path $OutRoot 'backfill_manifest.md')"
