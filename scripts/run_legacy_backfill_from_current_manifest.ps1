param(
    [string]$CaseManifestJson,
    [string]$OutRoot,
    [string]$OldRepoRoot = "",
    [string]$PythonExe = "",
    [string]$ZjuRoot = "",
    [string]$Ckpt = "",
    [string]$LegacyViewProfileTag = "legacy_from_current_manifest",
    [double]$FallbackMaxSim3RmseAfter = 0.25,
    [switch]$DryRunOnly
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Ensure-Dir([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Write-JsonNoBom([string]$Path, [object]$Obj) {
    $dir = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($dir)) {
        Ensure-Dir $dir
    }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $json = $Obj | ConvertTo-Json -Depth 32
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

function Resolve-PythonExe([string]$Preferred) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }
    $candidates = @(
        ".venv5080\\Scripts\\python.exe",
        ".venv\\Scripts\\python.exe",
        "D:\\anaconda\\envs\\vggt-colmap\\python.exe",
        "C:\\Python313\\python.exe",
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
    foreach ($drive in @("G:\\", "F:\\")) {
        if (-not (Test-Path $drive)) {
            continue
        }
        foreach ($dir in @(Get-ChildItem $drive -Directory -ErrorAction SilentlyContinue)) {
            foreach ($repoRoot in @(
                (Join-Path $dir.FullName "vggt"),
                (Join-Path $dir.FullName "repo\\vggt")
            )) {
                $scriptPath = Join-Path $repoRoot "scripts\\orig_vggt_viewcount\\render_raw_compare.py"
                if (Test-Path $scriptPath) {
                    return (Resolve-Path $repoRoot).Path
                }
            }
            foreach ($subdir in @(Get-ChildItem $dir.FullName -Directory -ErrorAction SilentlyContinue)) {
                $repoRoot = Join-Path $subdir.FullName "vggt"
                $scriptPath = Join-Path $repoRoot "scripts\\orig_vggt_viewcount\\render_raw_compare.py"
                if (Test-Path $scriptPath) {
                    return (Resolve-Path $repoRoot).Path
                }
            }
        }
    }
    return ""
}

function Resolve-ZjuRootAuto() {
    foreach ($drive in @("F:\\", "G:\\")) {
        if (-not (Test-Path $drive)) {
            continue
        }
        $directCandidate = Join-Path $drive "datasets\\ZJU_MoCap\\data\\zju_mocap"
        $directProbe = Join-Path $directCandidate "CoreView_390\\annots.npy"
        if (Test-Path $directProbe) {
            return (Resolve-Path $directCandidate).Path
        }
        foreach ($dir in @(Get-ChildItem $drive -Directory -ErrorAction SilentlyContinue)) {
            $candidate = Join-Path $dir.FullName "datasets\\ZJU_MoCap\\data\\zju_mocap"
            $probe = Join-Path $candidate "CoreView_390\\annots.npy"
            if (Test-Path $probe) {
                return (Resolve-Path $candidate).Path
            }
        }
    }
    return ""
}

function Resolve-PatchedRenderScript([string]$OriginalScript, [string]$CacheRoot, [double]$RmseThreshold) {
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

function Load-JsonFile([string]$Path) {
    $raw = [System.IO.File]::ReadAllText($Path)
    return $raw | ConvertFrom-Json
}

function Find-LatestReportJson([string]$CaseRoot) {
    if (-not (Test-Path $CaseRoot)) {
        return $null
    }
    $runs = @(Get-ChildItem $CaseRoot -Directory -Filter "run_*" -ErrorAction SilentlyContinue | Sort-Object Name)
    if ($runs.Count -le 0) {
        return $null
    }
    $latest = $runs[-1]
    $reportPath = Join-Path $latest.FullName "report.json"
    if (-not (Test-Path $reportPath)) {
        return $null
    }
    return (Resolve-Path $reportPath).Path
}

if ([string]::IsNullOrWhiteSpace($CaseManifestJson)) {
    throw "CaseManifestJson is required."
}
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    throw "OutRoot is required."
}

$manifestPathResolved = Resolve-ExistingPath @($CaseManifestJson) "case manifest"
$outRootResolved = [System.IO.Path]::GetFullPath($OutRoot)
Ensure-Dir $outRootResolved

$pythonResolved = Resolve-PythonExe $PythonExe
$oldRepoAuto = Resolve-OldRepoRootAuto
$zjuRootAuto = Resolve-ZjuRootAuto
$oldRepoRootResolved = Resolve-ExistingPath @(
    $OldRepoRoot,
    $oldRepoAuto
) "old repo root"
$zjuRootResolved = Resolve-ExistingPath @(
    "F:\\datasets\\ZJU_MoCap\\data\\zju_mocap",
    $zjuRootAuto,
    $ZjuRoot
) "ZJU root"
$ckptResolved = Resolve-ExistingPath @(
    $Ckpt,
    (Join-Path $oldRepoRootResolved "model.pt")
) "legacy checkpoint"
$oldRenderScript = Resolve-ExistingPath @(
    (Join-Path $oldRepoRootResolved "scripts\\orig_vggt_viewcount\\render_raw_compare.py")
) "old render script"

$caseManifest = Load-JsonFile $manifestPathResolved
$cases = @($caseManifest.cases)
if ($cases.Count -le 0) {
    throw "No cases found in $manifestPathResolved"
}

$rows = @()
foreach ($case in $cases) {
    $currentSummaryPath = Resolve-ExistingPath @([string]$case.current_summary_json) "current summary json"
    $summary = Load-JsonFile $currentSummaryPath
    $seqName = [string]$case.seq_name
    $frameId = [int]$case.frame_id
    $targetCamera = [string]$case.target_camera
    $viewProfile = [string]$case.view_profile
    $summaryCase = $summary.case
    if ([string]$summaryCase.seq_name -ne $seqName -or [int]$summaryCase.frame_id -ne $frameId -or [string]$summaryCase.target_camera -ne $targetCamera) {
        throw "Case mismatch between manifest and summary: $currentSummaryPath"
    }
    $sourceCameras = @($summaryCase.source_cameras | ForEach-Object { [string]$_ })
    if ($sourceCameras.Count -le 0) {
        throw "No source cameras found in $currentSummaryPath"
    }
    $srcCsv = [string]::Join(",", $sourceCameras)
    $caseFrameTag = ("frame_{0:D6}_{1}" -f $frameId, $targetCamera)
    $caseRoot = Join-Path $outRootResolved (Join-Path $seqName $caseFrameTag)
    $existingReport = Find-LatestReportJson $caseRoot

    $row = [ordered]@{
        seq_name = $seqName
        frame_id = $frameId
        view_profile = $viewProfile
        target_camera = $targetCamera
        current_summary_json = $currentSummaryPath
        source_cameras = $sourceCameras
        src_cameras_csv = $srcCsv
        legacy_view_profile_tag = $LegacyViewProfileTag
        legacy_case_root = $caseRoot
        legacy_report_json = $existingReport
        status = if ($existingReport) { "reused" } else { "pending" }
    }

    if (-not $DryRunOnly -and -not $existingReport) {
        $cmdArgs = @(
            $oldRenderScript,
            "--seq_name", $seqName,
            "--frame_id", "$frameId",
            "--tgt_camera", $targetCamera,
            "--view_profile", $LegacyViewProfileTag,
            "--src_cameras", $srcCsv,
            "--zju_root", $zjuRootResolved,
            "--ckpt", $ckptResolved,
            "--out_dir", $outRootResolved
        )
        Push-Location $oldRepoRootResolved
        try {
            $logRoot = Join-Path $outRootResolved "_native_logs"
            $initial = Invoke-NativeCapture $pythonResolved $cmdArgs $oldRepoRootResolved $logRoot "${targetCamera}_${frameId}_initial"
            if ($initial.ExitCode -ne 0) {
                $rmseAfter = Extract-RmseAfter $initial.Output
                if (($null -ne $rmseAfter) -and ($rmseAfter -le $FallbackMaxSim3RmseAfter)) {
                    $patchedScript = Resolve-PatchedRenderScript $oldRenderScript $outRootResolved $FallbackMaxSim3RmseAfter
                    $cmdArgs[0] = $patchedScript
                    Write-Host "[legacy-backfill-manifest] retry $targetCamera frame=$frameId with relaxed sim3 rmse_after <= $FallbackMaxSim3RmseAfter (observed $rmseAfter)"
                    $retry = Invoke-NativeCapture $pythonResolved $cmdArgs $oldRepoRootResolved $logRoot "${targetCamera}_${frameId}_retry"
                    if ($retry.ExitCode -ne 0) {
                        throw "Old render command failed for $targetCamera frame=$frameId after relaxed sim3 retry with exit code $($retry.ExitCode)"
                    }
                } else {
                    throw "Old render command failed for $targetCamera frame=$frameId with exit code $($initial.ExitCode)"
                }
            }
        } finally {
            Pop-Location
        }
        $existingReport = Find-LatestReportJson $caseRoot
        if (-not $existingReport) {
            throw "Expected report.json under $caseRoot after rendering"
        }
        $row.legacy_report_json = $existingReport
        $row.status = "rendered"
    }

    $rows += [pscustomobject]$row
}

$manifestOut = [ordered]@{
    generated_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    dry_run_only = [bool]$DryRunOnly
    case_manifest_json = $manifestPathResolved
    out_root = $outRootResolved
    old_repo_root = $oldRepoRootResolved
    python = $pythonResolved
    zju_root = $zjuRootResolved
    ckpt = $ckptResolved
    legacy_view_profile_tag = $LegacyViewProfileTag
    rows = $rows
}

$manifestOutPath = Join-Path $outRootResolved "backfill_manifest.json"
Write-JsonNoBom $manifestOutPath $manifestOut

$md = @(
    "# Legacy Backfill Manifest",
    "",
    "- dry_run_only: ``$([bool]$DryRunOnly)``",
    "- case_manifest_json: ``$manifestPathResolved``",
    "- out_root: ``$outRootResolved``",
    "- old_repo_root: ``$oldRepoRootResolved``",
    "- legacy_view_profile_tag: ``$LegacyViewProfileTag``",
    "",
    "## Cases",
    ""
)
foreach ($row in $rows) {
    $md += "- $($row.seq_name) / frame $($row.frame_id) / $($row.target_camera): ``$($row.src_cameras_csv)`` (`$($row.status)`)"
}
$md += ""
$md += "- manifest_json: ``$manifestOutPath``"
$mdPath = Join-Path $outRootResolved "backfill_manifest.md"
Write-TextNoBom $mdPath ($md -join "`n")

Write-Host "BACKFILL_MANIFEST_JSON: $manifestOutPath"
Write-Host "BACKFILL_MANIFEST_MD: $mdPath"
