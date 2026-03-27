param(
    [string]$ModalExe = "",
    [string]$AppDescription = "vggt-zju-geometry-minimal-finetune",
    [string]$DataVolume = "vggt-zju-data",
    [string]$OutputVolume = "vggt-out",
    [string]$LocalCheckpoint = "",
    [double]$MaxSafeLocalCheckpointGb = 1.0,
    [double]$WarnLargeArtifactGb = 2.0,
    [double]$MinFreeMemoryGb = 12.0,
    [int]$MinStaleDetachMinutes = 5,
    [switch]$Detach,
    [switch]$AllowLargeLocalUpload,
    [switch]$StopExistingApps,
    [switch]$StopRepoProcesses
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

    $command = Get-Command modal -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Path
    }

    throw "Unable to resolve modal.exe."
}

function Get-RepoProcesses([string]$RepoRoot) {
    $escapedRepo = [regex]::Escape($RepoRoot)
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID `
            -and $_.Name -match "powershell|python|modal" `
            -and $_.CommandLine `
            -and $_.CommandLine -match $escapedRepo `
            -and $_.CommandLine -notmatch "autopep8|isort|invoke_modal_zju_preflight\.ps1|run_zju_post_v9_residual_cluster_nightly\.py|run_zju_source_policy_rawpool_local_nightly\.py|run_zju_source_policy_rawpool_long_gate\.py|run_zju_source_policy_rawpool_overnight_watch\.py|run_zju_source_policy_rawpool_guard_daemon\.py|run_zju_source_policy_research_candidate\.py|arm_zju_source_policy_approved_problem\.py"
    } | Select-Object Name, ProcessId, ParentProcessId, CreationDate, CommandLine
}

function Get-ProcessTreeIds([int]$RootPid) {
    $children = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $RootPid })
    $ids = @()
    foreach ($child in $children) {
        $ids += Get-ProcessTreeIds -RootPid $child.ProcessId
        $ids += $child.ProcessId
    }
    return $ids
}

function Stop-ProcessTree([int]$RootPid) {
    $allIds = @((Get-ProcessTreeIds -RootPid $RootPid) + $RootPid) |
        Sort-Object -Descending -Unique
    foreach ($procId in $allIds) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "[modal-preflight] stopped local process pid=$procId"
        } catch {
            # best-effort cleanup; ignore already-exited processes
        }
    }
}

function Get-ProcessAgeMinutes($Proc) {
    try {
        if (-not $Proc -or [string]::IsNullOrWhiteSpace([string]$Proc.CreationDate)) {
            return $null
        }
        $createdAt = [Management.ManagementDateTimeConverter]::ToDateTime([string]$Proc.CreationDate)
        return ((Get-Date) - $createdAt).TotalMinutes
    } catch {
        return $null
    }
}

function Test-StaleDetachedLauncher($Proc, [string]$RepoRoot, [int]$MinAgeMinutes) {
    if (-not $Proc -or [string]::IsNullOrWhiteSpace($Proc.CommandLine)) {
        return $false
    }

    $cmd = [string]$Proc.CommandLine
    $isRepoLauncher = $cmd -match [regex]::Escape($RepoRoot) `
        -and $cmd -match "modal_zju_geometry_minimal_finetune\.py|run_modal_zju_geometry_minimal_finetune\.ps1|run_modal_zju_unproject_geometry_ablation_pair\.ps1"
    $usesDetach = $cmd -match "(^|[\s`"'])--detach($|[\s`"'])" -or $cmd -match "(^|[\s`"'])-Detach($|[\s`"'])"
    if (-not ($isRepoLauncher -and $usesDetach)) {
        return $false
    }

    $ageMinutes = Get-ProcessAgeMinutes $Proc
    return $ageMinutes -ne $null -and $ageMinutes -ge $MinAgeMinutes
}

function Get-LargeArtifacts([string]$RepoRoot, [double]$WarnGb) {
    $patterns = @("*.pt", "*.pth", "*.ckpt", "*.npz", "*.ply", "*.obj", "*.glb", "*.zip")
    Get-ChildItem -Path $RepoRoot -Recurse -File -Include $patterns -ErrorAction SilentlyContinue |
        Where-Object { $_.Length -ge ($WarnGb * 1GB) } |
        Sort-Object Length -Descending |
        Select-Object -First 12 FullName, @{N = "GB"; E = { [math]::Round($_.Length / 1GB, 3) } }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$os = Get-CimInstance Win32_OperatingSystem
$freeMemoryGb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$usedMemoryGb = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 2)

Write-Host "[modal-preflight] repo_root=$repoRoot"
Write-Host "[modal-preflight] modal=$modal"
Write-Host "[modal-preflight] free_memory_gb=$freeMemoryGb used_memory_gb=$usedMemoryGb"

if ($freeMemoryGb -lt $MinFreeMemoryGb) {
    throw "Free system memory is only ${freeMemoryGb}GB. Abort before launching cloud work."
}

$repoProcesses = @(Get-RepoProcesses $repoRoot)
if ($repoProcesses.Count -gt 0) {
    $staleDetachLaunchers = @(
        $repoProcesses | Where-Object {
            Test-StaleDetachedLauncher -Proc $_ -RepoRoot $repoRoot -MinAgeMinutes $MinStaleDetachMinutes
        }
    )

    if ($StopRepoProcesses -and $staleDetachLaunchers.Count -gt 0) {
        Write-Host "[modal-preflight] stopping stale detached repo launchers (older than ${MinStaleDetachMinutes} min):"
        $staleDetachLaunchers | Format-Table -AutoSize | Out-String | Write-Host
        foreach ($proc in ($staleDetachLaunchers | Sort-Object ProcessId -Unique)) {
            Stop-ProcessTree -RootPid $proc.ProcessId
        }
        Start-Sleep -Seconds 2
        $repoProcesses = @(Get-RepoProcesses $repoRoot)
    }
}

if ($repoProcesses.Count -gt 0) {
    $repoProcesses = @(
        $repoProcesses | Where-Object {
            $_.Name -notmatch "powershell"
        }
    )
}

if ($repoProcesses.Count -gt 0) {
    $repoProcesses | Format-Table -AutoSize | Out-String | Write-Host
    throw "Detected repo-scoped local powershell/python/modal processes. Stop them before launching cloud work."
}

$largeArtifacts = @(Get-LargeArtifacts $repoRoot $WarnLargeArtifactGb)
if ($largeArtifacts.Count -gt 0) {
    Write-Host "[modal-preflight] large local artifacts detected (safe if ignored, dangerous if uploaded):"
    $largeArtifacts | Format-Table -AutoSize | Out-String | Write-Host
}

if (-not [string]::IsNullOrWhiteSpace($LocalCheckpoint)) {
    $resolvedCheckpoint = (Resolve-Path $LocalCheckpoint).Path
    $checkpointItem = Get-Item $resolvedCheckpoint
    $checkpointGb = [math]::Round($checkpointItem.Length / 1GB, 3)
    Write-Host "[modal-preflight] local_checkpoint=$resolvedCheckpoint size_gb=$checkpointGb"
    if ($checkpointGb -gt $MaxSafeLocalCheckpointGb -and -not $AllowLargeLocalUpload) {
        throw "Refusing to upload a ${checkpointGb}GB local checkpoint. Leave -LocalCheckpoint empty or pass -AllowLargeLocalUpload explicitly."
    }
}

$profileName = & $modal profile current
Write-Host "[modal-preflight] modal_profile=$profileName"

$appJson = & $modal app list --json
$apps = @()
if (-not [string]::IsNullOrWhiteSpace($appJson)) {
    $apps = @($appJson | ConvertFrom-Json)
}

$activeApps = @(
    $apps | Where-Object {
        $_.Description -eq $AppDescription -and $_.'State' -ne "stopped"
    }
)

if ($activeApps.Count -gt 0) {
    Write-Host "[modal-preflight] active matching apps detected:"
    $activeApps | Format-Table -AutoSize | Out-String | Write-Host
    if ($StopExistingApps) {
        foreach ($app in $activeApps) {
            Write-Host "[modal-preflight] stopping app $($app.'App ID')"
            & $modal app stop $app.'App ID'
        }
    } else {
        throw "Found active Modal apps for $AppDescription. Re-run with -StopExistingApps or stop them manually."
    }
}

$weightProbe = $null
try {
    $weightProbe = & $modal volume ls $OutputVolume /weights
} catch {
    $weightProbe = $null
}

if ($weightProbe) {
    Write-Host "[modal-preflight] output volume has /weights entries."
} else {
    Write-Host "[modal-preflight] warning: could not confirm ${OutputVolume}:/weights"
}

if ($Detach) {
    Write-Host "[modal-preflight] detach enabled"
} else {
    Write-Host "[modal-preflight] detach disabled"
}

Write-Host "[modal-preflight] checks passed"
