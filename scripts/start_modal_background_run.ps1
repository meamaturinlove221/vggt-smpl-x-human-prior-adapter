param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("single", "pair")]
    [string]$Mode,
    [string]$LogPath = "",
    [string]$WorkingDirectory = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ForwardArgs
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    $WorkingDirectory = $repoRoot
}
$resolvedWorkdir = (Resolve-Path $WorkingDirectory).Path

$targetScript = switch ($Mode) {
    "single" { Join-Path $repoRoot "scripts\\run_modal_zju_geometry_minimal_finetune.ps1" }
    "pair" { Join-Path $repoRoot "scripts\\run_modal_zju_unproject_geometry_ablation_pair.ps1" }
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logDir = Join-Path $repoRoot "output"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $LogPath = Join-Path $logDir "modal_${Mode}_${timestamp}.out.log"
}

$logParent = Split-Path -Parent $LogPath
if (-not [string]::IsNullOrWhiteSpace($logParent)) {
    New-Item -ItemType Directory -Force -Path $logParent | Out-Null
}
$errorLogPath = if ($LogPath.EndsWith(".out.log")) {
    $LogPath.Substring(0, $LogPath.Length - ".out.log".Length) + ".err.log"
} else {
    $LogPath + ".err"
}

$argList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $targetScript
) + $ForwardArgs

$proc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $argList `
    -WorkingDirectory $resolvedWorkdir `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError $errorLogPath `
    -PassThru

[pscustomobject]@{
    ProcessId = $proc.Id
    Mode = $Mode
    StdoutLogPath = $LogPath
    StderrLogPath = $errorLogPath
    TargetScript = $targetScript
    WorkingDirectory = $resolvedWorkdir
    ForwardArgs = $ForwardArgs
} | ConvertTo-Json -Compress -Depth 4
