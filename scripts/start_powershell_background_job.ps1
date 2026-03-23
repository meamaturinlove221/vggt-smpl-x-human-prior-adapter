param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,
    [string[]]$ScriptArguments = @(),
    [string]$LogPath = "",
    [string]$WorkingDirectory = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

if (-not (Test-Path $ScriptPath)) {
    throw "ScriptPath does not exist: $ScriptPath"
}

$resolvedScript = (Resolve-Path $ScriptPath).Path
if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    $WorkingDirectory = Split-Path -Parent $resolvedScript
}
$resolvedWorkdir = (Resolve-Path $WorkingDirectory).Path

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logDir = Join-Path $resolvedWorkdir "output"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $LogPath = Join-Path $logDir "background_job_${timestamp}.log"
}

$logDirParent = Split-Path -Parent $LogPath
if (-not [string]::IsNullOrWhiteSpace($logDirParent)) {
    New-Item -ItemType Directory -Force -Path $logDirParent | Out-Null
}

$argList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $resolvedScript
) + $ScriptArguments

$proc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $argList `
    -WorkingDirectory $resolvedWorkdir `
    -RedirectStandardOutput $LogPath `
    -RedirectStandardError $LogPath `
    -PassThru

[pscustomobject]@{
    ProcessId = $proc.Id
    LogPath = $LogPath
    ScriptPath = $resolvedScript
    WorkingDirectory = $resolvedWorkdir
} | ConvertTo-Json -Compress
