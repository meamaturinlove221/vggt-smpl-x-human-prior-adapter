param(
    [string]$PythonExe = "",
    [string]$OutputJson = "output\\co3d_candidates_targeted.json",
    [int]$MaxDepth = 5
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Resolve-PythonExe $PythonExe

$gData = "G:\" + [char]0x6570 + [char]0x636E + [char]0x96C6
$gProjects = "G:\" + [char]0x9879 + [char]0x76EE + [char]0x5907 + [char]0x4EFD

$roots = @(
    $gData,
    $gProjects,
    "D:\model",
    "D:\BaiduNetdisk"
) | Where-Object { Test-Path $_ }

$rootsJsonPath = Join-Path $repoRoot "output\co3d_scan_roots.json"
$roots | ConvertTo-Json -Compress | Set-Content -Path $rootsJsonPath -Encoding utf8

Write-Host "[find-co3d] repo_root=$repoRoot"
Write-Host "[find-co3d] python=$python"
Write-Host "[find-co3d] roots=$($roots -join ', ')"

Push-Location $repoRoot
try {
    & $python .\scripts\find_co3d_candidates.py --roots-file $rootsJsonPath --max-depth $MaxDepth --output-json $OutputJson
} finally {
    Pop-Location
}
