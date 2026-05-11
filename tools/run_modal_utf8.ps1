param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ModalArgs
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

if ($ModalArgs.Count -eq 0) {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File tools\run_modal_utf8.ps1 <modal args...>"
    Write-Host "Example: powershell -ExecutionPolicy Bypass -File tools\run_modal_utf8.ps1 run modal_surface_research_preflight.py::run_research --help"
    exit 2
}

& modal @ModalArgs
exit $LASTEXITCODE
