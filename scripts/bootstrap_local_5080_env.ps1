param(
    [string]$PythonExe = "C:\Users\WINDOWS\AppData\Local\Programs\Python\Python313\python.exe",
    [string]$EnvDir = ".venv5080",
    [switch]$InstallDemoDeps,
    [switch]$InstallTrainingDeps,
    [switch]$InstallModalDeps
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envPath = Join-Path $repoRoot $EnvDir

Write-Host "[bootstrap-5080] repo_root=$repoRoot"
Write-Host "[bootstrap-5080] python=$PythonExe"
Write-Host "[bootstrap-5080] env_dir=$envPath"

if (-not (Test-Path $envPath)) {
    & $PythonExe -m venv $envPath
}

$venvPython = Join-Path $envPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtualenv python missing: $venvPython"
}

Push-Location $repoRoot
try {
    & $venvPython -m pip install --upgrade pip setuptools wheel
    & $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    & $venvPython -m pip install huggingface_hub einops safetensors

    if ($InstallDemoDeps) {
        & $venvPython -m pip install -r requirements_demo.txt
    }

    if ($InstallTrainingDeps) {
        & $venvPython -m pip install -r requirements_training.txt
    }

    if ($InstallModalDeps) {
        & $venvPython -m pip install modal
    }

    @'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("arch_list", torch.cuda.get_arch_list())
'@ | & $venvPython -
} finally {
    Pop-Location
}
