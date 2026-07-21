# Creates a local virtual environment (.venv) and installs requirements.txt into it.
# Usage: powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1

$ErrorActionPreference = "Stop"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Error "Python was not found on PATH. Install Python 3 and try again."
    exit 1
}

if (-not (Test-Path ".venv")) {
    & $pythonCmd.Source -m venv .venv
}

$venvPython = ".venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host ""
Write-Host "Virtual environment ready. Activate it with:"
Write-Host "  .venv\Scripts\Activate.ps1"
