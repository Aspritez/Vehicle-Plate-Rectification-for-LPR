<#
.SYNOPSIS
    One-click launcher for the Thai License Plate Deskew app (local development).

.DESCRIPTION
    1. Creates a .venv virtual environment if one does not already exist.
    2. Installs / updates dependencies from requirements-local.txt (full OpenCV).
    3. Launches the Streamlit app on http://localhost:8501.

    Run from the repository root:
        .\run_local.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root   = $PSScriptRoot
$Venv   = Join-Path $Root ".venv"
$Python = "python"

# ---------------------------------------------------------------------------
# 1. Locate Python
# ---------------------------------------------------------------------------
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found on PATH. Install Python 3.10+ and try again."
    exit 1
}
$PythonVersion = & $Python --version 2>&1
Write-Host "Using $PythonVersion" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# 2. Create virtual environment (skip if it already exists)
# ---------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $Venv "Scripts\Activate.ps1"))) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & $Python -m venv $Venv
}

# ---------------------------------------------------------------------------
# 3. Activate and install/update dependencies
# ---------------------------------------------------------------------------
$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate

Write-Host "Installing dependencies from requirements-local.txt..." -ForegroundColor Yellow
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r (Join-Path $Root "requirements-local.txt")

# ---------------------------------------------------------------------------
# 4. Launch the app
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Starting app at http://localhost:8501  (Ctrl+C to stop)" -ForegroundColor Green
Write-Host ""
streamlit run (Join-Path $Root "app.py")
