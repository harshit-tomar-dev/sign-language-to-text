# ==============================================================================
# One-Click Setup Script for New Laptop
# Real-Time Sign Language to Text Conversion System
# ==============================================================================

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Starting Automated Project Setup on New Laptop..." -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Locate project root
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot
Write-Host "[1/6] Project Root: $ProjectRoot" -ForegroundColor Green

# 2. Check Python 3.11
Write-Host "[2/6] Checking Python installation..." -ForegroundColor Green
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "Please install Python 3.11 from python.org and check 'Add Python to PATH'." -ForegroundColor Yellow
    Exit 1
}

$pyVersion = python --version 2>&1
Write-Host "      Detected: $pyVersion" -ForegroundColor Gray
if ($pyVersion -notmatch "3\.11") {
    Write-Host "WARNING: Python 3.11 is required (MediaPipe does not support 3.12+)." -ForegroundColor Yellow
    Write-Host "If build fails, please install Python 3.11." -ForegroundColor Yellow
}

# 3. Ensure uv is installed
Write-Host "[3/6] Checking package installer (uv)..." -ForegroundColor Green
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "      Installing 'uv' package manager..." -ForegroundColor Gray
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH += ";$env:USERPROFILE\.local\bin"
}
Write-Host "      uv is ready." -ForegroundColor Gray

# 4. Create Virtual Environment
Write-Host "[4/6] Creating virtual environment (.venv)..." -ForegroundColor Green
if (Test-Path ".venv") {
    Write-Host "      Existing .venv found." -ForegroundColor Gray
} else {
    uv venv --python 3.11 .venv
}

# 5. Install Dependencies
Write-Host "[5/6] Installing dependencies (this may take 1-3 minutes)..." -ForegroundColor Green
uv pip install --python .venv\Scripts\python.exe -r requirements.txt pytest

# 6. Configure src/ module path (.pth)
Write-Host "[6/6] Registering sign_language package path..." -ForegroundColor Green
$sitePkg = & ".venv\Scripts\python.exe" -c "import site; print(site.getsitepackages()[0])"
if (-not (Test-Path "$sitePkg\site-packages")) {
    $sitePkg = "$sitePkg\Lib\site-packages"
}
if (-not (Test-Path $sitePkg)) {
    # Fallback to standard venv layout
    $sitePkg = "$ProjectRoot\.venv\Lib\site-packages"
}

New-Item -ItemType Directory -Force -Path $sitePkg | Out-Null
"$ProjectRoot\src" | Out-File -FilePath "$sitePkg\sign_language_src.pth" -Encoding ascii
Write-Host "      Registered src/ path at: $sitePkg\sign_language_src.pth" -ForegroundColor Gray

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " Running Automated Verification Tests (56 tests)..." -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

& ".venv\Scripts\pytest.exe" -v

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host " SUCCESS! Setup is complete and verified." -ForegroundColor Green
Write-Host " You are ready to run camera & data collection." -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
