$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         Phanty-SDF-Cuda: Starting Web App" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$VenvDir = "venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "[*] Creating virtual environment (venv)..." -ForegroundColor Yellow
    & python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[-] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

Write-Host "[*] Activating virtual environment..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1

Write-Host "[*] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
# Run pip install, and check $LASTEXITCODE since it's an external process
$oldErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue" # allow external command failure checks
& python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Warning: Installing full requirements.txt failed. Installing core dependencies..." -ForegroundColor Yellow
    & python -m pip install numpy trimesh fastapi uvicorn python-multipart
}
$ErrorActionPreference = $oldErrorAction

# Ensure python-multipart is installed (FastAPI needs it for Form/File data)
python -m pip install python-multipart --quiet

# Check if binary exists
$BinaryRelease = "build\Release\dc_cli.exe"
$BinaryRoot = "build\dc_cli.exe"
if (-not (Test-Path $BinaryRelease) -and -not (Test-Path $BinaryRoot)) {
    Write-Host "[!] Warning: dc_cli.exe binary not found. Running build script first..." -ForegroundColor Yellow
    & powershell.exe -ExecutionPolicy Bypass -File .\setup.ps1
}

Write-Host "[+] Starting server at http://localhost:8080 ..." -ForegroundColor Green
Write-Host "[+] Press Ctrl+C to stop the server." -ForegroundColor Yellow
python webapp/server.py
