@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo          Phanty-SDF-Cuda: Starting Web App
echo ========================================================

if not exist venv (
    echo [*] Creating virtual environment (venv)...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [-] Failed to create virtual environment.
        exit /b 1
    )
)

echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [*] Installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [!] Warning: Installing full requirements.txt failed. Installing core dependencies...
    python -m pip install numpy trimesh fastapi uvicorn python-multipart
)

:: Ensure python-multipart is installed for forms/uploads
python -m pip install python-multipart --quiet

:: Check if binary exists
if not exist "build\Release\dc_cli.exe" (
    if not exist "build\dc_cli.exe" (
        echo [!] Warning: dc_cli.exe binary not found. Running build script first...
        powershell.exe -ExecutionPolicy Bypass -File .\setup.ps1
    )
)

echo [+] Starting server at http://localhost:8080 ...
echo [+] Press Ctrl+C to stop the server.
python webapp/server.py
