<#
.SYNOPSIS
    Phanty-SDF-Cuda: Automatic build script for dc_cli on Windows (PowerShell)
.DESCRIPTION
    Configures and builds the high-performance CUDA Dual Contouring CLI tool (`dc_cli.exe`)
    using CMake and MSVC / Visual Studio Build Tools.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "          Phanty-SDF-Cuda: Building dc_cli (MSVC/CUDA)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Check for CMake
if (-not (Get-Command "cmake" -ErrorAction SilentlyContinue)) {
    Write-Host "[-] Error: 'cmake' not found in PATH. Please install CMake from https://cmake.org/ or via 'winget install Kitware.CMake'." -ForegroundColor Red
    exit 1
}

# Check for Git
if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
    Write-Host "[-] Error: 'git' not found in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[*] Updating Git submodules..." -ForegroundColor Yellow
git submodule update --init --recursive 2>$null

$BuildDir = "build"
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}
Set-Location $BuildDir

Write-Host "[*] Configuring CMake (Release build, x64, Viewer OFF)..." -ForegroundColor Yellow
# If user passed arguments, include them; otherwise use default generator with -A x64
if ($args.Count -gt 0) {
    & cmake .. -DCMAKE_BUILD_TYPE=Release -DDC_ENABLE_VIEWER=OFF @args
} else {
    & cmake .. -DCMAKE_BUILD_TYPE=Release -DDC_ENABLE_VIEWER=OFF -A x64
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] CMake configuration failed. Ensure Visual Studio Build Tools C++ and CUDA Toolkit are installed." -ForegroundColor Red
    exit $LASTEXITCODE
}

$Cores = $env:NUMBER_OF_PROCESSORS
if (-not $Cores) { $Cores = 4 }

Write-Host "[*] Building dc_cli.exe using $Cores parallel jobs..." -ForegroundColor Yellow
& cmake --build . --target dc_cli --config Release -j $Cores

if ($LASTEXITCODE -ne 0) {
    Write-Host "[-] Build failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "========================================================" -ForegroundColor Green
$ExePath1 = Join-Path (Get-Location) "Release\dc_cli.exe"
$ExePath2 = Join-Path (Get-Location) "dc_cli.exe"

if (Test-Path $ExePath1) {
    Write-Host "[+] Success! dc_cli.exe built successfully at:" -ForegroundColor Green
    Write-Host "    $ExePath1" -ForegroundColor White
} elseif (Test-Path $ExePath2) {
    Write-Host "[+] Success! dc_cli.exe built successfully at:" -ForegroundColor Green
    Write-Host "    $ExePath2" -ForegroundColor White
} else {
    Write-Host "[+] Build complete. Check the build/ or build/Release/ folder for dc_cli.exe." -ForegroundColor Green
}
Write-Host "========================================================" -ForegroundColor Green
Set-Location ..
