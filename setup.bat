@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo           Phanty-SDF-Cuda: Building dc_cli (Windows CMD)
echo ========================================================

where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] cmake not found in PATH. Please install CMake from https://cmake.org/
    exit /b 1
)

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] git not found in PATH. Please install Git.
    exit /b 1
)

echo Updating submodules...
git submodule update --init --recursive

if not exist build mkdir build
cd build

echo Configuring CMake (Release build, x64, Viewer OFF)...
cmake .. -DCMAKE_BUILD_TYPE=Release -DDC_ENABLE_VIEWER=OFF -A x64 %*
if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed. Ensure Visual Studio C++ Build Tools and CUDA Toolkit are installed.
    cd ..
    exit /b %errorlevel%
)

if not defined NUMBER_OF_PROCESSORS set NUMBER_OF_PROCESSORS=4

echo Building dc_cli.exe using %NUMBER_OF_PROCESSORS% parallel jobs...
cmake --build . --target dc_cli --config Release -j %NUMBER_OF_PROCESSORS%
if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    cd ..
    exit /b %errorlevel%
)

echo ========================================================
if exist "Release\dc_cli.exe" (
    echo [SUCCESS] dc_cli.exe built successfully at:
    echo   %CD%\Release\dc_cli.exe
) else if exist "dc_cli.exe" (
    echo [SUCCESS] dc_cli.exe built successfully at:
    echo   %CD%\dc_cli.exe
) else (
    echo [SUCCESS] Build complete. Check the build\ or build\Release\ folder for dc_cli.exe.
)
echo ========================================================
cd ..
