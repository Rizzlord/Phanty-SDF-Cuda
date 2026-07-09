#!/usr/bin/env bash
# Phanty-SDF-Cuda: Automatic build script for dc_cli (Linux / macOS)
set -e

echo "========================================================"
echo "          Phanty-SDF-Cuda: Building dc_cli"
echo "========================================================"

if ! command -v cmake &> /dev/null; then
    echo "❌ Error: cmake not found. Please install cmake (e.g. 'sudo apt install cmake' or 'brew install cmake')."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Error: git not found. Please install git."
    exit 1
fi

echo "🔍 Checking submodules..."
git submodule update --init --recursive 2>/dev/null || true

BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "⚙️  Configuring CMake (Release build, Viewer OFF for clean CLI generation)..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DDC_ENABLE_VIEWER=OFF "$@"

CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "🚀 Building dc_cli using $CORES parallel jobs..."
cmake --build . --target dc_cli --config Release -j"$CORES"

echo "========================================================"
if [ -f "dc_cli" ] || [ -f "Release/dc_cli" ]; [ -f "dc_cli.exe" ] || [ -f "Release/dc_cli.exe" ]; then
    echo "✅ Success! dc_cli binary built successfully."
    echo "📍 Binary location: $(pwd)/dc_cli"
    echo "========================================================"
else
    echo "✅ Build complete. Check $(pwd) for the dc_cli binary."
fi
