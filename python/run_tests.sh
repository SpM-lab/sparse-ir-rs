#!/bin/bash
# Script to setup build environment and run tests with uv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================"
echo "Cleaning up previous build artifacts..."
echo "======================================"
# Remove copied source files and directories
[ -d "include" ] && rm -rf "include"
[ -d "src" ] && rm -rf "src"
[ -d "cmake" ] && rm -rf "cmake"
[ -f "LICENSE" ] && rm -f "LICENSE"

# Remove .venv directory if it exists
[ -d ".venv" ] && rm -rf ".venv"

# Remove build cache directories
[ -d "_skbuild" ] && rm -rf "_skbuild"
[ -d "dist" ] && rm -rf "dist"
[ -d "*.egg-info" ] && rm -rf *.egg-info 2>/dev/null || true

echo "Cleanup completed."

echo ""
echo "======================================"
echo "Setting up build environment..."
echo "======================================"
python3 setup_build.py

echo ""
echo "======================================"
echo "Running uv sync (with rebuild)..."
echo "======================================"
# Remove any cached build artifacts and force rebuild
uv sync --refresh

echo ""
echo "======================================"
echo "Running tests..."
echo "======================================"
uv run pytest tests/ -v

echo ""
echo "======================================"
echo "All tests completed successfully!"
echo "======================================"

