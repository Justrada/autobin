#!/bin/bash
# Build FrameX.app — standalone macOS application
#
# Usage:
#   ./build_app.sh        Build the standalone .app bundle
#
# Output: dist/FrameX.app

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check pyinstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Clean previous builds
rm -rf build dist

echo "Building FrameX.app..."
pyinstaller FrameX.spec 2>&1

# Check if build succeeded
if [ -d "dist/FrameX.app" ]; then
    echo ""
    echo "========================================="
    echo "  FrameX.app built successfully!"
    echo "  Location: dist/FrameX.app"
    echo "========================================="
    echo ""
    echo "To run:     open dist/FrameX.app"
    echo "To install: cp -R dist/FrameX.app /Applications/"
else
    echo "Build failed. Check the output above for errors."
    exit 1
fi
