#!/bin/bash
# Build AutoBin.app — standalone macOS application
#
# Usage:
#   ./build_app.sh        Build the standalone .app bundle
#
# Output: dist/AutoBin.app

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

echo "Building AutoBin.app..."
pyinstaller AutoBin.spec 2>&1

# Check if build succeeded
if [ -d "dist/AutoBin.app" ]; then
    echo ""
    echo "========================================="
    echo "  AutoBin.app built successfully!"
    echo "  Location: dist/AutoBin.app"
    echo "========================================="
    echo ""
    echo "To run:     open dist/AutoBin.app"
    echo "To install: cp -R dist/AutoBin.app /Applications/"
else
    echo "Build failed. Check the output above for errors."
    exit 1
fi
