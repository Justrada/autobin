#!/usr/bin/env python3
"""AutoBin — GUI entry point."""

import os
import sys

# ---------------------------------------------------------------------------
# Fix PATH for macOS .app bundles
# ---------------------------------------------------------------------------
# When launched as a .app, macOS provides a minimal PATH that doesn't include
# Homebrew or other common install locations.  We add them early so every
# shutil.which() / subprocess call in the app can find ffmpeg, ollama, etc.
_EXTRA_PATHS = [
    "/opt/homebrew/bin",      # Homebrew (Apple Silicon)
    "/opt/homebrew/sbin",
    "/usr/local/bin",         # Homebrew (Intel) / Ollama
    "/usr/local/sbin",
]
_current = os.environ.get("PATH", "")
_missing = [p for p in _EXTRA_PATHS if p not in _current.split(":")]
if _missing:
    os.environ["PATH"] = ":".join(_missing) + ":" + _current

from PySide6.QtWidgets import QApplication, QDialog

from gui.main_window import MainWindow
from gui.setup_wizard import SetupWizard, needs_setup


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AutoBin")

    # First-run setup wizard — checks for ffmpeg, Ollama, model
    if needs_setup():
        wizard = SetupWizard()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
