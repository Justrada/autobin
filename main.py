#!/usr/bin/env python3
"""AutoBin — GUI entry point."""

import sys

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
