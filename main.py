#!/usr/bin/env python3
"""VLM I-Frame Extractor — GUI entry point."""

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VLM I-Frame Extractor")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
