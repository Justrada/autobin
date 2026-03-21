"""First-run setup wizard for AutoBin.

Checks system dependencies (ffmpeg, Ollama, model) and offers
one-click installation via Homebrew before the main window opens.
"""

from __future__ import annotations

import shutil
import subprocess
import json
from typing import Optional

import requests
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QSpacerItem,
)


# ---------------------------------------------------------------------------
# Dark-theme palette
# ---------------------------------------------------------------------------
_BG = "#1e1e1e"
_CARD_BG = "#2d2d2d"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#999999"
_GREEN = "#4CAF50"
_RED = "#f44336"
_BTN_BG = "#3d3d3d"
_BTN_ACTION = "#4CAF50"
_ORANGE = "#FF9800"

_DIALOG_STYLE = f"""
    QDialog {{
        background-color: {_BG};
    }}
    QLabel {{
        color: {_TEXT};
        background: transparent;
    }}
"""

_CARD_STYLE = f"""
    background-color: {_CARD_BG};
    border-radius: 8px;
    padding: 12px;
"""

_CONTINUE_ENABLED = f"""
    QPushButton {{
        background-color: {_BTN_ACTION};
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 10px 32px;
        font-size: 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: #43A047;
    }}
"""

_CONTINUE_DISABLED = f"""
    QPushButton {{
        background-color: {_BTN_BG};
        color: #666666;
        border: none;
        border-radius: 6px;
        padding: 10px 32px;
        font-size: 14px;
        font-weight: 600;
    }}
"""

_REFRESH_STYLE = f"""
    QPushButton {{
        background-color: {_BTN_BG};
        color: {_TEXT};
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: #4a4a4a;
    }}
"""

_ACTION_BUTTON = f"""
    QPushButton {{
        background-color: {_BTN_ACTION};
        color: #ffffff;
        border: none;
        border-radius: 5px;
        padding: 6px 16px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: #43A047;
    }}
    QPushButton:disabled {{
        background-color: {_BTN_BG};
        color: #666666;
    }}
"""


# ---------------------------------------------------------------------------
# Worker thread for running install subprocesses
# ---------------------------------------------------------------------------
class _InstallWorker(QThread):
    """Runs a shell command in a background thread and emits the result."""

    finished = Signal(bool, str)  # success, output/error

    def __init__(self, command: list[str], detach: bool = False, parent=None):
        super().__init__(parent)
        self._command = command
        self._detach = detach

    def run(self):
        try:
            if self._detach:
                subprocess.Popen(
                    self._command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                # Give the server a moment to start
                import time
                time.sleep(2)
                self.finished.emit(True, "Background process started.")
            else:
                result = subprocess.run(
                    self._command,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    self.finished.emit(True, result.stdout)
                else:
                    self.finished.emit(False, result.stderr or "Unknown error.")
        except FileNotFoundError:
            self.finished.emit(False, "Command not found.")
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Operation timed out.")
        except Exception as exc:
            self.finished.emit(False, str(exc))


# ---------------------------------------------------------------------------
# Lightweight dependency check (no GUI)
# ---------------------------------------------------------------------------
def needs_setup() -> bool:
    """Return *True* if any required dependency is missing.

    This is intentionally fast so the main window can call it on startup
    to decide whether to show the wizard.
    """
    if not shutil.which("ffmpeg"):
        return True
    if not shutil.which("ollama"):
        return True
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            return True
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any("qwen3.5" in m for m in models):
            return True
    except Exception:
        return True
    return False


# ---------------------------------------------------------------------------
# Dependency row helper
# ---------------------------------------------------------------------------
class _DepRow(QWidget):
    """A single dependency row with status icon, label and action button."""

    install_requested = Signal()

    def __init__(
        self,
        name: str,
        description: str,
        action_label: str = "Install",
        parent=None,
    ):
        super().__init__(parent)
        self._passed = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(12)

        # Status icon
        self._icon = QLabel()
        self._icon.setFixedWidth(22)
        self._icon.setAlignment(Qt.AlignCenter)
        self._set_icon(False)
        layout.addWidget(self._icon)

        # Name + description
        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)

        name_label = QLabel(name)
        name_label.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {_TEXT};")
        text_layout.addWidget(name_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet(f"font-size: 11px; color: {_TEXT_DIM};")
        text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, stretch=1)

        # Action button
        self._btn = QPushButton(action_label)
        self._btn.setStyleSheet(_ACTION_BUTTON)
        self._btn.setFixedWidth(100)
        self._btn.setCursor(Qt.PointingHandCursor)
        self._btn.clicked.connect(self.install_requested.emit)
        layout.addWidget(self._btn)

    # -- public helpers --
    def set_passed(self, passed: bool):
        self._passed = passed
        self._set_icon(passed)
        self._btn.setVisible(not passed)

    def is_passed(self) -> bool:
        return self._passed

    def set_busy(self, busy: bool):
        if busy:
            self._btn.setEnabled(False)
            self._btn.setText("Installing\u2026")
        else:
            self._btn.setEnabled(True)
            self._btn.setText("Install")

    # -- internal --
    def _set_icon(self, ok: bool):
        if ok:
            self._icon.setText("\u2714")
            self._icon.setStyleSheet(f"font-size: 16px; color: {_GREEN};")
        else:
            self._icon.setText("\u2718")
            self._icon.setStyleSheet(f"font-size: 16px; color: {_RED};")


# ---------------------------------------------------------------------------
# Setup wizard dialog
# ---------------------------------------------------------------------------
class SetupWizard(QDialog):
    """Modal dialog shown on first run to verify / install dependencies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AutoBin Setup")
        self.setFixedSize(600, 500)
        self.setStyleSheet(_DIALOG_STYLE)
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint
        )

        self._workers: list[_InstallWorker] = []
        self._brew_available: bool = self._check_brew_installed()
        self._build_ui()
        self._run_checks()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(0)

        # Title
        title = QLabel("Welcome to AutoBin")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {_TEXT}; margin-bottom: 4px;"
        )
        root.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "AutoBin needs a few things before it can process your videos."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"font-size: 13px; color: {_TEXT_DIM}; margin-bottom: 20px;"
        )
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Checklist card
        card = QWidget()
        card.setStyleSheet(_CARD_STYLE)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(4, 8, 4, 8)
        card_layout.setSpacing(2)

        self._row_ffmpeg = _DepRow(
            "FFmpeg",
            "Video processing toolkit used to extract I-frames",
            action_label="Install",
        )
        self._row_ollama = _DepRow(
            "Ollama",
            "Local LLM runtime for on-device inference",
            action_label="Install",
        )
        self._row_serve = _DepRow(
            "Ollama Running",
            "The Ollama server must be running on localhost:11434",
            action_label="Start",
        )
        self._row_model = _DepRow(
            "Qwen 3.5 Model",
            "Vision-language model used to classify clips",
            action_label="Pull",
        )

        for row in (
            self._row_ffmpeg,
            self._row_ollama,
            self._row_serve,
            self._row_model,
        ):
            card_layout.addWidget(row)

        root.addWidget(card)

        # Brew warning (hidden by default)
        self._brew_warning = QLabel(
            "Homebrew is required for automatic installs. "
            'Visit <a style="color: #FF9800;" href="https://brew.sh">https://brew.sh</a> '
            "to install it."
        )
        self._brew_warning.setOpenExternalLinks(True)
        self._brew_warning.setWordWrap(True)
        self._brew_warning.setStyleSheet(
            f"font-size: 12px; color: {_ORANGE}; margin-top: 10px; padding: 0 4px;"
        )
        self._brew_warning.setVisible(not self._brew_available)
        root.addWidget(self._brew_warning)

        root.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Bottom button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setStyleSheet(_REFRESH_STYLE)
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.clicked.connect(self._run_checks)
        btn_row.addWidget(self._refresh_btn)

        btn_row.addStretch()

        self._continue_btn = QPushButton("Continue")
        self._continue_btn.setCursor(Qt.PointingHandCursor)
        self._continue_btn.clicked.connect(self.accept)
        self._update_continue_state()
        btn_row.addWidget(self._continue_btn)

        root.addLayout(btn_row)

        # Wire install buttons
        self._row_ffmpeg.install_requested.connect(self._install_ffmpeg)
        self._row_ollama.install_requested.connect(self._install_ollama)
        self._row_serve.install_requested.connect(self._start_ollama)
        self._row_model.install_requested.connect(self._pull_model)

    # ------------------------------------------------------------------
    # Dependency checks
    # ------------------------------------------------------------------
    def _run_checks(self):
        self._row_ffmpeg.set_passed(bool(shutil.which("ffmpeg")))
        self._row_ollama.set_passed(bool(shutil.which("ollama")))

        ollama_running = False
        model_found = False
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                ollama_running = True
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                model_found = any("qwen3.5" in m for m in models)
        except Exception:
            pass

        self._row_serve.set_passed(ollama_running)
        self._row_model.set_passed(model_found)

        self._brew_available = self._check_brew_installed()
        self._brew_warning.setVisible(not self._brew_available)
        self._update_continue_state()

    def _update_continue_state(self):
        ok = self.all_checks_passed()
        self._continue_btn.setEnabled(ok)
        self._continue_btn.setStyleSheet(
            _CONTINUE_ENABLED if ok else _CONTINUE_DISABLED
        )

    def all_checks_passed(self) -> bool:
        """Return True if every dependency check is satisfied."""
        return (
            self._row_ffmpeg.is_passed()
            and self._row_ollama.is_passed()
            and self._row_serve.is_passed()
            and self._row_model.is_passed()
        )

    # ------------------------------------------------------------------
    # Brew helper
    # ------------------------------------------------------------------
    @staticmethod
    def _check_brew_installed() -> bool:
        return shutil.which("brew") is not None

    # ------------------------------------------------------------------
    # Install actions
    # ------------------------------------------------------------------
    def _spawn_worker(
        self,
        command: list[str],
        row: _DepRow,
        detach: bool = False,
    ):
        row.set_busy(True)

        worker = _InstallWorker(command, detach=detach, parent=self)

        def _on_finished(success: bool, msg: str):
            row.set_busy(False)
            self._run_checks()

        worker.finished.connect(_on_finished)
        self._workers.append(worker)
        worker.start()

    def _install_ffmpeg(self):
        if not self._brew_available:
            return
        self._spawn_worker(["brew", "install", "ffmpeg"], self._row_ffmpeg)

    def _install_ollama(self):
        if not self._brew_available:
            return
        self._spawn_worker(["brew", "install", "ollama"], self._row_ollama)

    def _start_ollama(self):
        self._spawn_worker(["ollama", "serve"], self._row_serve, detach=True)

    def _pull_model(self):
        self._spawn_worker(
            ["ollama", "pull", "qwen3.5:latest"], self._row_model
        )
