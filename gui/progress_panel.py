"""Progress panel: airplane-console status dashboard with step indicators."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

_DARK_BG = "#1a1a2e"
_PANEL_BG = "#16213e"
_CARD_BG = "#0f3460"
_TEXT = "#e0e0e0"
_TEXT_DIM = "#8899aa"
_BORDER = "#1a3a5c"

_COLOR_IDLE = "#555566"
_COLOR_WORKING = "#4488ff"
_COLOR_DONE = "#44cc66"
_COLOR_ERROR = "#ee4444"

_INDICATOR_SIZE = 14

_PANEL_QSS = f"""
    QWidget#ProgressPanel {{
        background: {_DARK_BG};
    }}
    QLabel {{
        color: {_TEXT};
        font-size: 13px;
    }}
    QLabel#heading {{
        color: {_TEXT};
        font-weight: bold;
        font-size: 14px;
    }}
    QLabel#dimLabel {{
        color: {_TEXT_DIM};
        font-size: 12px;
    }}
    QProgressBar {{
        background: {_PANEL_BG};
        border: 1px solid {_BORDER};
        border-radius: 3px;
        height: 10px;
        text-align: center;
        font-size: 10px;
        color: {_TEXT_DIM};
    }}
    QProgressBar::chunk {{
        background: {_COLOR_WORKING};
        border-radius: 2px;
    }}
    QPlainTextEdit {{
        background: {_PANEL_BG};
        color: {_TEXT_DIM};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        font-family: "SF Mono", "Menlo", "Consolas", monospace;
        font-size: 11px;
    }}
    QPushButton#logToggle {{
        background: {_CARD_BG};
        color: {_TEXT_DIM};
        border: 1px solid {_BORDER};
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 11px;
    }}
    QPushButton#logToggle:hover {{
        background: {_PANEL_BG};
        color: {_TEXT};
    }}
    QTableWidget {{
        background: {_PANEL_BG};
        color: {_TEXT};
        border: 1px solid {_BORDER};
        gridline-color: {_BORDER};
        font-size: 12px;
        selection-background-color: {_CARD_BG};
    }}
    QHeaderView::section {{
        background: {_CARD_BG};
        color: {_TEXT};
        border: 1px solid {_BORDER};
        padding: 4px;
        font-size: 12px;
        font-weight: bold;
    }}
"""


# ---------------------------------------------------------------------------
# StatusLight — one row in the console grid
# ---------------------------------------------------------------------------

class StatusLight(QWidget):
    """Single pipeline-step indicator with dot, label, timer, and optional bar."""

    def __init__(self, step_name: str, parent=None):
        super().__init__(parent)
        self._step_name = step_name
        self._state = "idle"  # idle | working | done | error
        self._start_time: float | None = None
        self._elapsed: float = 0.0
        self._pulse_on = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        # Colored dot
        self._dot = QLabel()
        self._dot.setFixedSize(_INDICATOR_SIZE, _INDICATOR_SIZE)
        self._apply_dot_style(_COLOR_IDLE)
        layout.addWidget(self._dot)

        # Step name
        self._name_label = QLabel(step_name)
        self._name_label.setFixedWidth(180)
        layout.addWidget(self._name_label)

        # Timer label
        self._timer_label = QLabel("--:--")
        self._timer_label.setObjectName("dimLabel")
        self._timer_label.setFixedWidth(60)
        self._timer_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._timer_label)

        # Mini progress bar (hidden by default)
        self._bar = QProgressBar()
        self._bar.setFixedHeight(8)
        self._bar.setTextVisible(False)
        self._bar.setMaximumWidth(120)
        self._bar.hide()
        layout.addWidget(self._bar)

        layout.addStretch()

    # -- public API --

    def set_idle(self):
        self._state = "idle"
        self._start_time = None
        self._elapsed = 0.0
        self._apply_dot_style(_COLOR_IDLE)
        self._timer_label.setText("--:--")
        self._bar.hide()
        self._bar.setValue(0)

    def set_working(self):
        self._state = "working"
        self._start_time = time.monotonic()
        self._elapsed = 0.0
        self._pulse_on = True
        self._apply_dot_style(_COLOR_WORKING)
        self._timer_label.setText("00:00")

    def set_done(self):
        if self._start_time is not None:
            self._elapsed = time.monotonic() - self._start_time
        self._state = "done"
        self._start_time = None
        self._apply_dot_style(_COLOR_DONE)
        self._timer_label.setText(self._format_time(self._elapsed))
        self._bar.hide()

    def set_error(self):
        if self._start_time is not None:
            self._elapsed = time.monotonic() - self._start_time
        self._state = "error"
        self._start_time = None
        self._apply_dot_style(_COLOR_ERROR)
        self._timer_label.setText(self._format_time(self._elapsed))

    def set_progress(self, value: int, maximum: int = 100):
        self._bar.setMaximum(maximum)
        self._bar.setValue(value)
        if not self._bar.isVisible():
            self._bar.show()

    def tick(self):
        """Called every second by the parent timer."""
        if self._state == "working" and self._start_time is not None:
            self._elapsed = time.monotonic() - self._start_time
            self._timer_label.setText(self._format_time(self._elapsed))
            # Pulse effect
            self._pulse_on = not self._pulse_on
            color = _COLOR_WORKING if self._pulse_on else "#2266cc"
            self._apply_dot_style(color)

    # -- internal --

    def _apply_dot_style(self, color: str):
        self._dot.setStyleSheet(
            f"background: {color};"
            f"border-radius: {_INDICATOR_SIZE // 2}px;"
            f"min-width: {_INDICATOR_SIZE}px;"
            f"max-width: {_INDICATOR_SIZE}px;"
            f"min-height: {_INDICATOR_SIZE}px;"
            f"max-height: {_INDICATOR_SIZE}px;"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Step key mapping — maps log-message tags to step keys
# ---------------------------------------------------------------------------

_STEP_DEFS = [
    ("audio_check",  "Audio Check"),
    ("classify",     "Classification"),
    ("extract",      "Frame Extraction"),
    ("transcribe",   "Transcription"),
    ("llm_summary",  "LLM: Transcript Summary"),
    ("llm_keywords", "LLM: Image Keywords"),
    ("refine",       "LLM: Clip Refinement"),
    ("multicam",     "Multi-Camera Detection"),
]


# ---------------------------------------------------------------------------
# ProgressPanel — main widget
# ---------------------------------------------------------------------------

class ProgressPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ProgressPanel")
        self._lights: dict[str, StatusLight] = {}
        self._active_step: str | None = None
        self._build_ui()

        # 1-second tick timer for elapsed counters + pulse
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # -- Current video heading --
        self.current_label = QLabel("Idle")
        self.current_label.setObjectName("heading")
        root.addWidget(self.current_label)

        # -- Compact progress bars --
        bars = QGridLayout()
        bars.setSpacing(4)

        lbl_frame = QLabel("Frames")
        lbl_frame.setObjectName("dimLabel")
        self.frame_progress = QProgressBar()
        self.frame_progress.setFixedHeight(12)
        self.frame_progress.setTextVisible(False)
        bars.addWidget(lbl_frame, 0, 0)
        bars.addWidget(self.frame_progress, 0, 1)

        lbl_trans = QLabel("Transcript")
        lbl_trans.setObjectName("dimLabel")
        self.transcript_progress = QProgressBar()
        self.transcript_progress.setFixedHeight(12)
        self.transcript_progress.setTextVisible(False)
        bars.addWidget(lbl_trans, 1, 0)
        bars.addWidget(self.transcript_progress, 1, 1)

        lbl_llm = QLabel("LLM")
        lbl_llm.setObjectName("dimLabel")
        self.llm_progress = QProgressBar()
        self.llm_progress.setFixedHeight(12)
        self.llm_progress.setTextVisible(False)
        bars.addWidget(lbl_llm, 2, 0)
        bars.addWidget(self.llm_progress, 2, 1)

        bars.setColumnStretch(1, 1)
        root.addLayout(bars)

        # -- Separator --
        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {_BORDER};")
        root.addWidget(sep)

        # -- Status lights grid --
        lights_container = QVBoxLayout()
        lights_container.setSpacing(0)

        for key, name in _STEP_DEFS:
            light = StatusLight(name, self)
            self._lights[key] = light
            lights_container.addWidget(light)

        root.addLayout(lights_container)

        # -- Separator --
        sep2 = QLabel()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background: {_BORDER};")
        root.addWidget(sep2)

        # -- Collapsible log area --
        log_row = QHBoxLayout()
        self._log_toggle = QPushButton("Show Log")
        self._log_toggle.setObjectName("logToggle")
        self._log_toggle.setCheckable(True)
        self._log_toggle.toggled.connect(self._on_log_toggled)
        log_row.addWidget(self._log_toggle)
        log_row.addStretch()
        root.addLayout(log_row)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(2000)
        self.log_text.setFixedHeight(150)
        self.log_text.hide()
        root.addWidget(self.log_text)

        # -- Results table --
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(
            ["Video", "Keywords", "Summary", "Export"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self.results_table, stretch=1)

        # Apply master stylesheet
        self.setStyleSheet(_PANEL_QSS)

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    def log(self, message: str):
        """Append to hidden log and auto-drive status lights from tags."""
        self.log_text.appendPlainText(message)
        self._parse_log_message(message)

    def set_current_video(self, name: str):
        self.current_label.setText(f"Processing: {name}")
        # Reset all lights for new video
        for light in self._lights.values():
            light.set_idle()
        self._active_step = None

    def set_idle(self):
        self.current_label.setText("Idle")
        self.frame_progress.setValue(0)
        self.transcript_progress.setValue(0)
        self.llm_progress.setValue(0)
        for light in self._lights.values():
            light.set_idle()
        self._active_step = None

    def set_complete(self):
        self.current_label.setText("Queue complete")
        # Mark any still-working step as done
        if self._active_step and self._active_step in self._lights:
            self._lights[self._active_step].set_done()
            self._active_step = None

    def add_result_row(self, video_name: str, keywords: str, summary: str):
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(video_name))
        self.results_table.setItem(row, 1, QTableWidgetItem(keywords))
        self.results_table.setItem(row, 2, QTableWidgetItem(summary))
        self.results_table.setItem(row, 3, QTableWidgetItem("CSV"))

    # ------------------------------------------------------------------
    # Log parsing → light driving
    # ------------------------------------------------------------------

    def _parse_log_message(self, msg: str):
        lower = msg.lower()

        # Check for errors first
        if "[error]" in lower:
            if self._active_step and self._active_step in self._lights:
                self._lights[self._active_step].set_error()
                self._active_step = None
            return

        # Map log tags to step keys and detect start / done
        if "[audio-check]" in lower:
            self._transition_to("audio_check", msg)
        elif "[classify]" in lower:
            self._transition_to("classify", msg)
        elif "[extract]" in lower:
            self._transition_to("extract", msg)
        elif "[transcribe]" in lower:
            self._transition_to("transcribe", msg)
        elif "[llm]" in lower:
            if "step 1" in lower or "summariz" in lower:
                self._transition_to("llm_summary", msg)
            elif "step 2" in lower or "batch" in lower or "frames" in lower:
                self._transition_to("llm_keywords", msg)
            elif "done" in lower:
                # Final LLM done message — finish whichever LLM step is active
                if self._active_step in ("llm_summary", "llm_keywords"):
                    self._lights[self._active_step].set_done()
                    self._active_step = None
        elif "[refine]" in lower:
            self._transition_to("refine", msg)
        elif "[multicam]" in lower:
            self._transition_to("multicam", msg)

    def _transition_to(self, step_key: str, msg: str):
        """Transition active step: finish previous, start (or update) new."""
        # If we're already on this step, check for completion keywords
        if self._active_step == step_key:
            lower = msg.lower()
            # Detect completion patterns
            completion_hints = ("done", "found", "got ", "unique keywords",
                                "no multi-camera", "group", "subject:",
                                "speech ratio", "too short")
            if any(h in lower for h in completion_hints):
                self._lights[step_key].set_done()
                self._active_step = None
            return

        # Finish previous active step
        if self._active_step and self._active_step in self._lights:
            self._lights[self._active_step].set_done()

        # Start new step
        if step_key in self._lights:
            self._lights[step_key].set_working()
            self._active_step = step_key

    # ------------------------------------------------------------------
    # Timer tick
    # ------------------------------------------------------------------

    def _on_tick(self):
        for light in self._lights.values():
            light.tick()

    # ------------------------------------------------------------------
    # Log toggle
    # ------------------------------------------------------------------

    def _on_log_toggled(self, checked: bool):
        self.log_text.setVisible(checked)
        self._log_toggle.setText("Hide Log" if checked else "Show Log")
