"""Settings panel with tabs for Ingest, LLM, Transcription, and Export."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.schemas import AppSettings
from core.transcribe import is_whisper_model_downloaded, download_whisper_model


class SettingsPanel(QWidget):
    settings_changed = Signal()

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._build_ui()
        self._load_from_settings()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        tabs.addTab(self._build_ingest_tab(), "Ingest")
        tabs.addTab(self._build_llm_tab(), "LLM")
        tabs.addTab(self._build_transcription_tab(), "Transcription")
        tabs.addTab(self._build_export_tab(), "Export")

        layout.addWidget(tabs)

    # -- Ingest Tab --
    def _build_ingest_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.threshold_mode = QComboBox()
        self.threshold_mode.addItems(["AUTO", "Manual"])
        self.threshold_mode.currentTextChanged.connect(self._on_threshold_mode)
        form.addRow("Threshold Mode:", self.threshold_mode)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.50, 0.999)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setEnabled(False)
        form.addRow("Threshold:", self.threshold_spin)

        self.target_fpm_spin = QDoubleSpinBox()
        self.target_fpm_spin.setRange(0.5, 30.0)
        self.target_fpm_spin.setSingleStep(0.5)
        self.target_fpm_spin.setDecimals(1)
        form.addRow("Target Frames/Min:", self.target_fpm_spin)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["histogram", "ssim", "phash"])
        form.addRow("Metric:", self.metric_combo)

        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(1, 60)
        form.addRow("Frame Offset:", self.offset_spin)

        return w

    def _on_threshold_mode(self, mode: str):
        manual = mode == "Manual"
        self.threshold_spin.setEnabled(manual)
        self.target_fpm_spin.setEnabled(not manual)

    # -- LLM Tab --
    def _build_llm_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.llm_backend = QComboBox()
        self.llm_backend.addItems(["ollama", "openai", "anthropic"])
        self.llm_backend.currentTextChanged.connect(self._on_llm_backend)
        form.addRow("Backend:", self.llm_backend)

        self.llm_model = QLineEdit()
        form.addRow("Model:", self.llm_model)

        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_api_key.setPlaceholderText("Not needed for Ollama")
        form.addRow("API Key:", self.llm_api_key)

        self.llm_base_url = QLineEdit()
        form.addRow("Base URL:", self.llm_base_url)

        self.context_window_spin = QSpinBox()
        self.context_window_spin.setRange(2048, 131072)
        self.context_window_spin.setSingleStep(1024)
        form.addRow("Context Window (tokens):", self.context_window_spin)

        self.vlm_resolution_spin = QSpinBox()
        self.vlm_resolution_spin.setRange(240, 1080)
        self.vlm_resolution_spin.setSingleStep(120)
        form.addRow("VLM Input Resolution:", self.vlm_resolution_spin)

        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(1, 32)
        self.max_images_spin.setSingleStep(1)
        form.addRow("Max Images/Batch:", self.max_images_spin)
        batch_hint = QLabel("More = fewer LLM calls but more VRAM per call")
        batch_hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow("", batch_hint)

        return w

    def _on_llm_backend(self, backend: str):
        is_local = backend == "ollama"
        self.llm_api_key.setEnabled(not is_local)
        self.llm_base_url.setEnabled(is_local)
        if is_local:
            self.llm_api_key.setPlaceholderText("Not needed for Ollama")
        else:
            self.llm_api_key.setPlaceholderText("Enter API key")

    # -- Transcription Tab --
    def _build_transcription_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.transcription_backend = QComboBox()
        self.transcription_backend.addItems(["mlx-whisper", "faster-whisper"])
        form.addRow("Backend:", self.transcription_backend)

        self.transcription_model = QComboBox()
        self.transcription_model.addItems(["tiny", "base", "small", "medium", "large"])
        self.transcription_model.currentTextChanged.connect(self._update_model_status)
        form.addRow("Model Size:", self.transcription_model)

        # Model status + download button
        model_row = QHBoxLayout()
        self.model_status_label = QLabel("")
        self.model_status_label.setStyleSheet("font-size: 11px;")
        model_row.addWidget(self.model_status_label)
        self.download_btn = QPushButton("Download")
        self.download_btn.setFixedWidth(90)
        self.download_btn.clicked.connect(self._download_model)
        model_row.addWidget(self.download_btn)
        form.addRow("Model Status:", model_row)

        self.vocabulary_edit = QLineEdit()
        self.vocabulary_edit.setPlaceholderText("e.g. Niobe, Stranger Comics, Asunda")
        form.addRow("Custom Vocabulary:", self.vocabulary_edit)
        vocab_hint = QLabel("Comma-separated words to help with uncommon names/terms")
        vocab_hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow("", vocab_hint)

        # Audio check settings
        from PySide6.QtWidgets import QCheckBox
        self.audio_check_cb = QCheckBox("Check audio level before transcribing")
        self.audio_check_cb.setToolTip("Skip transcription on silent/ambient-only clips to avoid hallucinated text")
        form.addRow("Audio Check:", self.audio_check_cb)

        self.noise_floor_spin = QDoubleSpinBox()
        self.noise_floor_spin.setRange(-80.0, -10.0)
        self.noise_floor_spin.setSingleStep(5.0)
        self.noise_floor_spin.setSuffix(" dBFS")
        self.noise_floor_spin.setToolTip("Audio below this level is considered silence")
        form.addRow("Noise Floor:", self.noise_floor_spin)

        return w

    def _update_model_status(self, model_size: str = ""):
        if not model_size:
            model_size = self.transcription_model.currentText()
        if is_whisper_model_downloaded(model_size):
            self.model_status_label.setText("Downloaded")
            self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            self.download_btn.setEnabled(True)
            self.download_btn.setText("Re-download")
        else:
            self.model_status_label.setText("Not downloaded (will download on first use)")
            self.model_status_label.setStyleSheet("color: #FF9800; font-size: 11px;")
            self.download_btn.setEnabled(True)
            self.download_btn.setText("Download")

    def _download_model(self):
        model_size = self.transcription_model.currentText()
        self.download_btn.setEnabled(False)
        self.download_btn.setText("Downloading...")
        self.model_status_label.setText("Downloading...")
        self.model_status_label.setStyleSheet("color: gray; font-size: 11px;")

        self._download_thread = _WhisperDownloadThread(model_size)
        self._download_thread.finished_signal.connect(self._on_download_done)
        self._download_thread.start()

    def _on_download_done(self, success: bool, msg: str):
        if success:
            self.model_status_label.setText("Downloaded")
            self.model_status_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
        else:
            self.model_status_label.setText(f"Failed: {msg}")
            self.model_status_label.setStyleSheet("color: red; font-size: 11px;")
        self.download_btn.setEnabled(True)
        self._update_model_status()

    # -- Export Tab --
    def _build_export_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.export_format = QComboBox()
        self.export_format.addItems(["csv", "fcpxml (coming soon)"])
        form.addRow("Export Format:", self.export_format)

        # Output folder picker
        folder_row = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Choose a folder for all CSV exports...")
        self.output_folder_edit.setReadOnly(True)
        folder_row.addWidget(self.output_folder_edit)
        self.browse_folder_btn = QPushButton("Browse...")
        self.browse_folder_btn.setFixedWidth(80)
        self.browse_folder_btn.clicked.connect(self._browse_output_folder)
        folder_row.addWidget(self.browse_folder_btn)
        form.addRow("Output Folder:", folder_row)

        folder_hint = QLabel("All CSV files will be saved here, named after each clip")
        folder_hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow("", folder_hint)

        return w

    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder",
            self.output_folder_edit.text() or "",
        )
        if folder:
            self.output_folder_edit.setText(folder)

    # -- Load / Save --
    def _load_from_settings(self):
        s = self._settings

        # Ingest
        if s.ingest.threshold is None:
            self.threshold_mode.setCurrentText("AUTO")
        else:
            self.threshold_mode.setCurrentText("Manual")
            self.threshold_spin.setValue(s.ingest.threshold)
        self.target_fpm_spin.setValue(s.ingest.target_fpm)
        self.metric_combo.setCurrentText(s.ingest.metric)
        self.offset_spin.setValue(s.ingest.offset)

        # LLM
        self.llm_backend.setCurrentText(s.llm.backend)
        self.llm_model.setText(s.llm.model)
        self.llm_api_key.setText(s.llm.api_key or "")
        self.llm_base_url.setText(s.llm.base_url)
        self.context_window_spin.setValue(s.llm.context_window)
        self.vlm_resolution_spin.setValue(s.llm.vlm_input_resolution)
        self.max_images_spin.setValue(s.llm.max_images_per_batch)

        # Transcription
        self.transcription_backend.setCurrentText(s.transcription.backend)
        self.transcription_model.setCurrentText(s.transcription.model_size)
        self.vocabulary_edit.setText(s.transcription.vocabulary)
        self.audio_check_cb.setChecked(s.transcription.audio_check)
        self.noise_floor_spin.setValue(s.transcription.noise_floor_db)
        self._update_model_status(s.transcription.model_size)

        # Export
        self.export_format.setCurrentText(s.export.format)
        self.output_folder_edit.setText(s.export.output_folder or "")

    def save_to_settings(self) -> AppSettings:
        """Read current widget values back into an AppSettings object."""
        s = self._settings

        # Ingest
        if self.threshold_mode.currentText() == "AUTO":
            s.ingest.threshold = None
        else:
            s.ingest.threshold = self.threshold_spin.value()
        s.ingest.target_fpm = self.target_fpm_spin.value()
        s.ingest.metric = self.metric_combo.currentText()
        s.ingest.offset = self.offset_spin.value()

        # LLM
        s.llm.backend = self.llm_backend.currentText()
        s.llm.model = self.llm_model.text()
        api_key = self.llm_api_key.text().strip()
        s.llm.api_key = api_key if api_key else None
        s.llm.base_url = self.llm_base_url.text()
        s.llm.context_window = self.context_window_spin.value()
        s.llm.vlm_input_resolution = self.vlm_resolution_spin.value()
        s.llm.max_images_per_batch = self.max_images_spin.value()

        # Transcription
        s.transcription.backend = self.transcription_backend.currentText()
        s.transcription.model_size = self.transcription_model.currentText()
        s.transcription.vocabulary = self.vocabulary_edit.text().strip()
        s.transcription.audio_check = self.audio_check_cb.isChecked()
        s.transcription.noise_floor_db = self.noise_floor_spin.value()

        # Export
        fmt = self.export_format.currentText()
        s.export.format = fmt.split(" ")[0]  # strip "(coming soon)"
        s.export.output_folder = self.output_folder_edit.text().strip()

        return s


class _WhisperDownloadThread(QThread):
    finished_signal = Signal(bool, str)  # success, message

    def __init__(self, model_size: str):
        super().__init__()
        self.model_size = model_size

    def run(self):
        messages = []
        path = download_whisper_model(
            self.model_size,
            log=lambda msg: messages.append(msg),
        )
        if path:
            self.finished_signal.emit(True, "")
        else:
            self.finished_signal.emit(False, messages[-1] if messages else "Unknown error")
