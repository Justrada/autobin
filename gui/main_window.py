"""Main application window."""

from __future__ import annotations

import json
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.schemas import (
    AppSettings, ClipClassification, ClipRefinement,
    TranscriptSummary, UserOverrides, VideoResult,
)
from gui.filmstrip import FilmStrip
from gui.metadata_panel import MetadataPanel
from gui.orchestrator import PipelineOrchestrator
from gui.progress_panel import ProgressPanel
from gui.queue_panel import QueuePanel
from gui.settings_panel import SettingsPanel

SETTINGS_PATH = os.path.expanduser("~/.config/vlm_iframe/settings.json")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoBin")
        self.setMinimumSize(1100, 700)

        self._settings = self._load_settings()
        self._build_ui()
        self._orchestrator = PipelineOrchestrator(self._settings, parent=self)
        self._connect_orchestrator()

        self._active_index: int = -1
        self._multicam_groups: list = []

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Film strip at the top
        self.filmstrip = FilmStrip()
        main_layout.addWidget(self.filmstrip)

        # Main content area
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: queue + start button
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.queue_panel = QueuePanel()
        left_layout.addWidget(self.queue_panel)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self._start_processing)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_processing)
        btn_row.addWidget(self.stop_btn)

        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        # Center: tabs (progress + settings)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.tabs = QTabWidget()

        self.progress_panel = ProgressPanel()
        self.tabs.addTab(self.progress_panel, "Progress")

        self.settings_panel = SettingsPanel(self._settings)
        self.tabs.addTab(self.settings_panel, "Settings")

        center_layout.addWidget(self.tabs)
        splitter.addWidget(center)

        # Right: metadata panel
        self.metadata_panel = MetadataPanel()
        splitter.addWidget(self.metadata_panel)

        splitter.setSizes([250, 500, 350])
        main_layout.addWidget(splitter)

        # Connect queue click for viewing completed results
        self.queue_panel.list_widget.currentRowChanged.connect(self._on_queue_selection_changed)

        # Connect commit button
        self.metadata_panel.overrides_committed.connect(self._on_overrides_committed)

    def _connect_orchestrator(self):
        o = self._orchestrator
        o.log.connect(self.progress_panel.log)
        o.video_started.connect(self._on_video_started)
        o.video_skipped.connect(self._on_video_skipped)
        o.frame_progress.connect(self._on_frame_progress)
        o.transcript_progress.connect(self._on_transcript_progress)
        o.llm_progress.connect(self._on_llm_progress)
        o.classification_done.connect(self._on_classification_done)
        o.refinement_done.connect(self._on_refinement_done)
        o.transcript_summary_done.connect(self._on_transcript_summary_done)
        o.transcript_text_done.connect(self._on_transcript_text_done)
        o.keywords_done.connect(self._on_keywords_done)
        o.frame_count_update.connect(self._on_frame_count_update)
        o.frames_available.connect(self._on_frames_available)
        o.video_completed.connect(self._on_video_completed)
        o.video_error.connect(self._on_video_error)
        o.multicam_groups_found.connect(self._on_multicam_groups_found)
        o.queue_completed.connect(self._on_queue_completed)
        o.queue_progress.connect(self._on_queue_progress)

    def _ensure_output_folder(self) -> bool:
        """Make sure an output folder is set. Prompt if not."""
        if self._settings.export.output_folder:
            return True

        folder = QFileDialog.getExistingDirectory(
            self, "Choose Output Folder for CSV Exports",
            os.path.expanduser("~/Documents"),
        )
        if not folder:
            return False

        self._settings.export.output_folder = folder
        self.settings_panel.output_folder_edit.setText(folder)
        self._save_settings()
        return True

    def _start_processing(self):
        paths = self.queue_panel.get_video_paths()
        if not paths:
            QMessageBox.warning(self, "No Videos", "Add videos to the queue first.")
            return

        # Save current settings
        self._settings = self.settings_panel.save_to_settings()
        self._save_settings()

        # Ensure output folder is set
        if not self._ensure_output_folder():
            QMessageBox.information(
                self, "Output Folder Required",
                "Please set an output folder in Settings > Export before processing."
            )
            return

        # Collect folder tags for each queue item
        folder_tags_map = {}
        for i in range(len(paths)):
            tags = self.queue_panel.get_folder_tags(i)
            if tags:
                folder_tags_map[i] = tags

        self._orchestrator.settings = self._settings
        self._orchestrator.set_queue(paths, folder_tags_map)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setCurrentWidget(self.progress_panel)
        self.progress_panel.log("Starting pipeline...")

        self._orchestrator.start()

    def _stop_processing(self):
        self._orchestrator.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_panel.log("Stopped by user.")

    # -- Queue click handler --

    def _on_queue_selection_changed(self, row: int):
        if row < 0:
            return
        self.metadata_panel.set_current_index(row)
        if row == self._active_index and self._orchestrator._running:
            return
        result = self._orchestrator.get_result(row)
        if result:
            self.metadata_panel.set_result(result)
            # Show multicam group info if available
            if result.multicam_group_id and hasattr(self, '_multicam_groups'):
                for g in self._multicam_groups:
                    if result.video_path in g.clip_paths:
                        others = [p for p in g.clip_paths if p != result.video_path]
                        self.metadata_panel.set_multicam(g.group_id, others)
                        break

    def _on_overrides_committed(self, index: int, overrides: UserOverrides):
        self._orchestrator.apply_overrides(index, overrides)
        self.progress_panel.log(f"[commit] User edits saved for clip {index + 1}")

    # -- Orchestrator signal handlers --

    def _on_video_skipped(self, index: int, path: str):
        """Video was skipped because CSV already exists (resume mode)."""
        name = os.path.basename(path)
        self.queue_panel.set_item_status(index, "[SKIP]")
        self.progress_panel.log(f"[skip] {name} already processed, skipping")

    def _on_queue_progress(self, completed: int, total: int, est_remaining: float):
        """Update the queue-level progress bar and ETA."""
        self.progress_panel.set_queue_progress(completed, total, est_remaining)

    def _on_video_started(self, index: int, path: str):
        self._active_index = index
        name = os.path.basename(path)
        self.progress_panel.set_current_video(name)
        self.queue_panel.set_item_status(index, "[...]")
        self.progress_panel.frame_progress.setValue(0)
        self.progress_panel.transcript_progress.setValue(0)
        self.progress_panel.llm_progress.setValue(0)
        self.metadata_panel.clear()
        self.metadata_panel.set_current_index(index)
        self.metadata_panel.set_clip_name(name)
        self.filmstrip.clear()
        self.filmstrip.set_title(f"Frames: {name}")
        self.queue_panel.list_widget.setCurrentRow(index)

    def _on_frame_progress(self, current: int, total: int):
        self.progress_panel.frame_progress.setMaximum(total)
        self.progress_panel.frame_progress.setValue(current)

    def _on_transcript_progress(self, current: int, total: int):
        self.progress_panel.transcript_progress.setMaximum(total)
        self.progress_panel.transcript_progress.setValue(current)

    def _on_llm_progress(self, current: int, total: int):
        self.progress_panel.llm_progress.setMaximum(total)
        self.progress_panel.llm_progress.setValue(current)

    def _on_classification_done(self, classification: ClipClassification | None):
        if classification:
            self.metadata_panel.set_classification(classification)

    def _on_refinement_done(self, refinement: ClipRefinement | None):
        if refinement:
            self.metadata_panel.set_refinement(refinement)

    def _on_transcript_summary_done(self, summary: TranscriptSummary | None):
        if summary:
            self.metadata_panel.set_transcript_summary(summary)

    def _on_transcript_text_done(self, text: str):
        if text:
            self.metadata_panel.set_transcript_length(len(text))

    def _on_keywords_done(self, keywords: list[str]):
        self.metadata_panel.set_keywords(keywords)

    def _on_frame_count_update(self, count: int):
        self.metadata_panel.set_frame_count(count)

    def _on_frames_available(self, paths: list[str]):
        """Show extracted frames in the film strip."""
        self.filmstrip.add_frames(paths)

    def _on_video_completed(self, index: int, result: VideoResult):
        name = os.path.basename(result.video_path)
        keywords_str = ", ".join(result.keywords[:20])
        if len(result.keywords) > 20:
            keywords_str += f" ... (+{len(result.keywords) - 20} more)"
        summary_str = result.transcript_summary.summary if result.transcript_summary else ""
        self.progress_panel.add_result_row(name, keywords_str, summary_str)
        self.queue_panel.set_item_status(index, "[OK]")
        self.metadata_panel.set_duration(result.duration_seconds)

    def _on_video_error(self, index: int, msg: str):
        self.queue_panel.set_item_status(index, "[ERR]")
        self.progress_panel.log(f"[error] {msg}")

    def _on_multicam_groups_found(self, groups: list):
        """Store multicam groups and update the current metadata view."""
        self._multicam_groups = groups
        if groups:
            count = sum(len(g.clip_paths) for g in groups)
            self.progress_panel.log(
                f"[multicam] Found {len(groups)} group(s) containing {count} clips"
            )
            # If currently viewing a clip that's in a group, update the display
            row = self.queue_panel.list_widget.currentRow()
            result = self._orchestrator.get_result(row) if row >= 0 else None
            if result and result.multicam_group_id:
                for g in groups:
                    if result.video_path in g.clip_paths:
                        others = [p for p in g.clip_paths if p != result.video_path]
                        self.metadata_panel.set_multicam(g.group_id, others)
                        break

    def _on_queue_completed(self):
        self._active_index = -1
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_panel.set_complete()
        self.progress_panel.log("All videos processed.")

        # Show timing stats per file type
        stats = self._orchestrator.get_timing_stats()
        if stats:
            self.progress_panel.log("\n" + "=" * 50)
            self.progress_panel.log("Timing stats by file type:")
            self.progress_panel.log("=" * 50)
            for ext, s in stats.items():
                self.progress_panel.log(
                    f"  {ext or '(no ext)'} ({int(s['count'])} clips): "
                    f"avg total={s['avg_total']:.1f}s | "
                    f"frames={s['avg_frame_extraction']:.1f}s | "
                    f"transcribe={s['avg_transcription']:.1f}s | "
                    f"llm={s['avg_llm']:.1f}s | "
                    f"refine={s['avg_refinement']:.1f}s"
                )

    # -- Settings persistence --

    def _load_settings(self) -> AppSettings:
        if os.path.isfile(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH) as f:
                    return AppSettings.model_validate_json(f.read())
            except Exception:
                pass
        return AppSettings()

    def _save_settings(self):
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            f.write(self._settings.model_dump_json(indent=2))

    def closeEvent(self, event):
        self._settings = self.settings_panel.save_to_settings()
        self._save_settings()
        super().closeEvent(event)
