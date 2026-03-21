"""Video queue panel: add, remove, reorder videos."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

VIDEO_EXTENSIONS = "Video Files (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.mts *.ts);;All Files (*)"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mts", ".ts"}

# Qt.UserRole constants
ROLE_PATH = 256         # Qt.UserRole
ROLE_FOLDER_TAGS = 257  # Qt.UserRole + 1


class FolderScanWorker(QThread):
    """Background thread that recursively scans a folder for video files."""
    batch_found = Signal(list)    # list of (path, folder_tags) tuples
    scan_complete = Signal(int)   # total files found
    progress = Signal(str)        # status message

    def __init__(self, root_folder: str, parent=None):
        super().__init__(parent)
        self.root_folder = root_folder

    def run(self):
        root = self.root_folder
        results: list[tuple[str, list[str]]] = []
        batch: list[tuple[str, list[str]]] = []
        count = 0

        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            # Sort filenames for consistent ordering
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in VIDEO_SUFFIXES:
                    continue

                full_path = os.path.join(dirpath, fname)

                # Extract folder tags from relative path
                rel = os.path.relpath(dirpath, root)
                if rel == ".":
                    folder_tags = []
                else:
                    parts = Path(rel).parts
                    folder_tags = [
                        p.strip().replace("_", " ").replace("-", " ")
                        for p in parts
                        if p.strip() and p != "."
                    ]

                batch.append((full_path, folder_tags))
                count += 1

                # Emit in batches of 100 to avoid signal flooding
                if len(batch) >= 100:
                    self.progress.emit(f"Scanning... {count} videos found")
                    self.batch_found.emit(batch)
                    batch = []

        # Emit remaining
        if batch:
            self.batch_found.emit(batch)

        self.scan_complete.emit(count)


class QueuePanel(QWidget):
    queue_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._existing_paths: set[str] = set()  # dedup tracker
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Header with scan status
        self.header_label = QLabel("Queue")
        self.header_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.header_label)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Videos")
        self.add_btn.clicked.connect(self._add_videos)
        btn_row.addWidget(self.add_btn)

        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.clicked.connect(self._add_folder)
        btn_row.addWidget(self.add_folder_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self.clear_btn)

        layout.addLayout(btn_row)

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "", VIDEO_EXTENSIONS
        )
        added = 0
        for path in paths:
            if path in self._existing_paths:
                continue
            item = QListWidgetItem(os.path.basename(path))
            item.setData(ROLE_PATH, path)
            item.setData(ROLE_FOLDER_TAGS, [])
            item.setToolTip(path)
            self.list_widget.addItem(item)
            self._existing_paths.add(path)
            added += 1
        if added:
            self._update_header()
            self.queue_changed.emit()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Scan for Videos",
            os.path.expanduser("~/Documents"),
        )
        if not folder:
            return

        self.add_folder_btn.setEnabled(False)
        self.add_folder_btn.setText("Scanning...")
        self.header_label.setText("Queue — Scanning...")

        self._scan_worker = FolderScanWorker(folder, parent=self)
        self._scan_worker.batch_found.connect(self._on_scan_batch)
        self._scan_worker.progress.connect(
            lambda msg: self.header_label.setText(f"Queue — {msg}")
        )
        self._scan_worker.scan_complete.connect(self._on_scan_complete)
        self._scan_worker.start()

    def _on_scan_batch(self, batch: list[tuple[str, list[str]]]):
        """Add a batch of found videos to the queue."""
        self.list_widget.setUpdatesEnabled(False)
        added = 0
        for path, folder_tags in batch:
            if path in self._existing_paths:
                continue
            # Show folder context in display name
            display = os.path.basename(path)
            if folder_tags:
                tag_prefix = "/".join(folder_tags[-2:])  # show last 2 folder levels
                display = f"[{tag_prefix}] {display}"

            item = QListWidgetItem(display)
            item.setData(ROLE_PATH, path)
            item.setData(ROLE_FOLDER_TAGS, folder_tags)
            item.setToolTip(f"{path}\nTags: {', '.join(folder_tags)}" if folder_tags else path)
            self.list_widget.addItem(item)
            self._existing_paths.add(path)
            added += 1
        self.list_widget.setUpdatesEnabled(True)
        if added:
            self._update_header()

    def _on_scan_complete(self, total: int):
        self.add_folder_btn.setEnabled(True)
        self.add_folder_btn.setText("Add Folder")
        self._update_header()
        self.queue_changed.emit()

    def _remove_selected(self):
        for item in self.list_widget.selectedItems():
            path = item.data(ROLE_PATH)
            self._existing_paths.discard(path)
            self.list_widget.takeItem(self.list_widget.row(item))
        self._update_header()
        self.queue_changed.emit()

    def _clear_all(self):
        self.list_widget.clear()
        self._existing_paths.clear()
        self._update_header()
        self.queue_changed.emit()

    def _update_header(self):
        count = self.list_widget.count()
        self.header_label.setText(f"Queue ({count} clip{'s' if count != 1 else ''})")

    def get_video_paths(self) -> list[str]:
        """Return all queued video paths in order."""
        paths = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            paths.append(item.data(ROLE_PATH))
        return paths

    def get_folder_tags(self, index: int) -> list[str]:
        """Return folder tags for a queue item by index."""
        if 0 <= index < self.list_widget.count():
            item = self.list_widget.item(index)
            tags = item.data(ROLE_FOLDER_TAGS)
            return tags if tags else []
        return []

    def set_item_status(self, index: int, status: str):
        """Update the display text for a queue item with a status indicator."""
        if 0 <= index < self.list_widget.count():
            item = self.list_widget.item(index)
            path = item.data(ROLE_PATH)
            folder_tags = item.data(ROLE_FOLDER_TAGS) or []
            name = os.path.basename(path)
            if folder_tags:
                tag_prefix = "/".join(folder_tags[-2:])
                name = f"[{tag_prefix}] {name}"
            item.setText(f"{status} {name}")
