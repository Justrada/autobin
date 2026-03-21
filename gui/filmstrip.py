"""Film strip widget — horizontal scrolling strip showing I-frames as they're extracted."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


THUMB_HEIGHT = 80


class FilmStrip(QWidget):
    """Horizontal scrolling strip of frame thumbnails, like a film strip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(THUMB_HEIGHT + 30)  # thumbnails + label + margins
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        self._title = QLabel("Extracted Frames")
        self._title.setStyleSheet("font-weight: bold; font-size: 11px;")
        header.addWidget(self._title)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: gray; font-size: 11px;")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self._count_label)
        outer.addLayout(header)

        # Scrollable strip
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFixedHeight(THUMB_HEIGHT + 4)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("background: #1a1a1a; border-radius: 4px;")

        self._strip_container = QWidget()
        self._strip_layout = QHBoxLayout(self._strip_container)
        self._strip_layout.setContentsMargins(4, 2, 4, 2)
        self._strip_layout.setSpacing(3)
        self._strip_layout.addStretch()

        self._scroll.setWidget(self._strip_container)
        outer.addWidget(self._scroll)

        self._frame_count = 0

    def clear(self):
        """Remove all thumbnails."""
        while self._strip_layout.count() > 1:  # keep the stretch
            item = self._strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._frame_count = 0
        self._count_label.setText("")
        self._title.setText("Extracted Frames")

    def set_title(self, title: str):
        self._title.setText(title)

    def add_frame(self, image_path: str):
        """Add a single frame thumbnail to the strip."""
        if not os.path.isfile(image_path):
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return

        # Scale to thumbnail height, preserving aspect ratio
        thumb = pixmap.scaledToHeight(THUMB_HEIGHT - 4, Qt.TransformationMode.SmoothTransformation)

        label = QLabel()
        label.setPixmap(thumb)
        label.setFixedSize(thumb.size())
        label.setStyleSheet("border: 1px solid #333; border-radius: 2px;")
        label.setToolTip(os.path.basename(image_path))

        # Insert before the stretch
        self._strip_layout.insertWidget(self._strip_layout.count() - 1, label)
        self._frame_count += 1
        self._count_label.setText(f"{self._frame_count} frames")

        # Auto-scroll to the latest frame
        self._scroll.horizontalScrollBar().setValue(
            self._scroll.horizontalScrollBar().maximum()
        )

    def add_frames(self, image_paths: list[str]):
        """Add multiple frame thumbnails at once."""
        for path in image_paths:
            self.add_frame(path)
