"""Live metadata panel with Auto/User editable fields."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.schemas import (
    AudioCheckResult, ClipClassification, ClipRefinement,
    MultiCamGroup, TranscriptSummary, UserOverrides, VideoResult,
)


# ---------------------------------------------------------------------------
# Reusable editable field widgets
# ---------------------------------------------------------------------------

class EditableField(QWidget):
    """A field that toggles between Auto (read-only label) and User (editable input)."""

    def __init__(self, choices: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._auto_value = "—"
        self._is_user_mode = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toggle button — aligned to top so it stays in place when text wraps
        self.toggle_btn = QPushButton("A")
        self.toggle_btn.setFixedSize(22, 22)
        self.toggle_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 10px; "
            "border-radius: 3px; background: #4CAF50; color: white; }"
        )
        self.toggle_btn.setToolTip("Auto mode — click to edit manually")
        self.toggle_btn.clicked.connect(self._toggle_mode)
        layout.addWidget(self.toggle_btn, 0, Qt.AlignmentFlag.AlignTop)

        # Auto display label
        self.auto_label = QLabel("—")
        self.auto_label.setWordWrap(True)
        self.auto_label.setSizePolicy(
            self.auto_label.sizePolicy().horizontalPolicy(),
            self.auto_label.sizePolicy().verticalPolicy(),
        )
        self.auto_label.setMinimumWidth(50)
        self.auto_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.auto_label, 1)

        # User input (hidden by default)
        if choices:
            self.user_input = QComboBox()
            self.user_input.addItems(choices)
            self.user_input.setMinimumWidth(120)
        else:
            self.user_input = QLineEdit()
            self.user_input.setPlaceholderText("Enter value...")
        self.user_input.hide()
        layout.addWidget(self.user_input, 1)

    def _toggle_mode(self):
        self._is_user_mode = not self._is_user_mode
        if self._is_user_mode:
            self.toggle_btn.setText("U")
            self.toggle_btn.setStyleSheet(
                "QPushButton { font-weight: bold; font-size: 10px; "
                "border-radius: 3px; background: #FF9800; color: white; }"
            )
            self.toggle_btn.setToolTip("User mode — click to revert to auto")
            self.auto_label.hide()
            self.user_input.show()
            # Pre-fill user input with current auto value
            if isinstance(self.user_input, QComboBox):
                idx = self.user_input.findText(self._auto_value, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    self.user_input.setCurrentIndex(idx)
            else:
                if not self.user_input.text():
                    self.user_input.setText(self._auto_value if self._auto_value != "—" else "")
                self.user_input.setFocus()
        else:
            self.toggle_btn.setText("A")
            self.toggle_btn.setStyleSheet(
                "QPushButton { font-weight: bold; font-size: 10px; "
                "border-radius: 3px; background: #4CAF50; color: white; }"
            )
            self.toggle_btn.setToolTip("Auto mode — click to edit manually")
            self.user_input.hide()
            self.auto_label.show()

    def set_auto_value(self, value: str, style: str = ""):
        """Set the auto-filled value. Does NOT overwrite user input if in User mode."""
        self._auto_value = value
        self.auto_label.setText(value)
        if style:
            self.auto_label.setStyleSheet(style)

    def get_effective_value(self) -> str | None:
        """Return user value if in User mode, else None (meaning 'use auto')."""
        if not self._is_user_mode:
            return None
        if isinstance(self.user_input, QComboBox):
            return self.user_input.currentText()
        text = self.user_input.text().strip()
        return text if text else None

    def is_user_mode(self) -> bool:
        return self._is_user_mode

    def reset(self):
        """Reset to auto mode with default value."""
        self._is_user_mode = False
        self._auto_value = "—"
        self.auto_label.setText("—")
        self.auto_label.setStyleSheet("")
        self.auto_label.show()
        self.user_input.hide()
        if isinstance(self.user_input, QComboBox):
            self.user_input.setCurrentIndex(0)
        else:
            self.user_input.clear()
        self.toggle_btn.setText("A")
        self.toggle_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 10px; "
            "border-radius: 3px; background: #4CAF50; color: white; }"
        )

    def set_from_result(self, auto_value: str, user_value: str | None = None,
                        style: str = ""):
        """Set field from a saved result — restores user mode if override exists."""
        self.set_auto_value(auto_value, style)
        if user_value is not None and user_value != auto_value:
            if not self._is_user_mode:
                self._toggle_mode()
            if isinstance(self.user_input, QComboBox):
                idx = self.user_input.findText(user_value, Qt.MatchFlag.MatchFixedString)
                if idx >= 0:
                    self.user_input.setCurrentIndex(idx)
            else:
                self.user_input.setText(user_value)


class EditableListField(QWidget):
    """An editable list field (keywords, content tags) that merges user + auto values."""

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self._auto_values: list[str] = []
        self._is_user_mode = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header row with toggle
        header = QHBoxLayout()
        header.setSpacing(4)

        self.toggle_btn = QPushButton("A")
        self.toggle_btn.setFixedSize(22, 22)
        self.toggle_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 10px; "
            "border-radius: 3px; background: #4CAF50; color: white; }"
        )
        self.toggle_btn.setToolTip("Auto mode — click to add your own entries")
        self.toggle_btn.clicked.connect(self._toggle_mode)
        header.addWidget(self.toggle_btn)

        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: gray; font-size: 11px;")
        header.addWidget(self.count_label, 1)
        layout.addLayout(header)

        # Auto display (read-only)
        self.auto_text = QPlainTextEdit()
        self.auto_text.setReadOnly(True)
        self.auto_text.setMaximumHeight(80)
        self.auto_text.setPlaceholderText(placeholder)
        layout.addWidget(self.auto_text)

        # User input (hidden by default)
        self.user_input = QPlainTextEdit()
        self.user_input.setMaximumHeight(60)
        self.user_input.setPlaceholderText("Add comma-separated entries (merged with auto)...")
        self.user_input.hide()
        layout.addWidget(self.user_input)

    def _toggle_mode(self):
        self._is_user_mode = not self._is_user_mode
        if self._is_user_mode:
            self.toggle_btn.setText("U")
            self.toggle_btn.setStyleSheet(
                "QPushButton { font-weight: bold; font-size: 10px; "
                "border-radius: 3px; background: #FF9800; color: white; }"
            )
            self.toggle_btn.setToolTip("User mode — your entries will be merged with auto")
            self.user_input.show()
            self.user_input.setFocus()
        else:
            self.toggle_btn.setText("A")
            self.toggle_btn.setStyleSheet(
                "QPushButton { font-weight: bold; font-size: 10px; "
                "border-radius: 3px; background: #4CAF50; color: white; }"
            )
            self.toggle_btn.setToolTip("Auto mode — click to add your own entries")
            self.user_input.hide()

    def set_auto_values(self, values: list[str]):
        """Set auto-generated values."""
        self._auto_values = list(values)
        self.auto_text.setPlainText(", ".join(values))
        self.count_label.setText(f"{len(values)} auto")

    def get_user_entries(self) -> list[str] | None:
        """Return user-added entries if in User mode, else None."""
        if not self._is_user_mode:
            return None
        text = self.user_input.toPlainText().strip()
        if not text:
            return None
        return [w.strip() for w in text.split(",") if w.strip()]

    def get_merged_values(self) -> list[str]:
        """Return deduplicated merge of auto + user values."""
        seen = set()
        merged = []
        for v in self._auto_values:
            low = v.lower()
            if low not in seen:
                seen.add(low)
                merged.append(v)
        user = self.get_user_entries()
        if user:
            for v in user:
                low = v.lower()
                if low not in seen:
                    seen.add(low)
                    merged.append(v)
        return merged

    def is_user_mode(self) -> bool:
        return self._is_user_mode

    def reset(self):
        self._is_user_mode = False
        self._auto_values = []
        self.auto_text.setPlainText("")
        self.user_input.clear()
        self.user_input.hide()
        self.count_label.setText("")
        self.toggle_btn.setText("A")
        self.toggle_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 10px; "
            "border-radius: 3px; background: #4CAF50; color: white; }"
        )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class MetadataPanel(QWidget):
    overrides_committed = Signal(int, object)  # (queue_index, UserOverrides)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_index: int = -1
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)

        # Clip Classification group
        classify_group = QGroupBox("Clip Classification")
        classify_form = QFormLayout(classify_group)

        self.clip_name_label = QLabel("—")
        self.clip_name_label.setStyleSheet("font-weight: bold;")
        classify_form.addRow("Clip:", self.clip_name_label)

        self.roll_type_field = EditableField(choices=["A-ROLL", "B-ROLL", "N/A"])
        classify_form.addRow("Roll Type:", self.roll_type_field)

        self.shot_type_field = EditableField(
            choices=["Close Up", "Medium Close Up", "Medium Shot",
                     "Wide Shot", "Extreme Wide Shot", "N/A"]
        )
        classify_form.addRow("Shot Type:", self.shot_type_field)

        self.camera_angle_field = EditableField(
            choices=["Eye Level", "High Angle", "Low Angle",
                     "Dutch Angle", "Overhead", "N/A"]
        )
        classify_form.addRow("Camera Angle:", self.camera_angle_field)

        self.camera_movement_field = EditableField(
            choices=["Static", "Pan", "Tilt", "Dolly",
                     "Handheld", "Tracking", "N/A"]
        )
        classify_form.addRow("Movement:", self.camera_movement_field)

        self.lighting_field = EditableField(
            choices=["Natural", "Studio", "Mixed", "Low Key", "High Key", "N/A"]
        )
        classify_form.addRow("Lighting:", self.lighting_field)

        self.location_field = EditableField(choices=["Interior", "Exterior", "N/A"])
        classify_form.addRow("Location:", self.location_field)

        self.subject_field = EditableField()
        classify_form.addRow("Subject:", self.subject_field)

        self.talking_head_field = EditableField(choices=["Yes", "No"])
        classify_form.addRow("Talking Head:", self.talking_head_field)

        layout.addWidget(classify_group)

        # Transcript Refinement group
        refine_group = QGroupBox("Transcript Refinement")
        refine_form = QFormLayout(refine_group)

        self.subject_name_field = EditableField()
        refine_form.addRow("Subject Name:", self.subject_name_field)

        self.is_interview_field = EditableField(choices=["Yes", "No"])
        refine_form.addRow("Interview:", self.is_interview_field)

        self.refined_subject_field = EditableField()
        refine_form.addRow("Description:", self.refined_subject_field)

        self.content_tags_field = EditableListField(
            placeholder="Waiting for LLM processing..."
        )
        refine_form.addRow("Content Tags:", self.content_tags_field)

        layout.addWidget(refine_group)

        # Transcript Summary group
        summary_group = QGroupBox("Transcript Summary")
        summary_form = QFormLayout(summary_group)

        self.title_field = EditableField()
        summary_form.addRow("Title:", self.title_field)

        self.summary_field = EditableField()
        summary_form.addRow("Summary:", self.summary_field)

        self.topics_field = EditableField()
        summary_form.addRow("Topics:", self.topics_field)

        layout.addWidget(summary_group)

        # Keywords group
        keywords_group = QGroupBox("Keywords")
        keywords_layout = QVBoxLayout(keywords_group)
        self.keywords_field = EditableListField(
            placeholder="Waiting for LLM processing..."
        )
        keywords_layout.addWidget(self.keywords_field)
        layout.addWidget(keywords_group)

        # Commit button
        self.commit_btn = QPushButton("Commit User Edits")
        self.commit_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; "
            "background: #FF9800; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #F57C00; }"
        )
        self.commit_btn.setToolTip(
            "Save your manual edits. User values overwrite auto for text fields, "
            "merge for keywords/tags."
        )
        self.commit_btn.clicked.connect(self._on_commit)
        layout.addWidget(self.commit_btn)

        # Multi-Camera group
        multicam_group = QGroupBox("Multi-Camera")
        multicam_layout = QVBoxLayout(multicam_group)
        self.multicam_id_label = self._make_ro_label()
        multicam_form = QFormLayout()
        multicam_form.addRow("Group ID:", self.multicam_id_label)
        multicam_layout.addLayout(multicam_form)
        self.multicam_clips_label = QLabel("")
        self.multicam_clips_label.setWordWrap(True)
        self.multicam_clips_label.setStyleSheet("font-size: 11px; color: gray;")
        multicam_layout.addWidget(self.multicam_clips_label)
        layout.addWidget(multicam_group)

        # Pipeline Status group (read-only, no editing needed)
        status_group = QGroupBox("Pipeline Status")
        status_form = QFormLayout(status_group)
        self.audio_level_label = self._make_ro_label()
        status_form.addRow("Audio Level:", self.audio_level_label)
        self.speech_ratio_label = self._make_ro_label()
        status_form.addRow("Speech Detected:", self.speech_ratio_label)
        self.folder_tags_label = self._make_ro_label()
        self.folder_tags_label.setWordWrap(True)
        status_form.addRow("Folder Tags:", self.folder_tags_label)
        self.frame_count_label = self._make_ro_label()
        status_form.addRow("Frames Extracted:", self.frame_count_label)
        self.transcript_length_label = self._make_ro_label()
        status_form.addRow("Transcript Length:", self.transcript_length_label)
        self.duration_label = self._make_ro_label()
        status_form.addRow("Duration:", self.duration_label)
        layout.addWidget(status_group)

        layout.addStretch()

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _make_ro_label(self) -> QLabel:
        """Create a read-only label for pipeline status fields."""
        label = QLabel("—")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    # -- All editable fields for iteration --
    def _editable_fields(self) -> list[EditableField]:
        return [
            self.roll_type_field, self.shot_type_field,
            self.camera_angle_field, self.camera_movement_field,
            self.lighting_field, self.location_field,
            self.subject_field, self.talking_head_field,
            self.subject_name_field, self.is_interview_field,
            self.refined_subject_field,
            self.title_field, self.summary_field, self.topics_field,
        ]

    def _editable_list_fields(self) -> list[EditableListField]:
        return [self.content_tags_field, self.keywords_field]

    def _ro_labels(self) -> list[QLabel]:
        return [
            self.multicam_id_label,
            self.audio_level_label, self.speech_ratio_label,
            self.folder_tags_label, self.frame_count_label,
            self.transcript_length_label, self.duration_label,
        ]

    # -- Clear / Reset --
    def clear(self):
        """Reset all fields for a new video."""
        self.clip_name_label.setText("—")
        for field in self._editable_fields():
            field.reset()
        for field in self._editable_list_fields():
            field.reset()
        for label in self._ro_labels():
            label.setText("—")
            label.setStyleSheet("")
        self.multicam_clips_label.setText("")

    def set_current_index(self, index: int):
        self._current_index = index

    # -- Auto-fill setters (pipeline calls these as stages complete) --

    def set_clip_name(self, name: str):
        self.clip_name_label.setText(name)

    def set_classification(self, c: ClipClassification):
        """Fill classification fields. Respects user mode."""
        roll_upper = c.roll_type.upper()
        if roll_upper == "A-ROLL":
            style = "font-weight: bold; color: #4CAF50;"
        elif roll_upper == "B-ROLL":
            style = "font-weight: bold; color: #2196F3;"
        else:
            style = "color: #999;"
        self.roll_type_field.set_auto_value(roll_upper, style)

        self.shot_type_field.set_auto_value(c.shot_type.title())
        self.camera_angle_field.set_auto_value(c.camera_angle.title())
        self.camera_movement_field.set_auto_value(c.camera_movement.title())
        self.lighting_field.set_auto_value(c.lighting.title())
        self.location_field.set_auto_value(c.location.title())
        self.subject_field.set_auto_value(c.subject)

        th_text = "Yes" if c.is_talking_head else "No"
        th_color = "#FF9800" if c.is_talking_head else "#4CAF50"
        self.talking_head_field.set_auto_value(
            th_text, f"font-weight: bold; color: {th_color};"
        )

    def set_refinement(self, r: ClipRefinement):
        """Fill refinement fields. Respects user mode."""
        name = r.subject_name
        if name.lower() != "unknown":
            self.subject_name_field.set_auto_value(
                name, "font-weight: bold; color: #4CAF50;"
            )
        else:
            self.subject_name_field.set_auto_value("Unknown", "color: #999;")

        interview_text = "Yes" if r.is_interview else "No"
        interview_color = "#4CAF50" if r.is_interview else "#FF9800"
        self.is_interview_field.set_auto_value(
            interview_text, f"font-weight: bold; color: {interview_color};"
        )

        self.refined_subject_field.set_auto_value(r.refined_subject)
        self.content_tags_field.set_auto_values(r.content_tags)

    def set_transcript_summary(self, s: TranscriptSummary):
        self.title_field.set_auto_value(s.title, "font-weight: bold;")
        self.summary_field.set_auto_value(s.summary)
        self.topics_field.set_auto_value(", ".join(s.topics))

    def set_transcript_length(self, chars: int):
        self.transcript_length_label.setText(f"{chars:,} characters")

    def set_frame_count(self, count: int):
        self.frame_count_label.setText(str(count))

    def set_duration(self, seconds: float):
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        self.duration_label.setText(f"{mins}:{secs:02d}")

    def set_audio_check(self, ac: AudioCheckResult):
        level_text = f"{ac.rms_db:.1f} dBFS (peak: {ac.peak_db:.1f})"
        if ac.has_audio:
            self.audio_level_label.setText(level_text)
            self.audio_level_label.setStyleSheet("color: #4CAF50;")
        else:
            self.audio_level_label.setText(f"{level_text} — SILENT")
            self.audio_level_label.setStyleSheet("color: #F44336; font-weight: bold;")

        if ac.speech_ratio > 0:
            pct = f"{ac.speech_ratio:.0%}"
            color = "#4CAF50" if ac.speech_ratio > 0.1 else "#FF9800"
            self.speech_ratio_label.setText(pct)
            self.speech_ratio_label.setStyleSheet(f"color: {color};")
        else:
            self.speech_ratio_label.setText("None" if not ac.has_audio else "—")

    def set_keywords(self, keywords: list[str]):
        self.keywords_field.set_auto_values(keywords)

    def set_multicam(self, group_id: str, other_clips: list[str]):
        """Show multi-cam group info."""
        self.multicam_id_label.setText(group_id)
        self.multicam_id_label.setStyleSheet("font-weight: bold; color: #9C27B0;")
        if other_clips:
            names = [os.path.basename(p) for p in other_clips]
            self.multicam_clips_label.setText(
                f"Other angles: {', '.join(names)}"
            )

    # -- Commit --

    def _on_commit(self):
        """Collect all user overrides and emit signal."""
        if self._current_index < 0:
            return
        overrides = self.get_user_overrides()
        self.overrides_committed.emit(self._current_index, overrides)

    def get_user_overrides(self) -> UserOverrides:
        """Collect all fields in User mode into a UserOverrides object."""
        def _bool_val(field: EditableField) -> bool | None:
            v = field.get_effective_value()
            if v is None:
                return None
            return v.lower() == "yes"

        return UserOverrides(
            roll_type=self.roll_type_field.get_effective_value(),
            shot_type=self.shot_type_field.get_effective_value(),
            camera_angle=self.camera_angle_field.get_effective_value(),
            camera_movement=self.camera_movement_field.get_effective_value(),
            lighting=self.lighting_field.get_effective_value(),
            location=self.location_field.get_effective_value(),
            subject=self.subject_field.get_effective_value(),
            is_talking_head=_bool_val(self.talking_head_field),
            subject_name=self.subject_name_field.get_effective_value(),
            is_interview=_bool_val(self.is_interview_field),
            refined_subject=self.refined_subject_field.get_effective_value(),
            content_tags=self.content_tags_field.get_user_entries(),
            title=self.title_field.get_effective_value(),
            summary=self.summary_field.get_effective_value(),
            topics=self._parse_csv(self.topics_field.get_effective_value()),
            keywords=self.keywords_field.get_user_entries(),
        )

    def _parse_csv(self, val: str | None) -> list[str] | None:
        if val is None:
            return None
        return [w.strip() for w in val.split(",") if w.strip()]

    # -- Full result display (for click-to-review) --

    def set_result(self, result: VideoResult):
        """Fill everything from a complete result."""
        self.clear()
        self.set_clip_name(os.path.basename(result.video_path))
        if result.clip_classification:
            self.set_classification(result.clip_classification)
        if result.clip_refinement:
            self.set_refinement(result.clip_refinement)
        if result.audio_check:
            self.set_audio_check(result.audio_check)
        if result.folder_tags:
            self.folder_tags_label.setText(", ".join(result.folder_tags))
            self.folder_tags_label.setStyleSheet("color: #2196F3;")
        if result.transcript_summary:
            self.set_transcript_summary(result.transcript_summary)
        if result.transcript:
            self.set_transcript_length(len(result.transcript))
        self.set_frame_count(result.frame_count)
        self.set_duration(result.duration_seconds)
        if result.keywords:
            self.set_keywords(result.keywords)
        if result.multicam_group_id:
            self.set_multicam(result.multicam_group_id, [])
