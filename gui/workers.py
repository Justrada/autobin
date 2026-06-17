"""QThread workers for background processing."""

from __future__ import annotations

import tempfile
import time

from PySide6.QtCore import QThread, Signal

from core.frames import run_frame_pipeline, sample_fast_frames, get_video_info
from core.llm import classify_clip, refine_classification, run_llm_pipeline
from core.multicam import find_multicam_groups
from core.schemas import AppSettings, ClipClassification, ClipRefinement, TranscriptSummary, VideoResult
from core.transcribe import check_audio_level, transcribe_video


# ---------------------------------------------------------------------------
# Cancellable base class
# ---------------------------------------------------------------------------

class CancellableWorker(QThread):
    """Base QThread with a cooperative cancellation flag.

    Workers should periodically check ``self.is_cancelled`` and exit early
    when it returns True.  The core functions can't check this directly,
    but the long-running subprocess calls already have timeouts, and
    :pymethod:`terminate` acts as a hard stop for truly stuck threads.
    """

    _cancelled: bool = False

    def cancel(self):
        """Request cooperative cancellation."""
        self._cancelled = True

    @property
    def is_cancelled(self):
        return self._cancelled


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class AudioCheckWorker(CancellableWorker):
    """Quick ffmpeg-based audio level check. Runs before transcription."""
    log = Signal(str)
    finished = Signal(dict)  # {has_audio, rms_db, peak_db, speech_ratio}
    error = Signal(str)

    def __init__(self, video_path: str, settings: AppSettings):
        super().__init__()
        self.video_path = video_path
        self.settings = settings

    def run(self):
        try:
            result = check_audio_level(
                self.video_path,
                threshold_db=self.settings.transcription.noise_floor_db,
                log=lambda msg: self.log.emit(msg),
            )
            if not self.is_cancelled:
                self.finished.emit(result)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))


class ClipClassificationWorker(CancellableWorker):
    log = Signal(str)
    finished = Signal(object, list)  # (ClipClassification | None, frame_paths)
    error = Signal(str)

    def __init__(self, video_path: str, settings: AppSettings, keep_frames: bool = False):
        super().__init__()
        self.video_path = video_path
        self.settings = settings
        self.keep_frames = keep_frames

    def run(self):
        try:
            result, frame_paths = classify_clip(
                self.video_path,
                self.settings.llm,
                log=lambda msg: self.log.emit(msg),
                keep_frames=self.keep_frames,
            )
            if not self.is_cancelled:
                self.finished.emit(result, frame_paths)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))


class ClipRefinementWorker(CancellableWorker):
    log = Signal(str)
    finished = Signal(object)  # ClipRefinement | None
    error = Signal(str)

    def __init__(self, transcript: str, classification: ClipClassification,
                 settings: AppSettings):
        super().__init__()
        self.transcript = transcript
        self.classification = classification
        self.settings = settings

    def run(self):
        try:
            result = refine_classification(
                self.transcript,
                self.classification,
                self.settings.llm,
                log=lambda msg: self.log.emit(msg),
            )
            if not self.is_cancelled:
                self.finished.emit(result)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))


class FrameExtractionWorker(CancellableWorker):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(list)  # list of frame paths
    error = Signal(str)

    def __init__(self, video_path: str, settings: AppSettings, output_dir: str):
        super().__init__()
        self.video_path = video_path
        self.settings = settings
        self.output_dir = output_dir

    def run(self):
        try:
            ingest = self.settings.ingest
            saved = run_frame_pipeline(
                video_path=self.video_path,
                output_dir=self.output_dir,
                threshold=ingest.threshold,
                target_fpm=ingest.target_fpm,
                max_frames=ingest.max_frames,
                time_budget=ingest.time_budget,
                metric=ingest.metric,
                log=lambda msg: self.log.emit(msg),
                progress=lambda cur, tot: self.progress.emit(cur, tot),
            )
            if not self.is_cancelled:
                self.finished.emit(saved)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))


class TranscriptionWorker(CancellableWorker):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(str)  # transcript text
    error = Signal(str)

    def __init__(self, video_path: str, settings: AppSettings):
        super().__init__()
        self.video_path = video_path
        self.settings = settings
        # MLX-Whisper large model needs a bigger stack for recursive compile_dfs
        self.setStackSize(32 * 1024 * 1024)  # 32 MB stack

    def run(self):
        try:
            self.progress.emit(0, 1)
            text = transcribe_video(
                self.video_path,
                self.settings.transcription,
                log=lambda msg: self.log.emit(msg),
            )
            self.progress.emit(1, 1)
            if not self.is_cancelled:
                self.finished.emit(text)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))


class LLMWorker(CancellableWorker):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object, list)  # (TranscriptSummary | None, keywords)
    error = Signal(str)

    def __init__(self, transcript: str, frame_paths: list[str], settings: AppSettings):
        super().__init__()
        self.transcript = transcript
        self.frame_paths = frame_paths
        self.settings = settings

    def run(self):
        try:
            summary, keywords = run_llm_pipeline(
                transcript=self.transcript,
                frame_paths=self.frame_paths,
                settings=self.settings.llm,
                log=lambda msg: self.log.emit(msg),
                progress=lambda cur, tot: self.progress.emit(cur, tot),
            )
            if not self.is_cancelled:
                self.finished.emit(summary, keywords)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))


class FastFrameWorker(CancellableWorker):
    """Grab N equidistant frames via seeking. Used in fast mode."""
    log = Signal(str)
    finished = Signal(list)  # list of frame paths
    error = Signal(str)

    def __init__(self, video_path: str, output_dir: str,
                 n_frames: int = 5, max_width: int = 640):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.n_frames = n_frames
        self.max_width = max_width

    def run(self):
        try:
            info = get_video_info(self.video_path)
            duration = info.get("duration", 0)
            if duration <= 0:
                self.error.emit("Could not determine video duration")
                return
            frames = sample_fast_frames(
                self.video_path, self.output_dir, duration,
                n_frames=self.n_frames, max_width=self.max_width,
                log=lambda msg: self.log.emit(msg),
            )
            if not self.is_cancelled:
                self.finished.emit(frames)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))


class MultiCamDetectionWorker(CancellableWorker):
    """Post-queue worker that compares transcripts to find multi-cam groups."""
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(list)  # list of MultiCamGroup
    error = Signal(str)

    def __init__(self, results: list[VideoResult], threshold: float = 0.4):
        super().__init__()
        self.results = results
        self.threshold = threshold

    def run(self):
        try:
            groups = find_multicam_groups(
                self.results,
                threshold=self.threshold,
                log=lambda msg: self.log.emit(msg),
                progress=lambda cur, tot: self.progress.emit(cur, tot),
            )
            if not self.is_cancelled:
                self.finished.emit(groups)
        except Exception as e:
            if not self.is_cancelled:
                self.error.emit(str(e))
