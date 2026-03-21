"""QThread workers for background processing."""

from __future__ import annotations

import tempfile

from PySide6.QtCore import QThread, Signal

from core.frames import run_frame_pipeline
from core.llm import classify_clip, refine_classification, run_llm_pipeline
from core.multicam import find_multicam_groups
from core.schemas import AppSettings, ClipClassification, ClipRefinement, TranscriptSummary, VideoResult
from core.transcribe import check_audio_level, transcribe_video


class AudioCheckWorker(QThread):
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
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ClipClassificationWorker(QThread):
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
            self.finished.emit(result, frame_paths)
        except Exception as e:
            self.error.emit(str(e))


class ClipRefinementWorker(QThread):
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
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class FrameExtractionWorker(QThread):
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
                metric=ingest.metric,
                offset=ingest.offset,
                log=lambda msg: self.log.emit(msg),
                progress=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit(saved)
        except Exception as e:
            self.error.emit(str(e))


class TranscriptionWorker(QThread):
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
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class LLMWorker(QThread):
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
            self.finished.emit(summary, keywords)
        except Exception as e:
            self.error.emit(str(e))


class MultiCamDetectionWorker(QThread):
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
            self.finished.emit(groups)
        except Exception as e:
            self.error.emit(str(e))
