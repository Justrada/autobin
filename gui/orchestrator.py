"""Pipeline orchestrator: sequences workers per video, manages the queue."""

from __future__ import annotations

import os
import shutil
import tempfile
import time

from PySide6.QtCore import QObject, Signal

from core.resolve_export import export_csv, export_combined_csv
from core.schemas import (
    AppSettings, AudioCheckResult, ClipClassification, ClipRefinement,
    MultiCamGroup, StepTiming, TranscriptSummary, UserOverrides, VideoResult,
)
from gui.workers import (
    AudioCheckWorker,
    ClipClassificationWorker,
    ClipRefinementWorker,
    FrameExtractionWorker,
    LLMWorker,
    MultiCamDetectionWorker,
    TranscriptionWorker,
)


class PipelineOrchestrator(QObject):
    # Signals for GUI updates
    log = Signal(str)
    video_started = Signal(int, str)             # queue_index, video_path
    video_skipped = Signal(int, str)             # queue_index, video_path (already done)
    frame_progress = Signal(int, int)
    transcript_progress = Signal(int, int)
    llm_progress = Signal(int, int)
    classification_done = Signal(object)          # ClipClassification | None
    refinement_done = Signal(object)              # ClipRefinement | None
    transcript_summary_done = Signal(object)      # TranscriptSummary | None
    transcript_text_done = Signal(str)            # raw transcript text
    keywords_done = Signal(list)                  # final keywords list
    frame_count_update = Signal(int)              # number of frames extracted
    frames_available = Signal(list)               # frame paths for film strip
    video_completed = Signal(int, object)         # queue_index, VideoResult
    video_error = Signal(int, str)                # queue_index, error_msg
    multicam_groups_found = Signal(list)          # list of MultiCamGroup
    queue_completed = Signal()
    queue_progress = Signal(int, int, float)      # completed, total, est_remaining_s

    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._queue: list[str] = []
        self._current_index = 0
        self._running = False

        # Per-video state
        self._temp_frames_dir: str | None = None
        self._frame_paths: list[str] = []
        self._classify_frame_paths: list[str] = []
        self._transcript: str = ""
        self._classification: ClipClassification | None = None
        self._refinement: ClipRefinement | None = None
        self._audio_check_result: AudioCheckResult | None = None
        self._frames_done = False
        self._transcript_done = False
        self._classify_done = False

        # Step timing for current video
        self._step_timing = StepTiming()
        self._video_start_time: float = 0.0
        self._audio_check_start: float = 0.0
        self._classify_start: float = 0.0
        self._frames_start: float = 0.0
        self._transcribe_start: float = 0.0
        self._llm_start: float = 0.0
        self._refine_start: float = 0.0

        # Queue-level timing for ETA
        self._completed_count = 0
        self._completed_times: list[float] = []   # seconds per completed video
        self._timing_by_ext: dict[str, list[StepTiming]] = {}  # extension -> timings

        # Store completed results for click-to-view
        self._results: dict[int, VideoResult] = {}

    def set_queue(self, video_paths: list[str],
                  folder_tags_map: dict[int, list[str]] | None = None):
        self._queue = list(video_paths)
        self._current_index = 0
        self._results.clear()
        self._folder_tags_map = folder_tags_map or {}
        self._completed_count = 0
        self._completed_times.clear()

    def get_result(self, index: int) -> VideoResult | None:
        """Get completed result for a queue index (for click-to-view)."""
        return self._results.get(index)

    def start(self):
        if not self._queue:
            return
        self._running = True
        self._current_index = 0
        self._completed_count = 0
        self._completed_times.clear()
        self._process_next()

    def stop(self):
        self._running = False

    def _get_output_folder(self) -> str:
        """Return the centralized output folder for CSVs."""
        return self.settings.export.output_folder

    def _csv_exists_for(self, video_path: str) -> bool:
        """Check if a CSV has already been exported for this video."""
        output_folder = self._get_output_folder()
        if not output_folder:
            return False
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(output_folder, f"{video_name}.csv")
        return os.path.isfile(csv_path)

    def apply_overrides(self, index: int, overrides: UserOverrides):
        """Apply user overrides to a completed result and re-export CSV."""
        result = self._results.get(index)
        if not result:
            self.log.emit(f"[warning] No result at index {index} to apply overrides to")
            return

        # Apply classification overrides
        if result.clip_classification:
            c = result.clip_classification
            if overrides.roll_type is not None:
                c.roll_type = overrides.roll_type
            if overrides.shot_type is not None:
                c.shot_type = overrides.shot_type
            if overrides.camera_angle is not None:
                c.camera_angle = overrides.camera_angle
            if overrides.camera_movement is not None:
                c.camera_movement = overrides.camera_movement
            if overrides.lighting is not None:
                c.lighting = overrides.lighting
            if overrides.location is not None:
                c.location = overrides.location
            if overrides.subject is not None:
                c.subject = overrides.subject
            if overrides.is_talking_head is not None:
                c.is_talking_head = overrides.is_talking_head

        # Apply refinement overrides
        if result.clip_refinement:
            r = result.clip_refinement
            if overrides.subject_name is not None:
                r.subject_name = overrides.subject_name
            if overrides.is_interview is not None:
                r.is_interview = overrides.is_interview
            if overrides.refined_subject is not None:
                r.refined_subject = overrides.refined_subject
            # Content tags: merge
            if overrides.content_tags is not None:
                existing = {t.lower() for t in r.content_tags}
                for tag in overrides.content_tags:
                    if tag.lower() not in existing:
                        r.content_tags.append(tag)
                        existing.add(tag.lower())

        # Apply summary overrides
        if result.transcript_summary:
            s = result.transcript_summary
            if overrides.title is not None:
                s.title = overrides.title
            if overrides.summary is not None:
                s.summary = overrides.summary
            if overrides.topics is not None:
                s.topics = overrides.topics

        # Keywords: merge
        if overrides.keywords is not None:
            existing = {k.lower() for k in result.keywords}
            for kw in overrides.keywords:
                if kw.lower() not in existing:
                    result.keywords.append(kw)
                    existing.add(kw.lower())

        self.log.emit(f"[commit] User overrides applied to {os.path.basename(result.video_path)}")

        # Re-export CSV
        output_folder = self._get_output_folder()
        if output_folder:
            video_name = os.path.splitext(os.path.basename(result.video_path))[0]
            csv_path = os.path.join(output_folder, f"{video_name}.csv")
            export_csv(result, csv_path)
            self.log.emit(f"[export] CSV updated: {csv_path}")

            # Re-export combined CSV if all results exist
            if len(self._results) > 1:
                self._export_combined_csv()

    def _run_multicam_detection(self):
        """Run multi-cam detection after all clips are processed."""
        self.log.emit("\n" + "=" * 50)
        self.log.emit("Running multi-camera angle detection...")
        self.log.emit("=" * 50)

        all_results = [self._results[i] for i in sorted(self._results)]
        self._multicam_worker = MultiCamDetectionWorker(all_results)
        self._multicam_worker.log.connect(self.log.emit)
        self._multicam_worker.finished.connect(self._on_multicam_done)
        self._multicam_worker.error.connect(self._on_multicam_error)
        self._multicam_worker.start()

    def _on_multicam_done(self, groups: list):
        """Apply multicam group IDs to results, re-export, then finish."""
        if groups:
            # Assign group IDs to matching results
            for group in groups:
                for clip_path in group.clip_paths:
                    for result in self._results.values():
                        if result.video_path == clip_path:
                            result.multicam_group_id = group.group_id
                            break

            # Re-export everything with multicam info
            self._export_combined_csv()

            # Re-export individual CSVs with group IDs
            output_folder = self._get_output_folder()
            if output_folder:
                for result in self._results.values():
                    if result.multicam_group_id:
                        video_name = os.path.splitext(
                            os.path.basename(result.video_path)
                        )[0]
                        csv_path = os.path.join(output_folder, f"{video_name}.csv")
                        export_csv(result, csv_path)

        self.multicam_groups_found.emit(groups)
        self.queue_completed.emit()
        self._running = False

    def _on_multicam_error(self, msg: str):
        self.log.emit(f"[warning] Multi-cam detection failed: {msg}")
        self.queue_completed.emit()
        self._running = False

    def _export_combined_csv(self):
        """Write a single combined CSV with all completed results."""
        output_folder = self._get_output_folder()
        if not output_folder:
            return
        results = [self._results[i] for i in sorted(self._results) if i in self._results]
        if not results:
            return
        combined_path = os.path.join(output_folder, "resolve_metadata_all.csv")
        export_combined_csv(results, combined_path)
        self.log.emit(f"[export] Combined CSV saved: {combined_path} ({len(results)} clips)")

    def _estimate_remaining(self) -> float:
        """Estimate remaining seconds based on average per-video time."""
        if not self._completed_times:
            return 0.0
        avg = sum(self._completed_times) / len(self._completed_times)
        remaining_count = len(self._queue) - self._current_index
        return avg * remaining_count

    def _process_next(self):
        if not self._running or self._current_index >= len(self._queue):
            self._export_combined_csv()
            # Run multi-cam detection if we have 2+ results with transcripts
            results_with_transcripts = [
                r for r in self._results.values()
                if r.transcript and len(r.transcript.strip()) > 50
            ]
            if len(results_with_transcripts) >= 2:
                self._run_multicam_detection()
            else:
                self.queue_completed.emit()
                self._running = False
            return

        video_path = self._queue[self._current_index]

        # --- Resume support: skip if CSV already exists ---
        if self._csv_exists_for(video_path):
            self.log.emit(f"[skip] CSV already exists for {os.path.basename(video_path)}")
            self.video_skipped.emit(self._current_index, video_path)
            self._completed_count += 1
            est = self._estimate_remaining()
            self.queue_progress.emit(self._completed_count, len(self._queue), est)
            self._current_index += 1
            self._process_next()
            return

        self.video_started.emit(self._current_index, video_path)
        self.log.emit(f"\n{'='*50}")
        self.log.emit(f"Processing: {os.path.basename(video_path)}")
        self.log.emit(f"{'='*50}")

        # Reset per-video state
        self._frame_paths = []
        self._classify_frame_paths = []
        self._transcript = ""
        self._classification = None
        self._refinement = None
        self._audio_check_result = None
        self._frames_done = False
        self._transcript_done = False
        self._classify_done = False

        # Reset step timing
        self._step_timing = StepTiming()
        self._video_start_time = time.monotonic()

        # Temp dir for frames (cleaned up after LLM processing)
        self._temp_frames_dir = tempfile.mkdtemp(prefix="framex_")

        # Start audio check + classification in parallel (both are fast)
        self._classify_start = time.monotonic()
        self._classify_worker = ClipClassificationWorker(
            video_path, self.settings, keep_frames=True
        )
        self._classify_worker.log.connect(self.log.emit)
        self._classify_worker.finished.connect(self._on_classify_done)
        self._classify_worker.error.connect(self._on_classify_error)
        self._classify_worker.start()

        if self.settings.transcription.audio_check:
            self._audio_check_start = time.monotonic()
            self._audio_worker = AudioCheckWorker(video_path, self.settings)
            self._audio_worker.log.connect(self.log.emit)
            self._audio_worker.finished.connect(self._on_audio_check_done)
            self._audio_worker.error.connect(self._on_audio_check_error)
            self._audio_worker.start()
        else:
            # Skip audio check, go straight to transcription
            self._start_transcription(video_path)

    def _start_transcription(self, video_path: str):
        """Launch the transcription worker."""
        self._transcribe_start = time.monotonic()
        self._transcript_worker = TranscriptionWorker(video_path, self.settings)
        self._transcript_worker.log.connect(self.log.emit)
        self._transcript_worker.progress.connect(self.transcript_progress.emit)
        self._transcript_worker.finished.connect(self._on_transcript_done)
        self._transcript_worker.error.connect(self._on_transcript_error)
        self._transcript_worker.start()

    def _on_audio_check_done(self, result: dict):
        self._step_timing.audio_check_s = time.monotonic() - self._audio_check_start
        self._audio_check_result = AudioCheckResult(**result)
        video_path = self._queue[self._current_index]

        if result.get("has_audio", False):
            self._start_transcription(video_path)
        else:
            # BUG FIX: Always transcribe anyway, just log the warning.
            # The audio check being negative doesn't mean the transcript is useless —
            # 360 cameras and other sources may have speech that fails the
            # speech-ratio check due to ambient noise characteristics.
            # We still transcribe and let the downstream length checks
            # (len > 20 for refinement, strip() for summary) decide.
            self.log.emit("[orchestrator] Audio check: low/no speech detected. "
                          "Transcribing anyway (downstream checks will filter).")
            self._start_transcription(video_path)

    def _on_audio_check_error(self, msg: str):
        self.log.emit(f"[warning] Audio check failed: {msg} — transcribing anyway")
        video_path = self._queue[self._current_index]
        self._start_transcription(video_path)

    def _on_classify_done(self, result: ClipClassification | None, frame_paths: list[str]):
        self._step_timing.classification_s = time.monotonic() - self._classify_start
        self._classification = result
        self._classify_frame_paths = frame_paths
        self._classify_done = True

        # If audio check already ran and found no audio, override talking_head
        if result and self._audio_check_result and not self._audio_check_result.has_audio:
            result.is_talking_head = False
            result.roll_type = "b-roll"
            self.log.emit("[orchestrator] Low audio level — defaulting to b-roll")

        self.classification_done.emit(result)

        if result:
            self.log.emit(f"[orchestrator] Classification: {result.roll_type} | "
                          f"{result.shot_type} | talking_head={result.is_talking_head}")

        video_path = self._queue[self._current_index]
        if result and result.is_talking_head:
            self.log.emit("[orchestrator] Talking head detected — using classification frames for keywords")
            self._frame_paths = list(frame_paths)
            self._frames_done = True
            self.frame_count_update.emit(len(frame_paths))
            self.frames_available.emit(list(frame_paths))
            self._check_ready_for_llm()
        else:
            self._frames_start = time.monotonic()
            self._frame_worker = FrameExtractionWorker(
                video_path, self.settings,
                os.path.join(self._temp_frames_dir, "frames")
            )
            self._frame_worker.log.connect(self.log.emit)
            self._frame_worker.progress.connect(self.frame_progress.emit)
            self._frame_worker.finished.connect(self._on_frames_done)
            self._frame_worker.error.connect(self._on_frame_error)
            self._frame_worker.start()

    def _on_classify_error(self, msg: str):
        self.log.emit(f"[warning] Classification failed: {msg}")
        self._classification = None
        self._classify_done = True
        self.classification_done.emit(None)

        video_path = self._queue[self._current_index]
        self._frames_start = time.monotonic()
        self._frame_worker = FrameExtractionWorker(
            video_path, self.settings,
            os.path.join(self._temp_frames_dir, "frames")
        )
        self._frame_worker.log.connect(self.log.emit)
        self._frame_worker.progress.connect(self.frame_progress.emit)
        self._frame_worker.finished.connect(self._on_frames_done)
        self._frame_worker.error.connect(self._on_frame_error)
        self._frame_worker.start()

    def _on_frames_done(self, paths: list[str]):
        self._step_timing.frame_extraction_s = time.monotonic() - self._frames_start
        self._frame_paths = paths
        self._frames_done = True
        self.frame_count_update.emit(len(paths))
        self.frames_available.emit(paths)
        self.log.emit(f"[orchestrator] Frame extraction complete: {len(paths)} frames")
        self._check_ready_for_llm()

    def _on_frame_error(self, msg: str):
        self.log.emit(f"[error] Frame extraction failed: {msg}")
        self._frames_done = True
        self.frame_count_update.emit(0)
        self._check_ready_for_llm()

    def _on_transcript_done(self, text: str):
        self._step_timing.transcription_s = time.monotonic() - self._transcribe_start
        self._transcript = text
        self._transcript_done = True
        self.transcript_text_done.emit(text)
        self.log.emit(f"[orchestrator] Transcription complete: {len(text)} chars")
        self._check_ready_for_llm()

    def _on_transcript_error(self, msg: str):
        self.log.emit(f"[warning] Transcription failed: {msg}")
        self._transcript = ""
        self._transcript_done = True
        self.transcript_text_done.emit("")
        self._check_ready_for_llm()

    def _check_ready_for_llm(self):
        if not (self._frames_done and self._transcript_done and self._classify_done):
            return

        if not self._frame_paths and not self._transcript:
            self.log.emit("[warning] No frames or transcript — skipping LLM.")
            self._finalize_video(None, [])
            return

        self.log.emit("[orchestrator] Starting LLM pipeline...")
        self._llm_start = time.monotonic()
        self._llm_worker = LLMWorker(
            self._transcript, self._frame_paths, self.settings
        )
        self._llm_worker.log.connect(self.log.emit)
        self._llm_worker.progress.connect(self.llm_progress.emit)
        self._llm_worker.finished.connect(self._on_llm_done)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_worker.start()

    def _on_llm_done(self, summary, keywords: list[str]):
        self._step_timing.llm_s = time.monotonic() - self._llm_start
        if summary:
            self.transcript_summary_done.emit(summary)
        self.keywords_done.emit(keywords)

        if self._classification and self._transcript and len(self._transcript.strip()) > 20:
            self.log.emit("[orchestrator] Refining classification with transcript...")
            self._refine_start = time.monotonic()
            self._refine_worker = ClipRefinementWorker(
                self._transcript, self._classification, self.settings
            )
            self._refine_worker.log.connect(self.log.emit)
            self._refine_worker.finished.connect(
                lambda r: self._on_refinement_done(r, summary, keywords)
            )
            self._refine_worker.error.connect(
                lambda msg: self._on_refinement_error(msg, summary, keywords)
            )
            self._refine_worker.start()
        else:
            self._finalize_video(summary, keywords)

    def _on_refinement_done(self, refinement: ClipRefinement | None,
                            summary: TranscriptSummary | None, keywords: list[str]):
        self._step_timing.refinement_s = time.monotonic() - self._refine_start
        self._refinement = refinement
        if refinement:
            self.refinement_done.emit(refinement)
            for tag in refinement.content_tags:
                tag_lower = tag.strip().lower()
                if tag_lower and tag_lower not in {k.lower() for k in keywords}:
                    keywords.append(tag.strip())
            self.keywords_done.emit(keywords)
        self._finalize_video(summary, keywords)

    def _on_refinement_error(self, msg: str,
                              summary: TranscriptSummary | None, keywords: list[str]):
        self.log.emit(f"[warning] Refinement failed: {msg}")
        self._finalize_video(summary, keywords)

    def _on_llm_error(self, msg: str):
        self.log.emit(f"[error] LLM pipeline failed: {msg}")
        self._finalize_video(None, [])

    def _finalize_video(self, summary: TranscriptSummary | None, keywords: list[str]):
        video_path = self._queue[self._current_index]

        # Finalize total timing
        self._step_timing.total_s = time.monotonic() - self._video_start_time

        from core.frames import get_video_info
        info = get_video_info(video_path)

        # Inject folder tags into keywords and content tags
        folder_tags = self._folder_tags_map.get(self._current_index, [])
        if folder_tags:
            existing_lower = {k.lower() for k in keywords}
            for tag in folder_tags:
                if tag.lower() not in existing_lower:
                    keywords.append(tag)
                    existing_lower.add(tag.lower())
            # Also inject into refinement content_tags if available
            if self._refinement and self._refinement.content_tags is not None:
                existing_tags_lower = {t.lower() for t in self._refinement.content_tags}
                for tag in folder_tags:
                    if tag.lower() not in existing_tags_lower:
                        self._refinement.content_tags.append(tag)
                        existing_tags_lower.add(tag.lower())

        ext = os.path.splitext(video_path)[1].lower()

        result = VideoResult(
            video_path=video_path,
            transcript=self._transcript,
            transcript_summary=summary,
            clip_classification=self._classification,
            clip_refinement=self._refinement,
            audio_check=self._audio_check_result,
            keywords=keywords,
            folder_tags=folder_tags,
            frame_count=len(self._frame_paths),
            duration_seconds=info.get("duration", 0),
            timing=self._step_timing,
            file_extension=ext,
        )

        # Store result for click-to-view
        self._results[self._current_index] = result

        # Record timing for ETA and per-extension stats
        self._completed_count += 1
        self._completed_times.append(self._step_timing.total_s)
        if ext not in self._timing_by_ext:
            self._timing_by_ext[ext] = []
        self._timing_by_ext[ext].append(self._step_timing)

        # Log timing summary
        t = self._step_timing
        self.log.emit(
            f"[timing] {os.path.basename(video_path)}: "
            f"total={t.total_s:.1f}s | "
            f"audio={t.audio_check_s:.1f}s | "
            f"classify={t.classification_s:.1f}s | "
            f"frames={t.frame_extraction_s:.1f}s | "
            f"transcribe={t.transcription_s:.1f}s | "
            f"llm={t.llm_s:.1f}s | "
            f"refine={t.refinement_s:.1f}s"
        )

        # Emit queue-level progress with ETA
        est = self._estimate_remaining()
        self.queue_progress.emit(self._completed_count, len(self._queue), est)

        # Export CSV to centralized output folder
        output_folder = self._get_output_folder()
        if output_folder and (keywords or self._classification):
            os.makedirs(output_folder, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(output_folder, f"{video_name}.csv")
            export_csv(result, csv_path)
            self.log.emit(f"[export] CSV saved: {csv_path}")

        # Clean up temp frames
        if self._temp_frames_dir and os.path.isdir(self._temp_frames_dir):
            shutil.rmtree(self._temp_frames_dir, ignore_errors=True)

        # Clean up classification temp frames
        if self._classify_frame_paths:
            tmp_dir = os.path.dirname(self._classify_frame_paths[0])
            if os.path.isdir(tmp_dir) and tmp_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(tmp_dir, ignore_errors=True)

        self.video_completed.emit(self._current_index, result)

        self._current_index += 1
        self._process_next()

    def get_active_workers(self) -> list:
        """Return list of currently running QThread workers."""
        workers = []
        for attr in ("_audio_worker", "_classify_worker", "_transcript_worker",
                      "_frame_worker", "_llm_worker", "_refine_worker",
                      "_multicam_worker"):
            w = getattr(self, attr, None)
            if w is not None and w.isRunning():
                workers.append(w)
        return workers

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        """Return average timing per file extension for benchmarking display."""
        stats = {}
        for ext, timings in self._timing_by_ext.items():
            n = len(timings)
            if n == 0:
                continue
            stats[ext] = {
                "count": n,
                "avg_total": sum(t.total_s for t in timings) / n,
                "avg_audio_check": sum(t.audio_check_s for t in timings) / n,
                "avg_classification": sum(t.classification_s for t in timings) / n,
                "avg_frame_extraction": sum(t.frame_extraction_s for t in timings) / n,
                "avg_transcription": sum(t.transcription_s for t in timings) / n,
                "avg_llm": sum(t.llm_s for t in timings) / n,
                "avg_refinement": sum(t.refinement_s for t in timings) / n,
            }
        return stats
