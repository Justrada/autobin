"""Pydantic models for settings, LLM structured output, and pipeline results."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class IngestSettings(BaseModel):
    threshold: float | None = None  # None = auto-tune
    metric: str = "histogram"       # histogram | ssim | phash
    target_fpm: float = 4.0
    offset: int = 10


class LLMSettings(BaseModel):
    backend: str = "ollama"                # ollama | openai | anthropic
    model: str = "qwen3.5:latest"         # Ollama model name
    api_key: str | None = None
    base_url: str = "http://localhost:11434"
    context_window: int = 131072           # tokens (qwen3.5vl supports 262K)
    vlm_input_resolution: int = 480        # downscale height before sending to LLM
    max_images_per_batch: int = 4          # max images per vision request


class TranscriptionSettings(BaseModel):
    backend: str = "mlx-whisper"   # mlx-whisper | faster-whisper
    model_size: str = "base"       # tiny | base | small | medium | large
    vocabulary: str = ""           # comma-separated custom words (e.g. "Niobe, Stranger Comics")
    audio_check: bool = True       # check audio level before transcribing
    noise_floor_db: float = -50.0  # dBFS threshold — below this = silence


class ClipClassification(BaseModel):
    """Quick LLM classification from 4 evenly-spaced frames."""
    shot_type: str = Field(description="e.g. close up, medium close up, medium shot, wide shot, extreme wide shot, or N/A")
    camera_angle: str = Field(description="e.g. eye level, high angle, low angle, dutch angle, overhead, or N/A")
    camera_movement: str = Field(description="e.g. static, pan, tilt, dolly, handheld, tracking, or N/A")
    lighting: str = Field(description="e.g. natural, studio, mixed, low key, high key, or N/A")
    location: str = Field(description="interior, exterior, or N/A")
    subject: str = Field(description="Brief description of the main subject, or N/A")
    roll_type: str = Field(description="a-roll, b-roll, or N/A")
    is_talking_head: bool = Field(description="True if this is a talking head / interview shot, false otherwise")


class ClipRefinement(BaseModel):
    """Returned by transcript-based refinement of classification."""
    subject_name: str = Field(description="Name of the person if they introduced themselves, or 'unknown'")
    is_interview: bool = Field(description="True if this is an actual interview, false if background chatter")
    refined_subject: str = Field(description="Better description with transcript context")
    content_tags: list[str] = Field(description="3-8 topical tags from the transcript")


class ExportSettings(BaseModel):
    format: str = "csv"            # csv | fcpxml | resolve-api
    output_folder: str = ""        # centralized output folder for all CSVs (empty = prompt on first run)


class AppSettings(BaseModel):
    ingest: IngestSettings = IngestSettings()
    llm: LLMSettings = LLMSettings()
    transcription: TranscriptionSettings = TranscriptionSettings()
    export: ExportSettings = ExportSettings()


# ---------------------------------------------------------------------------
# LLM Structured Output Schemas
# ---------------------------------------------------------------------------

class TranscriptSummary(BaseModel):
    """Returned by LLM Step 1: summarize the audio transcript."""
    title: str = Field(description="Short descriptive title for the video")
    summary: str = Field(description="2-4 sentence summary of the video content")
    topics: list[str] = Field(description="Main topics discussed or shown")


class ImageKeywordBatch(BaseModel):
    """Returned by LLM Step 2: keywords for a batch of images."""
    keywords: list[str] = Field(description="DaVinci Resolve logging keywords for these frames")
    scene_descriptions: list[str] = Field(description="One short description per image")


# ---------------------------------------------------------------------------
# Pipeline Results
# ---------------------------------------------------------------------------

class UserOverrides(BaseModel):
    """User-entered overrides for auto-filled metadata fields.
    None means 'use auto value'. For keywords/content_tags, values are merged."""
    roll_type: str | None = None
    shot_type: str | None = None
    camera_angle: str | None = None
    camera_movement: str | None = None
    lighting: str | None = None
    location: str | None = None
    subject: str | None = None
    is_talking_head: bool | None = None
    subject_name: str | None = None
    is_interview: bool | None = None
    refined_subject: str | None = None
    content_tags: list[str] | None = None   # merged with auto
    title: str | None = None
    summary: str | None = None
    topics: list[str] | None = None
    keywords: list[str] | None = None       # merged with auto


class MultiCamMatch(BaseModel):
    """A pair of clips identified as multi-cam angles of the same scene."""
    video_path_a: str
    video_path_b: str
    similarity_score: float
    matched_trigrams: int
    total_trigrams_a: int
    total_trigrams_b: int


class MultiCamGroup(BaseModel):
    """A group of clips that are all angles of the same scene."""
    group_id: str                  # e.g. "MC_001"
    clip_paths: list[str]
    matches: list[MultiCamMatch]


class AudioCheckResult(BaseModel):
    """Result of pre-transcription audio level check."""
    has_audio: bool = False
    rms_db: float = -96.0
    peak_db: float = -96.0
    speech_ratio: float = 0.0


class VideoResult(BaseModel):
    """Complete result for one processed video."""
    video_path: str
    transcript: str = ""
    transcript_summary: TranscriptSummary | None = None
    clip_classification: ClipClassification | None = None
    clip_refinement: ClipRefinement | None = None
    audio_check: AudioCheckResult | None = None
    keywords: list[str] = Field(default_factory=list)
    folder_tags: list[str] = Field(default_factory=list)
    multicam_group_id: str | None = None
    frame_count: int = 0
    duration_seconds: float = 0.0
