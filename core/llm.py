"""
LLM client abstraction for Ollama (default), OpenAI, and Anthropic.
All clients produce structured output via JSON schema — no tool calling.
"""

from __future__ import annotations

import base64
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Callable

import cv2
import requests
from pydantic import BaseModel

from core.schemas import (
    ClipClassification,
    ClipRefinement,
    ImageKeywordBatch,
    LLMSettings,
    TranscriptSummary,
)
from core.token_budget import plan_batches

# Default max images per vision request (overridden by LLMSettings.max_images_per_batch)
DEFAULT_MAX_IMAGES_PER_REQUEST = 4


def _strip_schema_meta(schema_dict: dict) -> dict:
    """Remove Pydantic metadata fields that confuse local LLMs."""
    cleaned = {}
    for k, v in schema_dict.items():
        if k in ("title", "description"):
            continue
        if k == "properties":
            cleaned[k] = {
                pk: {fk: fv for fk, fv in pv.items() if fk != "title"}
                for pk, pv in v.items()
            }
        else:
            cleaned[k] = v
    return cleaned


def _clean_and_parse(content: str, schema: type[BaseModel]) -> BaseModel:
    """
    Try to parse LLM output as JSON. Handles common issues:
    - Markdown code fences
    - Extra wrapping keys (e.g. {"description": {actual data}})
    - Non-JSON text mixed in
    """
    content = content.strip()

    # Strip markdown code fences
    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)
    content = content.strip()

    # Try direct parse
    try:
        return schema.model_validate_json(content)
    except Exception:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        json_str = match.group()

        # Try direct
        try:
            return schema.model_validate_json(json_str)
        except Exception:
            pass

        # Try unwrapping: if the model wrapped in an extra key, dig one level in
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and len(parsed) == 1:
                inner = next(iter(parsed.values()))
                if isinstance(inner, dict):
                    return schema.model_validate(inner)
            # Also try the parsed dict directly with model_validate (not json)
            return schema.model_validate(parsed)
        except Exception:
            pass

    raise ValueError(f"Could not parse LLM response as {schema.__name__}: {content[:200]}")


class LLMClient(ABC):
    @abstractmethod
    def complete_text(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        ...

    @abstractmethod
    def complete_vision(self, prompt: str, image_paths: list[str],
                        schema: type[BaseModel]) -> BaseModel:
        ...


# ---------------------------------------------------------------------------
# Ollama (default — local)
# ---------------------------------------------------------------------------

class OllamaClient(LLMClient):
    def __init__(self, settings: LLMSettings):
        self.model = settings.model
        self.base_url = settings.base_url.rstrip("/")
        self.vlm_resolution = settings.vlm_input_resolution

    def _encode_image(self, path: str) -> str:
        """Read and optionally downscale image, return base64."""
        img = cv2.imread(path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        target_h = self.vlm_resolution
        if h > target_h:
            scale = target_h / h
            img = cv2.resize(img, (int(w * scale), target_h))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def complete_text(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        clean_schema = _strip_schema_meta(schema.model_json_schema())
        schema_json = json.dumps(clean_schema)
        system_msg = (
            "You are a JSON-only assistant. Respond with ONLY a valid JSON object, "
            "no markdown, no explanation, no extra text.\n"
            f"Required JSON schema: {schema_json}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "think": False,
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return _clean_and_parse(content, schema)

    def complete_vision(self, prompt: str, image_paths: list[str],
                        schema: type[BaseModel]) -> BaseModel:
        images = [self._encode_image(p) for p in image_paths if p]
        images = [img for img in images if img]

        clean_schema = _strip_schema_meta(schema.model_json_schema())
        schema_json = json.dumps(clean_schema)
        system_msg = (
            "You are a JSON-only assistant. Respond with ONLY a valid JSON object, "
            "no markdown, no explanation, no extra text.\n"
            f"Required JSON schema: {schema_json}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt, "images": images},
            ],
            "format": "json",
            "stream": False,
            "think": False,
        }
        resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return _clean_and_parse(content, schema)


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    def __init__(self, settings: LLMSettings):
        self.model = settings.model
        self.api_key = settings.api_key
        self.vlm_resolution = settings.vlm_input_resolution

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _encode_image(self, path: str) -> str:
        img = cv2.imread(path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        target_h = self.vlm_resolution
        if h > target_h:
            scale = target_h / h
            img = cv2.resize(img, (int(w * scale), target_h))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def complete_text(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _clean_and_parse(content, schema)

    def complete_vision(self, prompt: str, image_paths: list[str],
                        schema: type[BaseModel]) -> BaseModel:
        content_parts = [{"type": "text", "text": prompt}]
        for path in image_paths:
            b64 = self._encode_image(path)
            if b64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content_parts}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _clean_and_parse(content, schema)


# ---------------------------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    def __init__(self, settings: LLMSettings):
        self.model = settings.model
        self.api_key = settings.api_key
        self.vlm_resolution = settings.vlm_input_resolution

    def _headers(self):
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _encode_image(self, path: str) -> str:
        img = cv2.imread(path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        target_h = self.vlm_resolution
        if h > target_h:
            scale = target_h / h
            img = cv2.resize(img, (int(w * scale), target_h))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def complete_text(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        system = (f"Respond ONLY with valid JSON matching this schema:\n{schema_str}\n"
                  "No other text, just JSON.")
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post("https://api.anthropic.com/v1/messages",
                             headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        return _clean_and_parse(content, schema)

    def complete_vision(self, prompt: str, image_paths: list[str],
                        schema: type[BaseModel]) -> BaseModel:
        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        system = (f"Respond ONLY with valid JSON matching this schema:\n{schema_str}\n"
                  "No other text, just JSON.")

        content_parts = []
        for path in image_paths:
            b64 = self._encode_image(path)
            if b64:
                content_parts.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
        content_parts.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": content_parts}],
        }
        resp = requests.post("https://api.anthropic.com/v1/messages",
                             headers=self._headers(), json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["content"][0]["text"]
        return _clean_and_parse(content, schema)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(settings: LLMSettings) -> LLMClient:
    if settings.backend == "ollama":
        return OllamaClient(settings)
    elif settings.backend == "openai":
        return OpenAIClient(settings)
    elif settings.backend == "anthropic":
        return AnthropicClient(settings)
    else:
        return OllamaClient(settings)


# ---------------------------------------------------------------------------
# High-Level LLM Pipeline
# ---------------------------------------------------------------------------

TRANSCRIPT_PROMPT = (
    "Analyze this video transcript and return a JSON object with exactly these fields:\n"
    '- "title": a short descriptive title\n'
    '- "summary": a 2-4 sentence summary\n'
    '- "topics": an array of main topics discussed\n\n'
    "Transcript:\n{transcript}"
)

IMAGE_KEYWORD_PROMPT = (
    "Analyze these {count} video frames for DaVinci Resolve clip logging.\n"
    "Return a JSON object with exactly these fields:\n"
    '- "keywords": array of descriptive keywords (shot type, setting, lighting, '
    "subjects, actions, mood — e.g. \"interview\", \"wide shot\", \"outdoors\")\n"
    '- "scene_descriptions": array with one short description per image\n\n'
    "Focus on visual content only."
)

IMAGE_KEYWORD_PROMPT_WITH_CONTEXT = (
    "You are analyzing frames from a video: \"{title}\"\n"
    "Context: {summary}\n\n"
    "Analyze these {count} video frames for DaVinci Resolve clip logging.\n"
    "Return a JSON object with exactly these fields:\n"
    '- "keywords": array of descriptive keywords (shot type, setting, lighting, '
    "subjects, actions, mood — e.g. \"interview\", \"wide shot\", \"outdoors\")\n"
    '- "scene_descriptions": array with one short description per image\n\n'
    "Focus on visual content. Use the context to identify subjects and topics accurately."
)


CLASSIFY_PROMPT = (
    "These 4 frames are evenly sampled from a single video clip.\n"
    "Classify the clip by answering each field in the JSON schema.\n"
    "If you cannot determine a field, answer \"N/A\" (except is_talking_head which must be true or false).\n\n"
    "For shot_type pick ONE: close up, medium close up, medium shot, wide shot, extreme wide shot, or N/A.\n"
    "For camera_angle pick ONE: eye level, high angle, low angle, dutch angle, overhead, or N/A.\n"
    "For camera_movement pick ONE: static, pan, tilt, dolly, handheld, tracking, or N/A.\n"
    "For lighting pick ONE: natural, studio, mixed, low key, high key, or N/A.\n"
    "For location pick ONE: interior, exterior, or N/A.\n"
    "For subject: briefly describe the main subject (e.g. 'man at desk', 'city skyline'), or N/A.\n"
    "For roll_type pick ONE: a-roll (interviews, presentations, direct address), "
    "b-roll (cutaways, scenery, action without dialogue), or N/A.\n"
    "For is_talking_head: true if a person is speaking to camera in a mostly static shot, false otherwise."
)

REFINE_PROMPT = (
    "You are refining metadata for a video clip. The visual classification says it is "
    "\"{roll_type}\" with subject \"{subject}\".\n\n"
    "Now read this transcript from the same clip and answer:\n"
    "- subject_name: If anyone introduced themselves or was addressed by name, "
    "what is their name? If unclear, say \"unknown\".\n"
    "- is_interview: Is this an actual interview (someone answering questions / "
    "speaking directly about a topic to camera)? Or is it just background "
    "conversation/chatter? Answer true for interview, false otherwise.\n"
    "- refined_subject: A better description of who/what is in the clip, "
    "now that you have transcript context. Keep it short (e.g. "
    "\"Wayman Earl III discussing Stranger Comics\").\n"
    "- content_tags: 3-8 topical tags from the transcript content "
    "(e.g. \"comics\", \"cosplay\", \"Niobe\", \"fan community\").\n\n"
    "Transcript:\n{transcript}"
)


def sample_classification_frames(video_path: str, vlm_resolution: int = 480,
                                  log: Callable[[str], None] | None = None) -> list[str]:
    """
    Sample 4 evenly-spaced frames from a video for quick classification.
    Saves them as temp files and returns paths.
    """
    import tempfile
    _log = log or (lambda m: None)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 4:
        cap.release()
        return []

    # Sample from 4 equal segments (center of each quarter)
    positions = [int(total_frames * (i * 2 + 1) / 8) for i in range(4)]
    tmp_dir = tempfile.mkdtemp(prefix="classify_")
    paths = []

    for i, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        # Downscale for VLM
        h, w = frame.shape[:2]
        if h > vlm_resolution:
            scale = vlm_resolution / h
            frame = cv2.resize(frame, (int(w * scale), vlm_resolution))
        path = os.path.join(tmp_dir, f"classify_{i}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)

    cap.release()
    _log(f"[classify] Sampled {len(paths)} frames for classification")
    return paths


def classify_clip(video_path: str, settings: LLMSettings,
                  log: Callable[[str], None] | None = None,
                  keep_frames: bool = False) -> tuple[ClipClassification | None, list[str]]:
    """
    Quick clip classification using 4 evenly-spaced frames.
    Returns (classification, frame_paths).
    If keep_frames=True, temp frame files are NOT deleted (caller must clean up).
    """
    _log = log or (lambda m: None)
    _log("[classify] Running quick clip classification...")

    frame_paths = sample_classification_frames(video_path, settings.vlm_input_resolution, log=log)
    if not frame_paths:
        _log("[classify] Could not sample frames")
        return None, []

    client = get_client(settings)
    try:
        result = client.complete_vision(CLASSIFY_PROMPT, frame_paths, ClipClassification)
        _log(f"[classify] {result.roll_type.upper()} | {result.shot_type} | "
             f"{result.subject} | talking_head={result.is_talking_head}")
        if not keep_frames:
            import shutil
            tmp_dir = os.path.dirname(frame_paths[0])
            shutil.rmtree(tmp_dir, ignore_errors=True)
            frame_paths = []
        return result, frame_paths
    except Exception as e:
        _log(f"[classify] Classification failed: {e}")
        if not keep_frames:
            import shutil
            if frame_paths:
                tmp_dir = os.path.dirname(frame_paths[0])
                shutil.rmtree(tmp_dir, ignore_errors=True)
            frame_paths = []
        return None, frame_paths


def refine_classification(transcript: str, classification: ClipClassification,
                          settings: LLMSettings,
                          log: Callable[[str], None] | None = None) -> ClipRefinement | None:
    """
    Use the transcript to refine classification — extract subject name,
    determine if it's a real interview, and add content tags.
    """
    _log = log or (lambda m: None)

    if not transcript or len(transcript.strip()) < 20:
        _log("[refine] Transcript too short for refinement")
        return None

    _log("[refine] Refining classification with transcript context...")
    client = get_client(settings)
    prompt = REFINE_PROMPT.format(
        roll_type=classification.roll_type,
        subject=classification.subject,
        transcript=transcript[:6000],
    )
    try:
        result = client.complete_text(prompt, ClipRefinement)
        _log(f"[refine] Subject: {result.subject_name} | "
             f"Interview: {result.is_interview} | "
             f"Tags: {', '.join(result.content_tags)}")
        return result
    except Exception as e:
        _log(f"[refine] Refinement failed: {e}")
        return None


def run_llm_pipeline(transcript: str, frame_paths: list[str],
                     settings: LLMSettings,
                     log: Callable[[str], None] | None = None,
                     progress: Callable[[int, int], None] | None = None) -> tuple[TranscriptSummary | None, list[str]]:
    """
    Three-step LLM pipeline:
    1. Summarize transcript (fresh context)
    2. Process image batches for keywords (fresh context per batch)
    3. Deduplicate keywords

    Returns (TranscriptSummary, deduplicated_keywords).
    """
    _log = log or (lambda m: None)
    client = get_client(settings)

    # Step 1: Summarize transcript
    summary = None
    if transcript.strip():
        _log("[llm] Step 1: Summarizing transcript...")
        try:
            summary = client.complete_text(
                TRANSCRIPT_PROMPT.format(transcript=transcript[:8000]),
                TranscriptSummary,
            )
            _log(f"[llm] Summary: {summary.title}")
        except Exception as e:
            _log(f"[llm] Transcript summary failed: {e}")

    # Step 2: Process image batches
    all_keywords: list[str] = []
    max_per_batch = settings.max_images_per_batch
    if frame_paths:
        batches = []
        for i in range(0, len(frame_paths), max_per_batch):
            batches.append(frame_paths[i:i + max_per_batch])

        _log(f"[llm] Step 2: Processing {len(frame_paths)} frames in {len(batches)} batches "
             f"({max_per_batch} images/batch)...")

        for i, batch in enumerate(batches):
            if progress:
                progress(i + 1, len(batches))
            _log(f"[llm] Batch {i+1}/{len(batches)} ({len(batch)} images)...")
            try:
                if summary:
                    prompt = IMAGE_KEYWORD_PROMPT_WITH_CONTEXT.format(
                        title=summary.title, summary=summary.summary, count=len(batch))
                else:
                    prompt = IMAGE_KEYWORD_PROMPT.format(count=len(batch))
                result = client.complete_vision(
                    prompt,
                    batch,
                    ImageKeywordBatch,
                )
                all_keywords.extend(result.keywords)
                _log(f"[llm] Batch {i+1}: {len(result.keywords)} keywords")
            except Exception as e:
                _log(f"[llm] Batch {i+1} failed: {e}")
    else:
        _log("[llm] No frames to process for keywords.")

    # Step 3: Deduplicate keywords
    seen = set()
    unique_keywords = []
    for kw in all_keywords:
        kw_lower = kw.strip().lower()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw.strip())

    _log(f"[llm] Done. {len(unique_keywords)} unique keywords from {len(all_keywords)} total.")
    return summary, unique_keywords
