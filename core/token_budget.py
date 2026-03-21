"""Image token estimation and batch planning for VLM context windows."""

from __future__ import annotations

import math

from core.schemas import LLMSettings


def estimate_image_tokens(backend: str, width: int, height: int) -> int:
    """
    Estimate tokens consumed by one image for a given backend.

    Qwen2.5-VL: tokens ≈ ceil(h/28) * ceil(w/28) * 1.5
    OpenAI low-detail: 85 tokens flat
    Anthropic: ≈ (w * h) / 750
    """
    if backend == "ollama":
        # Qwen2.5-VL dynamic resolution formula
        return int(math.ceil(height / 28) * math.ceil(width / 28) * 1.5)
    elif backend == "openai":
        return 85  # low-detail mode
    elif backend == "anthropic":
        return max(1, int((width * height) / 750))
    else:
        # Conservative fallback
        return int(math.ceil(height / 28) * math.ceil(width / 28) * 1.5)


def plan_batches(frame_paths: list[str], settings: LLMSettings,
                 prompt_tokens: int = 800) -> list[list[str]]:
    """
    Split frame paths into batches that fit within the context window.
    Frames are assumed to be downscaled to settings.vlm_input_resolution height
    with 4:3 aspect ratio for estimation.
    """
    h = settings.vlm_input_resolution
    w = int(h * 4 / 3)

    tokens_per_image = estimate_image_tokens(settings.backend, w, h)
    # Reserve space for system prompt, user prompt, and response
    reserved = prompt_tokens + 1500  # prompt + expected response
    available = settings.context_window - reserved

    if available <= 0:
        # Context window too small — one image at a time
        max_per_batch = 1
    else:
        max_per_batch = max(1, available // tokens_per_image)

    batches = []
    for i in range(0, len(frame_paths), max_per_batch):
        batches.append(frame_paths[i:i + max_per_batch])
    return batches
