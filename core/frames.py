"""
Frame extraction and filtering pipeline.

Unified approach: binary subdivision seeking works for ANY codec.
Frames are grabbed in coverage-priority order (midpoint → quarters → eighths…),
similarity-filtered to drop near-duplicates, and returned at VLM resolution.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Callable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Video Probing
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    """Probe video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"
         and s.get("disposition", {}).get("attached_pic", 0) == 0),
        None,
    )
    fmt = data.get("format", {})

    if video_stream is None:
        return {}

    duration = float(video_stream.get("duration", 0) or fmt.get("duration", 0))
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 and float(fps_parts[1]) else 30.0

    return {
        "codec": video_stream.get("codec_name", "unknown"),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "bitrate": int(video_stream.get("bit_rate", 0) or fmt.get("bit_rate", 0)),
        "fps": fps,
        "duration": duration,
        "total_frames": int(video_stream.get("nb_frames", 0) or (duration * fps)),
        "profile": video_stream.get("profile", ""),
        "pix_fmt": video_stream.get("pix_fmt", ""),
    }


# ---------------------------------------------------------------------------
# Frame Sampling (binary subdivision)
# ---------------------------------------------------------------------------

def _build_subdivision_timestamps(duration: float) -> list[float]:
    """Build timestamps in binary subdivision order: midpoint, quarters, eighths…

    Returns timestamps ordered by coverage priority — the first N timestamps
    always give the best possible spatial coverage across the clip.
    """
    queue: collections.deque[tuple[float, float]] = collections.deque()
    queue.append((0.0, duration))

    # Endpoints first (always most useful)
    seek_order: list[float] = [0.0, max(0.0, duration - 0.5)]

    while queue:
        lo, hi = queue.popleft()
        mid = (lo + hi) / 2.0
        if (hi - lo) < 1.0:
            continue
        seek_order.append(mid)
        queue.append((lo, mid))
        queue.append((mid, hi))

    # Deduplicate while preserving order
    seen: set[float] = set()
    unique: list[float] = []
    for ts in seek_order:
        rounded = round(ts, 2)
        if rounded not in seen:
            seen.add(rounded)
            unique.append(rounded)
    return unique


def sample_frames(video_path: str, output_dir: str, duration: float,
                  max_frames: int = 16, time_budget: float = 10.0,
                  max_width: int = 640,
                  log: Callable[[str], None] | None = None) -> list[str]:
    """Sample frames via binary subdivision seeking.

    Grabs frames in coverage-priority order until either *max_frames* or
    *time_budget* (seconds) is reached, whichever comes first. Works for
    any codec — no I-frame detection needed.

    Returns paths sorted chronologically.
    """
    _log = log or (lambda m: None)
    os.makedirs(output_dir, exist_ok=True)
    deadline = time.time() + time_budget

    _log(f"[extract] Sampling frames (max {max_frames}, budget {time_budget:.0f}s)...")

    timestamps = _build_subdivision_timestamps(duration)
    grabbed: dict[float, str] = {}
    count = 0

    for ts in timestamps:
        if count >= max_frames:
            _log(f"[extract] Frame cap ({max_frames}) reached after {count} frames")
            break
        if time.time() >= deadline:
            _log(f"[extract] Time budget reached after {count} frames")
            break

        out_path = os.path.join(output_dir, f"frame_{count:04d}.png")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-vf", f"scale='min({max_width},iw)':-2",
            "-frames:v", "1",
            "-q:v", "2",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            grabbed[ts] = out_path
            count += 1

    # Sort chronologically and rename so numbers are sequential
    sorted_ts = sorted(grabbed.keys())
    frames: list[str] = []
    for i, ts in enumerate(sorted_ts):
        old_path = grabbed[ts]
        new_path = os.path.join(output_dir, f"sampled_{i:04d}.png")
        os.rename(old_path, new_path)
        frames.append(new_path)

    elapsed = time_budget - max(0, deadline - time.time())
    _log(f"[extract] Sampled {len(frames)} frames in {elapsed:.1f}s")
    return frames


def sample_fast_frames(video_path: str, output_dir: str, duration: float,
                       n_frames: int = 5, max_width: int = 640,
                       log: Callable[[str], None] | None = None) -> list[str]:
    """Fast mode: grab exactly n_frames equidistant frames. No filtering."""
    _log = log or (lambda m: None)
    _log(f"[extract] Fast mode — grabbing {n_frames} equidistant frames...")

    os.makedirs(output_dir, exist_ok=True)
    frames = []

    margin = min(1.0, duration * 0.05)
    usable = duration - 2 * margin
    if usable <= 0:
        usable = duration
        margin = 0

    for i in range(n_frames):
        if n_frames == 1:
            ts = duration / 2
        else:
            ts = margin + (usable * i / (n_frames - 1))

        out_path = os.path.join(output_dir, f"fast_{i:03d}.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-vf", f"scale='min({max_width},iw)':-2",
            "-frames:v", "1",
            "-q:v", "2",
            out_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(out_path):
            frames.append(out_path)

    _log(f"[extract] Fast mode: got {len(frames)} frames")
    return frames


# ---------------------------------------------------------------------------
# Similarity Metrics
# ---------------------------------------------------------------------------

def histogram_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compare using normalized HSV color histograms (0-1)."""
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def ssim_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Structural Similarity Index between two images."""
    from skimage.metrics import structural_similarity as ssim
    if img_a.shape != img_b.shape:
        h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
        img_a, img_b = cv2.resize(img_a, (w, h)), cv2.resize(img_b, (w, h))
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    return ssim(gray_a, gray_b)


def phash_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Perceptual hash similarity via DCT (0-1)."""
    def _phash(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(resized)
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        return (dct_low > median).flatten()

    hash_a = _phash(img_a)
    hash_b = _phash(img_b)
    return np.sum(hash_a == hash_b) / 64.0


METRICS = {"histogram": histogram_similarity, "ssim": ssim_similarity, "phash": phash_similarity}


# ---------------------------------------------------------------------------
# Similarity Filtering & Auto-Tuning
# ---------------------------------------------------------------------------

def _precompute_similarities(paths: list[str], metric_fn,
                             log: Callable[[str], None] | None = None
                             ) -> list[tuple[str, float]]:
    """Read each image once, compute similarity to predecessor.

    Returns [(path, similarity_to_previous), ...] — first entry has sim=0.0.
    """
    if not paths:
        return []
    result: list[tuple[str, float]] = [(paths[0], 0.0)]
    prev_img = cv2.imread(paths[0])
    for path in paths[1:]:
        curr_img = cv2.imread(path)
        if curr_img is None:
            continue
        sim = metric_fn(prev_img, curr_img)
        result.append((path, sim))
        prev_img = curr_img
    if log:
        log(f"[auto] Pre-computed {len(result)} pairwise similarities")
    return result


def _filter_from_scores(scores: list[tuple[str, float]], threshold: float) -> list[str]:
    """Filter frames using pre-computed scores.

    Keeps a frame when any frame in the run since the last kept frame had
    similarity below threshold (indicating a scene change occurred).
    """
    if not scores:
        return []
    kept = [scores[0][0]]
    last_kept_idx = 0

    for i in range(1, len(scores)):
        min_sim_in_run = min(scores[j][1] for j in range(last_kept_idx + 1, i + 1))
        if min_sim_in_run < threshold:
            kept.append(scores[i][0])
            last_kept_idx = i

    return kept


def _filter_guide_frames(scores: list[tuple[str, float]], threshold: float,
                         log: Callable[[str], None] | None = None) -> list[str]:
    """Filter using pre-computed scores and log result."""
    kept = _filter_from_scores(scores, threshold)
    if log:
        log(f"[filter] {len(kept)} unique frames after similarity filter "
            f"(from {len(scores)}, threshold={threshold:.4f})")
    return kept


def auto_tune_threshold(scores: list[tuple[str, float]], target_guides: int,
                        lo: float = 0.60, hi: float = 0.995, iterations: int = 15,
                        log: Callable[[str], None] | None = None) -> float:
    """Binary-search for threshold that produces closest to target_guides.

    Uses pre-computed scores (no re-reading images).
    """
    if not scores:
        return (lo + hi) / 2

    best_t = (lo + hi) / 2
    for _ in range(iterations):
        mid = (lo + hi) / 2
        n = len(_filter_from_scores(scores, mid))
        if n < target_guides:
            lo = mid
        else:
            hi = mid
        best_t = mid

    if log:
        log(f"[auto] Selected threshold: {best_t:.4f}")
    return best_t


# ---------------------------------------------------------------------------
# High-Level Pipeline
# ---------------------------------------------------------------------------

def run_frame_pipeline(video_path: str, output_dir: str,
                       threshold: float | None = None,
                       target_fpm: float = 4.0,
                       max_frames: int = 16,
                       time_budget: float = 10.0,
                       metric: str = "histogram",
                       log: Callable[[str], None] | None = None,
                       progress: Callable[[int, int], None] | None = None,
                       **_kwargs) -> list[str]:
    """Unified frame extraction pipeline.

    1. Probe video metadata
    2. Sample frames via binary subdivision (any codec)
    3. Similarity-filter to drop near-duplicates
    4. Copy final frames to output_dir

    Parameters
    ----------
    max_frames : int
        Hard cap on frames to sample (default 16).
    time_budget : float
        Max seconds to spend sampling (default 10s).
    threshold : float or None
        Similarity threshold. None → auto-tune to target_fpm.
    target_fpm : float
        Target frames-per-minute for auto-tuning (default 4.0).
    metric : str
        Similarity metric: histogram (fast), ssim, or phash.

    Returns list of saved frame paths.
    """
    _log = log or (lambda msg: None)

    info = get_video_info(video_path)
    if not info:
        _log("[error] Could not read video metadata.")
        return []

    duration = info["duration"]
    duration_min = duration / 60.0
    _log(f"[info] {info['codec'].upper()} {info['width']}x{info['height']}, "
         f"{info['bitrate']/1e6:.1f} Mbps, {info['fps']:.0f} fps, "
         f"{duration:.1f}s ({duration_min:.1f} min)")

    if duration <= 0:
        _log("[error] Video has zero duration.")
        return []

    metric_fn = METRICS.get(metric, histogram_similarity)

    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        # Step 1: Sample frames via binary subdivision
        sampled = sample_frames(
            video_path, tmp, duration,
            max_frames=max_frames, time_budget=time_budget,
            log=log,
        )

        if not sampled:
            _log("[error] No frames sampled.")
            return []

        # Step 2: Compute pairwise similarities (once)
        scores = _precompute_similarities(sampled, metric_fn, log=log)

        # Step 3: Auto-tune or use fixed threshold, then filter
        if threshold is None:
            target_guides = max(2, int(target_fpm * duration_min / 2))
            _log(f"[auto] Tuning for ~{target_fpm} frames/min (~{target_guides} guides)...")
            threshold = auto_tune_threshold(scores, target_guides, log=log)

        guide_paths = _filter_guide_frames(scores, threshold, log=log)

        # Step 4: Copy filtered frames to output dir
        os.makedirs(output_dir, exist_ok=True)
        saved: list[str] = []
        for i, path in enumerate(guide_paths):
            dst = os.path.join(output_dir, f"frame_{i+1:03d}.jpg")
            shutil.copy2(path, dst)
            saved.append(dst)
            if progress:
                progress(i + 1, len(guide_paths))

    _log(f"[done] {len(saved)} frames saved ({len(saved)/max(duration_min, 0.01):.1f}/min)")
    return saved
