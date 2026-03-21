"""
Frame extraction and filtering pipeline.
Refactored from extract_iframes.py with progress callbacks for GUI integration.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import tempfile
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
# I-Frame Extraction
# ---------------------------------------------------------------------------

def extract_iframes(video_path: str, temp_dir: str, max_width: int = 640,
                    log: Callable[[str], None] | None = None) -> list[str]:
    """
    Extract I-frames from video into temp_dir, downscaled to max_width for
    fast comparison. Returns sorted list of paths.
    """
    output_pattern = os.path.join(temp_dir, "iframe_%06d.png")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-skip_frame", "nokey",
        "-i", video_path,
        "-vsync", "vfr",
        "-frame_pts", "1",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-q:v", "2",
        output_pattern,
    ]

    if log:
        log("[extract] Finding I-frames...")
    subprocess.run(cmd, capture_output=True, text=True)

    frames = sorted(glob.glob(os.path.join(temp_dir, "iframe_*.png")))
    if log:
        log(f"[extract] Found {len(frames)} I-frames")
    return frames


def get_frame_num(path: str) -> int:
    """Extract frame number from 'iframe_000102.png' -> 102."""
    match = re.search(r"iframe_(\d+)\.png", os.path.basename(path))
    return int(match.group(1)) if match else 0


def is_dark(img: np.ndarray, threshold: float = 15.0) -> bool:
    """Check if an image is dark/black based on mean brightness."""
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold


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
# Filtering & Auto-Tuning
# ---------------------------------------------------------------------------

def _precompute_similarities(paths: list[str], metric_fn,
                             log: Callable[[str], None] | None = None) -> list[tuple[str, float]]:
    """
    Read each image once, compute similarity to its predecessor.
    Returns [(path, similarity_to_previous), ...] where the first entry
    has similarity=0.0 (always kept).
    """
    if not paths:
        return []
    result: list[tuple[str, float]] = [(paths[0], 0.0)]
    prev_img = cv2.imread(paths[0])
    for i, path in enumerate(paths[1:], 1):
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
    """
    Filter guide frames using pre-computed similarity scores.
    A frame is kept when its similarity to the *last kept frame* is below threshold.

    Because similarities are sequential (each vs its predecessor), we need to
    track accumulated similarity — when consecutive frames are all similar,
    skipping one means the next comparison is against a frame further back.
    We re-simulate the keep/skip logic using the original sequential scores.
    """
    if not scores:
        return []
    # We always keep the first frame
    kept = [scores[0][0]]
    last_kept_idx = 0

    for i in range(1, len(scores)):
        # Compute similarity between current frame and last kept frame.
        # If frames between last_kept and i were all skipped, we can't use
        # the precomputed score directly (it's vs the immediate predecessor).
        # However, for the common case the sequential score is a good proxy:
        # if sim(i-1, i) is low, sim(last_kept, i) is also likely low.
        # For perfect accuracy we'd need all-pairs, but that's O(n²).
        #
        # Optimization: use the minimum similarity in the run since last kept.
        # If any frame in the run had low similarity to its predecessor,
        # there was a scene change, so current frame differs from last kept.
        min_sim_in_run = min(scores[j][1] for j in range(last_kept_idx + 1, i + 1))

        if min_sim_in_run < threshold:
            kept.append(scores[i][0])
            last_kept_idx = i

    return kept


def filter_guide_paths(paths: list[str], threshold: float, metric_fn) -> list[str]:
    """Keep only unique guide frames where similarity to previous drops below threshold."""
    if not paths:
        return []
    kept = [paths[0]]
    last_img = cv2.imread(paths[0])
    for path in paths[1:]:
        curr_img = cv2.imread(path)
        if curr_img is None:
            continue
        if metric_fn(last_img, curr_img) < threshold:
            kept.append(path)
            last_img = curr_img
    return kept


def auto_tune_threshold(paths: list[str], metric_fn, target_guides: int,
                        lo: float = 0.60, hi: float = 0.995, iterations: int = 15,
                        log: Callable[[str], None] | None = None) -> float:
    """
    Binary-search for threshold that produces closest to target_guides.
    Pre-computes all pairwise similarities once, then searches over cached scores.
    """
    scores = _precompute_similarities(paths, metric_fn, log=log)
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
    return best_t


# ---------------------------------------------------------------------------
# Target Index Computation
# ---------------------------------------------------------------------------

def determine_target_indices(guide_paths: list[str], total_frames: int,
                             offset: int = 10,
                             log: Callable[[str], None] | None = None) -> list[int]:
    """
    For each guide I-frame, produce the frame offset before and after it.
    First frame: if dark, only emit +offset.
    """
    targets = set()
    for i, path in enumerate(guide_paths):
        idx = get_frame_num(path)
        if i == 0:
            img = cv2.imread(path)
            if is_dark(img):
                if log:
                    log(f"[logic] First frame ({idx}) is dark — using +{offset} only.")
                targets.add(min(total_frames - 1, idx + offset))
                continue
        targets.add(max(0, idx - offset))
        targets.add(min(total_frames - 1, idx + offset))
    return sorted(targets)


# ---------------------------------------------------------------------------
# Final Frame Extraction
# ---------------------------------------------------------------------------

def extract_final_frames(video_path: str, indices: list[int], output_dir: str,
                         log: Callable[[str], None] | None = None,
                         progress: Callable[[int, int], None] | None = None) -> list[str]:
    """Seek and save exact frames at full resolution. Returns list of saved paths."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    saved = []
    total = len(indices)

    for i, target_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        dst = os.path.join(output_dir, f"frame_{i+1:03d}_orig{target_idx:06d}.jpg")
        cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(dst)
        if progress:
            progress(i + 1, total)

    cap.release()
    if log:
        log(f"[output] Extracted {len(saved)} contextual frames to {output_dir}/")
    return saved


# ---------------------------------------------------------------------------
# High-Level Pipeline
# ---------------------------------------------------------------------------

def run_frame_pipeline(video_path: str, output_dir: str,
                       threshold: float | None = None,
                       target_fpm: float = 4.0,
                       metric: str = "histogram",
                       offset: int = 10,
                       log: Callable[[str], None] | None = None,
                       progress: Callable[[int, int], None] | None = None) -> list[str]:
    """
    Full frame extraction pipeline. Returns list of saved frame paths.
    If threshold is None, auto-tunes to target_fpm.
    """
    _log = log or (lambda msg: None)

    info = get_video_info(video_path)
    if not info:
        _log("[error] Could not read video metadata.")
        return []

    duration_min = info["duration"] / 60.0
    _log(f"[info] {info['codec'].upper()} {info['width']}x{info['height']}, "
         f"{info['bitrate']/1e6:.1f} Mbps, {info['fps']:.0f} fps, "
         f"{info['duration']:.1f}s ({duration_min:.1f} min)")

    metric_fn = METRICS[metric]

    with tempfile.TemporaryDirectory(prefix="iframes_") as tmp:
        iframe_paths = extract_iframes(video_path, tmp, log=log)
        if not iframe_paths:
            _log("[error] No I-frames found.")
            return []

        # Auto-tune or use fixed threshold
        if threshold is None:
            target_guides = max(2, int(target_fpm * duration_min / 2))
            _log(f"[auto] Tuning for ~{target_fpm} frames/min (~{target_guides} guides)...")
            threshold = auto_tune_threshold(iframe_paths, metric_fn, target_guides, log=log)
            _log(f"[auto] Selected threshold: {threshold:.4f}")
        else:
            _log(f"[filter] Using fixed threshold: {threshold}")

        guide_paths = filter_guide_paths(iframe_paths, threshold, metric_fn)
        _log(f"[filter] {len(guide_paths)} guide scenes identified.")

        target_indices = determine_target_indices(
            guide_paths, info["total_frames"], offset, log=log
        )

        saved = extract_final_frames(video_path, target_indices, output_dir,
                                     log=log, progress=progress)

    _log(f"[done] {len(saved)} contextual frames saved ({len(saved)/duration_min:.1f}/min)")
    return saved
