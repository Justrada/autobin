#!/usr/bin/env python3
"""
Contextual I-Frame Extractor & Filter
=====================================

1. Extracts I-frames using ffmpeg (very fast).
2. Filters out redundant I-frames using visual similarity.
3. For each unique "guide" I-frame, replaces it with the frames N frames
   before and after it to provide better context for a Vision-Language Model.
4. Includes a specialized rule to shift the first frame if it is dark/black.
5. Auto-tunes the similarity threshold to hit a target frames-per-minute rate,
   adapting to any codec, resolution, bitrate, or content style.

Usage:
    python extract_iframes.py video.mp4 -o results/
    python extract_iframes.py video.mp4 --target-fpm 6
    python extract_iframes.py video.mp4 -t 0.93 -m histogram

Dependencies:
    pip install opencv-python-headless scikit-image numpy
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Extraction & Utilities
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


def extract_iframes(video_path: str, temp_dir: str, max_width: int = 640) -> list[str]:
    """
    Use ffmpeg to extract ONLY I-frames from *video_path* into *temp_dir*.
    Uses -frame_pts 1 so filenames contain the absolute frame index (0-based).
    Downscales to max_width for fast comparison (final frames are extracted
    at full resolution separately).
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

    print("[extract] Finding I-frames...")
    subprocess.run(cmd, capture_output=True, text=True)

    frames = sorted(glob.glob(os.path.join(temp_dir, "iframe_*.png")))
    return frames


def get_frame_num(path: str) -> int:
    """Extract the 6-digit frame number from 'iframe_000102.png' -> 102."""
    match = re.search(r"iframe_(\d+)\.png", os.path.basename(path))
    return int(match.group(1)) if match else 0


def is_dark(img: np.ndarray, threshold: float = 15.0) -> bool:
    """Check if an image is 'black or very dark' based on mean brightness."""
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def histogram_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compare two BGR images using normalized HSV color histograms (0-1)."""
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def ssim_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two images."""
    from skimage.metrics import structural_similarity as ssim
    if img_a.shape != img_b.shape:
        h, w = min(img_a.shape[0], img_b.shape[0]), min(img_a.shape[1], img_b.shape[1])
        img_a, img_b = cv2.resize(img_a, (w, h)), cv2.resize(img_b, (w, h))
    gray_a, gray_b = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    return ssim(gray_a, gray_b)


def phash_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Perceptual hash similarity. Resizes to 32x32, applies DCT, and compares
    the top-left 8x8 block as a 64-bit hash. Returns 0-1 where 1 = identical.
    """
    def _phash(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(resized)
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        return (dct_low > median).flatten()

    hash_a = _phash(img_a)
    hash_b = _phash(img_b)
    matching_bits = np.sum(hash_a == hash_b)
    return matching_bits / 64.0


METRICS = {"histogram": histogram_similarity, "ssim": ssim_similarity, "phash": phash_similarity}

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def filter_guide_paths(paths: list[str], threshold: float, metric_fn) -> list[str]:
    """
    Filter out redundant I-frame paths to leave only unique 'guide' frames.
    A new guide is kept when its similarity to the previous guide drops BELOW threshold.
    """
    if not paths:
        return []
    kept = [paths[0]]
    last_img = cv2.imread(paths[0])
    for path in paths[1:]:
        curr_img = cv2.imread(path)
        if curr_img is None:
            continue
        sim = metric_fn(last_img, curr_img)
        if sim < threshold:
            kept.append(path)
            last_img = curr_img
    return kept


def auto_tune_threshold(paths: list[str], metric_fn, target_guides: int,
                        lo: float = 0.60, hi: float = 0.995, iterations: int = 15) -> float:
    """
    Binary-search for the similarity threshold that produces closest to
    `target_guides` guide frames. Higher threshold = more guides kept.
    """
    best_t = (lo + hi) / 2
    for _ in range(iterations):
        mid = (lo + hi) / 2
        n = len(filter_guide_paths(paths, mid, metric_fn))
        if n < target_guides:
            lo = mid
        else:
            hi = mid
        best_t = mid
    return best_t


def determine_target_indices(guide_paths: list[str], total_frames: int, offset: int = 10) -> list[int]:
    """
    For each guide I-frame, replace it with the frame `offset` frames before
    and `offset` frames after it. The guide frame itself is NOT included —
    the before/after pair provides better context for a VLM.

    Special case: if the first guide frame is dark/black, only emit idx+offset.
    """
    targets = set()
    for i, path in enumerate(guide_paths):
        idx = get_frame_num(path)

        if i == 0:
            img = cv2.imread(path)
            if is_dark(img):
                print(f"[logic] First frame ({idx}) is dark — using +{offset} only.")
                targets.add(min(total_frames - 1, idx + offset))
                continue

        before = max(0, idx - offset)
        after = min(total_frames - 1, idx + offset)
        targets.add(before)
        targets.add(after)

    return sorted(targets)


def extract_final_frames(video_path: str, indices: list[int], output_dir: str):
    """Seek and save exact frames using OpenCV for high precision indices."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    saved_count = 0
    for i, target_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        dst = os.path.join(output_dir, f"frame_{i+1:03d}_orig{target_idx:06d}.jpg")
        cv2.imwrite(dst, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_count += 1

    cap.release()
    print(f"[output] Extracted {saved_count} contextual frames to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Extract contextual frames around video scene changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="By default, auto-tunes the threshold to produce ~4 frames/min.\n"
               "Use -t to override with a fixed threshold instead.",
    )
    p.add_argument("video", help="Input video file.")
    p.add_argument("-o", "--output-dir", default="output_frames", help="Output directory.")
    p.add_argument("-t", "--threshold", type=float, default=None,
                   help="Fixed similarity threshold (0–1). Disables auto-tuning.")
    p.add_argument("--target-fpm", type=float, default=4.0,
                   help="Target frames-per-minute when auto-tuning (default: 4).")
    p.add_argument("-m", "--metric", choices=list(METRICS.keys()), default="histogram",
                   help="Similarity metric: histogram, ssim, or phash.")
    p.add_argument("--offset", type=int, default=10, help="Frame offset before/after I-frame.")
    args = p.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found {args.video}")
        sys.exit(1)

    # Probe video
    info = get_video_info(args.video)
    if not info:
        print("Error: could not read video metadata.")
        sys.exit(1)

    duration_min = info["duration"] / 60.0
    print(f"[info] {info['codec'].upper()} {info['width']}x{info['height']}, "
          f"{info['bitrate']/1e6:.1f} Mbps, {info['fps']:.0f} fps, "
          f"{info['duration']:.1f}s ({duration_min:.1f} min)")

    # Clear previous output
    if os.path.isdir(args.output_dir):
        for f in glob.glob(os.path.join(args.output_dir, "frame_*.jpg")):
            os.remove(f)

    metric_fn = METRICS[args.metric]

    # Pipeline
    with tempfile.TemporaryDirectory(prefix="iframes_") as tmp:
        # Step A: Get I-frames
        iframe_paths = extract_iframes(args.video, tmp)
        if not iframe_paths:
            print("No I-frames found.")
            return

        iframes_per_min = len(iframe_paths) / duration_min
        print(f"[extract] {len(iframe_paths)} I-frames ({iframes_per_min:.1f}/min)")

        # Step B: Determine threshold (auto-tune or fixed)
        if args.threshold is not None:
            threshold = args.threshold
            print(f"[filter] Using fixed threshold: {threshold}")
        else:
            # Target: each guide produces ~2 context frames, so target_guides ≈ target_fpm * duration / 2
            target_guides = max(2, int(args.target_fpm * duration_min / 2))
            print(f"[auto] Tuning for ~{args.target_fpm} frames/min "
                  f"(~{target_guides} guide scenes for {duration_min:.1f} min)...")
            threshold = auto_tune_threshold(iframe_paths, metric_fn, target_guides)
            print(f"[auto] Selected threshold: {threshold:.4f}")

        # Step C: Filter
        guide_paths = filter_guide_paths(iframe_paths, threshold, metric_fn)
        actual_fpm = (len(guide_paths) * 2) / duration_min
        print(f"[filter] {len(guide_paths)} guide scenes ({actual_fpm:.1f} context frames/min)")

        # Step D: Determine contextual frame indices
        total_frames = info["total_frames"]
        target_indices = determine_target_indices(guide_paths, total_frames, args.offset)

        # Step E: Extract final frames
        extract_final_frames(args.video, target_indices, args.output_dir)

    print("─" * 50)
    print(f"  Metric / Threshold      : {args.metric} @ {threshold:.4f}")
    print(f"  Guide Scenes Found      : {len(guide_paths)}")
    print(f"  Contextual Frames Saved : {len(target_indices)}")
    print(f"  Frames / Minute         : {len(target_indices) / duration_min:.1f}")
    print(f"  Output Directory        : {os.path.abspath(args.output_dir)}")
    print("─" * 50)

if __name__ == "__main__":
    main()
